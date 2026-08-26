from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import piq
import torch
import yaml
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader

from cleanfid import fid
from diffusers import AutoencoderKL

from .train import CocoCaptions, make_lpips, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--include-ms-ssim", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-path", default=None)
    return parser.parse_args()


def summarize(values: list[np.ndarray]) -> dict[str, float]:
    packed = np.concatenate(values).astype(np.float64)
    return {
        "mean": float(packed.mean()),
        "std": float(packed.std()),
        "p50": float(np.quantile(packed, 0.5)),
        "p90": float(np.quantile(packed, 0.9)),
    }


def save_png(image: torch.Tensor, path: Path) -> None:
    array = image.float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path, compress_level=0)


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray, eps: float = 1e-6) -> float:
    mean_a, mean_b = features_a.mean(0), features_b.mean(0)
    covariance_a = np.cov(features_a, rowvar=False)
    covariance_b = np.cov(features_b, rowvar=False)
    difference = mean_a - mean_b
    covariance_mean = linalg.sqrtm(covariance_a.dot(covariance_b))
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(covariance_a.shape[0]) * eps
        covariance_mean = linalg.sqrtm((covariance_a + offset).dot(covariance_b + offset))
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diagonal(covariance_mean).imag, 0, atol=1e-3):
            raise ValueError(f"FID covariance has imaginary component {np.abs(covariance_mean.imag).max()}")
        covariance_mean = covariance_mean.real
    return float(
        difference.dot(difference)
        + np.trace(covariance_a)
        + np.trace(covariance_b)
        - 2.0 * np.trace(covariance_mean)
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    seed_everything(int(config["seed"]))
    device = torch.device(args.device)
    model_path = args.model_path or config["model"]["checkpoint"]
    vae = AutoencoderKL.from_pretrained(model_path, local_files_only=True).eval().to(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        vae.load_state_dict(payload["vae"], strict=True)
    vae.requires_grad_(False)
    lpips_model = make_lpips(device)
    if lpips_model is None:
        raise RuntimeError("LPIPS is required for reconstruction evaluation")
    data_cfg = config["data"]
    dataset = CocoCaptions(
        args.data_root or data_cfg["root"], "val", int(data_cfg["image_size"]), False, args.limit
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    work_dir = Path(args.work_dir) / args.name
    real_dir = work_dir / "real"
    recon_dir = work_dir / "reconstruction"
    real_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, list[np.ndarray]] = {}
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            target = batch["image"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = vae.encode(target).latent_dist.mean
                reconstruction = vae.decode(latent).sample
            target_f = target.float()
            reconstruction_f = reconstruction.float().clamp(-1, 1)
            error = reconstruction_f - target_f
            mse = error.square().mean((1, 2, 3))
            target_01 = target_f.add(1).mul(0.5)
            reconstruction_01 = reconstruction_f.add(1).mul(0.5)
            batch_metrics = {
                "l1": error.abs().mean((1, 2, 3)),
                "mse": mse,
                "psnr": 10.0 * torch.log10(1.0 / mse.clamp_min(1e-8)),
                "lpips": lpips_model(reconstruction_f, target_f).flatten(),
                "ssim": piq.ssim(reconstruction_01, target_01, data_range=1.0, reduction="none"),
            }
            if args.include_ms_ssim:
                batch_metrics["ms_ssim"] = piq.multi_scale_ssim(
                    reconstruction_01, target_01, data_range=1.0, reduction="none"
                )
            for key, value in batch_metrics.items():
                metrics.setdefault(key, []).append(value.float().cpu().numpy())
            for index in range(len(target)):
                filename = f"{offset + index:06d}.png"
                save_png(target[index], real_dir / filename)
                save_png(reconstruction[index], recon_dir / filename)
            offset += len(target)
    payload = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "count": len(dataset),
        "rfid_clean": None,
        "metrics": {key: summarize(values) for key, values in metrics.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    del vae, lpips_model
    torch.cuda.empty_cache()
    feature_model = fid.build_feature_extractor("clean", device=device, use_dataparallel=False)
    real_features = fid.get_folder_features(
        str(real_dir),
        model=feature_model,
        mode="clean",
        batch_size=args.fid_batch_size,
        num_workers=args.num_workers,
        device=device,
        description=f"{args.name} real",
        verbose=True,
    )
    reconstruction_features = fid.get_folder_features(
        str(recon_dir),
        model=feature_model,
        mode="clean",
        batch_size=args.fid_batch_size,
        num_workers=args.num_workers,
        device=device,
        description=f"{args.name} reconstruction",
        verbose=True,
    )
    payload["rfid_clean"] = frechet_distance(real_features, reconstruction_features)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
