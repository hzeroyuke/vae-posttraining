from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image

from diffusers import AutoencoderKL

from .train import CocoCaptions, make_lpips, make_reward_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--reward-model-path", default=None)
    parser.add_argument("--posterior-sample", action="store_true")
    return parser.parse_args()


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
    }


def bootstrap(delta: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = []
    for _ in range((samples + 199) // 200):
        count = min(200, samples - len(means) * 200)
        if count <= 0:
            break
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means.append(delta[indices].mean(1))
    low, high = np.quantile(np.concatenate(means), [0.025, 0.975])
    return [float(low), float(high)]


def per_image_metrics(recon: torch.Tensor, target: torch.Tensor, lpips_model) -> dict[str, torch.Tensor]:
    recon_f = recon.float().clamp(-1, 1)
    target_f = target.float()
    error = recon_f - target_f
    mse = error.square().mean((1, 2, 3))
    recon_01 = recon_f.add(1).mul(0.5)
    return {
        "l1": error.abs().mean((1, 2, 3)),
        "mse": mse,
        "psnr": 10.0 * torch.log10(1.0 / mse.clamp_min(1e-8)),
        "lpips": lpips_model(recon_f, target_f).flatten(),
        "saturation": ((recon_01 <= 1.0 / 255.0) | (recon_01 >= 254.0 / 255.0)).float().mean((1, 2, 3)),
        "red_mean": recon_01[:, 0].mean((1, 2)),
        "green_mean": recon_01[:, 1].mean((1, 2)),
        "blue_mean": recon_01[:, 2].mean((1, 2)),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device)
    torch.manual_seed(int(config["seed"]))
    model_path = args.model_path or config["model"]["checkpoint"]
    base = AutoencoderKL.from_pretrained(model_path, local_files_only=True).eval().to(device)
    candidate = AutoencoderKL.from_pretrained(model_path, local_files_only=True).eval().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    candidate.load_state_dict(checkpoint["vae"], strict=True)
    base.requires_grad_(False)
    candidate.requires_grad_(False)
    reward_cfg = dict(config["reward"])
    if args.processor_path is not None:
        reward_cfg["processor_path"] = args.processor_path
    if args.reward_model_path is not None:
        reward_cfg["model_path"] = args.reward_model_path
    scorer, reward_metric = make_reward_scorer(reward_cfg, device)
    lpips_model = make_lpips(device)
    if lpips_model is None:
        raise RuntimeError("LPIPS is required for formal evaluation")
    data_cfg = config["data"]
    requested_limit = args.limit if args.limit is not None else data_cfg.get("val_subset", 96)
    dataset = CocoCaptions(
        args.data_root or data_cfg["root"], "val", data_cfg["image_size"], False,
        args.offset + requested_limit,
    )
    if args.offset:
        dataset = Subset(dataset, range(args.offset, args.offset + requested_limit))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    arrays: dict[str, dict[str, list[np.ndarray]]] = {"base": {}, "candidate": {}}
    comparison_images = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            target = batch["image"].to(device, non_blocking=True)
            prompts = list(batch["prompt"])
            outputs = {}
            shared_epsilon = None
            for name, model in (("base", base), ("candidate", candidate)):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    posterior = model.encode(target).latent_dist
                    latent = posterior.mean
                    if args.posterior_sample:
                        if shared_epsilon is None:
                            shared_epsilon = torch.randn_like(latent)
                        latent = latent + posterior.std * shared_epsilon
                    recon = model.decode(latent).sample
                outputs[name] = recon
                metrics = per_image_metrics(recon, target, lpips_model)
                metrics[reward_metric] = scorer.score_tensor(recon, prompts)
                for key, value in metrics.items():
                    arrays[name].setdefault(key, []).append(value.float().cpu().numpy())
            if batch_index < 4:
                for index in range(len(target)):
                    comparison_images.extend([target[index], outputs["base"][index], outputs["candidate"][index]])
    packed = {
        name: {key: np.concatenate(values).astype(np.float64) for key, values in metrics.items()}
        for name, metrics in arrays.items()
    }
    directions = {
        "l1": -1,
        "mse": -1,
        "psnr": 1,
        "lpips": -1,
        "saturation": -1,
        "pickscore": 1,
        "aes": 1,
        "musiq_ava": 1,
    }
    payload: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "posterior_sample": args.posterior_sample,
        "count": len(dataset),
        "base": {key: summarize(values) for key, values in packed["base"].items()},
        "candidate": {key: summarize(values) for key, values in packed["candidate"].items()},
        "delta": {},
    }
    for key in packed["base"]:
        delta = packed["candidate"][key] - packed["base"][key]
        item = {"mean": float(delta.mean()), "ci95": bootstrap(delta, args.bootstrap, config["seed"])}
        if key in directions:
            item["improved_fraction"] = float((directions[key] * delta > 0).mean())
        payload["delta"][key] = item
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    grid = make_grid(torch.stack(comparison_images).float().clamp(-1, 1), nrow=3)
    save_image(grid.add(1).div(2), output.with_suffix(".png"))
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
