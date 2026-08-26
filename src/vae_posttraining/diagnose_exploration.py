from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from diffusers import AutoencoderKL

from .train import CocoCaptions, latent_stats, make_lpips, make_reward_scorer, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--rhos", type=float, nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--pairwise-limit", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def summarize(values: list[np.ndarray]) -> dict[str, float]:
    packed = np.concatenate(values).astype(np.float64)
    return {
        "mean": float(packed.mean()),
        "p50": float(np.quantile(packed, 0.5)),
        "p90": float(np.quantile(packed, 0.9)),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device)
    seed_everything(int(config["seed"]))
    model_cfg = config["model"]
    scale = float(model_cfg.get("scaling_factor", 0.18215))
    vae = AutoencoderKL.from_pretrained(model_cfg["checkpoint"], local_files_only=True).eval().to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        vae.load_state_dict(checkpoint["vae"], strict=True)
    vae.requires_grad_(False)
    reward_scorer, reward_metric = make_reward_scorer(config["reward"], device)
    lpips_model = make_lpips(device)
    if lpips_model is None:
        raise RuntimeError("LPIPS is required for exploration diagnostics")
    data_cfg = config["data"]
    train_set = CocoCaptions(data_cfg["root"], "train", data_cfg["image_size"], True, min(256, data_cfg.get("train_subset", 256)))
    val_set = CocoCaptions(data_cfg["root"], "val", data_cfg["image_size"], False, args.limit)
    stats_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    _, channel_std = latent_stats(vae, stats_loader, device, scale, int(config["train"].get("stats_batches", 16)))
    train_cfg = config["train"]
    l1_threshold = float(train_cfg.get("recon_l1_threshold", 0.0))
    lpips_threshold = float(train_cfg.get("recon_lpips_threshold", 0.0))
    l1_weight = float(train_cfg.get("recon_penalty", 5.0))
    lpips_weight = float(train_cfg.get("lpips_recon_penalty", 0.0))
    collected = {rho: {} for rho in args.rhos}
    processed = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            prompts = list(batch["prompt"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu_scaled = vae.encode(images).latent_dist.mean * scale
            bsz = len(images)
            epsilon = torch.randn(bsz, args.group_size, *mu_scaled.shape[1:], device=device)
            for rho in args.rhos:
                sigma = float(rho) * channel_std
                latent = mu_scaled[:, None] + sigma[None, None] * epsilon
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    behavior = vae.decode(latent.reshape(-1, *latent.shape[2:]) / scale).sample
                grouped = behavior.view(bsz, args.group_size, *behavior.shape[1:])
                targets = images[:, None].expand_as(grouped)
                l1 = (grouped.float() - targets.float()).abs().mean((2, 3, 4))
                lpips_values = lpips_model(behavior.float(), targets.reshape_as(behavior).float()).view(bsz, args.group_size)
                flat_prompts = [prompt for prompt in prompts for _ in range(args.group_size)]
                scores = reward_scorer.score_tensor(behavior, flat_prompts).view(bsz, args.group_size)
                reward = scores - l1_weight * F.relu(l1 - l1_threshold) - lpips_weight * F.relu(lpips_values - lpips_threshold)
                metrics = {
                    f"{reward_metric}_group_std": scores.std(1, unbiased=False),
                    f"{reward_metric}_top_bottom_gap": scores.max(1).values - scores.min(1).values,
                    "reward_group_std": reward.std(1, unbiased=False),
                    "reward_top_bottom_gap": reward.max(1).values - reward.min(1).values,
                    "behavior_l1": l1.flatten(),
                    "behavior_lpips": lpips_values.flatten(),
                    "l1_hinge_fraction": (l1 > l1_threshold).float().flatten(),
                    "lpips_hinge_fraction": (lpips_values > lpips_threshold).float().flatten(),
                }
                if processed < args.pairwise_limit:
                    first, second = torch.triu_indices(args.group_size, args.group_size, offset=1, device=device)
                    pair_a = grouped[:, first].reshape(-1, *behavior.shape[1:])
                    pair_b = grouped[:, second].reshape_as(pair_a)
                    metrics["pairwise_lpips"] = lpips_model(pair_a.float(), pair_b.float()).flatten()
                    metrics["pairwise_l1"] = (pair_a.float() - pair_b.float()).abs().mean((1, 2, 3))
                for key, value in metrics.items():
                    collected[rho].setdefault(key, []).append(value.float().cpu().numpy())
            processed += bsz
    payload = {
        "checkpoint": args.checkpoint,
        "reward_metric": reward_metric,
        "count": len(val_set),
        "group_size": args.group_size,
        "latent_channel_std": channel_std.flatten().cpu().tolist(),
        "results": {
            str(rho): {
                "sigma_expl": (float(rho) * channel_std).flatten().cpu().tolist(),
                **{key: summarize(values) for key, values in metrics.items()},
            }
            for rho, metrics in collected.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
