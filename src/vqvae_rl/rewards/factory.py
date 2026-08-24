from __future__ import annotations

from typing import Any

from .pickscore import PickScoreReward


def build_reward(cfg: Any, device: str, cache_dir: str | None = None):
    if cfg.backend != "pickscore":
        raise ValueError(f"Unsupported reward backend in this experiment: {cfg.backend}")
    return PickScoreReward(
        processor_name_or_path=cfg.processor_name_or_path,
        model_name_or_path=cfg.model_name_or_path,
        device=device,
        cache_dir=cache_dir,
        dtype=getattr(cfg, "dtype", "bf16"),
        batch_size=int(getattr(cfg, "batch_size", 16)),
        differentiable_resize_mode=str(
            getattr(cfg, "differentiable_resize_mode", "bicubic")
        ),
    )
