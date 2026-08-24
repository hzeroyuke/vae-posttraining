from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ResumeGeometry:
    micro_steps: int
    saved_micro_steps: int
    consumed_samples_in_epoch: int
    source_world_size: int
    source_gradient_accumulation: int
    source_per_rank_batch_size: int


def _saved_argument(arguments: Any, name: str, default: int) -> int:
    if isinstance(arguments, Mapping):
        return int(arguments.get(name, default))
    return int(getattr(arguments, name, default))


def resolve_resume_geometry(
    checkpoint: Mapping[str, Any],
    *,
    fallback_micro_steps: int,
    current_world_size: int,
    current_gradient_accumulation: int,
    current_global_batch_size: int,
) -> ResumeGeometry:
    """Translate a rank-local checkpoint position to the current DDP geometry."""
    if current_world_size <= 0 or current_gradient_accumulation <= 0:
        raise ValueError("Current world size and gradient accumulation must be positive")

    current_effective_world = current_world_size * current_gradient_accumulation
    if current_global_batch_size % current_effective_world:
        raise ValueError(
            "Current global batch must be divisible by world size times gradient accumulation"
        )
    current_per_rank_batch = current_global_batch_size // current_effective_world

    saved_micro_steps = int(checkpoint.get("micro_steps_in_epoch", fallback_micro_steps))
    if saved_micro_steps < 0:
        raise ValueError("Checkpoint micro_steps_in_epoch must be non-negative")

    saved_arguments = checkpoint.get("args")
    source_world_size = _saved_argument(
        saved_arguments, "world_size", current_world_size
    )
    source_gradient_accumulation = _saved_argument(
        saved_arguments,
        "gradient_accumulation_steps",
        current_gradient_accumulation,
    )
    source_global_batch_size = _saved_argument(
        saved_arguments, "global_batch_size", current_global_batch_size
    )
    if source_world_size <= 0 or source_gradient_accumulation <= 0:
        raise ValueError("Checkpoint world size and gradient accumulation must be positive")

    source_effective_world = source_world_size * source_gradient_accumulation
    if source_global_batch_size % source_effective_world:
        raise ValueError(
            "Checkpoint global batch is not divisible by its world size and gradient accumulation"
        )
    source_per_rank_batch = source_global_batch_size // source_effective_world
    consumed_samples = saved_micro_steps * source_per_rank_batch * source_world_size

    current_samples_per_micro_step = current_per_rank_batch * current_world_size
    if consumed_samples % current_samples_per_micro_step:
        raise ValueError(
            "Checkpoint epoch position cannot be represented by the current DDP geometry"
        )
    current_micro_steps = consumed_samples // current_samples_per_micro_step

    return ResumeGeometry(
        micro_steps=current_micro_steps,
        saved_micro_steps=saved_micro_steps,
        consumed_samples_in_epoch=consumed_samples,
        source_world_size=source_world_size,
        source_gradient_accumulation=source_gradient_accumulation,
        source_per_rank_batch_size=source_per_rank_batch,
    )
