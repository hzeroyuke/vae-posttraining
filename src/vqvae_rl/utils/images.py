from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return images.detach().float().clamp(-1, 1).add(1).div(2)


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    from torchvision.utils import make_grid

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(denormalize(images).cpu(), nrow=nrow)
    array = grid.mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(array, mode="RGB").save(path)
