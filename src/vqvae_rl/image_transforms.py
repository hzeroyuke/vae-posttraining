from __future__ import annotations

import torch
import torch.nn.functional as F


def _contrast(images: torch.Tensor, amount: float) -> torch.Tensor:
    return (images * amount).clamp(-1, 1)


def _saturation(images: torch.Tensor, amount: float) -> torch.Tensor:
    luminance = (
        0.2126 * images[:, :1]
        + 0.7152 * images[:, 1:2]
        + 0.0722 * images[:, 2:3]
    )
    return (luminance + amount * (images - luminance)).clamp(-1, 1)


def _sharpen(images: torch.Tensor, amount: float) -> torch.Tensor:
    blurred = F.avg_pool2d(F.pad(images, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
    return (images + amount * (images - blurred)).clamp(-1, 1)


def _gamma(images: torch.Tensor, amount: float) -> torch.Tensor:
    values = images.add(1).div(2).clamp(0, 1).pow(amount)
    return values.mul(2).sub(1)


def geneval_transform_variants(images: torch.Tensor) -> dict[str, torch.Tensor]:
    variants = {"identity": images}
    for amount in (1.05, 1.10, 1.15):
        variants[f"contrast_{amount:.2f}"] = _contrast(images, amount)
    for amount in (1.05, 1.10, 1.15):
        variants[f"saturation_{amount:.2f}"] = _saturation(images, amount)
    for amount in (0.25, 0.50, 0.75, 1.00):
        variants[f"sharpen_{amount:.2f}"] = _sharpen(images, amount)
    for amount in (0.90, 1.10):
        variants[f"gamma_{amount:.2f}"] = _gamma(images, amount)
    variants["sharp_0.50_contrast_1.05"] = _contrast(_sharpen(images, 0.50), 1.05)
    variants["sharp_0.75_contrast_1.10"] = _contrast(_sharpen(images, 0.75), 1.10)
    return variants
