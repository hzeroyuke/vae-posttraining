from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPModel

from .pickscore import load_clip_image_config, unwrap_feature_output


def normalized_aesthetic_scores(image_embeddings: torch.Tensor, head: nn.Module) -> torch.Tensor:
    """Apply the LAION aesthetic head to normalized CLIP image embeddings."""
    normalized = F.normalize(image_embeddings.float(), dim=-1)
    return head(normalized).flatten().float()


class AestheticReward:
    """LAION aesthetic predictor using the official ViT-L/14 linear head."""

    def __init__(
        self,
        model_name_or_path: str,
        head_path: str,
        device: str,
        cache_dir: str | None,
        dtype: str = "bf16",
        batch_size: int = 16,
        differentiable_resize_mode: str = "bicubic",
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.differentiable_resize_mode = str(differentiable_resize_mode)
        if self.differentiable_resize_mode not in {"bicubic", "bilinear", "nearest-exact"}:
            raise ValueError(
                "differentiable_resize_mode must be bicubic, bilinear, or nearest-exact, got "
                f"{self.differentiable_resize_mode!r}"
            )
        torch_dtype = torch.float32
        if self.device.type == "cuda" and dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif self.device.type == "cuda" and dtype == "fp16":
            torch_dtype = torch.float16

        model_local = Path(model_name_or_path).exists()
        self.image_size, self.image_mean, self.image_std = load_clip_image_config(
            model_name_or_path
        )
        self.model = CLIPModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=model_local,
        ).eval().to(self.device)
        self.model.requires_grad_(False)

        projection_dim = int(self.model.config.projection_dim)
        self.head = nn.Linear(projection_dim, 1)
        state = torch.load(head_path, map_location="cpu", weights_only=True)
        self.head.load_state_dict(state, strict=True)
        self.head.eval().requires_grad_(False).to(self.device, dtype=torch.float32)

    def _pixel_values(self, images: torch.Tensor) -> torch.Tensor:
        values = images.float().clamp(-1, 1).add(1).div(2)
        if self.differentiable_resize_mode == "nearest-exact":
            values = F.interpolate(
                values,
                size=(self.image_size, self.image_size),
                mode="nearest-exact",
            )
        else:
            values = F.interpolate(
                values,
                size=(self.image_size, self.image_size),
                mode=self.differentiable_resize_mode,
                align_corners=False,
                antialias=self.differentiable_resize_mode == "bicubic",
            )
        mean = self.image_mean.to(device=values.device, dtype=values.dtype).view(1, 3, 1, 1)
        std = self.image_std.to(device=values.device, dtype=values.dtype).view(1, 3, 1, 1)
        model_dtype = next(self.model.parameters()).dtype
        return ((values - mean) / std).to(device=self.device, dtype=model_dtype)

    def score_tensor(
        self,
        images: torch.Tensor,
        prompts: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Return differentiable aesthetic scores for images in the [-1, 1] range."""
        if prompts is not None and len(images) != len(prompts):
            raise ValueError(f"Got {len(images)} images but {len(prompts)} prompts")
        parts = []
        for start in range(0, len(images), self.batch_size):
            pixel_values = self._pixel_values(images[start : start + self.batch_size])
            embeddings = unwrap_feature_output(
                self.model.get_image_features(pixel_values=pixel_values)
            )
            parts.append(normalized_aesthetic_scores(embeddings, self.head))
        return torch.cat(parts).to(device=images.device, dtype=torch.float32)

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        prompts: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        del metadatas
        if prompts is not None and len(images) != len(prompts):
            raise ValueError(f"Got {len(images)} images but {len(prompts)} prompts")
        return self.score_tensor(images, prompts)
