from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer


def tensor_to_pil_batch(images: torch.Tensor) -> list[Image.Image]:
    images = images.detach().float().cpu().clamp(-1, 1).add(1).div(2)
    images = images.mul(255).round().to(torch.uint8)
    images = images.permute(0, 2, 3, 1).contiguous().numpy()
    return [Image.fromarray(image, mode="RGB") for image in images]


class PickScoreReward:
    def __init__(
        self,
        processor_name_or_path: str,
        model_name_or_path: str,
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

        processor_local = Path(processor_name_or_path).exists()
        model_local = Path(model_name_or_path).exists()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            processor_name_or_path,
            cache_dir=cache_dir,
            local_files_only=processor_local,
        )
        self.image_size, self.image_mean, self.image_std = load_clip_image_config(processor_name_or_path)
        self.model = CLIPModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=model_local,
        ).eval().to(self.device)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        del metadatas
        if len(images) != len(prompts):
            raise ValueError(f"Got {len(images)} images but {len(prompts)} prompts")

        pil_images = tensor_to_pil_batch(images)
        scores: list[torch.Tensor] = []
        for start in range(0, len(pil_images), self.batch_size):
            end = start + self.batch_size
            batch_images = pil_images[start:end]
            batch_prompts = list(prompts[start:end])
            pixel_values = preprocess_pil_images(
                batch_images,
                image_size=self.image_size,
                mean=self.image_mean,
                std=self.image_std,
                device=self.device,
            )
            text_inputs = self.tokenizer(
                text=batch_prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            image_embs = unwrap_feature_output(self.model.get_image_features(pixel_values=pixel_values))
            text_embs = unwrap_feature_output(self.model.get_text_features(**text_inputs))
            image_embs = F.normalize(image_embs.float(), dim=-1)
            text_embs = F.normalize(text_embs.float(), dim=-1)
            pair_scores = self.model.logit_scale.exp().float() * (image_embs * text_embs).sum(dim=-1)
            scores.append(pair_scores.detach().cpu())

        return torch.cat(scores, dim=0).to(device=images.device, dtype=torch.float32)

    def score_tensor(self, images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        """Differentiable PickScore for optimization through images."""
        if len(images) != len(prompts):
            raise ValueError(f"Got {len(images)} images but {len(prompts)} prompts")

        scores: list[torch.Tensor] = []
        model_dtype = next(self.model.parameters()).dtype
        for start in range(0, len(images), self.batch_size):
            end = start + self.batch_size
            batch_images = images[start:end].float().clamp(-1, 1).add(1).div(2)
            if self.differentiable_resize_mode == "nearest-exact":
                pixel_values = F.interpolate(
                    batch_images,
                    size=(self.image_size, self.image_size),
                    mode="nearest-exact",
                )
            else:
                pixel_values = F.interpolate(
                    batch_images,
                    size=(self.image_size, self.image_size),
                    mode=self.differentiable_resize_mode,
                    align_corners=False,
                    antialias=self.differentiable_resize_mode == "bicubic",
                )
            mean = self.image_mean.to(device=images.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
            std = self.image_std.to(device=images.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
            pixel_values = ((pixel_values - mean) / std).to(device=self.device, dtype=model_dtype)
            text_inputs = self.tokenizer(
                text=list(prompts[start:end]),
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            image_embs = unwrap_feature_output(self.model.get_image_features(pixel_values=pixel_values))
            with torch.no_grad():
                text_embs = unwrap_feature_output(self.model.get_text_features(**text_inputs))
            image_embs = F.normalize(image_embs.float(), dim=-1)
            text_embs = F.normalize(text_embs.float(), dim=-1)
            pair_scores = self.model.logit_scale.exp().float() * (image_embs * text_embs).sum(dim=-1)
            scores.append(pair_scores)

        return torch.cat(scores, dim=0).to(device=images.device, dtype=torch.float32)

    def score_tensor_prompt_pair(
        self,
        images: torch.Tensor,
        primary_prompts: Sequence[str],
        baseline_prompts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score two prompts per image while sharing the differentiable image features."""
        if len(images) != len(primary_prompts) or len(images) != len(baseline_prompts):
            raise ValueError(
                "images, primary_prompts, and baseline_prompts must have the same length"
            )

        primary_scores: list[torch.Tensor] = []
        baseline_scores: list[torch.Tensor] = []
        model_dtype = next(self.model.parameters()).dtype
        logit_scale = self.model.logit_scale.exp().float()
        for start in range(0, len(images), self.batch_size):
            end = start + self.batch_size
            batch_images = images[start:end].float().clamp(-1, 1).add(1).div(2)
            if self.differentiable_resize_mode == "nearest-exact":
                pixel_values = F.interpolate(
                    batch_images,
                    size=(self.image_size, self.image_size),
                    mode="nearest-exact",
                )
            else:
                pixel_values = F.interpolate(
                    batch_images,
                    size=(self.image_size, self.image_size),
                    mode=self.differentiable_resize_mode,
                    align_corners=False,
                    antialias=self.differentiable_resize_mode == "bicubic",
                )
            mean = self.image_mean.to(device=images.device, dtype=pixel_values.dtype).view(
                1, 3, 1, 1
            )
            std = self.image_std.to(device=images.device, dtype=pixel_values.dtype).view(
                1, 3, 1, 1
            )
            pixel_values = ((pixel_values - mean) / std).to(
                device=self.device, dtype=model_dtype
            )
            primary_inputs = self.tokenizer(
                text=list(primary_prompts[start:end]),
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            baseline_inputs = self.tokenizer(
                text=list(baseline_prompts[start:end]),
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            image_embs = unwrap_feature_output(
                self.model.get_image_features(pixel_values=pixel_values)
            )
            with torch.no_grad():
                primary_text_embs = unwrap_feature_output(
                    self.model.get_text_features(**primary_inputs)
                )
                baseline_text_embs = unwrap_feature_output(
                    self.model.get_text_features(**baseline_inputs)
                )
            image_embs = F.normalize(image_embs.float(), dim=-1)
            primary_text_embs = F.normalize(primary_text_embs.float(), dim=-1)
            baseline_text_embs = F.normalize(baseline_text_embs.float(), dim=-1)
            primary_scores.append(
                logit_scale * (image_embs * primary_text_embs).sum(dim=-1)
            )
            baseline_scores.append(
                logit_scale * (image_embs * baseline_text_embs).sum(dim=-1)
            )

        return (
            torch.cat(primary_scores, dim=0).to(device=images.device, dtype=torch.float32),
            torch.cat(baseline_scores, dim=0).to(device=images.device, dtype=torch.float32),
        )


class PickScoreImageSimilarityReward(PickScoreReward):
    """Reference-aware PickScore using the frozen image encoder on both inputs."""

    requires_reference_images = True

    def score_tensor(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        reference_images: torch.Tensor,
    ) -> torch.Tensor:
        del prompts
        if len(images) != len(reference_images):
            raise ValueError(
                f"Got {len(images)} images but {len(reference_images)} reference images"
            )

        scores: list[torch.Tensor] = []
        model_dtype = next(self.model.parameters()).dtype
        logit_scale = self.model.logit_scale.exp().float()
        for start in range(0, len(images), self.batch_size):
            end = start + self.batch_size

            def image_features(batch: torch.Tensor) -> torch.Tensor:
                batch = batch.float().clamp(-1, 1).add(1).div(2)
                if self.differentiable_resize_mode == "nearest-exact":
                    pixel_values = F.interpolate(
                        batch,
                        size=(self.image_size, self.image_size),
                        mode="nearest-exact",
                    )
                else:
                    pixel_values = F.interpolate(
                        batch,
                        size=(self.image_size, self.image_size),
                        mode=self.differentiable_resize_mode,
                        align_corners=False,
                        antialias=self.differentiable_resize_mode == "bicubic",
                    )
                mean = self.image_mean.to(
                    device=pixel_values.device, dtype=pixel_values.dtype
                ).view(1, 3, 1, 1)
                std = self.image_std.to(
                    device=pixel_values.device, dtype=pixel_values.dtype
                ).view(1, 3, 1, 1)
                pixel_values = ((pixel_values - mean) / std).to(
                    device=self.device, dtype=model_dtype
                )
                return unwrap_feature_output(
                    self.model.get_image_features(pixel_values=pixel_values)
                )

            candidate_features = image_features(images[start:end])
            with torch.no_grad():
                reference_features = image_features(reference_images[start:end])
            candidate_features = F.normalize(candidate_features.float(), dim=-1)
            reference_features = F.normalize(reference_features.float(), dim=-1)
            scores.append(
                logit_scale * (candidate_features * reference_features).sum(dim=-1)
            )

        return torch.cat(scores, dim=0).to(device=images.device, dtype=torch.float32)

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        reference_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del metadatas
        if reference_images is None:
            raise ValueError("PickScore image similarity reward requires reference_images")
        return self.score_tensor(images, prompts, reference_images)


def load_clip_image_config(path_or_repo: str) -> tuple[int, torch.Tensor, torch.Tensor]:
    config_path = Path(path_or_repo) / "preprocessor_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        size_cfg = cfg.get("size", {})
        crop_cfg = cfg.get("crop_size", {})
        if isinstance(crop_cfg, dict):
            image_size = int(crop_cfg.get("height") or crop_cfg.get("width") or 224)
        elif isinstance(size_cfg, dict):
            image_size = int(size_cfg.get("shortest_edge") or size_cfg.get("height") or 224)
        else:
            image_size = 224
        mean = torch.tensor(cfg.get("image_mean", [0.48145466, 0.4578275, 0.40821073]))
        std = torch.tensor(cfg.get("image_std", [0.26862954, 0.26130258, 0.27577711]))
        return image_size, mean, std

    return (
        224,
        torch.tensor([0.48145466, 0.4578275, 0.40821073]),
        torch.tensor([0.26862954, 0.26130258, 0.27577711]),
    )


def preprocess_pil_images(
    images: Sequence[Image.Image],
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    resized = [image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC) for image in images]
    tensors = []
    for image in resized:
        data = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
        data = data.permute(2, 0, 1).float().div(255.0)
        tensors.append(data)
    batch = torch.stack(tensors, dim=0).to(device)
    mean = mean.to(device=device, dtype=batch.dtype).view(1, 3, 1, 1)
    std = std.to(device=device, dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch - mean) / std


def unwrap_feature_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unexpected CLIP feature output type: {type(output)!r}")
