from __future__ import annotations

import argparse
import copy
import ctypes
import glob
import io
import json
import math
import os
import random
import runpy
import time
from bisect import bisect_right
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

# The repository's pinned `.deps` torch build may pair with a torchvision wheel
# that lacks the nms schema. System torch/torchvision already owns it, so only
# install the compatibility schema for the pinned local build.
if ".deps" in str(getattr(torch, "__file__", "")):
    try:
        _torchvision_compat = torch.library.Library("torchvision", "DEF")
        _torchvision_compat.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        _torchvision_compat = None

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from transformers import CLIPModel, CLIPTokenizer

try:
    import pyarrow.parquet as parquet
except ImportError:  # pragma: no cover - only required for ImageNet parquet data
    parquet = None

from diffusers import AutoencoderKL


def setup_distributed() -> tuple[int, int, torch.device]:
    process_name = os.environ.get("VAE_PROCESS_NAME")
    if process_name:
        try:
            ctypes.CDLL("libc.so.6").prctl(15, process_name.encode()[:15], 0, 0, 0)
        except OSError:
            pass
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world, torch.device("cuda", local_rank)
    if not torch.cuda.is_available():
        return 0, 1, torch.device("cpu")
    return 0, 1, torch.device("cuda", 0)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value)
    return value


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CocoCaptions(Dataset):
    def __init__(self, root: str, split: str, image_size: int, train: bool, limit: int | None = None):
        root_path = Path(root)
        split_name = f"{split}2017" if split in {"train", "val"} else split
        annotation_path = root_path / "annotations" / f"captions_{split_name}.json"
        image_dir = root_path / split_name
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        captions: dict[int, list[str]] = {}
        for item in payload["annotations"]:
            text = str(item.get("caption", "")).strip()
            if text:
                captions.setdefault(int(item["image_id"]), []).append(text)
        entries = []
        for item in payload["images"]:
            image_id = int(item["id"])
            path = image_dir / item["file_name"]
            if path.exists() and image_id in captions:
                entries.append((path, captions[image_id][0]))
        if limit is not None:
            entries = entries[: int(limit)]
        if not entries:
            raise RuntimeError(f"No captioned images found under {root_path} ({split_name})")
        self.entries = entries
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
            ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, caption = self.entries[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
            tensor = self.to_tensor(image)
        return {"image": tensor, "prompt": caption, "index": index}


def load_imagenet_class_names(path: str) -> list[str]:
    payload = runpy.run_path(path)
    mapping = payload.get("IMAGENET2012_CLASSES")
    if mapping is None:
        raise ValueError(f"{path} does not define IMAGENET2012_CLASSES")
    return [str(value).split(",", 1)[0].strip() for value in mapping.values()]


class ImageNetParquet(Dataset):
    """Lazy random-access ImageNet Dataset backed by Hugging Face parquet shards."""

    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        train: bool,
        limit: int | None = None,
        class_names_path: str | None = None,
    ):
        if parquet is None:
            raise RuntimeError("pyarrow is required for ImageNet parquet training")
        root_path = Path(root)
        split_name = "validation" if split in {"val", "validation"} else split
        shard_paths = sorted(glob.glob(str(root_path / "data" / f"{split_name}-*.parquet")))
        if not shard_paths:
            raise FileNotFoundError(f"No ImageNet parquet shards found under {root_path} for {split_name}")
        self.shards = []
        self.cumulative_rows = []
        total = 0
        for shard_path in shard_paths:
            parquet_file = parquet.ParquetFile(shard_path)
            row_groups = [parquet_file.metadata.row_group(i).num_rows for i in range(parquet_file.num_row_groups)]
            self.shards.append((shard_path, row_groups))
            total += sum(row_groups)
            self.cumulative_rows.append(total)
        self.length = min(total, int(limit)) if limit is not None else total
        self.class_names = load_imagenet_class_names(class_names_path) if class_names_path else None
        self.cache_key = None
        self.cache_table = None
        self._class_indices: list[np.ndarray] | None = None
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
            ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return self.length

    def build_class_indices(self, num_classes: int = 1000) -> list[np.ndarray]:
        """Read only labels once; the resulting int16 index arrays are small."""
        if self._class_indices is not None:
            return self._class_indices
        labels = np.empty(self.length, dtype=np.int16)
        cursor = 0
        for shard_path, row_groups in self.shards:
            if cursor >= self.length:
                break
            parquet_file = parquet.ParquetFile(shard_path)
            for row_group, row_count in enumerate(row_groups):
                if cursor >= self.length:
                    break
                take = min(int(row_count), self.length - cursor)
                table = parquet_file.read_row_group(row_group, columns=["label"])
                values = table.column("label").to_numpy(zero_copy_only=False)
                labels[cursor : cursor + take] = np.asarray(values[:take], dtype=np.int16)
                cursor += take
        self._class_indices = [
            np.flatnonzero(labels == class_id).astype(np.int64, copy=False)
            for class_id in range(int(num_classes))
        ]
        return self._class_indices

    def _locate(self, index: int) -> tuple[str, int, int]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative_rows, index)
        previous_rows = 0 if shard_index == 0 else self.cumulative_rows[shard_index - 1]
        local_index = index - previous_rows
        shard_path, row_groups = self.shards[shard_index]
        row_group = 0
        for rows in row_groups:
            if local_index < rows:
                return shard_path, row_group, local_index
            local_index -= rows
            row_group += 1
        raise RuntimeError("Invalid ImageNet parquet row index")

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_path, row_group, row_offset = self._locate(index)
        key = (shard_path, row_group)
        if key != self.cache_key:
            table = parquet.ParquetFile(shard_path).read_row_group(row_group, columns=["image", "label"])
            self.cache_key = key
            self.cache_table = table
        assert self.cache_table is not None
        row = self.cache_table.slice(row_offset, 1).to_pydict()
        image_payload = row["image"][0]
        image_bytes = image_payload["bytes"] if isinstance(image_payload, dict) else image_payload
        label = int(row["label"][0])
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image = self.transform(image)
            tensor = self.to_tensor(image)
        class_name = self.class_names[label] if self.class_names and label < len(self.class_names) else f"class {label}"
        return {
            "image": tensor,
            "prompt": f"a photo of a {class_name}",
            "class_id": label,
            "index": index,
        }


class ImageNetClassBalancedBatchSampler(Sampler[list[int]]):
    """Yield same-class batches without materializing image tensors."""

    def __init__(
        self,
        dataset: ImageNetParquet,
        batch_size: int,
        num_batches: int,
        num_classes: int = 1000,
        seed: int = 0,
        rank: int = 0,
        class_repeat_batches: int = 1,
    ):
        if batch_size < 2 or num_batches < 1:
            raise ValueError("class-balanced sampler needs batch_size >= 2 and num_batches >= 1")
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.rank = int(rank)
        self.class_repeat_batches = max(1, int(class_repeat_batches))
        self.class_indices = dataset.build_class_indices(num_classes)
        self.classes = np.asarray(
            [class_id for class_id, indices in enumerate(self.class_indices) if len(indices) >= self.batch_size],
            dtype=np.int64,
        )
        if not len(self.classes):
            raise RuntimeError("No ImageNet classes contain enough samples for a balanced batch")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1009 * self.epoch + self.rank)
        class_id = None
        for batch_index in range(self.num_batches):
            if class_id is None or batch_index % self.class_repeat_batches == 0:
                class_id = int(rng.choice(self.classes))
            pool = self.class_indices[class_id]
            chosen = rng.choice(pool, size=self.batch_size, replace=False)
            yield chosen.tolist()

    def __len__(self) -> int:
        return self.num_batches


class PickScore:
    def __init__(self, processor_path: str, model_path: str, device: torch.device, batch_size: int):
        self.device = device
        self.batch_size = int(batch_size)
        self.tokenizer = CLIPTokenizer.from_pretrained(processor_path, local_files_only=True)
        processor_config = json.loads((Path(processor_path) / "preprocessor_config.json").read_text())
        size = processor_config.get("size", 224)
        self.image_size = int(size.get("shortest_edge", size) if isinstance(size, dict) else size)
        mean = processor_config.get("image_mean", [0.48145466, 0.4578275, 0.40821073])
        std = processor_config.get("image_std", [0.26862954, 0.26130258, 0.27577711])
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(
            model_path, local_files_only=True, torch_dtype=dtype, use_safetensors=True
        ).eval().to(device)
        self.model.requires_grad_(False)

    @staticmethod
    def _features(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0]
        if isinstance(output, (tuple, list)):
            return output[0]
        raise TypeError(f"Unsupported CLIP feature output: {type(output)}")

    def _image_pixels(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float().clamp(-1, 1).add(1).mul(0.5)
        images = F.interpolate(images, (self.image_size, self.image_size), mode="bicubic", align_corners=False, antialias=True)
        pixels = (images - self.mean.to(images.dtype)) / self.std.to(images.dtype)
        return pixels.to(device=self.device, dtype=next(self.model.parameters()).dtype)

    def score_tensor(
        self,
        images: torch.Tensor,
        prompts: list[str] | None = None,
        reference_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prompts is None or len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length")
        scores = []
        for start in range(0, len(images), self.batch_size):
            end = min(start + self.batch_size, len(images))
            pixels = self._image_pixels(images[start:end])
            text = self.tokenizer(
                prompts[start:end], padding=True, truncation=True, max_length=77, return_tensors="pt"
            ).to(self.device)
            image_features = self._features(self.model.get_image_features(pixel_values=pixels))
            with torch.no_grad():
                text_features = self._features(self.model.get_text_features(**text))
            image_features = F.normalize(image_features.float(), dim=-1)
            text_features = F.normalize(text_features.float(), dim=-1)
            scores.append(self.model.logit_scale.exp().float() * (image_features * text_features).sum(-1))
        return torch.cat(scores).to(images.device, dtype=torch.float32)


class AestheticScore:
    """LAION aesthetic predictor backed by CLIP ViT-L/14 and its linear head."""

    def __init__(self, model_path: str, head_path: str, device: torch.device, batch_size: int):
        self.device = device
        self.batch_size = int(batch_size)
        processor_config = json.loads((Path(model_path) / "preprocessor_config.json").read_text())
        size = processor_config.get("size", 224)
        self.image_size = int(size.get("shortest_edge", size) if isinstance(size, dict) else size)
        self.mean = torch.tensor(
            processor_config.get("image_mean", [0.48145466, 0.4578275, 0.40821073]),
            device=device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            processor_config.get("image_std", [0.26862954, 0.26130258, 0.27577711]),
            device=device,
        ).view(1, 3, 1, 1)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(
            model_path, local_files_only=True, torch_dtype=dtype, use_safetensors=True
        ).eval().to(device)
        self.model.requires_grad_(False)
        self.head = nn.Linear(int(self.model.config.projection_dim), 1)
        self.head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True), strict=True)
        self.head.eval().requires_grad_(False).to(device=device, dtype=torch.float32)

    def score_tensor(
        self,
        images: torch.Tensor,
        prompts: Sequence[str] | None = None,
        reference_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prompts is not None and len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length")
        scores = []
        for start in range(0, len(images), self.batch_size):
            pixels = images[start : start + self.batch_size].float().clamp(-1, 1).add(1).mul(0.5)
            pixels = F.interpolate(
                pixels,
                (self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            pixels = (pixels - self.mean.to(pixels.dtype)) / self.std.to(pixels.dtype)
            pixels = pixels.to(device=self.device, dtype=next(self.model.parameters()).dtype)
            embeddings = PickScore._features(self.model.get_image_features(pixel_values=pixels))
            scores.append(self.head(F.normalize(embeddings.float(), dim=-1)).flatten())
        return torch.cat(scores).to(images.device, dtype=torch.float32)


class MUSIQAestheticScore:
    """Prompt-free MUSIQ model trained on AVA human aesthetic ratings."""

    def __init__(self, device: torch.device, batch_size: int):
        try:
            import pyiqa
        except ImportError as error:
            raise RuntimeError("The musiq_ava backend requires pyiqa") from error
        self.device = device
        self.batch_size = int(batch_size)
        self.model = pyiqa.create_metric("musiq-ava", device=device)
        self.model.eval().requires_grad_(False)

    def score_tensor(
        self,
        images: torch.Tensor,
        prompts: Sequence[str] | None = None,
        reference_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prompts is not None and len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length")
        scores = []
        for start in range(0, len(images), self.batch_size):
            pixels = images[start : start + self.batch_size].float().clamp(-1, 1).add(1).mul(0.5)
            scores.append(self.model(pixels.to(self.device)).flatten())
        return torch.cat(scores).to(images.device, dtype=torch.float32)


class DINOFeatureScore:
    """Frozen DINOv2 feature similarity against the source image.

    Patch-token similarity is the default because the global CLS token is too
    invariant to the local reconstruction changes that a VAE can make.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        batch_size: int,
        feature_mode: str = "patch",
    ):
        self.device = device
        self.batch_size = int(batch_size)
        self.feature_mode = str(feature_mode).lower()
        if self.feature_mode not in {"patch", "cls"}:
            raise ValueError(f"Unsupported DINO feature_mode: {self.feature_mode}")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        self.model.eval().requires_grad_(False).to(device)
        self.image_size = 224
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _image_pixels(self, images: torch.Tensor) -> torch.Tensor:
        pixels = images.float().clamp(-1, 1).add(1).mul(0.5)
        pixels = F.interpolate(
            pixels,
            (self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        pixels = (pixels - self.mean.to(pixels.dtype)) / self.std.to(pixels.dtype)
        return pixels.to(device=self.device, dtype=next(self.model.parameters()).dtype)

    def _features(self, pixels: torch.Tensor) -> torch.Tensor:
        outputs = self.model.forward_features(pixels)
        if isinstance(outputs, dict):
            if self.feature_mode == "patch":
                features = outputs.get("x_norm_patchtokens")
                if features is None:
                    prenorm = outputs.get("x_prenorm")
                    if prenorm is None or prenorm.ndim != 3 or prenorm.shape[1] < 2:
                        raise RuntimeError("DINO forward_features did not expose patch tokens")
                    features = prenorm[:, 1:]
            else:
                features = outputs.get("x_norm_clstoken")
                if features is None:
                    prenorm = outputs.get("x_prenorm")
                    if prenorm is None or prenorm.ndim != 3:
                        raise RuntimeError("DINO forward_features did not expose a CLS token")
                    features = prenorm[:, 0]
        elif outputs.ndim == 3:
            features = outputs[:, 1:] if self.feature_mode == "patch" else outputs[:, 0]
        else:
            features = outputs
        if features.ndim not in {2, 3}:
            raise RuntimeError(f"Unexpected DINO feature shape: {tuple(features.shape)}")
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def score_tensor(
        self,
        images: torch.Tensor,
        prompts: Sequence[str] | None = None,
        reference_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reference_images is None or len(images) != len(reference_images):
            raise ValueError("DINO feature reward requires reference_images with matching length")
        scores = []
        for start in range(0, len(images), self.batch_size):
            end = min(start + self.batch_size, len(images))
            candidate_pixels = self._image_pixels(images[start:end])
            reference_pixels = self._image_pixels(reference_images[start:end])
            features = self._features(torch.cat([candidate_pixels, reference_pixels], dim=0))
            candidate_features, reference_features = features.chunk(2, dim=0)
            similarity = (candidate_features * reference_features).sum(-1)
            if similarity.ndim == 2:
                similarity = similarity.mean(-1)
            scores.append(similarity)
        return torch.cat(scores).to(images.device, dtype=torch.float32)


RewardScorer = PickScore | AestheticScore | MUSIQAestheticScore | DINOFeatureScore


def make_reward_scorer(config: dict[str, Any], device: torch.device) -> tuple[RewardScorer, str]:
    backend = str(config.get("backend", "pickscore")).lower()
    if backend in {"musiq", "musiq_ava", "musiq-ava"}:
        return (
            MUSIQAestheticScore(device, int(config.get("batch_size", 16))),
            "musiq_ava",
        )
    if backend in {"aes", "aesthetic"}:
        return (
            AestheticScore(
                config["model_path"],
                config["head_path"],
                device,
                int(config.get("batch_size", 16)),
            ),
            "aes",
        )
    if backend in {"dino", "dinov2", "dino_v2"}:
        return (
            DINOFeatureScore(
                str(config.get("model_name", "dinov2_vits14")),
                device,
                int(config.get("batch_size", 8)),
                str(config.get("feature_mode", "patch")),
            ),
            "dino",
        )
    if backend != "pickscore":
        raise ValueError(f"Unsupported reward backend: {backend}")
    return (
        PickScore(
            config["processor_path"],
            config["model_path"],
            device,
            int(config.get("batch_size", 4)),
        ),
        "pickscore",
    )


class VAEWrapper(nn.Module):
    def __init__(
        self,
        vae: AutoencoderKL,
        scale: float,
        group_size: int,
        reconstruction_mode: str = "posterior",
        behavior_decode_chunk_size: int = 0,
        behavior_decoder_grad: bool = True,
    ):
        super().__init__()
        self.vae = vae
        self.scale = float(scale)
        self.group_size = int(group_size)
        self.behavior_decode_chunk_size = int(behavior_decode_chunk_size)
        self.behavior_decoder_grad = bool(behavior_decoder_grad)
        if reconstruction_mode not in {"posterior", "mean"}:
            raise ValueError(f"Unsupported reconstruction_mode: {reconstruction_mode}")
        self.reconstruction_mode = reconstruction_mode

    def forward(self, images: torch.Tensor, eps_post: torch.Tensor, eps_expl: torch.Tensor, sigma_expl: torch.Tensor):
        posterior = self.vae.encode(images).latent_dist
        mu = posterior.mean
        logvar = posterior.logvar
        post_std = torch.exp(0.5 * logvar)
        u_post = mu if self.reconstruction_mode == "mean" else mu + post_std * eps_post
        reconstruction = self.vae.decode(u_post).sample
        mu_scaled = mu * self.scale
        mu_group = mu_scaled[:, None].expand(-1, self.group_size, -1, -1, -1)
        z_scaled = mu_group.detach() + sigma_expl[None, None] * eps_expl
        behavior_latents = z_scaled.reshape(-1, *z_scaled.shape[2:]) / self.scale
        decode_chunk_size = self.behavior_decode_chunk_size or len(behavior_latents)
        behavior_chunks = []
        decode_context = nullcontext() if self.behavior_decoder_grad else torch.no_grad()
        with decode_context:
            for start in range(0, len(behavior_latents), decode_chunk_size):
                behavior_chunks.append(
                    self.vae.decode(behavior_latents[start : start + decode_chunk_size]).sample
                )
        behavior = torch.cat(behavior_chunks)
        return {
            "mu": mu,
            "logvar": logvar,
            "reconstruction": reconstruction,
            "mu_scaled": mu_scaled,
            "z_scaled": z_scaled,
            "behavior": behavior,
        }


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 32, max_channels: int = 128):
        super().__init__()
        channels = [
            base_channels,
            min(base_channels * 2, max_channels),
            min(base_channels * 4, max_channels),
            min(base_channels * 4, max_channels),
        ]
        layers: list[nn.Module] = []
        current_channels = 3
        for index, next_channels in enumerate(channels):
            stride = 2 if index + 1 < len(channels) else 1
            layers.append(
                nn.utils.spectral_norm(nn.Conv2d(current_channels, next_channels, 4, stride=stride, padding=1))
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_channels = next_channels
        layers.append(nn.utils.spectral_norm(nn.Conv2d(current_channels, 1, 3, padding=1)))
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor, return_features: bool = False):
        value = images.float()
        features = []
        for layer in self.net:
            value = layer(value)
            if isinstance(layer, nn.LeakyReLU):
                features.append(value)
        return (value, features) if return_features else value


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def normalize_group_component(group_values: torch.Tensor, center_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Put a per-sample group reward component and its center on a common z-score scale."""
    mean = group_values.mean(dim=1)
    std = group_values.std(dim=1, unbiased=False).clamp_min(1e-6)
    return (group_values - mean[:, None]) / std[:, None], (center_value - mean) / std


def latent_stats(ref: AutoencoderKL, loader: DataLoader, device: torch.device, scale: float, batches: int) -> tuple[torch.Tensor, torch.Tensor]:
    channel_sum = None
    channel_square_sum = None
    count = torch.zeros((), device=device, dtype=torch.float64)
    iterator = iter(loader)
    for _ in range(int(batches)):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        images = batch["image"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            mu = ref.encode(images).latent_dist.mean.float() * scale
        if channel_sum is None:
            channel_sum = torch.zeros(mu.shape[1], device=device, dtype=torch.float64)
            channel_square_sum = torch.zeros_like(channel_sum)
        channel_sum += mu.sum((0, 2, 3)).double()
        channel_square_sum += mu.square().sum((0, 2, 3)).double()
        count += mu.shape[0] * mu.shape[2] * mu.shape[3]
    if channel_sum is None or channel_square_sum is None:
        raise RuntimeError("Cannot estimate latent statistics from an empty loader")
    all_reduce(channel_sum)
    all_reduce(channel_square_sum)
    all_reduce(count)
    mean = channel_sum / count.clamp_min(1)
    variance = (channel_square_sum / count.clamp_min(1) - mean.square()).clamp_min(1e-6)
    return mean.float()[:, None, None], variance.sqrt().float()[:, None, None]


def viv_diag_proxy(samples: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable diagonal VIV proxy over distinct image latents.

    The paper's exact VIV requires a class-conditional full covariance.  The
    current COCO caption loader has no reliable class labels, so this proxy
    uses the marginal per-latent-element variance.  Callers must not pass the
    GRPO exploration samples here: those measure fixed exploration noise rather
    than the encoder's data distribution.
    """
    if samples.ndim != 4 or samples.shape[0] < 2:
        return samples.float().sum() * 0.0
    variance = samples.float().var(dim=0, unbiased=False)
    return (math.pi / 2.0) * torch.sqrt(variance + float(eps)).mean()


def viv_channel_std(samples: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Channel marginal standard deviation used by the VIV variance anchor."""
    if samples.ndim != 4 or samples.shape[0] < 2:
        return samples.new_zeros((samples.shape[1],))
    flattened = samples.float().permute(0, 2, 3, 1).reshape(-1, samples.shape[1])
    return torch.sqrt(flattened.var(dim=0, unbiased=False) + float(eps))


def estimate_viv_reference(
    ref: AutoencoderKL,
    loader: DataLoader,
    device: torch.device,
    scale: float,
    batches: int,
) -> torch.Tensor:
    """Estimate a fixed global VIV proxy with distributed sufficient statistics."""
    sum_latent = None
    square_sum_latent = None
    count = torch.zeros((), device=device, dtype=torch.float64)
    iterator = iter(loader)
    for _ in range(int(batches)):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        images = batch["image"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            latent = ref.encode(images).latent_dist.mean.float() * float(scale)
        if sum_latent is None:
            sum_latent = torch.zeros_like(latent[0], dtype=torch.float64)
            square_sum_latent = torch.zeros_like(sum_latent)
        sum_latent += latent.double().sum(dim=0)
        square_sum_latent += latent.double().square().sum(dim=0)
        count += latent.shape[0]
    if sum_latent is None or square_sum_latent is None or count.item() < 2:
        raise RuntimeError("Cannot estimate VIV reference from fewer than two latent samples")
    all_reduce(sum_latent)
    all_reduce(square_sum_latent)
    all_reduce(count)
    mean = sum_latent / count.clamp_min(1.0)
    variance = (square_sum_latent / count.clamp_min(1.0) - mean.square()).clamp_min(1e-6)
    return (math.pi / 2.0) * torch.sqrt(variance.float()).mean()


def estimate_class_viv_reference(
    ref: AutoencoderKL,
    loader: DataLoader,
    device: torch.device,
    scale: float,
    batches: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate class-conditional diagonal VIV with distributed sufficient statistics."""
    sum_latent = None
    square_sum_latent = None
    counts = torch.zeros(int(num_classes), device=device, dtype=torch.float64)
    iterator = iter(loader)
    for _ in range(int(batches)):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["class_id"].to(device=device, dtype=torch.long)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            latent = ref.encode(images).latent_dist.mean.float() * float(scale)
        if sum_latent is None:
            sum_latent = torch.zeros(
                int(num_classes), *latent.shape[1:], device=device, dtype=torch.float64
            )
            square_sum_latent = torch.zeros_like(sum_latent)
        for class_id in labels.unique().tolist():
            mask = labels == int(class_id)
            values = latent[mask].double()
            sum_latent[int(class_id)] += values.sum(dim=0)
            square_sum_latent[int(class_id)] += values.square().sum(dim=0)
            counts[int(class_id)] += values.shape[0]
    if sum_latent is None or square_sum_latent is None or counts.sum().item() < 2:
        raise RuntimeError("Cannot estimate class-conditional VIV reference from too few samples")
    all_reduce(sum_latent)
    all_reduce(square_sum_latent)
    all_reduce(counts)
    eligible = counts >= 2
    means = sum_latent / counts.clamp_min(1.0).view(-1, 1, 1, 1)
    variances = (
        square_sum_latent / counts.clamp_min(1.0).view(-1, 1, 1, 1) - means.square()
    ).clamp_min(1e-6)
    class_viv = (math.pi / 2.0) * torch.sqrt(variances.float()).flatten(1).mean(dim=1)
    if not bool(eligible.any()):
        raise RuntimeError("Class-conditional VIV reference has no class with two samples")
    return class_viv[eligible].mean().detach(), counts.detach()


def vae_parameter_groups(model: DDP | VAEWrapper) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    module = model.module if isinstance(model, DDP) else model
    encoder = list(module.vae.encoder.parameters()) + list(module.vae.quant_conv.parameters())
    decoder = list(module.vae.decoder.parameters()) + list(module.vae.post_quant_conv.parameters())
    grouped = {id(parameter) for parameter in encoder + decoder}
    trainable = {id(parameter) for parameter in module.vae.parameters() if parameter.requires_grad}
    if grouped != trainable:
        raise RuntimeError("Encoder/decoder parameter groups do not cover the complete VAE")
    return encoder, decoder


def reduce_metrics(metrics: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
    result = {}
    for key, value in metrics.items():
        pair = torch.stack([value.detach().float().sum(), value.detach().float().new_tensor(value.numel())]).to(device)
        all_reduce(pair)
        result[key] = float((pair[0] / pair[1].clamp_min(1)).cpu())
    return result


def make_lpips(device: torch.device) -> nn.Module | None:
    try:
        import lpips
        model = lpips.LPIPS(net="vgg", verbose=False).eval().to(device)
        model.requires_grad_(False)
        return model
    except Exception as exc:
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[warn] LPIPS unavailable, using L1 only: {exc}", flush=True)
        return None


def compute_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    scorer: RewardScorer,
    prompts: list[str],
    lpips_model: nn.Module | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    l1 = (reconstruction.float() - target.float()).abs().mean((1, 2, 3))
    mse = (reconstruction.float() - target.float()).square().mean((1, 2, 3))
    metrics = {"recon_l1": l1, "recon_mse": mse, "recon_psnr": 10.0 * torch.log10(1.0 / mse.clamp_min(1e-8))}
    if lpips_model is not None:
        metrics["recon_lpips"] = lpips_model(reconstruction.float(), target.float()).flatten()
    score = scorer.score_tensor(reconstruction, prompts, reference_images=target)
    return metrics, score


def evaluate(
    model: AutoencoderKL,
    base_model: AutoencoderKL,
    loader: DataLoader,
    device: torch.device,
    scorer: RewardScorer,
    reward_metric: str,
    lpips_model: nn.Module | None,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    base_model.eval()
    sums: dict[str, list[torch.Tensor]] = {}
    processed = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            prompts = list(batch["prompt"])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                mu = model.encode(images).latent_dist.mean
                reconstruction = model.decode(mu).sample
                base_mu = base_model.encode(images).latent_dist.mean
                base_reconstruction = base_model.decode(base_mu).sample
            metrics, score = compute_metrics(reconstruction, images, scorer, prompts, lpips_model)
            base_metrics, base_score = compute_metrics(
                base_reconstruction, images, scorer, prompts, lpips_model
            )
            metrics[reward_metric] = score
            metrics[f"base_{reward_metric}"] = base_score
            metrics[f"delta_{reward_metric}"] = score - base_score
            for key, value in base_metrics.items():
                metrics[f"base_{key}"] = value
                metrics[f"delta_{key}"] = metrics[key] - value
            for key, value in metrics.items():
                sums.setdefault(key, []).append(value.detach())
            processed += 1
            if processed >= max_batches:
                break
    local = {key: torch.cat(values).mean() for key, values in sums.items()}
    return reduce_metrics(local, device)


def save_checkpoint(
    model: DDP | VAEWrapper,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: Path,
    sigma_expl: torch.Tensor,
    config: dict[str, Any],
    discriminator: DDP | PatchDiscriminator | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    beta_policy: float | None = None,
) -> None:
    module = model.module if isinstance(model, DDP) else model
    payload = {
        "step": step,
        "vae": module.vae.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sigma_expl": sigma_expl.detach().cpu(),
        "config": config,
    }
    if beta_policy is not None:
        payload["beta_policy"] = float(beta_policy)
    if discriminator is not None:
        discriminator_module = discriminator.module if isinstance(discriminator, DDP) else discriminator
        payload["discriminator"] = discriminator_module.state_dict()
    if discriminator_optimizer is not None:
        payload["discriminator_optimizer"] = discriminator_optimizer.state_dict()
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    rank, world, device = setup_distributed()
    seed_everything(int(config["seed"]), rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    output = Path(config["output_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(Path(args.config).read_text())
    barrier()

    model_cfg = config["model"]
    tc = config["train"]
    if float(tc.get("lambda_dec_reward", 0.0)) != 0.0:
        raise ValueError(
            "Direct reward gradients into the decoder are disabled; "
            "reward scores may only affect the encoder through GRPO."
        )
    vae_path = model_cfg["checkpoint"]
    dtype = torch.float32
    vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True, torch_dtype=dtype).to(device)
    reference = AutoencoderKL.from_pretrained(vae_path, local_files_only=True, torch_dtype=dtype).to(device).eval()
    reference.requires_grad_(False)
    policy_reference_update_every = int(tc.get("policy_reference_update_every", 0))
    if policy_reference_update_every < 0:
        raise ValueError("policy_reference_update_every must be non-negative")
    policy_reference = (
        copy.deepcopy(reference) if policy_reference_update_every > 0 else reference
    )
    scale = float(model_cfg.get("scaling_factor", 0.18215))
    group_size = int(tc["group_size"])
    policy_noise_size = int(tc.get("policy_noise_size", 0))
    if policy_noise_size < 0:
        raise ValueError("policy_noise_size must be non-negative")
    policy_noise_schedule = [
        (int(start_step), int(noise_size))
        for start_step, noise_size in tc.get("policy_noise_schedule", [])
    ]
    if any(start_step < 1 or noise_size < 0 for start_step, noise_size in policy_noise_schedule):
        raise ValueError("policy_noise_schedule entries must have start_step >= 1 and noise_size >= 0")
    if policy_noise_schedule != sorted(policy_noise_schedule):
        raise ValueError("policy_noise_schedule must be sorted by start_step")
    base_rho = float(tc["rho"])
    rho_schedule = [
        (int(start_step), float(rho))
        for start_step, rho in tc.get("rho_schedule", [[1, base_rho]])
    ]
    if any(start_step < 1 or rho <= 0 for start_step, rho in rho_schedule):
        raise ValueError("rho_schedule entries must have start_step >= 1 and rho > 0")
    if rho_schedule != sorted(rho_schedule):
        raise ValueError("rho_schedule must be sorted by start_step")
    encoder_lr_schedule = [
        (int(start_step), float(learning_rate))
        for start_step, learning_rate in tc.get("encoder_lr_schedule", [])
    ]
    if any(start_step < 1 or learning_rate <= 0 for start_step, learning_rate in encoder_lr_schedule):
        raise ValueError("encoder_lr_schedule entries must have start_step >= 1 and learning_rate > 0")
    if encoder_lr_schedule != sorted(encoder_lr_schedule):
        raise ValueError("encoder_lr_schedule must be sorted by start_step")
    antithetic_exploration = bool(tc.get("antithetic_exploration", False))
    if antithetic_exploration and group_size % 2 != 0:
        raise ValueError("antithetic_exploration requires an even group_size")
    advantage_mode = str(tc.get("advantage_mode", "standardized"))
    if advantage_mode not in {"standardized", "top1", "pairwise_center"}:
        raise ValueError(f"Unsupported advantage_mode: {advantage_mode}")
    if advantage_mode == "pairwise_center" and not antithetic_exploration:
        raise ValueError("pairwise_center advantage requires antithetic_exploration")
    policy_kl_target_min = float(tc.get("policy_kl_target_min", 0.0))
    policy_kl_target_max = float(tc.get("policy_kl_target_max", float("inf")))
    if policy_kl_target_min < 0 or policy_kl_target_max <= policy_kl_target_min:
        raise ValueError("policy KL target range must satisfy 0 <= min < max")
    behavior_decoder_grad = bool(tc.get("behavior_decoder_grad", True))
    if not behavior_decoder_grad and float(tc.get("lambda_dec_anchor", 0.0)) != 0.0:
        raise ValueError("lambda_dec_anchor must be zero when behavior_decoder_grad is disabled")
    wrapper = VAEWrapper(
        vae,
        scale,
        group_size,
        tc.get("reconstruction_mode", "posterior"),
        int(tc.get("behavior_decode_chunk_size", 0)),
        behavior_decoder_grad,
    ).to(device)
    if world > 1:
        wrapper = DDP(wrapper, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    encoder_parameters, decoder_parameters = vae_parameter_groups(wrapper)
    module = wrapper.module if isinstance(wrapper, DDP) else wrapper
    quant_conv_parameters = list(module.vae.quant_conv.parameters())
    if "encoder_policy_lr" in tc:
        encoder_lr = float(tc.get("encoder_trunk_lr", tc.get("encoder_lr", tc["lr"])))
        quant_conv_lr = float(tc["encoder_policy_lr"])
        quant_conv_logvar_lr = float(tc.get("quant_conv_logvar_lr", encoder_lr))
        decoder_lr = float(tc.get("decoder_lr", tc["lr"]))
        policy_modules = [
            module.vae.encoder.mid_block,
            module.vae.encoder.conv_norm_out,
            module.vae.encoder.conv_out,
            module.vae.quant_conv,
        ]
        policy_parameters = [
            parameter for policy_module in policy_modules for parameter in policy_module.parameters()
        ]
        policy_parameter_ids = {id(parameter) for parameter in policy_parameters}
        trunk_parameters = [
            parameter
            for parameter in module.vae.encoder.parameters()
            if id(parameter) not in policy_parameter_ids
        ]
        optimizer_parameters = [
            {"params": trunk_parameters, "lr": encoder_lr},
            {"params": policy_parameters, "lr": quant_conv_lr},
            {"params": decoder_parameters, "lr": decoder_lr},
        ]
        encoder_optimizer_group_indices = (0, 1)
        decoder_optimizer_group_indices = (2,)
    elif "encoder_trunk_lr" in tc or "quant_conv_lr" in tc:
        encoder_lr = float(tc.get("encoder_trunk_lr", tc.get("encoder_lr", tc["lr"])))
        quant_conv_lr = float(tc.get("quant_conv_lr", tc.get("encoder_lr", tc["lr"])))
        quant_conv_logvar_lr = float(tc.get("quant_conv_logvar_lr", quant_conv_lr))
        decoder_lr = float(tc.get("decoder_lr", tc["lr"]))
        optimizer_parameters = [
            {"params": module.vae.encoder.parameters(), "lr": encoder_lr},
            {"params": quant_conv_parameters, "lr": quant_conv_lr},
            {"params": decoder_parameters, "lr": decoder_lr},
        ]
        encoder_optimizer_group_indices = (0, 1)
        decoder_optimizer_group_indices = (2,)
    elif "encoder_lr" in tc or "decoder_lr" in tc:
        encoder_lr = float(tc.get("encoder_lr", tc["lr"]))
        quant_conv_lr = encoder_lr
        quant_conv_logvar_lr = quant_conv_lr
        decoder_lr = float(tc.get("decoder_lr", tc["lr"]))
        optimizer_parameters = [
            {"params": encoder_parameters, "lr": encoder_lr},
            {"params": decoder_parameters, "lr": decoder_lr},
        ]
        encoder_optimizer_group_indices = (0,)
        decoder_optimizer_group_indices = (1,)
    else:
        encoder_lr = decoder_lr = float(tc["lr"])
        quant_conv_lr = encoder_lr
        quant_conv_logvar_lr = quant_conv_lr
        optimizer_parameters = wrapper.parameters()
        encoder_optimizer_group_indices = ()
        decoder_optimizer_group_indices = ()
    if encoder_lr_schedule and not encoder_optimizer_group_indices:
        raise ValueError("encoder_lr_schedule requires separate encoder/decoder optimizer groups")
    if quant_conv_logvar_lr < 0 or quant_conv_lr <= 0:
        raise ValueError("quant_conv learning rates must be non-negative, with quant_conv_lr > 0")
    quant_conv_logvar_step_scale = quant_conv_logvar_lr / quant_conv_lr
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=float(tc["lr"]), weight_decay=float(tc.get("weight_decay", 0.01))
    )
    reward_mix_cfg = tc.get("reward_mix", {}) or {}
    reward_mix_enabled = bool(reward_mix_cfg.get("enabled", False))
    reward_mix_lpips_weight = float(reward_mix_cfg.get("lpips_weight", 0.0))
    reward_mix_discriminator_weight = float(reward_mix_cfg.get("discriminator_weight", 0.0))
    reward_mix_base_weight = float(reward_mix_cfg.get("base_weight", 1.0))
    reward_mix_normalize = bool(reward_mix_cfg.get("normalize_components", True))
    normalize_base_reward = bool(tc.get("normalize_base_reward", False))
    if reward_mix_lpips_weight == 0.0 and reward_mix_discriminator_weight == 0.0:
        reward_mix_enabled = False
    discriminator_gate_cfg = tc.get("discriminator_gate", {}) or {}
    discriminator_gate_enabled = bool(discriminator_gate_cfg.get("enabled", False))
    discriminator_gate_weight = float(discriminator_gate_cfg.get("weight", 0.0))
    discriminator_gate_margin = float(discriminator_gate_cfg.get("margin", 0.0))
    discriminator_gate_start_step = int(discriminator_gate_cfg.get("start_step", 1))
    if discriminator_gate_enabled and discriminator_gate_weight <= 0.0:
        raise ValueError("discriminator_gate.weight must be positive when the gate is enabled")
    if discriminator_gate_margin < 0.0 or discriminator_gate_start_step < 1:
        raise ValueError("discriminator_gate.margin must be non-negative and start_step >= 1")
    if reward_mix_enabled and reward_mix_base_weight == 0.0:
        raise ValueError("reward_mix.base_weight must be non-zero when reward mixing is enabled")
    gan_weight = float(tc.get("gan_weight", 0.0))
    discriminator = None
    discriminator_optimizer = None
    if gan_weight > 0 or reward_mix_discriminator_weight != 0.0 or discriminator_gate_enabled:
        discriminator = PatchDiscriminator(
            base_channels=int(tc.get("gan_d_base_channels", 32)),
            max_channels=int(tc.get("gan_d_max_channels", 128)),
        ).to(device)
        if world > 1:
            discriminator = DDP(
                discriminator, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False
            )
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(tc.get("gan_d_lr", 1e-4)),
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )
    start_step = 0
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        module = wrapper.module if isinstance(wrapper, DDP) else wrapper
        module.vae.load_state_dict(resume_payload["vae"], strict=True)
        if policy_reference_update_every > 0:
            policy_reference.encoder.load_state_dict(module.vae.encoder.state_dict(), strict=True)
            policy_reference.quant_conv.load_state_dict(module.vae.quant_conv.state_dict(), strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        if "encoder_lr_resume_override" in tc:
            encoder_lr = float(tc["encoder_lr_resume_override"])
            for group_index in encoder_optimizer_group_indices:
                optimizer.param_groups[group_index]["lr"] = encoder_lr
        if "decoder_lr_resume_override" in tc:
            if not decoder_optimizer_group_indices:
                raise ValueError("decoder_lr_resume_override requires separate encoder/decoder optimizer groups")
            decoder_lr = float(tc["decoder_lr_resume_override"])
            for group_index in decoder_optimizer_group_indices:
                optimizer.param_groups[group_index]["lr"] = decoder_lr
        if discriminator is not None and "discriminator" in resume_payload:
            discriminator_module = discriminator.module if isinstance(discriminator, DDP) else discriminator
            discriminator_module.load_state_dict(resume_payload["discriminator"], strict=True)
        if discriminator_optimizer is not None and "discriminator_optimizer" in resume_payload:
            discriminator_optimizer.load_state_dict(resume_payload["discriminator_optimizer"])
        start_step = int(resume_payload["step"])
        if rank == 0:
            print(f"[resume] checkpoint={args.resume} step={start_step}", flush=True)

    reward_cfg = config["reward"]
    reward_scorer, reward_metric = make_reward_scorer(reward_cfg, device)
    lpips_model = make_lpips(device) if bool(config["train"].get("use_lpips", True)) else None

    data_cfg = config["data"]
    dataset_kind = str(data_cfg.get("backend", "coco")).lower()
    if dataset_kind in {"imagenet", "imagenet_parquet"}:
        class_names_path = data_cfg.get("class_names_path")
        train_set = ImageNetParquet(
            data_cfg["root"],
            "train",
            int(data_cfg["image_size"]),
            True,
            data_cfg.get("train_subset"),
            class_names_path,
        )
        val_set = ImageNetParquet(
            data_cfg.get("val_root", data_cfg["root"]),
            "validation",
            int(data_cfg["image_size"]),
            False,
            data_cfg.get("val_subset", 5000),
            class_names_path,
        )
        num_classes = int(data_cfg.get("num_classes", 1000))
    else:
        train_set = CocoCaptions(data_cfg["root"], "train", int(data_cfg["image_size"]), True, data_cfg.get("train_subset"))
        val_set = CocoCaptions(data_cfg["root"], "val", int(data_cfg["image_size"]), False, data_cfg.get("val_subset", 96))
        num_classes = 0
    class_conditional_viv = dataset_kind in {"imagenet", "imagenet_parquet"} and bool(tc.get("viv_class_conditional", True))
    class_balanced_viv_batches = class_conditional_viv and bool(tc.get("viv_class_balanced_batches", False))
    train_sampler = None
    train_batch_sampler = None
    val_sampler = DistributedSampler(val_set, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None
    loader_kwargs = dict(num_workers=int(data_cfg.get("num_workers", 4)), pin_memory=True)
    if class_balanced_viv_batches:
        batches = int(config["train"]["steps"]) * int(tc.get("gradient_accumulation_steps", 1)) + 1
        train_batch_sampler = ImageNetClassBalancedBatchSampler(
            train_set,
            batch_size=int(data_cfg["batch_size"]),
            num_batches=max(1, batches),
            num_classes=num_classes,
            seed=int(config["seed"]),
            rank=rank,
            class_repeat_batches=int(tc.get("viv_class_repeat_batches", 1)),
        )
        train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, **loader_kwargs)
    else:
        train_sampler = DistributedSampler(
            train_set, num_replicas=world, rank=rank, shuffle=True, seed=int(config["seed"])
        ) if world > 1 else None
        train_loader = DataLoader(
            train_set,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            batch_size=int(data_cfg["batch_size"]),
            drop_last=True,
            **loader_kwargs,
        )
    val_loader = DataLoader(val_set, sampler=val_sampler, shuffle=False, batch_size=int(data_cfg.get("eval_batch_size", 2)), num_workers=int(data_cfg.get("num_workers", 4)), pin_memory=True)
    viv_reference_loader = train_loader
    if dataset_kind in {"imagenet", "imagenet_parquet"}:
        if class_balanced_viv_batches and bool(tc.get("viv_reference_class_balanced", False)):
            reference_sampler = ImageNetClassBalancedBatchSampler(
                train_set,
                batch_size=int(data_cfg.get("viv_reference_batch_size", data_cfg["batch_size"])),
                num_batches=int(tc.get("viv_reference_batches", 128)) + 1,
                num_classes=num_classes,
                seed=int(config["seed"]) + 17,
                rank=rank,
                class_repeat_batches=int(tc.get("viv_class_repeat_batches", 1)),
            )
            viv_reference_loader = DataLoader(
                train_set,
                batch_sampler=reference_sampler,
                num_workers=0,
                pin_memory=True,
            )
        else:
            viv_reference_loader = DataLoader(
                train_set,
                shuffle=False,
                batch_size=int(data_cfg.get("viv_reference_batch_size", 4)),
                num_workers=0,
                pin_memory=True,
                drop_last=False,
            )
    if resume_payload is not None:
        sigma_expl = resume_payload["sigma_expl"].to(device=device, dtype=torch.float32)
        # sigma_expl is initialized as rho * latent_std; recover the reference
        # channel scale when resuming so VIV variance anchoring remains defined.
        latent_std = (sigma_expl / base_rho).clamp_min(float(tc.get("min_sigma", 1e-3)))
    else:
        _, latent_std = latent_stats(reference, train_loader, device, scale, int(config["train"].get("stats_batches", 16)))
        sigma_expl = (base_rho * latent_std).clamp_min(float(config["train"].get("min_sigma", 1e-3)))
    lambda_viv = float(tc.get("lambda_viv", 0.0))
    viv_target_ratio = float(tc.get("viv_target_ratio", 1.0))
    viv_objective = str(tc.get("viv_objective", "hinge")).lower()
    if viv_objective not in {"hinge", "absolute"}:
        raise ValueError("viv_objective must be hinge or absolute")
    lambda_viv_var_anchor = float(tc.get("lambda_viv_var_anchor", 0.0))
    viv_buffer_size = int(tc.get("viv_buffer_size", 128))
    viv_min_samples = int(tc.get("viv_min_samples", 16))
    viv_reference_batches = int(tc.get("viv_reference_batches", 128))
    viv_eps = float(tc.get("viv_eps", 1e-6))
    if lambda_viv < 0 or lambda_viv_var_anchor < 0:
        raise ValueError("VIV weights must be non-negative")
    if not (0 < viv_target_ratio <= 2.0):
        raise ValueError("viv_target_ratio must be in (0, 2]")
    if viv_buffer_size < 2 or viv_min_samples < 2 or viv_min_samples > viv_buffer_size:
        raise ValueError("VIV buffer requires 2 <= viv_min_samples <= viv_buffer_size")
    viv_enabled = lambda_viv > 0 or lambda_viv_var_anchor > 0
    viv_reference = None
    viv_reference_counts = None
    if viv_enabled:
        if class_conditional_viv:
            viv_reference, viv_reference_counts = estimate_class_viv_reference(
                reference,
                viv_reference_loader,
                device,
                scale,
                viv_reference_batches,
                num_classes,
            )
            viv_reference = viv_reference.clamp_min(viv_eps)
        else:
            viv_reference = estimate_viv_reference(
                reference,
                train_loader,
                device,
                scale,
                viv_reference_batches,
            ).detach().clamp_min(viv_eps)
        if rank == 0:
            print(
                f"[viv] {'class-conditional' if class_conditional_viv else 'global'} "
                f"diagonal proxy reference={float(viv_reference):.6f} "
                f"target_ratio={viv_target_ratio} buffer={viv_buffer_size} "
                f"min_samples={viv_min_samples} "
                f"eligible_reference_classes="
                f"{int((viv_reference_counts >= 2).sum().item()) if viv_reference_counts is not None else 0}",
                flush=True,
            )
    policy_kl_sigma_mode = str(tc.get("policy_kl_sigma_mode", "behavior"))
    if policy_kl_sigma_mode not in {"behavior", "fixed"}:
        raise ValueError("policy_kl_sigma_mode must be behavior or fixed")
    policy_kl_anchor_rho = float(tc.get("policy_kl_anchor_rho", base_rho))
    policy_kl_anchor_sigma = (
        sigma_expl * (policy_kl_anchor_rho / base_rho)
    ).clamp_min(float(tc.get("min_sigma", 1e-3)))
    if resume_payload is not None and "beta_policy_resume_override" in tc:
        beta_policy = float(tc["beta_policy_resume_override"])
    elif resume_payload is not None:
        beta_policy = float(resume_payload.get("beta_policy", tc.get("beta_policy", 0.01)))
    else:
        beta_policy = float(tc.get("beta_policy", 0.01))
    if rank == 0:
        print(
            f"[setup] world={world} device={device} train={len(train_set)} "
            f"rho={base_rho} rho_schedule={rho_schedule} policy_noise_size={policy_noise_size} "
            f"policy_noise_schedule={policy_noise_schedule} "
            f"encoder_lr_schedule={encoder_lr_schedule} "
            f"antithetic={antithetic_exploration} advantage_mode={advantage_mode} "
            f"reward_metric={reward_metric} policy_kl_sigma_mode={policy_kl_sigma_mode} "
            f"policy_reference_update_every={policy_reference_update_every} "
            f"behavior_decoder_grad={behavior_decoder_grad} "
            f"reward_mix_enabled={reward_mix_enabled} "
            f"normalize_base_reward={normalize_base_reward} "
            f"reward_mix_lpips_weight={reward_mix_lpips_weight} "
            f"reward_mix_discriminator_weight={reward_mix_discriminator_weight} "
            f"discriminator_gate_enabled={discriminator_gate_enabled} "
            f"discriminator_gate_weight={discriminator_gate_weight} "
            f"discriminator_gate_margin={discriminator_gate_margin} "
            f"discriminator_gate_start_step={discriminator_gate_start_step} "
            f"encoder_lr={encoder_lr} quant_conv_lr={quant_conv_lr} "
            f"quant_conv_logvar_lr={quant_conv_logvar_lr} decoder_lr={decoder_lr} "
            f"sigma_expl={sigma_expl.tolist()}",
            flush=True,
        )
    if train_sampler is not None:
        train_sampler.set_epoch(0)
    if train_batch_sampler is not None:
        train_batch_sampler.set_epoch(0)

    wandb_run = None
    if rank == 0 and bool(config.get("wandb", {}).get("enabled", True)):
        try:
            import wandb
            wandb_run = wandb.init(
                entity=config["wandb"].get("entity"), project=config["wandb"].get("project", "vae-posttraining"),
                name=config["wandb"].get("name"), config=config, dir=str(output), mode=os.environ.get("WANDB_MODE", "offline"),
            )
        except Exception as exc:
            print(f"[warn] W&B disabled: {exc}", flush=True)

    ref_logvar_cache = None
    max_steps = int(config["train"]["steps"])
    iterator = iter(train_loader)
    viv_buffer: torch.Tensor | None = None
    viv_class_buffers: dict[int, torch.Tensor] = {}
    eval_every = int(config["train"].get("eval_every", 100))
    save_every = int(config["train"].get("save_every", 100))
    accumulation_steps = int(tc.get("gradient_accumulation_steps", 1))
    gan_warmup_steps = int(tc.get("gan_warmup_steps", 0))
    gan_d_start_step = int(tc.get("gan_d_start_step", 1))
    gan_feature_matching_weight = float(tc.get("gan_feature_matching_weight", 0.0))
    beta_policy_increase = float(
        tc.get("beta_policy_increase", tc.get("beta_policy_adjustment", 1.05))
    )
    beta_policy_decrease = float(
        tc.get("beta_policy_decrease", tc.get("beta_policy_adjustment", 1.05))
    )
    beta_policy_min = float(tc.get("beta_policy_min", 1e-6))
    beta_policy_max = float(tc.get("beta_policy_max", 10.0))
    if (
        beta_policy_increase < 1.0
        or beta_policy_decrease < 1.0
        or not (0 < beta_policy_min <= beta_policy_max)
    ):
        raise ValueError("invalid adaptive beta_policy controller settings")
    discriminator_module = discriminator.module if isinstance(discriminator, DDP) else discriminator
    start = time.time()
    for step in range(start_step + 1, max_steps + 1):
        active_policy_noise_size = policy_noise_size
        for schedule_start, schedule_size in policy_noise_schedule:
            if step >= schedule_start:
                active_policy_noise_size = schedule_size
            else:
                break
        active_rho = base_rho
        for schedule_start, schedule_rho in rho_schedule:
            if step >= schedule_start:
                active_rho = schedule_rho
            else:
                break
        active_encoder_lr = encoder_lr
        for schedule_start, schedule_lr in encoder_lr_schedule:
            if step >= schedule_start:
                active_encoder_lr = schedule_lr
            else:
                break
        for group_index in encoder_optimizer_group_indices:
            optimizer.param_groups[group_index]["lr"] = active_encoder_lr
        optimizer.zero_grad(set_to_none=True)
        metric_sums: dict[str, torch.Tensor] = {}
        outputs = None
        discriminator_real_batches = []
        discriminator_fake_batches = []
        if discriminator_module is not None:
            set_requires_grad(discriminator_module, False)
        for micro_step in range(accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                if train_sampler is not None:
                    train_sampler.set_epoch(step)
                if train_batch_sampler is not None:
                    train_batch_sampler.set_epoch(step)
                iterator = iter(train_loader)
                batch = next(iterator)
            images = batch["image"].to(device, non_blocking=True)
            prompts = list(batch["prompt"])
            class_ids = (
                batch["class_id"].to(device=device, dtype=torch.long)
                if class_conditional_viv
                else None
            )
            bsz = images.shape[0]
            with torch.no_grad():
                ref_dist = policy_reference.encode(images).latent_dist
                ref_mu = ref_dist.mean.float()
                ref_logvar = ref_dist.logvar.float()
            eps_post = torch.randn_like(ref_mu)
            noise_height = active_policy_noise_size or ref_mu.shape[-2]
            noise_width = active_policy_noise_size or ref_mu.shape[-1]
            if antithetic_exploration:
                eps_half = torch.randn(
                    bsz,
                    group_size // 2,
                    ref_mu.shape[1],
                    noise_height,
                    noise_width,
                    device=device,
                    dtype=ref_mu.dtype,
                )
                if active_policy_noise_size:
                    eps_half = F.interpolate(
                        eps_half.flatten(0, 1),
                        size=ref_mu.shape[-2:],
                        mode="nearest",
                    ).view(bsz, group_size // 2, *ref_mu.shape[1:])
                eps_expl = torch.cat([eps_half, -eps_half], dim=1)
            else:
                eps_expl = torch.randn(
                    bsz,
                    group_size,
                    ref_mu.shape[1],
                    noise_height,
                    noise_width,
                    device=device,
                    dtype=ref_mu.dtype,
                )
                if active_policy_noise_size:
                    eps_expl = F.interpolate(
                        eps_expl.flatten(0, 1),
                        size=ref_mu.shape[-2:],
                        mode="nearest",
                    ).view(bsz, group_size, *ref_mu.shape[1:])
            sigma = (
                sigma_expl.to(device=device, dtype=ref_mu.dtype) * (active_rho / base_rho)
            ).clamp_min(float(tc.get("min_sigma", 1e-3)))
            sync_context = wrapper.no_sync() if isinstance(wrapper, DDP) and micro_step + 1 < accumulation_steps else nullcontext()
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    outputs = wrapper(images, eps_post, eps_expl, sigma)
                mu = outputs["mu"].float()
                logvar = outputs["logvar"].float()
                reconstruction = outputs["reconstruction"]
                behavior = outputs["behavior"]
                center_l1 = (reconstruction.float() - images.float()).abs().mean((1, 2, 3))
                center_mse = (reconstruction.float() - images.float()).square().mean((1, 2, 3))
                center_lpips = torch.zeros_like(center_l1)
                if lpips_model is not None:
                    center_lpips = lpips_model(reconstruction.float(), images.float()).flatten()
                l1_rec = center_l1.mean()
                mse_rec = center_mse.mean()
                lpips_rec = center_lpips.mean()
                behavior_group = behavior.view(bsz, group_size, *behavior.shape[1:])
                behavior_l1 = (behavior_group.float() - images[:, None].float()).abs().mean((2, 3, 4))
                repeated_images = images[:, None].expand(-1, group_size, -1, -1, -1).reshape_as(behavior)
                behavior_lpips = torch.zeros_like(behavior_l1)
                needs_behavior_lpips = (
                    float(tc.get("lpips_recon_penalty", 0.0)) > 0
                    or "elite_lpips_margin" in tc
                    or (reward_mix_enabled and reward_mix_lpips_weight != 0.0)
                )
                if lpips_model is not None and needs_behavior_lpips:
                    with torch.no_grad():
                        behavior_lpips = lpips_model(behavior.float(), repeated_images.float()).view(bsz, group_size)
                flat_prompts = [prompt for prompt in prompts for _ in range(group_size)]
                with torch.no_grad():
                    reward_scores = reward_scorer.score_tensor(
                        behavior,
                        flat_prompts,
                        reference_images=repeated_images,
                    ).view(bsz, group_size)
                    center_reward_score = reward_scorer.score_tensor(
                        reconstruction.detach(),
                        prompts,
                        reference_images=images,
                    )
                    behavior_discriminator_score = torch.zeros_like(behavior_l1)
                    center_discriminator_score = torch.zeros_like(center_l1)
                    if (
                        (reward_mix_enabled and reward_mix_discriminator_weight != 0.0)
                        or discriminator_gate_enabled
                    ):
                        behavior_discriminator_score = discriminator_module(behavior.float()).flatten(1).mean(1).view(
                            bsz, group_size
                        )
                        center_discriminator_score = discriminator_module(reconstruction.float()).flatten(1).mean(1)
                discriminator_gate_deficit = torch.zeros_like(behavior_l1)
                if discriminator_gate_enabled and step >= discriminator_gate_start_step:
                    discriminator_gate_deficit = F.relu(
                        center_discriminator_score[:, None]
                        - behavior_discriminator_score
                        - discriminator_gate_margin
                    )
                l1_excess = F.relu(behavior_l1.detach() - float(tc.get("recon_l1_threshold", 0.0)))
                lpips_excess = F.relu(behavior_lpips.detach() - float(tc.get("recon_lpips_threshold", 0.0)))
                center_l1_excess = F.relu(center_l1.detach() - float(tc.get("recon_l1_threshold", 0.0)))
                center_lpips_excess = F.relu(
                    center_lpips.detach() - float(tc.get("recon_lpips_threshold", 0.0))
                )
                if reward_mix_enabled:
                    if reward_mix_normalize:
                        base_group, base_center = normalize_group_component(
                            reward_scores.detach(), center_reward_score.detach()
                        )
                    else:
                        base_group, base_center = reward_scores.detach(), center_reward_score.detach()
                    shaped = reward_mix_base_weight * base_group
                    center_shaped = reward_mix_base_weight * base_center
                    if reward_mix_lpips_weight != 0.0:
                        lpips_group = -behavior_lpips.detach()
                        lpips_center = -center_lpips.detach()
                        if reward_mix_normalize:
                            lpips_group, lpips_center = normalize_group_component(lpips_group, lpips_center)
                        shaped = shaped + reward_mix_lpips_weight * lpips_group
                        center_shaped = center_shaped + reward_mix_lpips_weight * lpips_center
                    if reward_mix_discriminator_weight != 0.0:
                        discriminator_group = behavior_discriminator_score.detach()
                        discriminator_center = center_discriminator_score.detach()
                        if reward_mix_normalize:
                            discriminator_group, discriminator_center = normalize_group_component(
                                discriminator_group, discriminator_center
                            )
                        shaped = shaped + reward_mix_discriminator_weight * discriminator_group
                        center_shaped = center_shaped + reward_mix_discriminator_weight * discriminator_center
                else:
                    if normalize_base_reward:
                        shaped, center_shaped = normalize_group_component(
                            reward_scores.detach(), center_reward_score.detach()
                        )
                    else:
                        shaped = reward_scores.detach()
                        center_shaped = center_reward_score.detach()
                shaped = (
                    shaped
                    - float(tc.get("recon_penalty", 5.0)) * l1_excess
                    - float(tc.get("lpips_recon_penalty", 0.0)) * lpips_excess
                )
                center_shaped = (
                    center_shaped
                    - float(tc.get("recon_penalty", 5.0)) * center_l1_excess
                    - float(tc.get("lpips_recon_penalty", 0.0)) * center_lpips_excess
                )
                if discriminator_gate_enabled:
                    shaped = shaped - discriminator_gate_weight * discriminator_gate_deficit.detach()
                if advantage_mode == "top1":
                    advantage = torch.full_like(shaped, -1.0 / (group_size - 1))
                    advantage.scatter_(1, shaped.argmax(1, keepdim=True), 1.0)
                    advantage = advantage / (advantage.std(1, keepdim=True, unbiased=False) + 1e-6)
                elif advantage_mode == "standardized":
                    advantage = (shaped - shaped.mean(1, keepdim=True)) / (
                        shaped.std(1, keepdim=True, unbiased=False) + 1e-6
                    )
                else:
                    half_group = group_size // 2
                    center_relative = shaped - center_shaped[:, None]
                    center_advantage = center_relative / (
                        center_relative.std(1, keepdim=True, unbiased=False) + 1e-6
                    )
                    pair_delta = shaped[:, :half_group] - shaped[:, half_group:]
                    pair_scale = pair_delta.std(1, keepdim=True, unbiased=False) + 1e-6
                    pair_advantage = torch.cat(
                        [pair_delta / pair_scale, -pair_delta / pair_scale], dim=1
                    )
                    advantage = (
                        float(tc.get("center_advantage_weight", 0.5)) * center_advantage
                        + float(tc.get("pairwise_advantage_weight", 0.5)) * pair_advantage
                    )
                    advantage = advantage.clamp(
                        -float(tc.get("advantage_clip", 5.0)),
                        float(tc.get("advantage_clip", 5.0)),
                    )
                mu_scaled = outputs["mu_scaled"].float()
                z_scaled = outputs["z_scaled"].float()
                sigma_f = sigma.float()
                viv_proxy = reconstruction.new_zeros(())
                viv_ratio = reconstruction.new_zeros(())
                viv_loss = reconstruction.new_zeros(())
                viv_var_anchor = reconstruction.new_zeros(())
                viv_buffer_samples = 0
                viv_group_count = 0
                if viv_enabled:
                    reference_channel_std = latent_std.flatten().to(
                        device=mu_scaled.device,
                        dtype=mu_scaled.dtype,
                    )
                    viv_values = []
                    viv_anchor_values = []
                    if class_conditional_viv:
                        assert class_ids is not None
                        for class_id in class_ids.unique().tolist():
                            class_key = int(class_id)
                            current_class = mu_scaled[class_ids == class_key]
                            previous_class = viv_class_buffers.get(class_key)
                            class_samples = (
                                current_class
                                if previous_class is None
                                else torch.cat(
                                    [previous_class.to(dtype=current_class.dtype), current_class], dim=0
                                )
                            )
                            if class_samples.shape[0] >= viv_min_samples:
                                viv_values.append(viv_diag_proxy(class_samples, viv_eps))
                                if lambda_viv_var_anchor > 0:
                                    current_channel_std = viv_channel_std(class_samples, viv_eps)
                                    viv_anchor_values.append(
                                        (
                                            current_channel_std
                                            / reference_channel_std.clamp_min(viv_eps)
                                            - 1.0
                                        ).square().mean()
                                    )
                            # Cached history is detached and half precision to
                            # keep the 1000-class path bounded in GPU memory.
                            viv_class_buffers[class_key] = class_samples.detach().half()[-viv_buffer_size:]
                        if viv_values:
                            viv_proxy = torch.stack(viv_values).mean()
                            viv_group_count = len(viv_values)
                            viv_buffer_samples = sum(
                                viv_class_buffers[class_key].shape[0]
                                for class_key in class_ids.unique().tolist()
                            )
                        if viv_anchor_values:
                            viv_var_anchor = torch.stack(viv_anchor_values).mean()
                    else:
                        current_viv_samples = (
                            mu_scaled
                            if viv_buffer is None
                            else torch.cat([viv_buffer, mu_scaled], dim=0)
                        )
                        viv_buffer_samples = int(current_viv_samples.shape[0])
                        if viv_buffer_samples >= viv_min_samples:
                            viv_proxy = viv_diag_proxy(current_viv_samples, viv_eps)
                            viv_group_count = 1
                            if lambda_viv_var_anchor > 0:
                                current_channel_std = viv_channel_std(current_viv_samples, viv_eps)
                                viv_var_anchor = (
                                    current_channel_std / reference_channel_std.clamp_min(viv_eps) - 1.0
                                ).square().mean()
                        detached_current = mu_scaled.detach()
                        viv_buffer = detached_current if viv_buffer is None else torch.cat(
                            [viv_buffer, detached_current], dim=0
                        )
                        if viv_buffer.shape[0] > viv_buffer_size:
                            viv_buffer = viv_buffer[-viv_buffer_size:]
                    if viv_group_count > 0:
                        viv_ratio = viv_proxy / viv_reference.clamp_min(viv_eps)
                        viv_loss = (
                            viv_ratio
                            if viv_objective == "absolute"
                            else F.relu(viv_ratio - viv_target_ratio)
                        )
                if active_policy_noise_size:
                    policy_shape = (active_policy_noise_size, active_policy_noise_size)
                    mu_policy = F.adaptive_avg_pool2d(mu_scaled, policy_shape)
                    ref_mu_policy = F.adaptive_avg_pool2d(ref_mu * scale, policy_shape)
                    z_policy = F.adaptive_avg_pool2d(
                        z_scaled.flatten(0, 1).detach(), policy_shape
                    ).view(bsz, group_size, mu_scaled.shape[1], *policy_shape)
                else:
                    mu_policy = mu_scaled
                    ref_mu_policy = ref_mu * scale
                    z_policy = z_scaled.detach()
                logp_elements = -0.5 * ((z_policy - mu_policy[:, None]) / sigma_f[None, None]).square()
                logp_elements = logp_elements - sigma_f.log()[None, None] - 0.5 * math.log(2.0 * math.pi)
                policy_reduction = tc.get("policy_reduction", "mean")
                if policy_reduction == "sum":
                    logp = logp_elements.sum((2, 3, 4))
                elif policy_reduction == "mean":
                    logp = logp_elements.mean((2, 3, 4))
                else:
                    raise ValueError(f"Unsupported policy_reduction: {policy_reduction}")
                l_grpo = -(advantage * logp).mean()
                policy_kl_sigma = (
                    policy_kl_anchor_sigma.to(device=device, dtype=mu_policy.dtype)
                    if policy_kl_sigma_mode == "fixed"
                    else sigma_f
                )
                policy_kl_elements = 0.5 * ((mu_policy - ref_mu_policy) / policy_kl_sigma).square()
                policy_kl_per_element = policy_kl_elements.mean()
                if policy_reduction == "sum":
                    policy_kl = policy_kl_elements.sum((1, 2, 3)).mean()
                else:
                    policy_kl = policy_kl_elements.mean()
                elite_mask = shaped.gt(center_shaped[:, None])
                if "elite_l1_margin" in tc:
                    elite_mask = elite_mask & behavior_l1.detach().le(
                        center_l1.detach()[:, None] + float(tc["elite_l1_margin"])
                    )
                if "elite_lpips_margin" in tc:
                    elite_mask = elite_mask & behavior_lpips.detach().le(
                        center_lpips.detach()[:, None] + float(tc["elite_lpips_margin"])
                    )
                elite_normalized_error = 0.5 * (
                    (z_policy - mu_policy[:, None]) / sigma_f[None, None]
                ).square().sum((2, 3, 4))
                elite_count = elite_mask.float().sum()
                elite_distill = (
                    elite_normalized_error * elite_mask.float()
                ).sum() / elite_count.clamp_min(1.0)
                post_var = logvar.exp().clamp_min(1e-12)
                ref_var = ref_logvar.exp().clamp_min(1e-12)
                post_kl = 0.5 * (ref_logvar - logvar + (post_var + (mu - ref_mu).square()) / ref_var - 1.0).mean()
                prior_kl = 0.5 * (mu.square() + post_var - logvar - 1.0).mean()
                l_dec_anchor = behavior_l1.float().mean()
                gan_g_loss = reconstruction.new_zeros(())
                gan_feature_matching_loss = reconstruction.new_zeros(())
                if gan_weight > 0 and discriminator_module is not None and step > gan_warmup_steps:
                    fake_logits, fake_features = discriminator_module(reconstruction.float(), return_features=True)
                    with torch.no_grad():
                        _, real_features = discriminator_module(images.float(), return_features=True)
                    gan_g_loss = generator_hinge_loss(fake_logits)
                    gan_feature_matching_loss = torch.stack(
                        [
                            F.l1_loss(fake_feature, real_feature.detach())
                            for fake_feature, real_feature in zip(fake_features, real_features)
                        ]
                    ).mean()
                loss = (
                    float(tc.get("lambda_grpo", 10.0)) * l_grpo
                    + beta_policy * policy_kl
                    + float(tc.get("lambda_elite_distill", 0.0)) * elite_distill
                    + float(tc.get("lambda_l1", 1.0)) * l1_rec
                    + float(tc.get("lambda_lpips", 0.1)) * lpips_rec
                    + float(tc.get("lambda_prior", 1e-6)) * prior_kl
                    + float(tc.get("lambda_dec_anchor", 0.5)) * l_dec_anchor
                    + lambda_viv * viv_loss
                    + lambda_viv_var_anchor * viv_var_anchor
                    + gan_weight * (gan_g_loss + gan_feature_matching_weight * gan_feature_matching_loss)
                )
                if float(tc.get("lambda_post", 0.0)) != 0:
                    loss = loss + float(tc["lambda_post"]) * post_kl
                (loss / accumulation_steps).backward()
            if discriminator is not None:
                discriminator_real_batches.append(images.detach())
                discriminator_fake_batches.append(reconstruction.detach())
            micro_values = {
                "train/loss": loss,
                "train/grpo": l_grpo,
                "train/reward_score": reward_scores.mean(),
                "train/mixed_reward": shaped.mean(),
                f"train/{reward_metric}": reward_scores.mean(),
                f"train/center_{reward_metric}": center_reward_score.mean(),
                f"train/{reward_metric}_vs_center": (
                    reward_scores - center_reward_score[:, None]
                ).mean(),
                "train/above_center_fraction": shaped.gt(center_shaped[:, None]).float().mean(),
                f"train/group_{reward_metric}_std": reward_scores.std(1, unbiased=False).mean(),
                f"train/group_{reward_metric}_gap": (
                    reward_scores.max(1).values - reward_scores.min(1).values
                ).mean(),
                "train/group_reward_std": shaped.std(1, unbiased=False).mean(),
                "train/group_reward_gap": (shaped.max(1).values - shaped.min(1).values).mean(),
                "train/recon_l1": l1_rec,
                "train/recon_mse": mse_rec,
                "train/lpips": lpips_rec,
                "train/policy_kl": policy_kl,
                "train/policy_kl_per_element": policy_kl_per_element,
                "train/beta_policy": reconstruction.new_tensor(beta_policy),
                "train/active_rho": reconstruction.new_tensor(active_rho),
                "train/elite_distill": elite_distill,
                "train/elite_fraction": elite_mask.float().mean(),
                "train/post_kl": post_kl,
                "train/prior_kl": prior_kl,
                "train/behavior_l1": l_dec_anchor,
                "train/behavior_lpips": behavior_lpips.mean(),
                "train/behavior_discriminator_score": behavior_discriminator_score.mean(),
                "train/center_discriminator_score": center_discriminator_score.mean(),
                "train/discriminator_gate_deficit": discriminator_gate_deficit.mean(),
                "train/discriminator_gate_active_fraction": discriminator_gate_deficit.gt(0).float().mean(),
                "train/l1_hinge_fraction": (l1_excess > 0).float().mean(),
                "train/lpips_hinge_fraction": (lpips_excess > 0).float().mean(),
                "train/advantage_std": advantage.std(unbiased=False),
                "train/policy_noise_size": reconstruction.new_tensor(float(active_policy_noise_size)),
                "train/encoder_lr": reconstruction.new_tensor(active_encoder_lr),
                "train/gan_g_loss": gan_g_loss,
                "train/gan_feature_matching_loss": gan_feature_matching_loss,
                "train/viv_proxy": viv_proxy,
                "train/viv_ratio": viv_ratio,
                "train/viv_loss": viv_loss,
                "train/viv_variance_anchor": viv_var_anchor,
                "train/viv_buffer_samples": reconstruction.new_tensor(float(viv_buffer_samples)),
                "train/viv_group_count": reconstruction.new_tensor(float(viv_group_count)),
            }
            for key, value in micro_values.items():
                metric_sums[key] = metric_sums.get(key, torch.zeros_like(value.detach())) + value.detach()
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder_parameters, float(tc.get("max_encoder_grad_norm", tc.get("max_grad_norm", 1.0))))
        decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder_parameters, float(tc.get("max_decoder_grad_norm", tc.get("max_grad_norm", 1.0))))
        logvar_rows_before = []
        if quant_conv_logvar_step_scale != 1.0:
            with torch.no_grad():
                for parameter in quant_conv_parameters:
                    split = parameter.shape[0] // 2
                    logvar_rows_before.append((parameter, split, parameter[split:].clone()))
        optimizer.step()
        if logvar_rows_before:
            with torch.no_grad():
                for parameter, split, before in logvar_rows_before:
                    parameter[split:].copy_(
                        before + quant_conv_logvar_step_scale * (parameter[split:] - before)
                    )
        gan_d_loss = torch.zeros((), device=device)
        gan_d_grad_norm = torch.zeros((), device=device)
        if (
            discriminator is not None
            and discriminator_module is not None
            and discriminator_optimizer is not None
            and step >= gan_d_start_step
        ):
            set_requires_grad(discriminator_module, True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            real_images = torch.cat(discriminator_real_batches)
            fake_images = torch.cat(discriminator_fake_batches)
            real_logits = discriminator(real_images.float())
            fake_logits = discriminator(fake_images.float())
            gan_d_loss = discriminator_hinge_loss(real_logits, fake_logits)
            (float(tc.get("gan_d_weight", 1.0)) * gan_d_loss).backward()
            gan_d_grad_norm = torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), float(tc.get("gan_d_max_grad_norm", 1.0))
            )
            discriminator_optimizer.step()
            set_requires_grad(discriminator_module, False)
        log_values = {key: value / accumulation_steps for key, value in metric_sums.items()}
        log_values["train/encoder_grad_norm"] = encoder_grad_norm
        log_values["train/decoder_grad_norm"] = decoder_grad_norm
        log_values["train/gan_d_loss"] = gan_d_loss
        log_values["train/gan_d_grad_norm"] = gan_d_grad_norm
        observed_policy_kl = log_values["train/policy_kl_per_element"].detach().clone()
        all_reduce(observed_policy_kl)
        observed_policy_kl /= world
        if observed_policy_kl > policy_kl_target_max:
            beta_policy = min(beta_policy * beta_policy_increase, beta_policy_max)
        elif observed_policy_kl < policy_kl_target_min:
            beta_policy = max(beta_policy / beta_policy_decrease, beta_policy_min)
        if step == 1 and rank == 0:
            print(
                f"[step 1] loss={float(log_values['train/loss']):.5f} "
                f"grpo={float(log_values['train/grpo']):.5f} "
                f"{reward_metric}={float(log_values[f'train/{reward_metric}']):.5f} "
                f"viv_ratio={float(log_values['train/viv_ratio']):.5f} "
                f"viv_loss={float(log_values['train/viv_loss']):.5f} "
                f"enc_grad={float(encoder_grad_norm):.5f} dec_grad={float(decoder_grad_norm):.5f}",
                flush=True,
            )
        if rank == 0 and wandb_run is not None:
            wandb_run.log({key: float(value.detach().float()) for key, value in log_values.items()}, step=step)
        if step % eval_every == 0:
            barrier()
            eval_metrics = evaluate(
                wrapper.module.vae if isinstance(wrapper, DDP) else wrapper.vae,
                reference,
                val_loader,
                device,
                reward_scorer,
                reward_metric,
                lpips_model,
                int(config["train"].get("eval_batches", 24)),
            )
            if rank == 0:
                elapsed = time.time() - start
                print(f"[eval {step}] {json.dumps(eval_metrics, sort_keys=True)} elapsed={elapsed/60:.1f}m", flush=True)
                eval_history_path = output / "eval_history.json"
                eval_history = []
                if eval_history_path.exists():
                    eval_history = json.loads(eval_history_path.read_text())
                eval_history = [item for item in eval_history if int(item["step"]) != step]
                eval_history.append({"step": step, **eval_metrics})
                eval_history_path.write_text(json.dumps(sorted(eval_history, key=lambda item: item["step"]), indent=2))
                if wandb_run is not None:
                    wandb_run.log({f"eval/{key}": value for key, value in eval_metrics.items()}, step=step)
            wrapper.train()
            barrier()
            if policy_reference_update_every > 0 and step % policy_reference_update_every == 0:
                module = wrapper.module if isinstance(wrapper, DDP) else wrapper
                policy_reference.encoder.load_state_dict(module.vae.encoder.state_dict(), strict=True)
                policy_reference.quant_conv.load_state_dict(module.vae.quant_conv.state_dict(), strict=True)
                policy_reference.eval().requires_grad_(False)
                beta_policy = float(tc.get("policy_reference_reset_beta", tc.get("beta_policy", 0.01)))
                if rank == 0:
                    print(
                        f"[policy reference {step}] refreshed encoder; beta_policy={beta_policy:.6g}",
                        flush=True,
                    )
                barrier()
        if step % save_every == 0 and rank == 0:
            save_checkpoint(
                wrapper,
                optimizer,
                step,
                output / f"checkpoint_{step:06d}.pt",
                sigma_expl,
                config,
                discriminator,
                discriminator_optimizer,
                beta_policy,
            )
            assert outputs is not None
            with torch.no_grad():
                grid = make_grid(outputs["behavior"][: min(8, outputs["behavior"].shape[0])].float().clamp(-1, 1), nrow=group_size)
                save_image(grid.add(1).div(2), output / f"samples_{step:06d}.png")
    if rank == 0 and wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
