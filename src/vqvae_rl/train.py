from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.distributions import Categorical
from tqdm import tqdm

from vqvae_rl.config import load_config
from vqvae_rl.data import build_dataloader
from vqvae_rl.image_transforms import geneval_transform_variants
from vqvae_rl.models import build_model
from vqvae_rl.rewards import build_reward
from vqvae_rl.utils.dist import barrier, cleanup, init_distributed, is_main_process, reduce_mean
from vqvae_rl.utils.images import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-model-only", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rl-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-resume-batch-skip", action="store_true")
    parser.add_argument("--resume-batches-to-skip", type=int, default=None)
    parser.add_argument("--wandb-id", default=None)
    return parser.parse_args()


def seed_all(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_deterministic_algorithms(enabled: bool) -> None:
    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def cycle_loader(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def skip_batches(batches, count: int) -> None:
    if count < 0:
        raise ValueError("batch skip count must be non-negative")
    for _ in range(count):
        next(batches)


def to_device_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, list[str], list[dict[str, Any]], torch.Tensor | None, torch.Tensor | None]:
    images = batch["image"].to(device, non_blocking=True)
    prompts = list(batch["prompt"])
    raw_metadatas = batch.get("metadata")
    if raw_metadatas is None:
        metadatas = [{} for _ in prompts]
    else:
        metadatas = []
        for metadata in raw_metadatas:
            if isinstance(metadata, str):
                metadatas.append(json.loads(metadata))
            elif isinstance(metadata, dict):
                metadatas.append(metadata)
            else:
                raise TypeError(f"Unsupported metadata type: {type(metadata)!r}")
    llamagen_codes = batch.get("llamagen_code")
    if llamagen_codes is not None:
        llamagen_codes = llamagen_codes.to(device=device, dtype=torch.long, non_blocking=True)
    geneval_targets = batch.get("geneval_target")
    if geneval_targets is not None:
        geneval_targets = geneval_targets.to(device=device, non_blocking=True)
    return images, prompts, metadatas, llamagen_codes, geneval_targets


def build_wandb_settings(wandb_module):
    http_proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
    https_proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if not http_proxy and not https_proxy:
        return None

    proxies = {}
    kwargs: dict[str, Any] = {
        "init_timeout": float(os.environ.get("WANDB_INIT_TIMEOUT", "120")),
        "x_service_wait": float(os.environ.get("WANDB_SERVICE_WAIT", "60")),
        "x_graphql_timeout_seconds": float(os.environ.get("WANDB_GRAPHQL_TIMEOUT", "30")),
        "x_graphql_retry_max": int(os.environ.get("WANDB_GRAPHQL_RETRY_MAX", "3")),
    }
    if http_proxy:
        proxies["http"] = http_proxy
        kwargs["http_proxy"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
        kwargs["https_proxy"] = https_proxy
    if proxies:
        kwargs["x_proxies"] = proxies
    return wandb_module.Settings(**kwargs)


def make_repeated_prompts(prompts: list[str], num_samples: int) -> list[str]:
    return list(itertools.chain.from_iterable([[prompt] * num_samples for prompt in prompts]))


def make_repeated_metadata(metadatas: list[dict[str, Any]], num_samples: int) -> list[dict[str, Any]]:
    return list(itertools.chain.from_iterable([[metadata] * num_samples for metadata in metadatas]))


def score_reward(
    reward_model,
    images: torch.Tensor,
    prompts: list[str],
    metadatas: list[dict[str, Any]] | None = None,
    reference_images: torch.Tensor | None = None,
) -> torch.Tensor:
    if bool(getattr(reward_model, "requires_reference_images", False)):
        if reference_images is None:
            raise ValueError("This reward requires reference_images")
        return reward_model(
            images,
            prompts,
            metadatas,
            reference_images=reference_images,
        )
    return reward_model(images, prompts, metadatas)


def score_reward_tensor(
    reward_model,
    images: torch.Tensor,
    prompts: list[str],
    reference_images: torch.Tensor | None = None,
) -> torch.Tensor:
    if bool(getattr(reward_model, "requires_reference_images", False)):
        if reference_images is None:
            raise ValueError("This reward requires reference_images")
        return reward_model.score_tensor(images, prompts, reference_images)
    return reward_model.score_tensor(images, prompts)


def score_reward_tensor_prompt_pair(
    reward_model,
    images: torch.Tensor,
    primary_prompts: list[str],
    baseline_prompts: list[str],
    reference_images: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_scorer = getattr(reward_model, "score_tensor_prompt_pair", None)
    if callable(pair_scorer) and not bool(
        getattr(reward_model, "requires_reference_images", False)
    ):
        return pair_scorer(images, primary_prompts, baseline_prompts)
    return (
        score_reward_tensor(
            reward_model,
            images,
            primary_prompts,
            reference_images=reference_images,
        ),
        score_reward_tensor(
            reward_model,
            images,
            baseline_prompts,
            reference_images=reference_images,
        ),
    )


def group_advantages(scores: torch.Tensor) -> torch.Tensor:
    centered = scores - scores.mean(dim=1, keepdim=True)
    denom = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    return centered / denom


def topk_group_advantages(scores: torch.Tensor, topk: int, negative_value: float) -> torch.Tensor:
    if topk <= 0 or topk >= scores.shape[1]:
        return group_advantages(scores)
    advantages = torch.full_like(scores, -abs(float(negative_value)))
    top_indices = scores.topk(k=topk, dim=1).indices
    advantages.scatter_(1, top_indices, 1.0)
    return advantages


def mean_group_std(values: torch.Tensor) -> torch.Tensor:
    if values.shape[1] <= 1:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return values.float().std(dim=1, unbiased=False).mean()


def mean_group_corr(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.float()
    right = right.float()
    left_centered = left - left.mean(dim=1, keepdim=True)
    right_centered = right - right.mean(dim=1, keepdim=True)
    numerator = (left_centered * right_centered).mean(dim=1)
    denominator = left_centered.square().mean(dim=1).sqrt() * right_centered.square().mean(dim=1).sqrt()
    return (numerator / denominator.clamp_min(1e-6)).mean()


def load_llamagen_prior_pool(path: str | None) -> tuple[torch.Tensor | None, list[str] | None]:
    if path is None or str(path).lower() in {"", "none", "null"}:
        return None, None
    pool_path = Path(path).expanduser()
    if not pool_path.exists():
        raise FileNotFoundError(f"LlamaGen prior code pool not found: {pool_path}")
    pool = torch.load(pool_path, map_location="cpu", weights_only=False)
    if not isinstance(pool, dict) or "codes" not in pool or "prompts" not in pool:
        raise RuntimeError(f"Expected prior pool with 'codes' and 'prompts': {pool_path}")
    codes = pool["codes"].long()
    prompts = [str(prompt) for prompt in pool["prompts"]]
    if codes.ndim != 3:
        raise RuntimeError(f"Expected prior codes with shape [N,H,W], got {tuple(codes.shape)}")
    if len(prompts) != codes.shape[0]:
        raise RuntimeError(f"Prior pool has {codes.shape[0]} codes but {len(prompts)} prompts")
    return codes, prompts


def validate_no_generated_code_training(cfg: Any) -> None:
    """Reject every training path that can consume externally generated codes."""
    if not bool(getattr(cfg.train, "forbid_generated_code_training", False)):
        return

    path_fields = {
        "train.llamagen_prior_code_path": getattr(cfg.train, "llamagen_prior_code_path", None),
        "data.code_pool": getattr(cfg.data, "code_pool", None),
    }
    enabled_paths = {
        name: value
        for name, value in path_fields.items()
        if value is not None and str(value).lower() not in {"", "none", "null"}
    }
    weight_fields = {
        "train.llamagen_prior_pickscore_weight": getattr(cfg.train, "llamagen_prior_pickscore_weight", 0.0),
        "train.geneval_prior_distill_weight": getattr(cfg.train, "geneval_prior_distill_weight", 0.0),
        "train.geneval_offline_distill_weight": getattr(cfg.train, "geneval_offline_distill_weight", 0.0),
    }
    enabled_weights = {name: float(value) for name, value in weight_fields.items() if float(value) != 0.0}
    if enabled_paths or enabled_weights:
        raise ValueError(
            "forbid_generated_code_training=true rejects external code training: "
            f"paths={enabled_paths}, weights={enabled_weights}"
        )


@torch.no_grad()
def greedy_reference_codes(reference_model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    logits = reference_model.encode_logits(images)
    if logits.ndim != 4:
        raise RuntimeError(f"Expected reference logits [B,K,H,W], got {tuple(logits.shape)}")
    return logits.argmax(dim=1)


@torch.no_grad()
def reference_teacher_code_batch(
    reference_model: nn.Module,
    images: torch.Tensor,
    include_greedy: bool,
    posterior_samples: int,
    temperature: float,
) -> torch.Tensor:
    logits = reference_model.encode_logits(images)
    if logits.ndim != 4:
        raise RuntimeError(f"Expected reference logits [B,K,H,W], got {tuple(logits.shape)}")
    if posterior_samples < 0:
        raise ValueError("teacher_code_posterior_samples must be nonnegative")
    if posterior_samples > 0 and temperature <= 0.0:
        raise ValueError("teacher_code_temperature must be positive when posterior sampling is enabled")

    bsz, codebook_size, height, width = logits.shape
    batches = []
    if include_greedy:
        batches.append(logits.argmax(dim=1)[:, None])
    if posterior_samples > 0:
        flat_logits = logits.permute(0, 2, 3, 1).reshape(bsz * height * width, codebook_size).float()
        sampled = Categorical(logits=flat_logits / temperature).sample((posterior_samples,))
        sampled = sampled.view(posterior_samples, bsz, height, width).permute(1, 0, 2, 3).contiguous()
        batches.append(sampled)
    if not batches:
        raise ValueError("teacher-code training requires greedy codes or at least one posterior sample")
    return torch.cat(batches, dim=1)


@torch.no_grad()
def reference_teacher_soft_support(
    reference_model: nn.Module,
    images: torch.Tensor,
    topk: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = reference_model.encode_logits(images)
    if logits.ndim != 4:
        raise RuntimeError(f"Expected reference logits [B,K,H,W], got {tuple(logits.shape)}")
    if topk <= 0 or topk > logits.shape[1]:
        raise ValueError(f"teacher_code_soft_topk must be in [1,{logits.shape[1]}], got {topk}")
    if temperature <= 0.0:
        raise ValueError("teacher_code_temperature must be positive for soft posterior support")
    values, indices = logits.float().topk(topk, dim=1)
    weights = torch.softmax(values / temperature, dim=1)
    return indices, weights


def select_advantage_scores(
    mode: str,
    objective_scores: torch.Tensor,
    pick_scores: torch.Tensor,
    gated_pick_scores: torch.Tensor,
) -> torch.Tensor:
    mode = mode.lower()
    if mode == "objective":
        return objective_scores
    if mode == "pickscore":
        return pick_scores
    if mode == "gated_pickscore":
        return gated_pick_scores
    raise ValueError(f"Unsupported advantage_score_mode: {mode}")


def apply_recon_gate(
    pick_scores: torch.Tensor,
    recon_losses: torch.Tensor,
    threshold: float,
    mode: str,
    sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if threshold <= 0.0:
        return pick_scores, torch.ones_like(pick_scores)
    if mode == "hard":
        gate = (recon_losses.detach() <= threshold).float()
    elif mode == "soft":
        gate = torch.sigmoid((threshold - recon_losses.detach()) * sharpness)
    else:
        raise ValueError(f"Unsupported recon_gate_mode: {mode}")
    return pick_scores * gate, gate


def teacher_code_quality_gate(
    mse_per_image: torch.Tensor,
    lap_energy_ratio_per_image: torch.Tensor,
    mse_budget: float,
    lap_energy_floor: float,
    mode: str,
    sharpness: float,
) -> torch.Tensor:
    if mode not in {"hard", "soft"}:
        raise ValueError("teacher_code_pickscore_gate_mode must be hard or soft")
    gate = torch.ones_like(mse_per_image)
    if mse_budget > 0.0:
        if mode == "hard":
            gate = gate * (mse_per_image <= mse_budget).float()
        else:
            gate = gate * torch.sigmoid((mse_budget - mse_per_image) * sharpness)
    if lap_energy_floor > 0.0:
        if mode == "hard":
            gate = gate * (lap_energy_ratio_per_image >= lap_energy_floor).float()
        else:
            gate = gate * torch.sigmoid(
                (lap_energy_ratio_per_image - lap_energy_floor) * sharpness
            )
    return gate


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, max_channels: int = 128):
        super().__init__()
        channels = [
            base_channels,
            min(base_channels * 2, max_channels),
            min(base_channels * 4, max_channels),
            min(base_channels * 4, max_channels),
        ]
        layers: list[nn.Module] = []
        current_channels = in_channels
        for index, next_channels in enumerate(channels):
            stride = 2 if index < len(channels) - 1 else 1
            layers.append(nn.utils.spectral_norm(nn.Conv2d(current_channels, next_channels, 4, stride=stride, padding=1)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_channels = next_channels
        layers.append(nn.utils.spectral_norm(nn.Conv2d(current_channels, 1, 3, padding=1)))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        value = images.float()
        features = []
        for layer in self.net:
            value = layer(value)
            if isinstance(layer, nn.LeakyReLU):
                features.append(value)
        if return_features:
            return value, features
        return value


class FIDFeatureDiscriminator(nn.Module):
    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.utils.spectral_norm(nn.Linear(feature_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float())


def discriminator_input(
    source_images: torch.Tensor,
    candidate_images: torch.Tensor,
    conditional: bool,
) -> torch.Tensor:
    if conditional:
        return torch.cat([source_images, candidate_images], dim=1)
    return candidate_images


class LPIPSStylePerceptualLoss(nn.Module):
    def __init__(self, scaling_layer: nn.Module, feature_network: nn.Module):
        super().__init__()
        self.scaling_layer = scaling_layer
        self.feature_network = feature_network

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        input_features = self.feature_network(self.scaling_layer(inputs))
        with torch.no_grad():
            target_features = self.feature_network(self.scaling_layer(targets))
        layer_losses = []
        for input_feature, target_feature in zip(input_features, target_features):
            normalized_input = F.normalize(input_feature.float(), dim=1)
            normalized_target = F.normalize(target_feature.float(), dim=1)
            layer_losses.append(
                (normalized_input - normalized_target).square().mean(dim=(1, 2, 3))
            )
        return torch.stack(layer_losses, dim=0).mean(dim=0)


def build_lpips_loss(device: torch.device) -> nn.Module:
    llamagen_root = Path(__file__).resolve().parents[2] / "external" / "LlamaGen"
    if not llamagen_root.is_dir():
        raise FileNotFoundError(f"LlamaGen LPIPS source not found: {llamagen_root}")
    if str(llamagen_root) not in sys.path:
        sys.path.insert(0, str(llamagen_root))
    from tokenizer.tokenizer_image.lpips import LPIPS

    lpips = LPIPS(use_dropout=False).eval().to(device)
    loss = LPIPSStylePerceptualLoss(lpips.scaling_layer, lpips.net).eval().to(device)
    loss.requires_grad_(False)
    return loss


def build_official_lpips_loss(device: torch.device) -> nn.Module:
    import lpips

    loss = lpips.LPIPS(net="vgg", version="0.1", verbose=False).eval().to(device)
    loss.requires_grad_(False)
    return loss


def build_dists_loss(device: torch.device) -> nn.Module:
    import DISTS_pytorch
    from DISTS_pytorch import DISTS

    loss = DISTS(load_weights=False)
    weights_path = Path(DISTS_pytorch.__file__).resolve().parent / "weights.pt"
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    loss.alpha.data.copy_(weights["alpha"])
    loss.beta.data.copy_(weights["beta"])
    loss = loss.eval().to(device)
    loss.requires_grad_(False)
    return loss


class FIDInceptionFeatureLoss(nn.Module):
    def __init__(self, feature_network: nn.Module):
        super().__init__()
        self.feature_network = feature_network

    def features(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(
            images.float().add(1.0).mul(0.5),
            size=(299, 299),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
        return self.feature_network(images)[0].flatten(1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        input_features = self.features(inputs)
        with torch.no_grad():
            target_features = self.features(targets)
        return F.mse_loss(input_features.float(), target_features.float())


def build_fid_inception_feature_loss(device: torch.device) -> nn.Module:
    from cleanfid.inception_pytorch import InceptionV3

    feature_network = InceptionV3(
        output_blocks=[3],
        resize_input=False,
        normalize_input=True,
        requires_grad=False,
        use_fid_inception=True,
    ).eval().to(device)
    loss = FIDInceptionFeatureLoss(feature_network).eval().to(device)
    loss.requires_grad_(False)
    return loss


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def restrict_trainable_parameters(model: nn.Module, patterns_value: Any) -> list[str]:
    patterns = [value.strip() for value in str(patterns_value or "").split(",") if value.strip()]
    if not patterns:
        return [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    trainable = []
    for name, parameter in model.named_parameters():
        enabled = parameter.requires_grad and any(pattern in name for pattern in patterns)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if not trainable:
        raise ValueError(f"No trainable parameters matched patterns: {patterns}")
    return trainable


def validate_decoder_only_routing(cfg: Any, trainable_names: list[str]) -> None:
    if not bool(getattr(cfg.train, "strict_decoder_only", False)):
        return

    errors = []
    if not bool(getattr(cfg.model, "freeze_encoder", False)):
        errors.append("model.freeze_encoder must be true")
    if bool(getattr(cfg.model, "freeze_decoder", False)):
        errors.append("model.freeze_decoder must be false")
    if not bool(getattr(cfg.model, "freeze_codebook", False)):
        errors.append("model.freeze_codebook must be true")
    unexpected = [name for name in trainable_names if not name.startswith("tokenizer.decoder.")]
    if unexpected:
        errors.append(f"non-decoder trainable parameters: {unexpected[:5]}")
    if float(getattr(cfg.train, "policy_loss_weight", 0.0)) != 0.0:
        errors.append("train.policy_loss_weight must be zero")
    if float(getattr(cfg.train, "ref_kl_weight", 0.0)) != 0.0:
        errors.append("train.ref_kl_weight must be zero")
    if float(getattr(cfg.train, "grpo_image_pickscore_weight", 0.0)) != 0.0:
        errors.append("train.grpo_image_pickscore_weight must be zero")
    if float(getattr(cfg.train, "llamagen_prior_pickscore_weight", 0.0)) != 0.0:
        errors.append("train.llamagen_prior_pickscore_weight must be zero")
    if (
        float(getattr(cfg.train, "teacher_code_pickscore_weight", 0.0)) <= 0.0
        and float(getattr(cfg.train, "tokenizer_reward_weight", 0.0)) <= 0.0
    ):
        errors.append(
            "train.teacher_code_pickscore_weight or train.tokenizer_reward_weight must be positive"
        )
    if not bool(getattr(cfg.train, "teacher_code_detach_codebook", False)):
        errors.append("train.teacher_code_detach_codebook must be true")
    encoder_anchor_fields = (
        "encoder_anchor_recon_weight",
        "encoder_anchor_l1_weight",
        "encoder_anchor_grad_weight",
        "encoder_anchor_lap_weight",
        "encoder_anchor_kl_weight",
    )
    if any(float(getattr(cfg.train, field, 0.0)) != 0.0 for field in encoder_anchor_fields):
        errors.append("all encoder-anchor weights must be zero")
    if errors:
        raise ValueError("strict decoder-only routing failed: " + "; ".join(errors))


def clip_tokenizer_gradient_groups(model: nn.Module, max_norm: float) -> None:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    group_prefixes = (
        ("tokenizer.encoder.", "tokenizer.quant_conv."),
        ("tokenizer.quantize.",),
        ("tokenizer.decoder.", "tokenizer.post_quant_conv."),
    )
    named_parameters = list(module.named_parameters())
    matched: set[str] = set()
    for prefixes in group_prefixes:
        parameters = []
        for name, parameter in named_parameters:
            if parameter.requires_grad and name.startswith(prefixes):
                parameters.append(parameter)
                matched.add(name)
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    remaining = [
        parameter
        for name, parameter in named_parameters
        if parameter.requires_grad and name not in matched
    ]
    if remaining:
        torch.nn.utils.clip_grad_norm_(remaining, max_norm)


def tokenizer_gradient_group_norms(model: nn.Module) -> dict[str, torch.Tensor]:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    groups = {
        "encoder": ("tokenizer.encoder.", "tokenizer.quant_conv."),
        "codebook": ("tokenizer.quantize.",),
        "decoder": ("tokenizer.decoder.", "tokenizer.post_quant_conv."),
    }
    norms = {}
    for group_name, prefixes in groups.items():
        squared_norm = torch.zeros((), device=next(module.parameters()).device)
        for name, parameter in module.named_parameters():
            if parameter.grad is not None and name.startswith(prefixes):
                squared_norm = squared_norm + parameter.grad.detach().float().square().sum()
        norms[group_name] = squared_norm.sqrt()
    return norms


def build_output_saturation_targets(model: nn.Module, amount: float) -> tuple[torch.Tensor, torch.Tensor]:
    conv_out = model.tokenizer.decoder.conv_out
    if conv_out.out_channels != 3 or conv_out.bias is None:
        raise ValueError("Output saturation targeting requires a biased three-channel decoder.conv_out")
    luminance = conv_out.weight.new_tensor([0.2126, 0.7152, 0.0722])
    matrix = amount * torch.eye(3, device=conv_out.weight.device, dtype=conv_out.weight.dtype)
    matrix = matrix + (1.0 - amount) * torch.ones(
        3, 1, device=conv_out.weight.device, dtype=conv_out.weight.dtype
    ) * luminance[None, :]
    target_weight = torch.einsum("ij,jchw->ichw", matrix, conv_out.weight.detach())
    target_bias = matrix @ conv_out.bias.detach()
    return target_weight, target_bias


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def build_optimizer(model, cfg: Any) -> torch.optim.Optimizer:
    base_lr = float(cfg.train.lr)
    weight_decay = float(cfg.train.weight_decay)
    decoder_lr = float(getattr(cfg.train, "decoder_lr", 0.0))
    decoder_patterns = tuple(str(getattr(cfg.train, "decoder_lr_patterns", "decoder,post_quant_conv")).split(","))
    module = model.module if isinstance(model, DistributedDataParallel) else model
    if decoder_lr <= 0.0:
        return torch.optim.AdamW(
            [{"params": module.parameters(), "lr": base_lr, "weight_decay": weight_decay, "group_name": "base"}],
            lr=base_lr,
            weight_decay=weight_decay,
        )

    base_params = []
    decoder_params = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(pattern and pattern in name for pattern in decoder_patterns):
            decoder_params.append(parameter)
        else:
            base_params.append(parameter)
    param_groups = []
    if base_params:
        param_groups.append(
            {"params": base_params, "lr": base_lr, "weight_decay": weight_decay, "group_name": "base"}
        )
    if decoder_params:
        param_groups.append(
            {"params": decoder_params, "lr": decoder_lr, "weight_decay": weight_decay, "group_name": "decoder"}
        )
    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)


def scheduled_train_value(cfg: Any, name: str, step: int, default: float) -> float:
    """Read a piecewise-constant train weight from train.loss_schedules."""
    schedules = getattr(cfg.train, "loss_schedules", None)
    if schedules is None:
        return float(default)
    schedule = schedules.get(name) if hasattr(schedules, "get") else None
    if not schedule:
        return float(default)
    value = float(default)
    starts = []
    for stage in schedule:
        start_step = int(stage.start_step)
        if starts and start_step <= starts[-1]:
            raise ValueError(f"train.loss_schedules.{name} must have increasing start_step values")
        starts.append(start_step)
        if step >= start_step:
            value = float(stage.value)
        else:
            break
    return value


def learning_rates_for_step(cfg: Any, step: int) -> tuple[float, float]:
    base_lr = float(cfg.train.lr)
    configured_decoder_lr = float(getattr(cfg.train, "decoder_lr", 0.0))
    decoder_lr = configured_decoder_lr if configured_decoder_lr > 0.0 else base_lr
    schedule = list(getattr(cfg.train, "lr_schedule", []))
    if not schedule:
        return base_lr, decoder_lr

    starts = [int(stage.start_step) for stage in schedule]
    if starts != sorted(set(starts)) or starts[0] != 1:
        raise ValueError("train.lr_schedule must have unique increasing start_step values beginning at 1")
    for stage in schedule:
        if step < int(stage.start_step):
            break
        base_lr = float(stage.lr)
        stage_decoder_lr = getattr(stage, "decoder_lr", None)
        decoder_lr = float(stage_decoder_lr) if stage_decoder_lr is not None else base_lr
    return base_lr, decoder_lr


def apply_learning_rate_schedule(optimizer: torch.optim.Optimizer, cfg: Any, step: int) -> tuple[float, float]:
    base_lr, decoder_lr = learning_rates_for_step(cfg, step)
    for group in optimizer.param_groups:
        group["lr"] = decoder_lr if group.get("group_name") == "decoder" else base_lr
    return base_lr, decoder_lr


def image_gradient_loss_per_image(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    channels = recon.shape[1]
    dtype = recon.dtype
    device = recon.device
    sobel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)
    recon_x = F.conv2d(recon.float(), sobel_x.float(), padding=1, groups=channels)
    recon_y = F.conv2d(recon.float(), sobel_y.float(), padding=1, groups=channels)
    target_x = F.conv2d(target.float(), sobel_x.float(), padding=1, groups=channels)
    target_y = F.conv2d(target.float(), sobel_y.float(), padding=1, groups=channels)
    grad_x = (recon_x - target_x).abs().mean(dim=(1, 2, 3))
    grad_y = (recon_y - target_y).abs().mean(dim=(1, 2, 3))
    return grad_x + grad_y


def image_gradient_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return image_gradient_loss_per_image(recon, target).mean()


def image_laplacian_loss_per_image(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    channels = recon.shape[1]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=recon.device,
        dtype=recon.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(channels, 1, 1, 1)
    recon_lap = F.conv2d(recon.float(), kernel.float(), padding=1, groups=channels)
    target_lap = F.conv2d(target.float(), kernel.float(), padding=1, groups=channels)
    return (recon_lap - target_lap).abs().mean(dim=(1, 2, 3))


def image_laplacian_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return image_laplacian_loss_per_image(recon, target).mean()


def image_laplacian_energy_per_image(images: torch.Tensor) -> torch.Tensor:
    channels = images.shape[1]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(channels, 1, 1, 1)
    response = F.conv2d(images.float(), kernel.float(), padding=1, groups=channels)
    return response.abs().mean(dim=(1, 2, 3))


def gaussian_highpass(images: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError(f"highpass sigma must be positive, got {sigma}")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=images.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * coordinates.square() / (sigma * sigma))
    kernel = (kernel / kernel.sum()).to(dtype=images.dtype)
    channels = images.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    blurred = F.pad(images, (radius, radius, 0, 0), mode="reflect")
    blurred = F.conv2d(blurred, horizontal, groups=channels)
    blurred = F.pad(blurred, (0, 0, radius, radius), mode="reflect")
    blurred = F.conv2d(blurred, vertical, groups=channels)
    return images - blurred


def image_highpass_loss(recon: torch.Tensor, target: torch.Tensor, sigma: float) -> torch.Tensor:
    return F.l1_loss(
        gaussian_highpass(recon.float(), sigma),
        gaussian_highpass(target.float(), sigma),
    )


def module_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(module, DistributedDataParallel):
        module = module.module
    return module.state_dict()


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    step: int,
    phase: str,
    cfg: Any,
    discriminator=None,
    discriminator_optimizer=None,
    fid_discriminator=None,
    fid_discriminator_optimizer=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": module_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "phase": phase,
        "config": to_plain_dict(cfg),
    }
    if discriminator is not None:
        checkpoint["discriminator"] = module_state_dict(discriminator)
    if discriminator_optimizer is not None:
        checkpoint["discriminator_optimizer"] = discriminator_optimizer.state_dict()
    if fid_discriminator is not None:
        checkpoint["fid_discriminator"] = module_state_dict(fid_discriminator)
    if fid_discriminator_optimizer is not None:
        checkpoint["fid_discriminator_optimizer"] = fid_discriminator_optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    discriminator=None,
    discriminator_optimizer=None,
    fid_discriminator=None,
    fid_discriminator_optimizer=None,
) -> tuple[int, str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    module = model.module if isinstance(model, DistributedDataParallel) else model
    module.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if discriminator is not None and "discriminator" in checkpoint:
        (discriminator.module if isinstance(discriminator, DistributedDataParallel) else discriminator).load_state_dict(
            checkpoint["discriminator"]
        )
    if discriminator_optimizer is not None and "discriminator_optimizer" in checkpoint:
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
    if fid_discriminator is not None and "fid_discriminator" in checkpoint:
        (fid_discriminator.module if isinstance(fid_discriminator, DistributedDataParallel) else fid_discriminator).load_state_dict(
            checkpoint["fid_discriminator"]
        )
    if fid_discriminator_optimizer is not None and "fid_discriminator_optimizer" in checkpoint:
        fid_discriminator_optimizer.load_state_dict(checkpoint["fid_discriminator_optimizer"])
    return int(checkpoint.get("step", 0)), str(checkpoint.get("phase", "pretrain"))


def to_plain_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    return value


def log_metrics(metrics: dict[str, torch.Tensor | float], step: int, prefix: str, wandb_run) -> None:
    reduced: dict[str, float] = {}
    for key, value in metrics.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(float(value))
        if tensor.ndim == 0:
            reduced[f"{prefix}/{key}"] = float(reduce_mean(tensor.float()).cpu())
    if is_main_process():
        text = " ".join(f"{k}={v:.4f}" for k, v in reduced.items())
        print(f"step={step} {text}", flush=True)
        if wandb_run is not None:
            wandb_run.log(reduced, step=step)


def maybe_log_samples(model, images: torch.Tensor, output_dir: Path, step: int, wandb_run) -> None:
    if not is_main_process():
        return
    module = model.module if isinstance(model, DistributedDataParallel) else model
    module.eval()
    with torch.no_grad():
        recon = module(images[:8], mode="greedy")
    module.train()
    grid_images = torch.cat([images[:8], recon[:8]], dim=0)
    path = output_dir / "samples" / f"step_{step:07d}.png"
    save_image_grid(grid_images, path, nrow=8)
    if wandb_run is not None:
        import wandb

        wandb_run.log({"samples/recon": wandb.Image(str(path))}, step=step)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        cfg.train.batch_size = args.batch_size
    if args.rl_steps is not None:
        if args.rl_steps < 0:
            raise ValueError("--rl-steps must be non-negative")
        cfg.train.rl_steps = args.rl_steps
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.no_resume_batch_skip:
        cfg.train.skip_batches_on_resume = False
    if args.resume_batches_to_skip is not None:
        if args.resume_batches_to_skip < 0:
            raise ValueError("--resume-batches-to-skip must be non-negative")
        cfg.train.resume_batches_to_skip = args.resume_batches_to_skip
    if args.wandb_id is not None:
        cfg.wandb.id = args.wandb_id
        cfg.wandb.name = args.wandb_id
    configure_deterministic_algorithms(bool(getattr(cfg.train, "deterministic_algorithms", False)))
    distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed_all(int(cfg.seed), rank)

    output_dir = Path(cfg.output_dir)
    if bool(getattr(cfg.train, "fresh_run_only", False)) and any((output_dir / "checkpoints").glob("*.pt")):
        raise RuntimeError(f"Fresh run required, but checkpoints already exist in {output_dir}")
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    loader_bundle = build_dataloader(
        cfg.data,
        cfg.reward,
        batch_size=int(cfg.train.batch_size),
        distributed=distributed,
    )
    batches = cycle_loader(loader_bundle.dataloader, loader_bundle.sampler)

    model = build_model(cfg.model).to(device)
    trainable_names = restrict_trainable_parameters(
        model,
        getattr(cfg.train, "trainable_parameter_patterns", None),
    )
    validate_decoder_only_routing(cfg, trainable_names)
    if is_main_process() and getattr(cfg.train, "trainable_parameter_patterns", None):
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"Restricted training to {trainable_count} parameters: {trainable_names}")
    output_saturation_weight = float(getattr(cfg.train, "output_saturation_weight", 0.0))
    output_saturation_targets = None
    if output_saturation_weight > 0.0:
        output_saturation_targets = build_output_saturation_targets(
            model,
            float(cfg.train.output_saturation_amount),
        )
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    gan_weight = float(getattr(cfg.train, "gan_weight", 0.0))
    gan_warmup_steps = int(getattr(cfg.train, "gan_warmup_steps", 0))
    gan_d_start_step = int(
        getattr(cfg.train, "gan_d_start_step", gan_warmup_steps + 1)
    )
    gan_d_weight = float(getattr(cfg.train, "gan_d_weight", 1.0))
    gan_d_lr = float(getattr(cfg.train, "gan_d_lr", cfg.train.lr))
    gan_d_base_channels = int(getattr(cfg.train, "gan_d_base_channels", 32))
    gan_d_max_channels = int(getattr(cfg.train, "gan_d_max_channels", 128))
    gan_conditional = bool(getattr(cfg.train, "gan_conditional", False))
    gan_feature_matching_weight = float(getattr(cfg.train, "gan_feature_matching_weight", 0.0))
    discriminator = None
    discriminator_optimizer = None
    if gan_weight > 0.0:
        discriminator = PatchDiscriminator(
            in_channels=6 if gan_conditional else 3,
            base_channels=gan_d_base_channels,
            max_channels=gan_d_max_channels,
        ).to(device)
        if distributed:
            discriminator = DistributedDataParallel(discriminator, device_ids=[local_rank], output_device=local_rank)
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=gan_d_lr,
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )

    fid_gan_weight = float(getattr(cfg.train, "fid_gan_weight", 0.0))
    fid_gan_warmup_steps = int(getattr(cfg.train, "fid_gan_warmup_steps", 0))
    fid_gan_d_start_step = int(
        getattr(cfg.train, "fid_gan_d_start_step", fid_gan_warmup_steps + 1)
    )
    fid_gan_d_weight = float(getattr(cfg.train, "fid_gan_d_weight", 1.0))
    fid_gan_d_lr = float(getattr(cfg.train, "fid_gan_d_lr", cfg.train.lr))
    fid_gan_hidden_dim = int(getattr(cfg.train, "fid_gan_hidden_dim", 512))
    fid_gan_batch_size = int(getattr(cfg.train, "fid_gan_batch_size", cfg.train.batch_size))
    if fid_gan_batch_size <= 0:
        raise ValueError("fid_gan_batch_size must be positive")
    fid_discriminator = None
    fid_discriminator_optimizer = None
    if fid_gan_weight > 0.0:
        fid_discriminator = FIDFeatureDiscriminator(hidden_dim=fid_gan_hidden_dim).to(device)
        if distributed:
            fid_discriminator = DistributedDataParallel(
                fid_discriminator,
                device_ids=[local_rank],
                output_device=local_rank,
            )
        fid_discriminator_optimizer = torch.optim.AdamW(
            fid_discriminator.parameters(),
            lr=fid_gan_d_lr,
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )

    ref_model = None
    ref_checkpoint = getattr(cfg.train, "ref_checkpoint", None)
    if ref_checkpoint and str(ref_checkpoint).lower() not in {"none", "null", ""}:
        ref_model = build_model(cfg.model).to(device)
        if str(ref_checkpoint).lower() != "base":
            load_checkpoint(str(ref_checkpoint), ref_model, optimizer=None)
        ref_model.requires_grad_(False)
        ref_model.eval()

    detail_ref_model = ref_model
    detail_ref_checkpoint = getattr(cfg.train, "teacher_code_detail_ref_checkpoint", None)
    if detail_ref_checkpoint and str(detail_ref_checkpoint).lower() not in {"none", "null", ""}:
        if str(detail_ref_checkpoint) != str(ref_checkpoint):
            detail_ref_model = build_model(cfg.model).to(device)
            if str(detail_ref_checkpoint).lower() != "base":
                load_checkpoint(str(detail_ref_checkpoint), detail_ref_model, optimizer=None)
            detail_ref_model.requires_grad_(False)
            detail_ref_model.eval()

    optimizer = build_optimizer(model, cfg)

    resume_step = 0
    resume_phase = "pretrain"
    if args.resume is not None:
        resume_step, resume_phase = load_checkpoint(
            args.resume,
            model,
            None if args.resume_model_only else optimizer,
            discriminator=None if args.resume_model_only else discriminator,
            discriminator_optimizer=(
                None if args.resume_model_only else discriminator_optimizer
            ),
            fid_discriminator=None if args.resume_model_only else fid_discriminator,
            fid_discriminator_optimizer=(
                None if args.resume_model_only else fid_discriminator_optimizer
            ),
        )
    elif is_main_process():
        save_checkpoint(
            output_dir / "checkpoints" / "rl_0000000.pt",
            model,
            optimizer,
            0,
            "rl",
            cfg,
            discriminator,
            discriminator_optimizer,
            fid_discriminator,
            fid_discriminator_optimizer,
        )
    barrier()

    wandb_run = None
    use_wandb = bool(cfg.wandb.enabled) and not args.no_wandb and is_main_process()
    if use_wandb:
        import wandb

        wandb_settings = build_wandb_settings(wandb)
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            id=getattr(cfg.wandb, "id", None),
            resume=getattr(cfg.wandb, "resume", None),
            config=dict(cfg),
            settings=wandb_settings,
        )

    precision = str(cfg.train.precision)
    total_pretrain_steps = int(cfg.train.pretrain_steps)
    total_rl_steps = int(cfg.train.rl_steps)
    recon_reward_weight = float(cfg.train.recon_reward_weight)
    pickscore_reward_weight = float(cfg.train.pickscore_reward_weight)
    recon_gate_threshold = float(getattr(cfg.train, "recon_gate_threshold", 0.0))
    recon_gate_penalty = float(getattr(cfg.train, "recon_gate_penalty", 0.0))
    recon_gate_mode = str(getattr(cfg.train, "recon_gate_mode", "hard"))
    recon_gate_sharpness = float(getattr(cfg.train, "recon_gate_sharpness", 200.0))
    rl_update_epochs = int(getattr(cfg.train, "rl_update_epochs", 1))
    advantage_clip = float(getattr(cfg.train, "advantage_clip", 0.0))
    advantage_mode = str(getattr(cfg.train, "advantage_mode", "standard"))
    advantage_score_mode = str(getattr(cfg.train, "advantage_score_mode", "objective"))
    advantage_topk = int(getattr(cfg.train, "advantage_topk", 1))
    advantage_topk_negative = float(getattr(cfg.train, "advantage_topk_negative", 0.25))
    target_kl = float(getattr(cfg.train, "target_kl", 0.0))
    normalize_logprob_by_sites = bool(getattr(cfg.train, "normalize_logprob_by_sites", False))
    policy_loss_weight = float(getattr(cfg.train, "policy_loss_weight", 1.0))
    ref_kl_weight = float(getattr(cfg.train, "ref_kl_weight", 0.0))
    sample_grad_reward_weight = float(getattr(cfg.train, "sample_grad_reward_weight", 0.0))
    sample_lap_reward_weight = float(getattr(cfg.train, "sample_lap_reward_weight", 0.0))
    tokenizer_recon_weight = float(getattr(cfg.train, "tokenizer_recon_weight", 0.0))
    tokenizer_l1_weight = float(getattr(cfg.train, "tokenizer_l1_weight", 0.0))
    tokenizer_grad_weight = float(getattr(cfg.train, "tokenizer_grad_weight", 0.0))
    tokenizer_lap_weight = float(getattr(cfg.train, "tokenizer_lap_weight", 0.0))
    tokenizer_perceptual_weight = float(getattr(cfg.train, "tokenizer_perceptual_weight", 0.0))
    tokenizer_official_lpips_weight = float(
        getattr(cfg.train, "tokenizer_official_lpips_weight", 0.0)
    )
    tokenizer_official_lpips_batch_size = int(
        getattr(cfg.train, "tokenizer_official_lpips_batch_size", cfg.train.batch_size)
    )
    tokenizer_dists_weight = float(getattr(cfg.train, "tokenizer_dists_weight", 0.0))
    tokenizer_dists_batch_size = int(
        getattr(cfg.train, "tokenizer_dists_batch_size", cfg.train.batch_size)
    )
    tokenizer_ssim_weight = float(getattr(cfg.train, "tokenizer_ssim_weight", 0.0))
    tokenizer_ms_ssim_weight = float(getattr(cfg.train, "tokenizer_ms_ssim_weight", 0.0))
    tokenizer_ssim_batch_size = int(
        getattr(cfg.train, "tokenizer_ssim_batch_size", cfg.train.batch_size)
    )
    tokenizer_reward_weight = float(getattr(cfg.train, "tokenizer_reward_weight", 0.0))
    tokenizer_reward_prompt_mode = str(
        getattr(cfg.train, "tokenizer_reward_prompt_mode", "matched")
    ).lower()
    if tokenizer_reward_prompt_mode not in {"matched", "batch_roll", "constant"}:
        raise ValueError(
            "tokenizer_reward_prompt_mode must be matched, batch_roll, or constant, got "
            f"{tokenizer_reward_prompt_mode!r}"
        )
    tokenizer_reward_constant_prompt = str(
        getattr(
            cfg.train,
            "tokenizer_reward_constant_prompt",
            getattr(cfg.reward, "default_prompt", "a photo"),
        )
    )
    tokenizer_reward_baseline_prompt_mode = str(
        getattr(cfg.train, "tokenizer_reward_baseline_prompt_mode", "none")
    ).lower()
    if tokenizer_reward_baseline_prompt_mode not in {"none", "batch_roll", "constant"}:
        raise ValueError(
            "tokenizer_reward_baseline_prompt_mode must be none, batch_roll, or constant, got "
            f"{tokenizer_reward_baseline_prompt_mode!r}"
        )
    tokenizer_reward_baseline_weight = float(
        getattr(cfg.train, "tokenizer_reward_baseline_weight", 1.0)
    )
    if tokenizer_reward_baseline_weight < 0.0:
        raise ValueError("tokenizer_reward_baseline_weight must be non-negative")
    tokenizer_reward_project_against_quality = bool(
        getattr(cfg.train, "tokenizer_reward_project_against_quality", False)
    )
    tokenizer_quality_guard_against_base = bool(
        getattr(cfg.train, "tokenizer_quality_guard_against_base", False)
    )
    quality_guard_tolerances = {
        name: float(getattr(cfg.train, f"tokenizer_quality_guard_{name}_tolerance", 0.0))
        for name in (
            "mse",
            "l1",
            "grad",
            "lap",
            "lpips",
            "dists",
            "ssim",
            "ms_ssim",
            "fid_feature",
        )
    }
    if tokenizer_quality_guard_against_base and ref_model is None:
        raise ValueError("tokenizer_quality_guard_against_base requires train.ref_checkpoint")
    if tokenizer_official_lpips_batch_size <= 0:
        raise ValueError("tokenizer_official_lpips_batch_size must be positive")
    if tokenizer_dists_batch_size <= 0:
        raise ValueError("tokenizer_dists_batch_size must be positive")
    if tokenizer_ssim_batch_size <= 0:
        raise ValueError("tokenizer_ssim_batch_size must be positive")
    tokenizer_fid_feature_weight = float(
        getattr(cfg.train, "tokenizer_fid_feature_weight", 0.0)
    )
    tokenizer_fid_feature_batch_size = int(
        getattr(cfg.train, "tokenizer_fid_feature_batch_size", cfg.train.batch_size)
    )
    if tokenizer_fid_feature_batch_size <= 0:
        raise ValueError("tokenizer_fid_feature_batch_size must be positive")
    tokenizer_vq_weight = float(getattr(cfg.train, "tokenizer_vq_weight", 0.0))
    encoder_anchor_recon_weight = float(getattr(cfg.train, "encoder_anchor_recon_weight", 0.0))
    encoder_anchor_l1_weight = float(getattr(cfg.train, "encoder_anchor_l1_weight", 0.0))
    encoder_anchor_grad_weight = float(getattr(cfg.train, "encoder_anchor_grad_weight", 0.0))
    encoder_anchor_lap_weight = float(getattr(cfg.train, "encoder_anchor_lap_weight", 0.0))
    encoder_anchor_kl_weight = float(getattr(cfg.train, "encoder_anchor_kl_weight", 0.0))
    encoder_anchor_temperature = float(getattr(cfg.train, "encoder_anchor_temperature", 0.07))
    encoder_anchor_reference_quantize = bool(
        getattr(cfg.train, "encoder_anchor_reference_quantize", False)
    )
    encoder_anchor_enabled = any(
        weight > 0.0
        for weight in (
            encoder_anchor_recon_weight,
            encoder_anchor_l1_weight,
            encoder_anchor_grad_weight,
            encoder_anchor_lap_weight,
            encoder_anchor_kl_weight,
        )
    )
    teacher_code_pickscore_weight = float(getattr(cfg.train, "teacher_code_pickscore_weight", 0.0))
    teacher_code_recon_weight = float(getattr(cfg.train, "teacher_code_recon_weight", 0.0))
    teacher_code_l1_weight = float(getattr(cfg.train, "teacher_code_l1_weight", 0.0))
    teacher_code_grad_weight = float(getattr(cfg.train, "teacher_code_grad_weight", 0.0))
    teacher_code_lap_weight = float(getattr(cfg.train, "teacher_code_lap_weight", 0.0))
    teacher_code_include_greedy = bool(getattr(cfg.train, "teacher_code_include_greedy", True))
    teacher_code_posterior_samples = int(getattr(cfg.train, "teacher_code_posterior_samples", 0))
    teacher_code_soft_topk = int(getattr(cfg.train, "teacher_code_soft_topk", 0))
    teacher_code_temperature = float(getattr(cfg.train, "teacher_code_temperature", 0.07))
    teacher_code_detach_codebook = bool(getattr(cfg.train, "teacher_code_detach_codebook", False))
    teacher_code_pickscore_mse_budget = float(
        getattr(cfg.train, "teacher_code_pickscore_mse_budget", 0.0)
    )
    teacher_code_pickscore_lap_energy_floor = float(
        getattr(cfg.train, "teacher_code_pickscore_lap_energy_floor", 0.0)
    )
    teacher_code_pickscore_gate_mode = str(
        getattr(cfg.train, "teacher_code_pickscore_gate_mode", "hard")
    )
    teacher_code_pickscore_gate_sharpness = float(
        getattr(cfg.train, "teacher_code_pickscore_gate_sharpness", 80.0)
    )
    teacher_code_lap_energy_penalty_weight = float(
        getattr(cfg.train, "teacher_code_lap_energy_penalty_weight", 0.0)
    )
    teacher_code_lap_energy_penalty_floor = float(
        getattr(cfg.train, "teacher_code_lap_energy_penalty_floor", 0.0)
    )
    teacher_code_reference_l1_weight = float(
        getattr(cfg.train, "teacher_code_reference_l1_weight", 0.0)
    )
    teacher_code_reference_grad_weight = float(
        getattr(cfg.train, "teacher_code_reference_grad_weight", 0.0)
    )
    teacher_code_reference_lap_weight = float(
        getattr(cfg.train, "teacher_code_reference_lap_weight", 0.0)
    )
    teacher_code_reference_detail_weight = float(
        getattr(cfg.train, "teacher_code_reference_detail_weight", 0.0)
    )
    teacher_code_reference_detail_sigma = float(
        getattr(cfg.train, "teacher_code_reference_detail_sigma", 1.0)
    )
    if teacher_code_reference_detail_weight > 0.0 and teacher_code_reference_detail_sigma <= 0.0:
        raise ValueError("teacher_code_reference_detail_sigma must be positive")
    teacher_code_reference_enabled = any(
        weight > 0.0
        for weight in (
            teacher_code_reference_l1_weight,
            teacher_code_reference_grad_weight,
            teacher_code_reference_lap_weight,
            teacher_code_reference_detail_weight,
        )
    )
    teacher_code_enabled = any(
        weight > 0.0
        for weight in (
            teacher_code_pickscore_weight,
            teacher_code_recon_weight,
            teacher_code_l1_weight,
            teacher_code_grad_weight,
            teacher_code_lap_weight,
            teacher_code_lap_energy_penalty_weight,
            teacher_code_reference_l1_weight,
            teacher_code_reference_grad_weight,
            teacher_code_reference_lap_weight,
            teacher_code_reference_detail_weight,
        )
    )
    perceptual_loss = None
    if tokenizer_perceptual_weight > 0.0:
        perceptual_loss = build_lpips_loss(device)
    official_lpips_loss = None
    if tokenizer_official_lpips_weight > 0.0:
        official_lpips_loss = build_official_lpips_loss(device)
    dists_loss = None
    if tokenizer_dists_weight > 0.0:
        dists_loss = build_dists_loss(device)
    fid_feature_loss = None
    if tokenizer_fid_feature_weight > 0.0 or fid_gan_weight > 0.0:
        fid_feature_loss = build_fid_inception_feature_loss(device)
    grpo_image_pickscore_weight = float(getattr(cfg.train, "grpo_image_pickscore_weight", 0.0))
    grpo_image_positive_advantage = bool(getattr(cfg.train, "grpo_image_positive_advantage", True))
    grpo_image_topk = int(getattr(cfg.train, "grpo_image_topk", 0))
    llamagen_prior_pickscore_weight = float(getattr(cfg.train, "llamagen_prior_pickscore_weight", 0.0))
    llamagen_prior_batch_size = int(getattr(cfg.train, "llamagen_prior_batch_size", 0))
    geneval_prior_distill_weight = float(getattr(cfg.train, "geneval_prior_distill_weight", 0.0))
    geneval_prior_transform_distill = bool(getattr(cfg.train, "geneval_prior_transform_distill", False))
    geneval_offline_distill_weight = float(getattr(cfg.train, "geneval_offline_distill_weight", 0.0))
    validate_no_generated_code_training(cfg)
    if bool(getattr(cfg.train, "strict_reward_gradient_routing", False)):
        routing_errors = []
        if not bool(getattr(cfg.model, "freeze_encoder", False)):
            routing_errors.append("model.freeze_encoder must be true")
        if bool(getattr(cfg.model, "freeze_decoder", False)):
            routing_errors.append("model.freeze_decoder must be false")
        if bool(getattr(cfg.model, "freeze_codebook", False)):
            routing_errors.append("model.freeze_codebook must be false")
        if bool(getattr(cfg.model, "policy_detach_codebook", False)):
            routing_errors.append("model.policy_detach_codebook must be false")
        if pickscore_reward_weight <= 0.0:
            routing_errors.append("train.pickscore_reward_weight must be positive")
        if teacher_code_pickscore_weight <= 0.0:
            routing_errors.append("train.teacher_code_pickscore_weight must be positive")
        if not teacher_code_detach_codebook:
            routing_errors.append("train.teacher_code_detach_codebook must be true")
        if grpo_image_pickscore_weight != 0.0:
            routing_errors.append("train.grpo_image_pickscore_weight must be zero")
        if encoder_anchor_enabled:
            routing_errors.append("all encoder-anchor weights must be zero")
        if routing_errors:
            raise ValueError("strict reward-gradient routing failed: " + "; ".join(routing_errors))
    if teacher_code_enabled and ref_model is None:
        raise ValueError("teacher-code training requires train.ref_checkpoint")
    if teacher_code_reference_detail_weight > 0.0 and detail_ref_model is None:
        raise ValueError(
            "teacher-code detail training requires train.teacher_code_detail_ref_checkpoint "
            "or train.ref_checkpoint"
        )
    if encoder_anchor_enabled and ref_model is None:
        raise ValueError("encoder-anchor training requires train.ref_checkpoint")
    if encoder_anchor_kl_weight > 0.0 and encoder_anchor_temperature <= 0.0:
        raise ValueError("encoder_anchor_temperature must be positive")
    if (
        teacher_code_enabled
        and not teacher_code_include_greedy
        and teacher_code_posterior_samples <= 0
        and teacher_code_soft_topk <= 0
    ):
        raise ValueError("teacher-code training requires greedy codes, posterior samples, or soft support")
    llamagen_prior_codes, llamagen_prior_prompts = load_llamagen_prior_pool(
        getattr(cfg.train, "llamagen_prior_code_path", None)
    )
    if llamagen_prior_pickscore_weight > 0.0 and (llamagen_prior_codes is None or llamagen_prior_prompts is None):
        raise ValueError("llamagen_prior_pickscore_weight > 0 requires train.llamagen_prior_code_path")
    if llamagen_prior_codes is not None and is_main_process():
        print(f"Loaded LlamaGen prior code pool with {llamagen_prior_codes.shape[0]} samples", flush=True)
    global_step = resume_step if resume_phase == "rl" else 0

    model.train()
    if total_pretrain_steps > 0 and resume_phase != "pretrain_done":
        pretrain_iter = range(1, total_pretrain_steps + 1)
        if is_main_process():
            pretrain_iter = tqdm(pretrain_iter, desc="pretrain")
        for step in pretrain_iter:
            images, _, _, _, _ = to_device_batch(next(batches), device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                recon, _, entropy = model(images, mode="pretrain", temperature=float(cfg.train.temperature))
                recon_loss = F.mse_loss(recon.float(), images.float())
                loss = recon_loss - float(cfg.train.entropy_weight) * entropy.float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.max_grad_norm))
            optimizer.step()
            global_step += 1

            if step % int(cfg.train.log_every) == 0:
                log_metrics(
                    {"loss": loss.detach(), "recon_mse": recon_loss.detach(), "entropy": entropy.detach()},
                    global_step,
                    "pretrain",
                    wandb_run,
                )
            if step % int(cfg.train.sample_every) == 0:
                maybe_log_samples(model, images, output_dir, global_step, wandb_run)
            if is_main_process() and step % int(cfg.train.save_every) == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / f"pretrain_{step:07d}.pt",
                    model,
                    optimizer,
                    step,
                    "pretrain",
                    cfg,
                    discriminator,
                    discriminator_optimizer,
                    fid_discriminator,
                    fid_discriminator_optimizer,
                )

        if is_main_process():
            save_checkpoint(
                output_dir / "checkpoints" / "pretrain_last.pt",
                model,
                optimizer,
                total_pretrain_steps,
                "pretrain_done",
                cfg,
                discriminator,
                discriminator_optimizer,
                fid_discriminator,
                fid_discriminator_optimizer,
            )
        barrier()
    elif is_main_process():
        print(f"Skipping pretrain phase from resume phase={resume_phase} step={resume_step}", flush=True)

    reward_model = build_reward(cfg.reward, device=str(device), cache_dir=str(cfg.cache_dir))
    if teacher_code_pickscore_weight > 0.0 and not hasattr(reward_model, "score_tensor"):
        raise ValueError("teacher_code_pickscore_weight requires a differentiable reward.score_tensor")
    model.train()
    rl_start_step = resume_step + 1 if resume_phase == "rl" else 1
    if is_main_process() and resume_phase == "rl":
        print(f"Resuming rl phase from step={resume_step}; next step={rl_start_step}", flush=True)
    if resume_phase == "rl" and bool(getattr(cfg.train, "skip_batches_on_resume", False)):
        resume_batches_to_skip = int(getattr(cfg.train, "resume_batches_to_skip", resume_step))
        if is_main_process():
            print(f"Advancing dataloader by {resume_batches_to_skip} consumed batches", flush=True)
        skip_batches(batches, resume_batches_to_skip)
        if is_main_process():
            print("Dataloader resume advance complete", flush=True)
    rl_iter = range(rl_start_step, total_rl_steps + 1)
    if is_main_process():
        rl_iter = tqdm(rl_iter, desc="rl")
    for step in rl_iter:
        current_lr, current_decoder_lr = apply_learning_rate_schedule(optimizer, cfg, step)
        teacher_code_pickscore_weight_step = scheduled_train_value(
            cfg, "teacher_code_pickscore_weight", step, teacher_code_pickscore_weight
        )
        teacher_code_recon_weight_step = scheduled_train_value(
            cfg, "teacher_code_recon_weight", step, teacher_code_recon_weight
        )
        teacher_code_l1_weight_step = scheduled_train_value(
            cfg, "teacher_code_l1_weight", step, teacher_code_l1_weight
        )
        teacher_code_grad_weight_step = scheduled_train_value(
            cfg, "teacher_code_grad_weight", step, teacher_code_grad_weight
        )
        teacher_code_lap_weight_step = scheduled_train_value(
            cfg, "teacher_code_lap_weight", step, teacher_code_lap_weight
        )
        teacher_code_lap_energy_penalty_weight_step = scheduled_train_value(
            cfg,
            "teacher_code_lap_energy_penalty_weight",
            step,
            teacher_code_lap_energy_penalty_weight,
        )
        teacher_code_reference_l1_weight_step = scheduled_train_value(
            cfg, "teacher_code_reference_l1_weight", step, teacher_code_reference_l1_weight
        )
        teacher_code_reference_grad_weight_step = scheduled_train_value(
            cfg, "teacher_code_reference_grad_weight", step, teacher_code_reference_grad_weight
        )
        teacher_code_reference_lap_weight_step = scheduled_train_value(
            cfg, "teacher_code_reference_lap_weight", step, teacher_code_reference_lap_weight
        )
        teacher_code_reference_detail_weight_step = scheduled_train_value(
            cfg,
            "teacher_code_reference_detail_weight",
            step,
            teacher_code_reference_detail_weight,
        )
        images, prompts, metadatas, batch_llamagen_codes, batch_geneval_targets = to_device_batch(
            next(batches), device
        )
        bsz = images.shape[0]
        num_samples = int(cfg.train.num_samples)
        if geneval_prior_distill_weight > 0.0 and batch_llamagen_codes is None:
            raise ValueError("geneval_prior_distill_weight > 0 requires data.code_pool")
        if geneval_offline_distill_weight > 0.0 and (
            batch_llamagen_codes is None or batch_geneval_targets is None
        ):
            raise ValueError("geneval_offline_distill_weight > 0 requires data.code_pool and data.target_root")

        teacher_codes = None
        teacher_soft_indices = None
        teacher_soft_weights = None
        ref_posterior_logits = None
        with torch.no_grad(), autocast_context(device, precision):
            sample_out = model(
                images,
                mode="sample",
                num_samples=num_samples,
                temperature=float(cfg.train.temperature),
            )
            recon = sample_out.recon
            recon_losses = F.mse_loss(
                recon.float(),
                images[:, None].expand_as(recon).float(),
                reduction="none",
            ).mean(dim=(2, 3, 4))
            sample_recon_losses = recon_losses.detach()

            if teacher_code_enabled:
                if teacher_code_include_greedy or teacher_code_posterior_samples > 0:
                    teacher_codes = reference_teacher_code_batch(
                        ref_model,
                        images,
                        include_greedy=teacher_code_include_greedy,
                        posterior_samples=teacher_code_posterior_samples,
                        temperature=teacher_code_temperature,
                    )
                if teacher_code_soft_topk > 0:
                    teacher_soft_indices, teacher_soft_weights = reference_teacher_soft_support(
                        ref_model,
                        images,
                        topk=teacher_code_soft_topk,
                        temperature=teacher_code_temperature,
                    )
            if encoder_anchor_kl_weight > 0.0:
                ref_posterior_logits = ref_model.encode_logits(images).detach()

            prior_reward_recon = None
            prior_reward_variants = None
            if geneval_prior_distill_weight > 0.0:
                module = model.module if isinstance(model, DistributedDataParallel) else model
                prior_reward_recon = module.decode_codes(batch_llamagen_codes)
                if geneval_prior_transform_distill:
                    prior_reward_variants = torch.stack(
                        list(geneval_transform_variants(prior_reward_recon).values()),
                        dim=1,
                    )
                else:
                    prior_reward_variants = prior_reward_recon[:, None]

        if str(cfg.reward.backend) == "code_target" and sample_out.score_inputs is not None:
            reward_inputs = sample_out.score_inputs
        else:
            reward_inputs = recon.detach()
        reward_count = num_samples
        prior_variant_count = 0
        if prior_reward_variants is not None:
            prior_variant_count = prior_reward_variants.shape[1]
            reward_inputs = torch.cat([reward_inputs, prior_reward_variants], dim=1)
            reward_count += prior_variant_count
        flat_reward_inputs = reward_inputs.reshape(bsz * reward_count, *reward_inputs.shape[2:])
        flat_prompts = make_repeated_prompts(prompts, num_samples)
        reward_prompts = make_repeated_prompts(prompts, reward_count)
        reward_metadatas = make_repeated_metadata(metadatas, reward_count)
        flat_reward_references = images[:, None].expand(
            -1, reward_count, -1, -1, -1
        ).reshape(bsz * reward_count, *images.shape[1:])
        all_reward_scores = score_reward(
            reward_model,
            flat_reward_inputs,
            reward_prompts,
            reward_metadatas,
            reference_images=flat_reward_references,
        ).view(bsz, reward_count)
        pick_scores = all_reward_scores[:, :num_samples]
        prior_geneval_scores = None
        prior_best_scores = None
        reward_metrics = dict(getattr(reward_model, "last_metrics", {}))

        best_sample_indices = pick_scores.argmax(dim=1)
        best_sample_rewards = pick_scores.gather(1, best_sample_indices[:, None]).squeeze(1)
        prior_reward_gains = None
        prior_distill_targets = None
        if prior_variant_count > 0:
            prior_variant_scores = all_reward_scores[:, num_samples:]
            prior_geneval_scores = prior_variant_scores[:, 0]
            rows = torch.arange(bsz, device=device)
            if geneval_prior_transform_distill:
                prior_best_indices = prior_variant_scores.argmax(dim=1)
                prior_best_scores = prior_variant_scores[rows, prior_best_indices]
                prior_distill_targets = prior_reward_variants[rows, prior_best_indices].detach()
            else:
                prior_best_scores = best_sample_rewards
                prior_distill_targets = recon[rows, best_sample_indices].detach()
            prior_reward_gains = (prior_best_scores - prior_geneval_scores).clamp_min(0.0).detach()

        gated_pick_scores, recon_gate = apply_recon_gate(
            pick_scores=pick_scores,
            recon_losses=recon_losses,
            threshold=recon_gate_threshold,
            mode=recon_gate_mode,
            sharpness=recon_gate_sharpness,
        )
        gate_penalty = recon_gate_penalty * (recon_losses.detach() - recon_gate_threshold).clamp_min(0.0)
        sample_grad_losses = torch.zeros_like(recon_losses)
        sample_lap_losses = torch.zeros_like(recon_losses)
        if sample_grad_reward_weight > 0.0 or sample_lap_reward_weight > 0.0:
            flat_recon = recon.detach().reshape(bsz * num_samples, *recon.shape[2:])
            flat_target = images[:, None].expand_as(recon).detach().reshape(bsz * num_samples, *images.shape[1:])
            if sample_grad_reward_weight > 0.0:
                sample_grad_losses = image_gradient_loss_per_image(flat_recon, flat_target).view(bsz, num_samples)
            if sample_lap_reward_weight > 0.0:
                sample_lap_losses = image_laplacian_loss_per_image(flat_recon, flat_target).view(bsz, num_samples)
        objective_scores = (
            pickscore_reward_weight * gated_pick_scores
            - recon_reward_weight * recon_losses.detach()
            - sample_grad_reward_weight * sample_grad_losses.detach()
            - sample_lap_reward_weight * sample_lap_losses.detach()
            - gate_penalty
        )
        advantage_scores = select_advantage_scores(
            advantage_score_mode,
            objective_scores=objective_scores,
            pick_scores=pick_scores,
            gated_pick_scores=gated_pick_scores,
        )
        if advantage_mode == "topk":
            advantages = topk_group_advantages(advantage_scores, topk=advantage_topk, negative_value=advantage_topk_negative)
        else:
            advantages = group_advantages(advantage_scores)
        if advantage_clip > 0.0:
            advantages = advantages.clamp(-advantage_clip, advantage_clip)
        old_log_probs = sample_out.log_probs.detach()
        logprob_scale = 1.0
        if normalize_logprob_by_sites:
            logprob_scale = float(sample_out.codes.shape[-1] * sample_out.codes.shape[-2])
        ref_log_probs = None
        if ref_model is not None and ref_kl_weight > 0.0:
            with torch.no_grad(), autocast_context(device, precision):
                _, ref_log_probs, _ = ref_model(
                    images,
                    mode="score_codes",
                    codes=sample_out.codes,
                    temperature=float(cfg.train.temperature),
                )
            ref_log_probs = ref_log_probs.detach()

        last_metrics: dict[str, torch.Tensor] = {}
        for update_epoch in range(rl_update_epochs):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                if float(cfg.train.recon_weight) > 0.0 or grpo_image_pickscore_weight > 0.0:
                    recon, log_probs, entropy = model(
                        images,
                        mode="score_codes",
                        codes=sample_out.codes,
                        temperature=float(cfg.train.temperature),
                    )
                    recon_losses = F.mse_loss(
                        recon.float(),
                        images[:, None].expand_as(recon).float(),
                        reduction="none",
                    ).mean(dim=(2, 3, 4))
                else:
                    log_probs, entropy = model(
                        images,
                        mode="score_log_probs",
                        codes=sample_out.codes,
                        temperature=float(cfg.train.temperature),
                    )
                    recon_losses = torch.zeros_like(log_probs)
                tokenizer_recon_loss = torch.zeros((), device=device)
                tokenizer_l1_loss = torch.zeros((), device=device)
                tokenizer_grad_loss = torch.zeros((), device=device)
                tokenizer_lap_loss = torch.zeros((), device=device)
                tokenizer_perceptual_loss = torch.zeros((), device=device)
                tokenizer_official_lpips_loss = torch.zeros((), device=device)
                tokenizer_dists_loss = torch.zeros((), device=device)
                tokenizer_ssim_loss = torch.zeros((), device=device)
                tokenizer_ms_ssim_loss = torch.zeros((), device=device)
                tokenizer_reward = torch.zeros((), device=device)
                tokenizer_reward_primary = torch.zeros((), device=device)
                tokenizer_reward_baseline = torch.zeros((), device=device)
                tokenizer_reward_loss = torch.zeros((), device=device)
                tokenizer_reward_loss_for_backward = torch.zeros((), device=device)
                tokenizer_reward_quality_cosine = torch.zeros((), device=device)
                tokenizer_reward_projection_active = torch.zeros((), device=device)
                tokenizer_reward_projection_ratio = torch.zeros((), device=device)
                tokenizer_fid_feature_loss = torch.zeros((), device=device)
                tokenizer_quality_guard_active_fraction = torch.zeros((), device=device)
                tokenizer_quality_guard_max_gap = torch.zeros((), device=device)
                tokenizer_quality_guard_gaps = {
                    name: torch.zeros((), device=device) for name in quality_guard_tolerances
                }
                tokenizer_quality_objectives = {
                    "mse": tokenizer_recon_loss,
                    "l1": tokenizer_l1_loss,
                    "grad": tokenizer_grad_loss,
                    "lap": tokenizer_lap_loss,
                    "lpips": tokenizer_official_lpips_loss,
                    "dists": tokenizer_dists_loss,
                    "ssim": tokenizer_ssim_loss,
                    "ms_ssim": tokenizer_ms_ssim_loss,
                    "fid_feature": tokenizer_fid_feature_loss,
                }
                tokenizer_vq_loss = torch.zeros((), device=device)
                encoder_anchor_recon_loss = torch.zeros((), device=device)
                encoder_anchor_l1_loss = torch.zeros((), device=device)
                encoder_anchor_grad_loss = torch.zeros((), device=device)
                encoder_anchor_lap_loss = torch.zeros((), device=device)
                encoder_anchor_kl_loss = torch.zeros((), device=device)
                teacher_code_pickscore = torch.zeros((), device=device)
                teacher_code_pickscore_loss = torch.zeros((), device=device)
                teacher_code_recon_loss = torch.zeros((), device=device)
                teacher_code_l1_loss = torch.zeros((), device=device)
                teacher_code_grad_loss = torch.zeros((), device=device)
                teacher_code_lap_loss = torch.zeros((), device=device)
                teacher_code_count_metric = torch.zeros((), device=device)
                teacher_code_pickscore_gate = torch.ones((), device=device)
                teacher_code_lap_energy_ratio = torch.ones((), device=device)
                teacher_code_lap_energy_penalty = torch.zeros((), device=device)
                teacher_code_reference_l1_loss = torch.zeros((), device=device)
                teacher_code_reference_grad_loss = torch.zeros((), device=device)
                teacher_code_reference_lap_loss = torch.zeros((), device=device)
                teacher_code_reference_detail_loss = torch.zeros((), device=device)
                gan_g_loss = torch.zeros((), device=device)
                gan_feature_matching_loss = torch.zeros((), device=device)
                fid_gan_g_loss = torch.zeros((), device=device)
                grpo_image_pickscore = torch.zeros((), device=device)
                grpo_image_pickscore_loss = torch.zeros((), device=device)
                llamagen_prior_pickscore = torch.zeros((), device=device)
                llamagen_prior_loss = torch.zeros((), device=device)
                geneval_prior_distill_loss = torch.zeros((), device=device)
                geneval_offline_distill_loss = torch.zeros((), device=device)
                output_saturation_loss = torch.zeros((), device=device)
                if encoder_anchor_enabled:
                    module = model.module if isinstance(model, DistributedDataParallel) else model
                    anchor_latents = module.encode_continuous_latents(images)
                    if encoder_anchor_reference_quantize:
                        anchor_latents, _, _ = ref_model.tokenizer.quantize(anchor_latents)
                    anchor_recon = ref_model.tokenizer.decode(anchor_latents).clamp(-1, 1)
                    if encoder_anchor_recon_weight > 0.0:
                        encoder_anchor_recon_loss = F.mse_loss(anchor_recon.float(), images.float())
                    if encoder_anchor_l1_weight > 0.0:
                        encoder_anchor_l1_loss = F.l1_loss(anchor_recon.float(), images.float())
                    if encoder_anchor_grad_weight > 0.0:
                        encoder_anchor_grad_loss = image_gradient_loss(anchor_recon, images)
                    if encoder_anchor_lap_weight > 0.0:
                        encoder_anchor_lap_loss = image_laplacian_loss(anchor_recon, images)
                    if encoder_anchor_kl_weight > 0.0:
                        current_logits = module.encode_logits(images).float() / encoder_anchor_temperature
                        reference_logits = ref_posterior_logits.float() / encoder_anchor_temperature
                        reference_log_probs = F.log_softmax(reference_logits, dim=1)
                        reference_probs = reference_log_probs.exp()
                        current_log_probs = F.log_softmax(current_logits, dim=1)
                        encoder_anchor_kl_loss = (
                            reference_probs * (reference_log_probs - current_log_probs)
                        ).sum(dim=1).mean()
                if teacher_code_enabled:
                    module = model.module if isinstance(model, DistributedDataParallel) else model
                    teacher_recons = []
                    if teacher_codes is not None:
                        teacher_recons.append(
                            module.decode_codes(
                                teacher_codes,
                                detach_codebook=teacher_code_detach_codebook,
                            )
                        )
                    if teacher_soft_indices is not None and teacher_soft_weights is not None:
                        teacher_recons.append(
                            module.decode_code_mixture(
                                teacher_soft_indices,
                                teacher_soft_weights,
                                detach_codebook=teacher_code_detach_codebook,
                            )[:, None]
                        )
                    teacher_recon = torch.cat(teacher_recons, dim=1)
                    teacher_count = teacher_recon.shape[1]
                    teacher_code_count_metric = torch.tensor(float(teacher_count), device=device)
                    flat_teacher_recon = teacher_recon.reshape(
                        bsz * teacher_count, *teacher_recon.shape[2:]
                    )
                    flat_teacher_targets = images[:, None].expand_as(teacher_recon).reshape(
                        bsz * teacher_count, *images.shape[1:]
                    )
                    teacher_code_lap_energy_ratio_per_image = (
                        image_laplacian_energy_per_image(flat_teacher_recon)
                        / image_laplacian_energy_per_image(flat_teacher_targets).clamp_min(1e-8)
                    )
                    teacher_code_lap_energy_ratio = teacher_code_lap_energy_ratio_per_image.mean()
                    if teacher_code_lap_energy_penalty_weight_step > 0.0:
                        teacher_code_lap_energy_penalty = F.relu(
                            teacher_code_lap_energy_penalty_floor
                            - teacher_code_lap_energy_ratio_per_image
                        ).mean()
                    if teacher_code_reference_enabled:
                        with torch.no_grad():
                            reference_teacher_recons = []
                            if teacher_codes is not None:
                                reference_teacher_recons.append(
                                    ref_model.decode_codes(
                                        teacher_codes,
                                        detach_codebook=True,
                                    )
                                )
                            if teacher_soft_indices is not None and teacher_soft_weights is not None:
                                reference_teacher_recons.append(
                                    ref_model.decode_code_mixture(
                                        teacher_soft_indices,
                                        teacher_soft_weights,
                                        detach_codebook=True,
                                    )[:, None]
                                )
                            flat_reference_teacher_recon = torch.cat(
                                reference_teacher_recons, dim=1
                            ).reshape_as(flat_teacher_recon)
                        if teacher_code_reference_l1_weight_step > 0.0:
                            teacher_code_reference_l1_loss = F.l1_loss(
                                flat_teacher_recon.float(),
                                flat_reference_teacher_recon.float(),
                            )
                        if teacher_code_reference_grad_weight_step > 0.0:
                            teacher_code_reference_grad_loss = image_gradient_loss(
                                flat_teacher_recon,
                                flat_reference_teacher_recon,
                            )
                        if teacher_code_reference_lap_weight_step > 0.0:
                            teacher_code_reference_lap_loss = image_laplacian_loss(
                                flat_teacher_recon,
                                flat_reference_teacher_recon,
                            )
                        if teacher_code_reference_detail_weight_step > 0.0:
                            flat_detail_reference_teacher_recon = flat_reference_teacher_recon
                            if detail_ref_model is not ref_model:
                                with torch.no_grad():
                                    detail_reference_teacher_recons = []
                                    if teacher_codes is not None:
                                        detail_reference_teacher_recons.append(
                                            detail_ref_model.decode_codes(
                                                teacher_codes,
                                                detach_codebook=True,
                                            )
                                        )
                                    if teacher_soft_indices is not None and teacher_soft_weights is not None:
                                        detail_reference_teacher_recons.append(
                                            detail_ref_model.decode_code_mixture(
                                                teacher_soft_indices,
                                                teacher_soft_weights,
                                                detach_codebook=True,
                                            )[:, None]
                                        )
                                    flat_detail_reference_teacher_recon = torch.cat(
                                        detail_reference_teacher_recons,
                                        dim=1,
                                    ).reshape_as(flat_teacher_recon)
                            teacher_code_reference_detail_loss = image_highpass_loss(
                                flat_teacher_recon,
                                flat_detail_reference_teacher_recon,
                                sigma=teacher_code_reference_detail_sigma,
                            )
                    if teacher_code_pickscore_weight_step > 0.0:
                        teacher_scores = score_reward_tensor(
                            reward_model,
                            flat_teacher_recon,
                            make_repeated_prompts(prompts, teacher_count),
                            reference_images=flat_teacher_targets,
                        )
                        teacher_mse_per_image = F.mse_loss(
                            flat_teacher_recon.float(),
                            flat_teacher_targets.float(),
                            reduction="none",
                        ).mean(dim=(1, 2, 3))
                        gate = teacher_code_quality_gate(
                            teacher_mse_per_image,
                            teacher_code_lap_energy_ratio_per_image,
                            mse_budget=teacher_code_pickscore_mse_budget,
                            lap_energy_floor=teacher_code_pickscore_lap_energy_floor,
                            mode=teacher_code_pickscore_gate_mode,
                            sharpness=teacher_code_pickscore_gate_sharpness,
                        )
                        gate = gate.detach()
                        teacher_code_pickscore_gate = gate.mean()
                        teacher_code_pickscore = teacher_scores.mean()
                        teacher_code_pickscore_loss = -(teacher_scores * gate).mean()
                    if teacher_code_recon_weight_step > 0.0:
                        teacher_code_recon_loss = F.mse_loss(
                            flat_teacher_recon.float(), flat_teacher_targets.float()
                        )
                    if teacher_code_l1_weight_step > 0.0:
                        teacher_code_l1_loss = F.l1_loss(
                            flat_teacher_recon.float(), flat_teacher_targets.float()
                        )
                    if teacher_code_grad_weight_step > 0.0:
                        teacher_code_grad_loss = image_gradient_loss(flat_teacher_recon, flat_teacher_targets)
                    if teacher_code_lap_weight_step > 0.0:
                        teacher_code_lap_loss = image_laplacian_loss(flat_teacher_recon, flat_teacher_targets)
                if grpo_image_pickscore_weight > 0.0:
                    image_advantages = advantages.detach()
                    if grpo_image_positive_advantage:
                        image_advantages = image_advantages.clamp_min(0.0)
                    direct_recon = recon
                    direct_prompts = flat_prompts
                    direct_advantages = image_advantages
                    if grpo_image_topk > 0 and grpo_image_topk < num_samples:
                        top_indices = image_advantages.topk(k=grpo_image_topk, dim=1).indices
                        gather_shape = [bsz, grpo_image_topk] + [1] * (recon.ndim - 2)
                        direct_recon = recon.gather(
                            dim=1,
                            index=top_indices.view(*gather_shape).expand(-1, -1, *recon.shape[2:]),
                        )
                        direct_advantages = image_advantages.gather(1, top_indices)
                        direct_prompts = make_repeated_prompts(prompts, grpo_image_topk)
                    direct_count = direct_recon.shape[1]
                    flat_train_recon = direct_recon.reshape(bsz * direct_count, *direct_recon.shape[2:])
                    direct_targets = images[:, None].expand(
                        -1, direct_count, -1, -1, -1
                    ).reshape(bsz * direct_count, *images.shape[1:])
                    image_pick_scores = score_reward_tensor(
                        reward_model,
                        flat_train_recon,
                        direct_prompts,
                        reference_images=direct_targets,
                    ).view(bsz, direct_count)
                    normalizer = image_advantages.abs().mean().clamp_min(1e-6)
                    grpo_image_pickscore = image_pick_scores.mean()
                    grpo_image_pickscore_loss = -(direct_advantages * image_pick_scores).mean() / normalizer
                if llamagen_prior_pickscore_weight > 0.0:
                    prior_count = max(1, min(llamagen_prior_batch_size, int(llamagen_prior_codes.shape[0])))
                    prior_indices = torch.randint(0, int(llamagen_prior_codes.shape[0]), (prior_count,))
                    prior_codes = llamagen_prior_codes[prior_indices].to(device=device, non_blocking=True)
                    prior_prompts = [llamagen_prior_prompts[int(index)] for index in prior_indices.tolist()]
                    module = model.module if isinstance(model, DistributedDataParallel) else model
                    prior_recon = module.decode_codes(prior_codes)
                    prior_scores = score_reward_tensor(
                        reward_model,
                        prior_recon,
                        prior_prompts,
                    )
                    llamagen_prior_pickscore = prior_scores.mean()
                    llamagen_prior_loss = -llamagen_prior_pickscore
                if geneval_prior_distill_weight > 0.0:
                    module = model.module if isinstance(model, DistributedDataParallel) else model
                    prior_direct_recon = module.decode_codes(batch_llamagen_codes)
                    distill_per_image = F.mse_loss(
                        prior_direct_recon.float(),
                        prior_distill_targets.float(),
                        reduction="none",
                    ).mean(dim=(1, 2, 3))
                    gain_sum = prior_reward_gains.sum()
                    if float(gain_sum.detach()) > 0.0:
                        geneval_prior_distill_loss = (
                            distill_per_image * prior_reward_gains
                        ).sum() / gain_sum
                if geneval_offline_distill_weight > 0.0:
                    module = model.module if isinstance(model, DistributedDataParallel) else model
                    offline_direct_recon = module.decode_codes(batch_llamagen_codes)
                    geneval_offline_distill_loss = F.mse_loss(
                        offline_direct_recon.float(), batch_geneval_targets.float()
                    )
                if output_saturation_targets is not None:
                    module = model.module if isinstance(model, DistributedDataParallel) else model
                    conv_out = module.tokenizer.decoder.conv_out
                    target_weight, target_bias = output_saturation_targets
                    output_saturation_loss = F.mse_loss(
                        conv_out.weight.float(), target_weight.float()
                    ) + F.mse_loss(conv_out.bias.float(), target_bias.float())
                if (
                    tokenizer_recon_weight > 0.0
                    or tokenizer_l1_weight > 0.0
                    or tokenizer_grad_weight > 0.0
                    or tokenizer_lap_weight > 0.0
                    or tokenizer_perceptual_weight > 0.0
                    or tokenizer_official_lpips_weight > 0.0
                    or tokenizer_dists_weight > 0.0
                    or tokenizer_ssim_weight > 0.0
                    or tokenizer_ms_ssim_weight > 0.0
                    or tokenizer_reward_weight > 0.0
                    or tokenizer_fid_feature_weight > 0.0
                    or tokenizer_vq_weight > 0.0
                    or (gan_weight > 0.0 and step > gan_warmup_steps)
                    or (fid_gan_weight > 0.0 and step > fid_gan_warmup_steps)
                ):
                    tokenizer_recon, tokenizer_vq_loss = model(images, mode="tokenizer")
                    tokenizer_recon_loss = F.mse_loss(tokenizer_recon.float(), images.float())
                    if tokenizer_l1_weight > 0.0:
                        tokenizer_l1_loss = F.l1_loss(tokenizer_recon.float(), images.float())
                    if tokenizer_grad_weight > 0.0:
                        tokenizer_grad_loss = image_gradient_loss(tokenizer_recon, images)
                    if tokenizer_lap_weight > 0.0:
                        tokenizer_lap_loss = image_laplacian_loss(tokenizer_recon, images)
                    if perceptual_loss is not None:
                        with torch.autocast(device_type=device.type, enabled=False):
                            tokenizer_perceptual_loss = perceptual_loss(
                                tokenizer_recon.float(), images.float()
                            ).mean()
                    if official_lpips_loss is not None:
                        metric_batch_size = min(
                            tokenizer_official_lpips_batch_size, tokenizer_recon.shape[0]
                        )
                        with torch.autocast(device_type=device.type, enabled=False):
                            tokenizer_official_lpips_loss = official_lpips_loss(
                                tokenizer_recon[:metric_batch_size].float(),
                                images[:metric_batch_size].float(),
                            ).mean()
                    if dists_loss is not None:
                        metric_batch_size = min(
                            tokenizer_dists_batch_size, tokenizer_recon.shape[0]
                        )
                        with torch.autocast(device_type=device.type, enabled=False):
                            tokenizer_dists_loss = dists_loss(
                                tokenizer_recon[:metric_batch_size].float().clamp(-1, 1).add(1).mul(0.5),
                                images[:metric_batch_size].float().clamp(-1, 1).add(1).mul(0.5),
                            ).mean()
                    if tokenizer_ssim_weight > 0.0 or tokenizer_ms_ssim_weight > 0.0:
                        from pytorch_msssim import ms_ssim, ssim

                        metric_batch_size = min(
                            tokenizer_ssim_batch_size, tokenizer_recon.shape[0]
                        )
                        metric_recon = (
                            tokenizer_recon[:metric_batch_size]
                            .float()
                            .clamp(-1, 1)
                            .add(1)
                            .mul(0.5)
                        )
                        metric_target = (
                            images[:metric_batch_size].float().clamp(-1, 1).add(1).mul(0.5)
                        )
                        if tokenizer_ssim_weight > 0.0:
                            tokenizer_ssim_loss = 1.0 - ssim(
                                metric_recon,
                                metric_target,
                                data_range=1.0,
                                size_average=True,
                            )
                        if tokenizer_ms_ssim_weight > 0.0:
                            tokenizer_ms_ssim_loss = 1.0 - ms_ssim(
                                metric_recon,
                                metric_target,
                                data_range=1.0,
                                size_average=True,
                            )
                    if tokenizer_reward_weight > 0.0:
                        tokenizer_reward_prompts = prompts
                        if tokenizer_reward_prompt_mode == "batch_roll":
                            tokenizer_reward_prompts = (
                                prompts[1:] + prompts[:1] if len(prompts) > 1 else prompts
                            )
                        elif tokenizer_reward_prompt_mode == "constant":
                            tokenizer_reward_prompts = [
                                tokenizer_reward_constant_prompt
                            ] * len(prompts)
                        if tokenizer_reward_baseline_prompt_mode == "none":
                            tokenizer_reward_scores = score_reward_tensor(
                                reward_model,
                                tokenizer_recon,
                                tokenizer_reward_prompts,
                                reference_images=images,
                            )
                            tokenizer_reward_primary = tokenizer_reward_scores.mean()
                        else:
                            if tokenizer_reward_baseline_prompt_mode == "batch_roll":
                                baseline_prompts = (
                                    prompts[1:] + prompts[:1] if len(prompts) > 1 else prompts
                                )
                            else:
                                baseline_prompts = [tokenizer_reward_constant_prompt] * len(
                                    prompts
                                )
                            primary_scores, baseline_scores = score_reward_tensor_prompt_pair(
                                reward_model,
                                tokenizer_recon,
                                tokenizer_reward_prompts,
                                baseline_prompts,
                                reference_images=images,
                            )
                            tokenizer_reward_primary = primary_scores.mean()
                            tokenizer_reward_baseline = baseline_scores.mean()
                            tokenizer_reward_scores = primary_scores - (
                                tokenizer_reward_baseline_weight * baseline_scores
                            )
                        tokenizer_reward = tokenizer_reward_scores.mean()
                        tokenizer_reward_loss = -tokenizer_reward
                        tokenizer_reward_loss_for_backward = tokenizer_reward_loss
                    if fid_feature_loss is not None and tokenizer_fid_feature_weight > 0.0:
                        feature_batch_size = min(
                            tokenizer_fid_feature_batch_size, tokenizer_recon.shape[0]
                        )
                        tokenizer_fid_feature_loss = fid_feature_loss(
                            tokenizer_recon[:feature_batch_size],
                            images[:feature_batch_size],
                        )

                    tokenizer_quality_objectives = {
                        "mse": tokenizer_recon_loss,
                        "l1": tokenizer_l1_loss,
                        "grad": tokenizer_grad_loss,
                        "lap": tokenizer_lap_loss,
                        "lpips": tokenizer_official_lpips_loss,
                        "dists": tokenizer_dists_loss,
                        "ssim": tokenizer_ssim_loss,
                        "ms_ssim": tokenizer_ms_ssim_loss,
                        "fid_feature": tokenizer_fid_feature_loss,
                    }
                    if tokenizer_quality_guard_against_base:
                        # Match the BF16 reconstruction path used by training and evaluation;
                        # metric implementations themselves still run in FP32 below.
                        with torch.no_grad(), autocast_context(device, precision):
                            base_tokenizer_recon, _ = ref_model(images, mode="tokenizer")
                        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                            base_quality_losses = {
                                "mse": F.mse_loss(
                                    base_tokenizer_recon.float(), images.float()
                                ),
                                "l1": F.l1_loss(
                                    base_tokenizer_recon.float(), images.float()
                                ) if tokenizer_l1_weight > 0.0 else torch.zeros((), device=device),
                                "grad": image_gradient_loss(
                                    base_tokenizer_recon, images
                                ) if tokenizer_grad_weight > 0.0 else torch.zeros((), device=device),
                                "lap": image_laplacian_loss(
                                    base_tokenizer_recon, images
                                ) if tokenizer_lap_weight > 0.0 else torch.zeros((), device=device),
                                "lpips": torch.zeros((), device=device),
                                "dists": torch.zeros((), device=device),
                                "ssim": torch.zeros((), device=device),
                                "ms_ssim": torch.zeros((), device=device),
                                "fid_feature": torch.zeros((), device=device),
                            }
                            if official_lpips_loss is not None:
                                metric_batch_size = min(
                                    tokenizer_official_lpips_batch_size,
                                    base_tokenizer_recon.shape[0],
                                )
                                base_quality_losses["lpips"] = official_lpips_loss(
                                    base_tokenizer_recon[:metric_batch_size].float(),
                                    images[:metric_batch_size].float(),
                                ).mean()
                            if dists_loss is not None:
                                metric_batch_size = min(
                                    tokenizer_dists_batch_size,
                                    base_tokenizer_recon.shape[0],
                                )
                                base_quality_losses["dists"] = dists_loss(
                                    base_tokenizer_recon[:metric_batch_size]
                                    .float()
                                    .clamp(-1, 1)
                                    .add(1)
                                    .mul(0.5),
                                    images[:metric_batch_size]
                                    .float()
                                    .clamp(-1, 1)
                                    .add(1)
                                    .mul(0.5),
                                ).mean()
                            if tokenizer_ssim_weight > 0.0 or tokenizer_ms_ssim_weight > 0.0:
                                metric_batch_size = min(
                                    tokenizer_ssim_batch_size,
                                    base_tokenizer_recon.shape[0],
                                )
                                base_metric_recon = (
                                    base_tokenizer_recon[:metric_batch_size]
                                    .float()
                                    .clamp(-1, 1)
                                    .add(1)
                                    .mul(0.5)
                                )
                                base_metric_target = (
                                    images[:metric_batch_size]
                                    .float()
                                    .clamp(-1, 1)
                                    .add(1)
                                    .mul(0.5)
                                )
                                if tokenizer_ssim_weight > 0.0:
                                    base_quality_losses["ssim"] = 1.0 - ssim(
                                        base_metric_recon,
                                        base_metric_target,
                                        data_range=1.0,
                                        size_average=True,
                                    )
                                if tokenizer_ms_ssim_weight > 0.0:
                                    base_quality_losses["ms_ssim"] = 1.0 - ms_ssim(
                                        base_metric_recon,
                                        base_metric_target,
                                        data_range=1.0,
                                        size_average=True,
                                    )
                            if fid_feature_loss is not None and tokenizer_fid_feature_weight > 0.0:
                                feature_batch_size = min(
                                    tokenizer_fid_feature_batch_size,
                                    base_tokenizer_recon.shape[0],
                                )
                                base_quality_losses["fid_feature"] = fid_feature_loss(
                                    base_tokenizer_recon[:feature_batch_size],
                                    images[:feature_batch_size],
                                )

                        enabled_guard_names = []
                        quality_weights = {
                            "mse": tokenizer_recon_weight,
                            "l1": tokenizer_l1_weight,
                            "grad": tokenizer_grad_weight,
                            "lap": tokenizer_lap_weight,
                            "lpips": tokenizer_official_lpips_weight,
                            "dists": tokenizer_dists_weight,
                            "ssim": tokenizer_ssim_weight,
                            "ms_ssim": tokenizer_ms_ssim_weight,
                            "fid_feature": tokenizer_fid_feature_weight,
                        }
                        for name, current_quality_loss in tokenizer_quality_objectives.items():
                            if quality_weights[name] <= 0.0:
                                continue
                            gap = current_quality_loss - base_quality_losses[name]
                            tokenizer_quality_guard_gaps[name] = gap.detach()
                            tokenizer_quality_objectives[name] = F.relu(
                                gap - quality_guard_tolerances[name]
                            )
                            enabled_guard_names.append(name)
                        if enabled_guard_names:
                            guard_gaps = torch.stack(
                                [tokenizer_quality_guard_gaps[name] for name in enabled_guard_names]
                            )
                            guard_tolerances = torch.tensor(
                                [quality_guard_tolerances[name] for name in enabled_guard_names],
                                device=device,
                            )
                            tokenizer_quality_guard_active_fraction = (
                                guard_gaps > guard_tolerances
                            ).float().mean()
                            tokenizer_quality_guard_max_gap = (
                                guard_gaps - guard_tolerances
                            ).max()
                    if discriminator is not None and gan_weight > 0.0 and step > gan_warmup_steps:
                        set_requires_grad(discriminator, False)
                        fake_input = discriminator_input(images, tokenizer_recon, gan_conditional)
                        if gan_feature_matching_weight > 0.0:
                            fake_logits, fake_features = discriminator(
                                fake_input, return_features=True
                            )
                            with torch.no_grad():
                                _, real_features = discriminator(
                                    discriminator_input(images, images, gan_conditional),
                                    return_features=True,
                                )
                            gan_feature_matching_loss = torch.stack(
                                [
                                    F.l1_loss(fake_feature, real_feature.detach())
                                    for fake_feature, real_feature in zip(fake_features, real_features)
                                ]
                            ).mean()
                        else:
                            fake_logits = discriminator(fake_input)
                        gan_g_loss = generator_hinge_loss(fake_logits)
                    if (
                        fid_discriminator is not None
                        and fid_feature_loss is not None
                        and fid_gan_weight > 0.0
                        and step > fid_gan_warmup_steps
                    ):
                        set_requires_grad(fid_discriminator, False)
                        feature_batch_size = min(fid_gan_batch_size, tokenizer_recon.shape[0])
                        fake_fid_features = fid_feature_loss.features(
                            tokenizer_recon[:feature_batch_size]
                        )
                        fid_gan_g_loss = generator_hinge_loss(
                            fid_discriminator(fake_fid_features)
                        )
                    if tokenizer_reward_project_against_quality and tokenizer_reward_weight > 0.0:
                        tokenizer_quality_objective = (
                            tokenizer_recon_weight * tokenizer_quality_objectives["mse"]
                            + tokenizer_l1_weight * tokenizer_quality_objectives["l1"]
                            + tokenizer_grad_weight * tokenizer_quality_objectives["grad"]
                            + tokenizer_lap_weight * tokenizer_quality_objectives["lap"]
                            + tokenizer_perceptual_weight * tokenizer_perceptual_loss
                            + tokenizer_official_lpips_weight * tokenizer_quality_objectives["lpips"]
                            + tokenizer_dists_weight * tokenizer_quality_objectives["dists"]
                            + tokenizer_ssim_weight * tokenizer_quality_objectives["ssim"]
                            + tokenizer_ms_ssim_weight * tokenizer_quality_objectives["ms_ssim"]
                            + tokenizer_fid_feature_weight * tokenizer_quality_objectives["fid_feature"]
                        )
                        quality_image_grad = torch.autograd.grad(
                            tokenizer_quality_objective,
                            tokenizer_recon,
                            retain_graph=True,
                        )[0].float()
                        reward_image_grad = torch.autograd.grad(
                            tokenizer_reward_loss,
                            tokenizer_recon,
                            retain_graph=True,
                        )[0].float()
                        grad_dot = (quality_image_grad * reward_image_grad).sum()
                        quality_grad_norm_sq = quality_image_grad.square().sum().clamp_min(1e-12)
                        reward_grad_norm = reward_image_grad.square().sum().sqrt().clamp_min(1e-12)
                        quality_grad_norm = quality_grad_norm_sq.sqrt()
                        tokenizer_reward_quality_cosine = (
                            grad_dot / (quality_grad_norm * reward_grad_norm)
                        ).detach()
                        tokenizer_reward_projection_active = (grad_dot < 0.0).float().detach()
                        projection_coefficient = (-grad_dot).clamp_min(0.0) / quality_grad_norm_sq
                        projected_reward_grad = (
                            reward_image_grad + projection_coefficient * quality_image_grad
                        )
                        tokenizer_reward_projection_ratio = (
                            (projected_reward_grad - reward_image_grad).square().sum().sqrt()
                            / reward_grad_norm
                        ).detach()
                        # Preserve the scalar value while replacing only its image-space gradient.
                        tokenizer_reward_loss_for_backward = tokenizer_reward_loss.detach() + (
                            (tokenizer_recon.float() - tokenizer_recon.float().detach())
                            * projected_reward_grad.detach()
                        ).sum()
            log_ratio = (log_probs.float() - old_log_probs.float()) / logprob_scale
            ratio = torch.exp(log_ratio)
            clipped_ratio = ratio.clamp(1.0 - float(cfg.train.grpo_clip_eps), 1.0 + float(cfg.train.grpo_clip_eps))
            policy_objective = torch.minimum(ratio * advantages.detach(), clipped_ratio * advantages.detach())
            policy_loss = -policy_objective.mean()
            recon_loss = recon_losses.mean()
            entropy = entropy.float()
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > float(cfg.train.grpo_clip_eps)).float().mean()
            ref_kl = torch.zeros((), device=device)
            if ref_log_probs is not None:
                ref_log_ratio = (log_probs.float() - ref_log_probs.float()) / logprob_scale
                ref_kl = ref_log_ratio.square().mean()
            loss = (
                policy_loss_weight * policy_loss
                + float(cfg.train.recon_weight) * recon_loss
                + tokenizer_recon_weight * tokenizer_quality_objectives["mse"]
                + tokenizer_l1_weight * tokenizer_quality_objectives["l1"]
                + tokenizer_grad_weight * tokenizer_quality_objectives["grad"]
                + tokenizer_lap_weight * tokenizer_quality_objectives["lap"]
                + tokenizer_perceptual_weight * tokenizer_perceptual_loss
                + tokenizer_official_lpips_weight * tokenizer_quality_objectives["lpips"]
                + tokenizer_dists_weight * tokenizer_quality_objectives["dists"]
                + tokenizer_ssim_weight * tokenizer_quality_objectives["ssim"]
                + tokenizer_ms_ssim_weight * tokenizer_quality_objectives["ms_ssim"]
                + tokenizer_reward_weight * tokenizer_reward_loss_for_backward
                + tokenizer_fid_feature_weight * tokenizer_quality_objectives["fid_feature"]
                + tokenizer_vq_weight * tokenizer_vq_loss
                + encoder_anchor_recon_weight * encoder_anchor_recon_loss
                + encoder_anchor_l1_weight * encoder_anchor_l1_loss
                + encoder_anchor_grad_weight * encoder_anchor_grad_loss
                + encoder_anchor_lap_weight * encoder_anchor_lap_loss
                + encoder_anchor_kl_weight * encoder_anchor_kl_loss
                + teacher_code_pickscore_weight_step * teacher_code_pickscore_loss
                + teacher_code_recon_weight_step * teacher_code_recon_loss
                + teacher_code_l1_weight_step * teacher_code_l1_loss
                + teacher_code_grad_weight_step * teacher_code_grad_loss
                + teacher_code_lap_weight_step * teacher_code_lap_loss
                + teacher_code_lap_energy_penalty_weight_step * teacher_code_lap_energy_penalty
                + teacher_code_reference_l1_weight_step * teacher_code_reference_l1_loss
                + teacher_code_reference_grad_weight_step * teacher_code_reference_grad_loss
                + teacher_code_reference_lap_weight_step * teacher_code_reference_lap_loss
                + teacher_code_reference_detail_weight_step * teacher_code_reference_detail_loss
                + grpo_image_pickscore_weight * grpo_image_pickscore_loss
                + llamagen_prior_pickscore_weight * llamagen_prior_loss
                + geneval_prior_distill_weight * geneval_prior_distill_loss
                + geneval_offline_distill_weight * geneval_offline_distill_loss
                + output_saturation_weight * output_saturation_loss
                + gan_weight * (
                    gan_g_loss + gan_feature_matching_weight * gan_feature_matching_loss
                )
                + fid_gan_weight * fid_gan_g_loss
                + ref_kl_weight * ref_kl
                - float(cfg.train.entropy_weight) * entropy
            )

            loss.backward()
            gradient_norms = tokenizer_gradient_group_norms(model)
            if bool(getattr(cfg.train, "separate_gradient_clipping", False)):
                clip_tokenizer_gradient_groups(model, float(cfg.train.max_grad_norm))
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.max_grad_norm))
            optimizer.step()
            gan_d_loss = torch.zeros((), device=device)
            if (
                discriminator is not None
                and discriminator_optimizer is not None
                and gan_d_weight > 0.0
                and gan_weight > 0.0
                and step >= gan_d_start_step
            ):
                set_requires_grad(discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, precision):
                    with torch.no_grad():
                        fake_images, _ = model(images, mode="tokenizer")
                    real_logits = discriminator(
                        discriminator_input(images.detach(), images.detach(), gan_conditional)
                    )
                    fake_logits = discriminator(
                        discriminator_input(images.detach(), fake_images.detach(), gan_conditional)
                    )
                    gan_d_loss = discriminator_hinge_loss(real_logits, fake_logits)
                    weighted_gan_d_loss = gan_d_weight * gan_d_loss
                weighted_gan_d_loss.backward()
                discriminator_optimizer.step()
            fid_gan_d_loss = torch.zeros((), device=device)
            if (
                fid_discriminator is not None
                and fid_discriminator_optimizer is not None
                and fid_feature_loss is not None
                and fid_gan_d_weight > 0.0
                and fid_gan_weight > 0.0
                and step >= fid_gan_d_start_step
            ):
                set_requires_grad(fid_discriminator, True)
                fid_discriminator_optimizer.zero_grad(set_to_none=True)
                feature_batch_size = min(fid_gan_batch_size, images.shape[0])
                with torch.no_grad(), autocast_context(device, precision):
                    fake_images, _ = model(images, mode="tokenizer")
                    real_fid_features = fid_feature_loss.features(
                        images[:feature_batch_size]
                    )
                    fake_fid_features = fid_feature_loss.features(
                        fake_images[:feature_batch_size]
                    )
                real_fid_logits = fid_discriminator(real_fid_features.detach())
                fake_fid_logits = fid_discriminator(fake_fid_features.detach())
                fid_gan_d_loss = discriminator_hinge_loss(
                    real_fid_logits,
                    fake_fid_logits,
                )
                (fid_gan_d_weight * fid_gan_d_loss).backward()
                fid_discriminator_optimizer.step()
            last_metrics = {
                "loss": loss.detach(),
                "policy_loss": policy_loss.detach(),
                "recon_mse": recon_loss.detach(),
                "tokenizer_recon_mse": tokenizer_recon_loss.detach(),
                "tokenizer_l1_loss": tokenizer_l1_loss.detach(),
                "tokenizer_grad_loss": tokenizer_grad_loss.detach(),
                "tokenizer_lap_loss": tokenizer_lap_loss.detach(),
                "tokenizer_perceptual_loss": tokenizer_perceptual_loss.detach(),
                "tokenizer_official_lpips_loss": tokenizer_official_lpips_loss.detach(),
                "tokenizer_dists_loss": tokenizer_dists_loss.detach(),
                "tokenizer_ssim_loss": tokenizer_ssim_loss.detach(),
                "tokenizer_ms_ssim_loss": tokenizer_ms_ssim_loss.detach(),
                "tokenizer_reward": tokenizer_reward.detach(),
                "tokenizer_reward_primary": tokenizer_reward_primary.detach(),
                "tokenizer_reward_baseline": tokenizer_reward_baseline.detach(),
                "tokenizer_reward_loss": tokenizer_reward_loss.detach(),
                "tokenizer_reward_quality_cosine": tokenizer_reward_quality_cosine,
                "tokenizer_reward_projection_active": tokenizer_reward_projection_active,
                "tokenizer_reward_projection_ratio": tokenizer_reward_projection_ratio,
                "tokenizer_fid_feature_loss": tokenizer_fid_feature_loss.detach(),
                "tokenizer_quality_guard_active_fraction": (
                    tokenizer_quality_guard_active_fraction.detach()
                ),
                "tokenizer_quality_guard_max_gap": tokenizer_quality_guard_max_gap.detach(),
                **{
                    f"tokenizer_quality_guard_{name}_gap": gap.detach()
                    for name, gap in tokenizer_quality_guard_gaps.items()
                },
                "tokenizer_vq_loss": tokenizer_vq_loss.detach(),
                "encoder_anchor_recon_mse": encoder_anchor_recon_loss.detach(),
                "encoder_anchor_l1_loss": encoder_anchor_l1_loss.detach(),
                "encoder_anchor_grad_loss": encoder_anchor_grad_loss.detach(),
                "encoder_anchor_lap_loss": encoder_anchor_lap_loss.detach(),
                "encoder_anchor_kl_loss": encoder_anchor_kl_loss.detach(),
                "teacher_code_pickscore": teacher_code_pickscore.detach(),
                "teacher_code_pickscore_loss": teacher_code_pickscore_loss.detach(),
                "teacher_code_recon_mse": teacher_code_recon_loss.detach(),
                "teacher_code_l1_loss": teacher_code_l1_loss.detach(),
                "teacher_code_grad_loss": teacher_code_grad_loss.detach(),
                "teacher_code_lap_loss": teacher_code_lap_loss.detach(),
                "teacher_code_count": teacher_code_count_metric.detach(),
                "teacher_code_pickscore_gate": teacher_code_pickscore_gate.detach(),
                "teacher_code_lap_energy_ratio": teacher_code_lap_energy_ratio.detach(),
                "teacher_code_lap_energy_penalty": teacher_code_lap_energy_penalty.detach(),
                "teacher_code_reference_l1_loss": teacher_code_reference_l1_loss.detach(),
                "teacher_code_reference_grad_loss": teacher_code_reference_grad_loss.detach(),
                "teacher_code_reference_lap_loss": teacher_code_reference_lap_loss.detach(),
                "teacher_code_reference_detail_loss": teacher_code_reference_detail_loss.detach(),
                "teacher_code_pickscore_weight": torch.tensor(
                    teacher_code_pickscore_weight_step, device=device
                ),
                "teacher_code_reference_weight_sum": torch.tensor(
                    teacher_code_reference_l1_weight_step
                    + teacher_code_reference_grad_weight_step
                    + teacher_code_reference_lap_weight_step
                    + teacher_code_reference_detail_weight_step,
                    device=device,
                ),
                "grpo_image_pickscore": grpo_image_pickscore.detach(),
                "grpo_image_pickscore_loss": grpo_image_pickscore_loss.detach(),
                "llamagen_prior_pickscore": llamagen_prior_pickscore.detach(),
                "llamagen_prior_loss": llamagen_prior_loss.detach(),
                "geneval_prior_distill_loss": geneval_prior_distill_loss.detach(),
                "geneval_offline_distill_loss": geneval_offline_distill_loss.detach(),
                "output_saturation_loss": output_saturation_loss.detach(),
                "gan_g_loss": gan_g_loss.detach(),
                "gan_feature_matching_loss": gan_feature_matching_loss.detach(),
                "gan_d_loss": gan_d_loss.detach(),
                "fid_gan_g_loss": fid_gan_g_loss.detach(),
                "fid_gan_d_loss": fid_gan_d_loss.detach(),
                "entropy": entropy.detach(),
                "approx_kl": approx_kl.detach(),
                "ref_kl": ref_kl.detach(),
                "clip_fraction": clip_fraction.detach(),
                "update_epoch": torch.tensor(float(update_epoch + 1), device=device),
                "encoder_grad_norm": gradient_norms["encoder"],
                "codebook_grad_norm": gradient_norms["codebook"],
                "decoder_grad_norm": gradient_norms["decoder"],
            }
            if target_kl > 0.0 and float(approx_kl.detach().cpu()) > target_kl:
                break
        global_step += 1

        if step % int(cfg.train.log_every) == 0:
            train_metrics = {
                    "loss": last_metrics["loss"],
                    "policy_loss": last_metrics["policy_loss"],
                    "sample_recon_mse": sample_recon_losses.mean().detach(),
                    "train_recon_mse": last_metrics["recon_mse"],
                    "tokenizer_recon_mse": last_metrics["tokenizer_recon_mse"],
                    "tokenizer_l1_loss": last_metrics["tokenizer_l1_loss"],
                    "tokenizer_grad_loss": last_metrics["tokenizer_grad_loss"],
                    "tokenizer_lap_loss": last_metrics["tokenizer_lap_loss"],
                    "tokenizer_perceptual_loss": last_metrics["tokenizer_perceptual_loss"],
                    "tokenizer_official_lpips_loss": last_metrics[
                        "tokenizer_official_lpips_loss"
                    ],
                    "tokenizer_dists_loss": last_metrics["tokenizer_dists_loss"],
                    "tokenizer_ssim_loss": last_metrics["tokenizer_ssim_loss"],
                    "tokenizer_ms_ssim_loss": last_metrics["tokenizer_ms_ssim_loss"],
                    "tokenizer_reward": last_metrics["tokenizer_reward"],
                    "tokenizer_reward_primary": last_metrics["tokenizer_reward_primary"],
                    "tokenizer_reward_baseline": last_metrics["tokenizer_reward_baseline"],
                    "tokenizer_reward_loss": last_metrics["tokenizer_reward_loss"],
                    "tokenizer_reward_quality_cosine": last_metrics[
                        "tokenizer_reward_quality_cosine"
                    ],
                    "tokenizer_reward_projection_active": last_metrics[
                        "tokenizer_reward_projection_active"
                    ],
                    "tokenizer_reward_projection_ratio": last_metrics[
                        "tokenizer_reward_projection_ratio"
                    ],
                    "tokenizer_fid_feature_loss": last_metrics[
                        "tokenizer_fid_feature_loss"
                    ],
                    "tokenizer_quality_guard_active_fraction": last_metrics[
                        "tokenizer_quality_guard_active_fraction"
                    ],
                    "tokenizer_quality_guard_max_gap": last_metrics[
                        "tokenizer_quality_guard_max_gap"
                    ],
                    **{
                        f"tokenizer_quality_guard_{name}_gap": last_metrics[
                            f"tokenizer_quality_guard_{name}_gap"
                        ]
                        for name in quality_guard_tolerances
                    },
                    "tokenizer_vq_loss": last_metrics["tokenizer_vq_loss"],
                    "encoder_anchor_recon_mse": last_metrics["encoder_anchor_recon_mse"],
                    "encoder_anchor_l1_loss": last_metrics["encoder_anchor_l1_loss"],
                    "encoder_anchor_grad_loss": last_metrics["encoder_anchor_grad_loss"],
                    "encoder_anchor_lap_loss": last_metrics["encoder_anchor_lap_loss"],
                    "encoder_anchor_kl_loss": last_metrics["encoder_anchor_kl_loss"],
                    "teacher_code_pickscore": last_metrics["teacher_code_pickscore"],
                    "teacher_code_pickscore_loss": last_metrics["teacher_code_pickscore_loss"],
                    "teacher_code_recon_mse": last_metrics["teacher_code_recon_mse"],
                    "teacher_code_l1_loss": last_metrics["teacher_code_l1_loss"],
                    "teacher_code_grad_loss": last_metrics["teacher_code_grad_loss"],
                    "teacher_code_lap_loss": last_metrics["teacher_code_lap_loss"],
                    "teacher_code_count": last_metrics["teacher_code_count"],
                    "teacher_code_pickscore_gate": last_metrics["teacher_code_pickscore_gate"],
                    "teacher_code_lap_energy_ratio": last_metrics["teacher_code_lap_energy_ratio"],
                    "teacher_code_lap_energy_penalty": last_metrics[
                        "teacher_code_lap_energy_penalty"
                    ],
                    "teacher_code_reference_l1_loss": last_metrics[
                        "teacher_code_reference_l1_loss"
                    ],
                    "teacher_code_reference_grad_loss": last_metrics[
                        "teacher_code_reference_grad_loss"
                    ],
                    "teacher_code_reference_lap_loss": last_metrics[
                        "teacher_code_reference_lap_loss"
                    ],
                    "teacher_code_reference_detail_loss": last_metrics[
                        "teacher_code_reference_detail_loss"
                    ],
                    "teacher_code_pickscore_weight": last_metrics[
                        "teacher_code_pickscore_weight"
                    ],
                    "teacher_code_reference_weight_sum": last_metrics[
                        "teacher_code_reference_weight_sum"
                    ],
                    "grpo_image_pickscore": last_metrics["grpo_image_pickscore"],
                    "grpo_image_pickscore_loss": last_metrics["grpo_image_pickscore_loss"],
                    "llamagen_prior_pickscore": last_metrics["llamagen_prior_pickscore"],
                    "llamagen_prior_loss": last_metrics["llamagen_prior_loss"],
                    "geneval_prior_distill_loss": last_metrics["geneval_prior_distill_loss"],
                    "geneval_offline_distill_loss": last_metrics["geneval_offline_distill_loss"],
                    "output_saturation_loss": last_metrics["output_saturation_loss"],
                    "gan_g_loss": last_metrics["gan_g_loss"],
                    "gan_feature_matching_loss": last_metrics["gan_feature_matching_loss"],
                    "gan_d_loss": last_metrics["gan_d_loss"],
                    "fid_gan_g_loss": last_metrics["fid_gan_g_loss"],
                    "fid_gan_d_loss": last_metrics["fid_gan_d_loss"],
                    "sample_grad_loss": sample_grad_losses.mean().detach(),
                    "sample_lap_loss": sample_lap_losses.mean().detach(),
                    "sampled_reward": pick_scores.mean().detach(),
                    "sampled_reward_best": pick_scores.max(dim=1).values.mean().detach(),
                    "sampled_reward_std": pick_scores.std(dim=1, unbiased=False).mean().detach(),
                    "sampled_reward_objective": objective_scores.mean().detach(),
                    "pickscore": pick_scores.mean().detach(),
                    "pickscore_best": pick_scores.max(dim=1).values.mean().detach(),
                    "pickscore_std": pick_scores.std(dim=1, unbiased=False).mean().detach(),
                    "pickscore_group_std": mean_group_std(pick_scores).detach(),
                    "recon_group_std": mean_group_std(recon_losses.detach()).detach(),
                    "gate_penalty_group_std": mean_group_std(gate_penalty.detach()).detach(),
                    "objective_group_std": mean_group_std(objective_scores.detach()).detach(),
                    "gated_pickscore": gated_pick_scores.mean().detach(),
                    "recon_gate": recon_gate.mean().detach(),
                    "objective_reward": objective_scores.mean().detach(),
                    "advantage_score": advantage_scores.mean().detach(),
                    "advantage_score_best": advantage_scores.max(dim=1).values.mean().detach(),
                    "adv_std": advantages.std(unbiased=False).detach(),
                    "adv_pickscore_corr": mean_group_corr(advantages.detach(), pick_scores.detach()).detach(),
                    "adv_recon_loss_corr": mean_group_corr(advantages.detach(), recon_losses.detach()).detach(),
                    "adv_objective_corr": mean_group_corr(advantages.detach(), objective_scores.detach()).detach(),
                    "entropy": last_metrics["entropy"],
                    "approx_kl": last_metrics["approx_kl"],
                    "ref_kl": last_metrics["ref_kl"],
                    "clip_fraction": last_metrics["clip_fraction"],
                    "update_epoch": last_metrics["update_epoch"],
                    "learning_rate": torch.tensor(current_lr, device=device),
                    "decoder_learning_rate": torch.tensor(current_decoder_lr, device=device),
                    "encoder_grad_norm": last_metrics["encoder_grad_norm"],
                    "codebook_grad_norm": last_metrics["codebook_grad_norm"],
                    "decoder_grad_norm": last_metrics["decoder_grad_norm"],
                }
            if prior_geneval_scores is not None:
                train_metrics.update(
                    {
                        "geneval_prior_reward": prior_geneval_scores.mean().detach(),
                        "geneval_prior_best_candidate_reward": prior_best_scores.mean().detach(),
                        "geneval_prior_gain": prior_reward_gains.mean().detach(),
                        "geneval_prior_improvement_rate": (prior_reward_gains > 0).float().mean().detach(),
                    }
                )
            train_metrics.update(reward_metrics)
            log_metrics(
                train_metrics,
                global_step,
                "rl",
                wandb_run,
            )
        if step % int(cfg.train.sample_every) == 0:
            maybe_log_samples(model, images, output_dir, global_step, wandb_run)
        if is_main_process() and step % int(cfg.train.save_every) == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"rl_{step:07d}.pt",
                model,
                optimizer,
                step,
                "rl",
                cfg,
                discriminator,
                discriminator_optimizer,
                fid_discriminator,
                fid_discriminator_optimizer,
            )

    if is_main_process():
        save_checkpoint(
            output_dir / "checkpoints" / "rl_last.pt",
            model,
            optimizer,
            total_rl_steps,
            "rl",
            cfg,
            discriminator,
            discriminator_optimizer,
            fid_discriminator,
            fid_discriminator_optimizer,
        )
        if device.type == "cuda":
            allocated_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
            reserved_gib = torch.cuda.max_memory_reserved(device) / (1024**3)
            print(
                f"Peak CUDA memory: allocated={allocated_gib:.2f}GiB reserved={reserved_gib:.2f}GiB",
                flush=True,
            )
    if wandb_run is not None:
        wandb_run.finish()
    cleanup()


if __name__ == "__main__":
    main()
