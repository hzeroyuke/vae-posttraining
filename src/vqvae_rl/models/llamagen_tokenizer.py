from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .vendor.llamagen_vq_model import VQ_models
from .vqvae import SampleOutput, categorical_entropy


def _sum_optional_losses(losses: Any, device: torch.device) -> torch.Tensor:
    if not isinstance(losses, (tuple, list)):
        return torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    for loss in losses:
        if torch.is_tensor(loss):
            total = total + loss.float()
    return total


class StochasticLlamaGenTokenizer(nn.Module):
    """Policy wrapper around the LlamaGen image tokenizer.

    LlamaGen's tokenizer encodes images to continuous latents and picks the
    nearest codebook entry. For RL we treat the negative codebook distance at
    each spatial location as categorical logits, then decode sampled codes with
    the frozen-compatible tokenizer decoder.
    """

    def __init__(
        self,
        vq_model: str = "VQ-16",
        checkpoint: str | None = None,
        codebook_size: int = 16384,
        codebook_embed_dim: int = 8,
        codebook_l2_norm: bool = True,
        codebook_show_usage: bool = False,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        freeze_codebook: bool = True,
        policy_detach_codebook: bool = False,
    ):
        super().__init__()
        if vq_model not in VQ_models:
            raise ValueError(f"Unsupported LlamaGen tokenizer {vq_model!r}; available: {sorted(VQ_models)}")
        self.vq_model = vq_model
        self.codebook_size = int(codebook_size)
        self.latent_dim = int(codebook_embed_dim)
        self.policy_detach_codebook = bool(policy_detach_codebook)
        self.tokenizer = VQ_models[vq_model](
            codebook_size=self.codebook_size,
            codebook_embed_dim=self.latent_dim,
            codebook_l2_norm=bool(codebook_l2_norm),
            codebook_show_usage=bool(codebook_show_usage),
        )
        if checkpoint:
            load_llamagen_checkpoint(self.tokenizer, checkpoint)
        if freeze_encoder:
            self.tokenizer.encoder.requires_grad_(False)
            self.tokenizer.quant_conv.requires_grad_(False)
        if freeze_decoder:
            self.tokenizer.decoder.requires_grad_(False)
            self.tokenizer.post_quant_conv.requires_grad_(False)
        if freeze_codebook:
            self.tokenizer.quantize.requires_grad_(False)

    def encode_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tokenizer.encoder(x)
        z = self.tokenizer.quant_conv(h)
        z = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z.view(-1, z.shape[-1]).float()
        embedding = self.tokenizer.quantize.embedding.weight
        if self.policy_detach_codebook:
            embedding = embedding.detach()
        embedding = embedding.float()
        if self.tokenizer.quantize.l2_norm:
            flat_z = F.normalize(flat_z, p=2, dim=-1)
            embedding = F.normalize(embedding, p=2, dim=-1)
        distances = (
            flat_z.square().sum(dim=1, keepdim=True)
            + embedding.square().sum(dim=1)
            - 2.0 * flat_z @ embedding.t()
        )
        logits = -distances.view(x.shape[0], z.shape[1], z.shape[2], self.codebook_size)
        return logits.permute(0, 3, 1, 2).contiguous()

    def encode_continuous_latents(self, x: torch.Tensor) -> torch.Tensor:
        latents = self.tokenizer.quant_conv(self.tokenizer.encoder(x))
        if self.tokenizer.quantize.l2_norm:
            latents = F.normalize(latents, p=2, dim=1)
        return latents

    def decode_codes(self, codes: torch.Tensor, detach_codebook: bool = False) -> torch.Tensor:
        def decode_flat(flat_codes: torch.Tensor, shape: tuple[int, int, int, int]) -> torch.Tensor:
            if not detach_codebook:
                return self.tokenizer.decode_code(flat_codes, shape=shape, channel_first=True)
            embedding = self.tokenizer.quantize.embedding.weight.detach()
            if self.tokenizer.quantize.l2_norm:
                embedding = F.normalize(embedding, p=2, dim=-1)
            quant = embedding[flat_codes].reshape(shape[0], shape[2], shape[3], shape[1])
            quant = quant.permute(0, 3, 1, 2).contiguous()
            return self.tokenizer.decode(quant)

        if codes.ndim == 3:
            bsz, height, width = codes.shape
            flat_codes = codes.reshape(-1)
            return decode_flat(flat_codes, (bsz, self.latent_dim, height, width)).clamp(-1, 1)
        if codes.ndim != 4:
            raise ValueError(f"Expected codes with 3 or 4 dims, got {tuple(codes.shape)}")

        bsz, num_samples, height, width = codes.shape
        flat_codes = codes.reshape(-1)
        recon = decode_flat(
            flat_codes,
            (bsz * num_samples, self.latent_dim, height, width),
        ).clamp(-1, 1)
        return recon.view(bsz, num_samples, *recon.shape[1:])

    def decode_code_mixture(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        detach_codebook: bool = False,
    ) -> torch.Tensor:
        if indices.ndim != 4 or weights.shape != indices.shape:
            raise ValueError(
                f"Expected matching [B,K,H,W] indices and weights, got {tuple(indices.shape)} and "
                f"{tuple(weights.shape)}"
            )
        embedding = self.tokenizer.quantize.embedding.weight
        if detach_codebook:
            embedding = embedding.detach()
        if self.tokenizer.quantize.l2_norm:
            embedding = F.normalize(embedding, p=2, dim=-1)
        quant = (embedding[indices] * weights[..., None].to(embedding.dtype)).sum(dim=1)
        quant = quant.permute(0, 3, 1, 2).contiguous()
        return self.tokenizer.decode(quant).clamp(-1, 1)

    def pretrain_forward(self, x: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon, _vq_loss = self.tokenizer(x)
        recon = recon.clamp(-1, 1)
        logits = self.encode_logits(x)
        entropy = categorical_entropy(logits).mean()
        return recon, logits, entropy

    def tokenizer_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon, vq_losses = self.tokenizer(x)
        return recon.clamp(-1, 1), _sum_optional_losses(vq_losses, x.device)

    def sample_forward(self, x: torch.Tensor, num_samples: int, temperature: float) -> SampleOutput:
        logits = self.encode_logits(x)
        bsz, codebook_size, height, width = logits.shape
        flat_logits = logits.permute(0, 2, 3, 1).reshape(bsz * height * width, codebook_size).float()
        dist = Categorical(logits=flat_logits / temperature)
        sampled = dist.sample((num_samples,))
        log_probs = dist.log_prob(sampled)
        codes = sampled.view(num_samples, bsz, height, width).permute(1, 0, 2, 3).contiguous()
        per_site_log_probs = log_probs.view(num_samples, bsz, height * width).permute(1, 0, 2).contiguous()
        recon = self.decode_codes(codes)
        entropy = dist.entropy().view(bsz, height * width).mean()
        return SampleOutput(
            recon=recon,
            logits=logits,
            codes=codes,
            log_probs=per_site_log_probs.sum(dim=-1),
            entropy=entropy,
            score_inputs=codes.float(),
        )

    def log_probs_for_codes(self, x: torch.Tensor, codes: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.encode_logits(x)
        bsz, codebook_size, height, width = logits.shape
        if codes.shape[:1] != (bsz,) or codes.shape[2:] != (height, width):
            raise ValueError(
                f"Codes shape {tuple(codes.shape)} is incompatible with logits {(bsz, codebook_size, height, width)}"
            )
        num_samples = codes.shape[1]
        flat_logits = logits.permute(0, 2, 3, 1).reshape(bsz * height * width, codebook_size).float()
        dist = Categorical(logits=flat_logits / temperature)
        flat_codes = codes.permute(1, 0, 2, 3).reshape(num_samples, bsz * height * width)
        log_probs = dist.log_prob(flat_codes)
        per_site_log_probs = log_probs.view(num_samples, bsz, height * width).permute(1, 0, 2).contiguous()
        entropy = dist.entropy().view(bsz, height * width).mean()
        return per_site_log_probs.sum(dim=-1), entropy

    def score_codes(self, x: torch.Tensor, codes: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon = self.decode_codes(codes)
        log_probs, entropy = self.log_probs_for_codes(x, codes, temperature=temperature)
        return recon, log_probs, entropy

    def score_log_probs(self, x: torch.Tensor, codes: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        return self.log_probs_for_codes(x, codes, temperature=temperature)

    def greedy_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.encode_logits(x)
        codes = logits.argmax(dim=1)
        return self.decode_codes(codes)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "pretrain",
        temperature: float = 1.0,
        num_samples: int = 1,
        codes: torch.Tensor | None = None,
    ):
        if mode == "pretrain":
            return self.pretrain_forward(x, temperature)
        if mode == "sample":
            return self.sample_forward(x, num_samples=num_samples, temperature=temperature)
        if mode == "score_codes":
            if codes is None:
                raise ValueError("codes must be provided for score_codes mode")
            return self.score_codes(x, codes=codes, temperature=temperature)
        if mode == "score_log_probs":
            if codes is None:
                raise ValueError("codes must be provided for score_log_probs mode")
            return self.score_log_probs(x, codes=codes, temperature=temperature)
        if mode == "tokenizer":
            return self.tokenizer_forward(x)
        if mode == "greedy":
            return self.greedy_reconstruct(x)
        raise ValueError(f"Unsupported forward mode: {mode}")


def load_llamagen_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"LlamaGen tokenizer checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = extract_state_dict(checkpoint)
    cleaned = {}
    for key, value in state.items():
        key = str(key)
        for prefix in ("module.", "model.", "tokenizer.", "vq_model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    missing, _unexpected = model.load_state_dict(cleaned, strict=False)
    required_prefixes = ("encoder.", "decoder.", "quantize.", "quant_conv.", "post_quant_conv.")
    missing_required = [key for key in missing if key.startswith(required_prefixes)]
    if missing_required:
        raise RuntimeError(f"Missing required keys in LlamaGen checkpoint {path}: {missing_required[:10]}")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("ema", "model", "state_dict", "tokenizer", "vq_model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(isinstance(k, str) for k in value.keys()):
                return value
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            tensor_values = [value for value in checkpoint.values() if isinstance(value, torch.Tensor)]
            if tensor_values:
                return checkpoint
    raise RuntimeError("Could not find a model state dict in LlamaGen checkpoint")
