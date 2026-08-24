from __future__ import annotations

from typing import Any

from .llamagen_tokenizer import StochasticLlamaGenTokenizer
from .vqvae import StochasticCodebookVQVAE


def build_model(cfg: Any):
    model_type = str(getattr(cfg, "type", "vqvae"))
    if model_type == "vqvae":
        return StochasticCodebookVQVAE(
            in_channels=int(cfg.in_channels),
            hidden_channels=int(cfg.hidden_channels),
            latent_dim=int(cfg.latent_dim),
            codebook_size=int(cfg.codebook_size),
        )
    if model_type == "llamagen_tokenizer":
        return StochasticLlamaGenTokenizer(
            vq_model=str(getattr(cfg, "vq_model", "VQ-16")),
            checkpoint=getattr(cfg, "checkpoint", None),
            codebook_size=int(getattr(cfg, "codebook_size", 16384)),
            codebook_embed_dim=int(getattr(cfg, "codebook_embed_dim", 8)),
            codebook_l2_norm=bool(getattr(cfg, "codebook_l2_norm", True)),
            codebook_show_usage=bool(getattr(cfg, "codebook_show_usage", False)),
            freeze_encoder=bool(getattr(cfg, "freeze_encoder", False)),
            freeze_decoder=bool(getattr(cfg, "freeze_decoder", False)),
            freeze_codebook=bool(getattr(cfg, "freeze_codebook", True)),
            policy_detach_codebook=bool(getattr(cfg, "policy_detach_codebook", False)),
        )
    raise ValueError(f"Unsupported model.type: {model_type}")
