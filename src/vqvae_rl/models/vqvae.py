from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, codebook_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, codebook_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_channels, 3, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@dataclass
class SampleOutput:
    recon: torch.Tensor
    logits: torch.Tensor
    codes: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor
    score_inputs: torch.Tensor | None = None


class StochasticCodebookVQVAE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int, codebook_size: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, hidden_channels, codebook_size)
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        self.decoder = Decoder(in_channels, hidden_channels, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def encode_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim == 3:
            z = self.codebook(codes).permute(0, 3, 1, 2).contiguous()
            return self.decoder(z)
        if codes.ndim != 4:
            raise ValueError(f"Expected codes with 3 or 4 dims, got {tuple(codes.shape)}")

        bsz, num_samples, height, width = codes.shape
        flat_codes = codes.reshape(bsz * num_samples, height, width)
        z = self.codebook(flat_codes).permute(0, 3, 1, 2).contiguous()
        recon = self.decoder(z)
        return recon.view(bsz, num_samples, *recon.shape[1:])

    def decode_soft_assignments(self, assignments: torch.Tensor) -> torch.Tensor:
        z = torch.einsum("bhwk,kd->bdhw", assignments, self.codebook.weight)
        return self.decoder(z)

    def pretrain_forward(self, x: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.encode_logits(x)
        assignments = F.gumbel_softmax(
            logits.permute(0, 2, 3, 1).float(),
            tau=temperature,
            hard=True,
            dim=-1,
        ).to(dtype=logits.dtype)
        recon = self.decode_soft_assignments(assignments)
        entropy = categorical_entropy(logits).mean()
        return recon, logits, entropy

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
        if mode == "greedy":
            return self.greedy_reconstruct(x)
        raise ValueError(f"Unsupported forward mode: {mode}")


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.float().softmax(dim=1)
    log_probs = logits.float().log_softmax(dim=1)
    return -(probs * log_probs).sum(dim=1)
