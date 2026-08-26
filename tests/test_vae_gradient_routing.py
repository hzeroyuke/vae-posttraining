from types import SimpleNamespace

import torch
from torch import nn

from vae_posttraining.train import VAEWrapper


class TinyVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, kernel_size=1)
        self.decoder = nn.Conv2d(4, 3, kernel_size=1)

    def encode(self, images: torch.Tensor) -> SimpleNamespace:
        mean = self.encoder(images)
        logvar = torch.full_like(mean, -4.0)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=mean, logvar=logvar))

    def decode(self, latents: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(sample=self.decoder(latents))


def _has_nonzero_grad(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in module.parameters()
    )


def test_grpo_policy_gradient_reaches_encoder_but_not_decoder() -> None:
    vae = TinyVAE()
    wrapper = VAEWrapper(
        vae,
        scale=1.0,
        group_size=2,
        reconstruction_mode="mean",
        behavior_decoder_grad=False,
    )
    images = torch.randn(1, 3, 4, 4)
    eps_post = torch.zeros(1, 4, 4, 4)
    eps_expl = torch.stack(
        [torch.ones(1, 4, 4, 4), -torch.ones(1, 4, 4, 4)], dim=1
    )
    sigma = torch.full((4, 1, 1), 0.1)

    outputs = wrapper(images, eps_post, eps_expl, sigma)
    mu = outputs["mu_scaled"]
    sampled_z = outputs["z_scaled"].detach()
    advantages = torch.tensor([[1.0, -1.0]])
    logp = -0.5 * ((sampled_z - mu[:, None]) / sigma[None, None]).square().sum((2, 3, 4))
    policy_loss = -(advantages * logp).mean()
    policy_loss.backward()

    assert _has_nonzero_grad(vae.encoder)
    assert not _has_nonzero_grad(vae.decoder)
    assert not outputs["behavior"].requires_grad


def test_reconstruction_gradient_reaches_encoder_and_decoder() -> None:
    vae = TinyVAE()
    wrapper = VAEWrapper(
        vae,
        scale=1.0,
        group_size=2,
        reconstruction_mode="mean",
        behavior_decoder_grad=False,
    )
    images = torch.randn(1, 3, 4, 4)
    outputs = wrapper(
        images,
        torch.zeros(1, 4, 4, 4),
        torch.randn(1, 2, 4, 4, 4),
        torch.full((4, 1, 1), 0.1),
    )

    reconstruction_loss = (outputs["reconstruction"] - images).abs().mean()
    reconstruction_loss.backward()

    assert _has_nonzero_grad(vae.encoder)
    assert _has_nonzero_grad(vae.decoder)
