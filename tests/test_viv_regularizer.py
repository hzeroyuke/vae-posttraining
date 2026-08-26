import math

import torch

from vae_posttraining.train import viv_channel_std, viv_diag_proxy


def test_viv_proxy_is_positive_and_differentiable() -> None:
    samples = torch.tensor(
        [
            [[[0.0, 1.0]], [[2.0, 3.0]]],
            [[[1.0, 3.0]], [[4.0, 8.0]]],
            [[[2.0, 5.0]], [[6.0, 13.0]]],
        ],
        requires_grad=True,
    )
    value = viv_diag_proxy(samples)
    assert value.item() > 0
    value.backward()
    assert samples.grad is not None
    assert torch.isfinite(samples.grad).all()


def test_viv_proxy_matches_diagonal_formula() -> None:
    samples = torch.tensor([[[[0.0]], [[0.0]]], [[[2.0]], [[4.0]]]])
    expected = math.pi / 2.0 * torch.sqrt(torch.tensor([1.0, 4.0])).mean()
    assert torch.allclose(viv_diag_proxy(samples), expected, atol=1e-6)
    assert torch.allclose(viv_channel_std(samples), torch.tensor([1.0, 2.0]), atol=1e-6)


def test_viv_proxy_returns_zero_for_single_sample() -> None:
    samples = torch.zeros(1, 4, 2, 2, requires_grad=True)
    value = viv_diag_proxy(samples)
    assert value.item() == 0.0
    value.backward()
    assert samples.grad is not None
