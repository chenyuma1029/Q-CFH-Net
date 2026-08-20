import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcfhnet.training import (
    build_reconstruction_loss,
    mixed_mse_nmse_loss,
    reconstruction_mse,
    reconstruction_nmse_loss,
)


def test_reconstruction_nmse_loss_matches_manual_ratio():
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    pred = target + 1.0
    expected = torch.tensor(4.0 / 30.0)
    assert torch.allclose(reconstruction_nmse_loss(pred, target), expected)


def test_mixed_mse_nmse_loss_combines_terms():
    target = torch.ones(1, 2, 2, 2)
    pred = torch.zeros_like(target)
    expected = reconstruction_mse(pred, target) + 0.25 * reconstruction_nmse_loss(pred, target)
    assert torch.allclose(mixed_mse_nmse_loss(pred, target, lambda_nmse=0.25), expected)


def test_build_reconstruction_loss_modes():
    target = torch.ones(1, 2, 2, 2)
    pred = torch.zeros_like(target)
    assert torch.allclose(build_reconstruction_loss("mse")(pred, target), reconstruction_mse(pred, target))
    assert torch.allclose(build_reconstruction_loss("nmse")(pred, target), reconstruction_nmse_loss(pred, target))
    assert torch.allclose(
        build_reconstruction_loss("mixed", lambda_nmse=0.25)(pred, target),
        mixed_mse_nmse_loss(pred, target, lambda_nmse=0.25),
    )
