import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcfhnet.quantization import UniformSTEQuantizer


def test_quantizer_forward_shape():
    quantizer = UniformSTEQuantizer(bitwidth=4, enabled=True)
    z = torch.randn(5, 16)
    out = quantizer(z)
    assert out.shape == z.shape


def test_quantizer_backward():
    quantizer = UniformSTEQuantizer(bitwidth=4, enabled=True)
    z = torch.randn(5, 16, requires_grad=True)
    loss = quantizer(z).pow(2).mean()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()

