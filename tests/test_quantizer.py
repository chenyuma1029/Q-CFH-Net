import sys
from pathlib import Path

import torch
import pytest

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


def test_quantizer_rejects_nonpositive_initial_scale():
    with pytest.raises(ValueError, match="init_scale"):
        UniformSTEQuantizer(bitwidth=4, init_scale=0.0)


def test_quantizer_integer_index_roundtrip_matches_forward_values():
    quantizer = UniformSTEQuantizer(bitwidth=4, enabled=True, learn_scale=False)
    z = torch.linspace(-3.0, 3.0, 25)
    indices = quantizer.encode_indices(z)
    decoded = quantizer.decode_indices(indices)

    assert indices.dtype == torch.int64
    assert int(indices.min()) >= quantizer.qmin
    assert int(indices.max()) <= quantizer.qmax
    assert torch.allclose(decoded, quantizer(z))


def test_quantizer_rejects_out_of_range_indices():
    quantizer = UniformSTEQuantizer(bitwidth=4)
    with pytest.raises(ValueError, match="indices"):
        quantizer.decode_indices(torch.tensor([quantizer.qmax + 1]))
