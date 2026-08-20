import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcfhnet.models import build_model


def test_qcfhnet_shape_32x32():
    config = {
        "data": {"nc": 32, "nt": 32},
        "model": {
            "name": "qcfhnet",
            "latent_dim": 16,
            "hidden_dim": 32,
            "num_wire_layers": 1,
            "coarse_grid": 2,
            "seed_channels": 8,
        },
        "quantization": {"enabled": True, "bitwidth": 4},
    }
    model = build_model(config)
    x = torch.randn(3, 2, 32, 32)
    out = model(x)
    assert out["h_hat"].shape == x.shape
    assert out["z"].shape == (3, 16)


def test_qcfhnet_shape_32x64():
    config = {
        "data": {"nc": 32, "nt": 64},
        "model": {
            "name": "qcfhnet",
            "latent_dim": 16,
            "hidden_dim": 32,
            "num_wire_layers": 1,
            "coarse_grid": 2,
            "seed_channels": 8,
        },
        "quantization": {"enabled": True, "bitwidth": 4},
    }
    model = build_model(config)
    x = torch.randn(2, 2, 32, 64)
    out = model(x)
    assert out["h_hat"].shape == x.shape


def test_qcfhnet_deconv_shape_32x32():
    config = {
        "data": {"nc": 32, "nt": 32},
        "model": {
            "name": "qcfhnet",
            "latent_dim": 16,
            "hidden_dim": 32,
            "num_wire_layers": 1,
            "coarse_grid": 2,
            "seed_channels": 8,
            "upsampler": "deconv_cascade",
        },
        "quantization": {"enabled": True, "bitwidth": 4},
    }
    model = build_model(config)
    x = torch.randn(2, 2, 32, 32)
    out = model(x)
    assert out["h_hat"].shape == x.shape
    assert model.upsampler_fallback_interpolate is False


def test_qcfhnet_interp_shape_32x64():
    config = {
        "data": {"nc": 32, "nt": 64},
        "model": {
            "name": "qcfhnet",
            "latent_dim": 16,
            "hidden_dim": 32,
            "num_wire_layers": 1,
            "coarse_grid": 2,
            "seed_channels": 8,
            "upsampler": "interp_conv",
        },
        "quantization": {"enabled": True, "bitwidth": 4},
    }
    model = build_model(config)
    x = torch.randn(2, 2, 32, 64)
    out = model(x)
    assert out["h_hat"].shape == x.shape
    assert model.upsampler_fallback_interpolate is False


def test_qcfhnet_deconv_fallback_shape_32x64():
    config = {
        "data": {"nc": 32, "nt": 64},
        "model": {
            "name": "qcfhnet",
            "latent_dim": 16,
            "hidden_dim": 32,
            "num_wire_layers": 1,
            "coarse_grid": 2,
            "seed_channels": 8,
            "upsampler": "deconv_cascade",
        },
        "quantization": {"enabled": True, "bitwidth": 4},
    }
    model = build_model(config)
    x = torch.randn(2, 2, 32, 64)
    out = model(x)
    assert out["h_hat"].shape == x.shape
    assert model.upsampler_fallback_interpolate is True


def test_qcfhnet_rect_deconv_shape_32x64_without_fallback():
    config = {
        "data": {"nc": 32, "nt": 64},
        "model": {
            "name": "qcfhnet",
            "latent_dim": 16,
            "hidden_dim": 32,
            "num_wire_layers": 1,
            "coarse_grid": 2,
            "seed_channels": 8,
            "upsampler": "deconv_rect_cascade",
        },
        "quantization": {"enabled": True, "bitwidth": 4},
    }
    model = build_model(config)
    x = torch.randn(2, 2, 32, 64)
    out = model(x)
    assert out["h_hat"].shape == x.shape
    assert model.upsampler_fallback_interpolate is False


def test_qcfhnet_split_quantized_shape():
    config = {
        "data": {"nc": 32, "nt": 32},
        "model": {
            "name": "qcfhnet",
            "split_real_imag": True,
            "latent_dim": 8,
            "hidden_dim": 32,
            "num_wire_layers": 1,
            "coarse_grid": 2,
            "seed_channels": 8,
            "upsampler": "deconv_cascade",
        },
        "quantization": {"enabled": True, "bitwidth": 4},
    }
    model = build_model(config)
    x = torch.randn(2, 2, 32, 32)
    out = model(x)
    assert out["h_hat"].shape == x.shape
    assert out["z"].shape == (2, 16)
    assert out["z_tilde"].shape == (2, 16)
    assert out["z_real"].shape == (2, 8)
    assert out["z_imag"].shape == (2, 8)
    assert model.feedback_scalars == 16
    assert model.quantized_feedback_bits == 64


def test_split_original_shape():
    config = {
        "data": {"nc": 32, "nt": 32},
        "model": {
            "name": "cfhnet",
            "split_real_imag": True,
            "latent_dim": 8,
            "hidden_dim": 32,
            "num_wire_layers": 1,
            "coarse_grid": 2,
            "seed_channels": 8,
        },
    }
    model = build_model(config)
    x = torch.randn(2, 2, 32, 32)
    out = model(x)
    assert out["h_hat"].shape == x.shape
    assert out["z"].shape == (2, 16)


def test_csi_inr_ff_quantized_shape_and_query_count():
    config = {
        "data": {"nc": 32, "nt": 32},
        "model": {
            "name": "csi_inr_ff",
            "split_real_imag": True,
            "latent_dim": 8,
            "hidden_dim": 32,
            "num_wire_layers": 1,
            "backbone": "wire",
            "use_hyperfilm": True,
        },
        "quantization": {"enabled": True, "bitwidth": 4},
    }
    model = build_model(config)
    x = torch.randn(2, 2, 32, 32)
    out = model(x)
    assert out["h_hat"].shape == x.shape
    assert out["z"].shape == (2, 16)
    assert model.feedback_scalars == 16
    assert model.quantized_feedback_bits == 64
    assert model.query_count == 32 * 32
