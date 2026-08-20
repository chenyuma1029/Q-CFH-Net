import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcfhnet.datasets import load_npz_csi


def load_train_module():
    spec = importlib.util.spec_from_file_location("train_script_for_npz_test", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_prepare_deepmimo_module():
    spec = importlib.util.spec_from_file_location("prepare_deepmimo_for_test", ROOT / "scripts" / "prepare_deepmimo.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_npz_csi_with_metadata(tmp_path):
    h_train = np.zeros((4, 2, 3, 5), dtype=np.float32)
    path = tmp_path / "csi.npz"
    np.savez(path, h_train=h_train, metadata=json.dumps({"source": "unit"}))
    h, meta = load_npz_csi(path, "h_train", nc=3, nt=5)
    assert h.shape == (4, 2, 3, 5)
    assert meta["source_metadata"]["source"] == "unit"


def test_train_load_split_supports_npz(tmp_path):
    h_train = np.zeros((4, 2, 3, 5), dtype=np.float32)
    h_test = np.ones((2, 2, 3, 5), dtype=np.float32)
    path = tmp_path / "csi.npz"
    np.savez(path, h_train=h_train, h_test=h_test, metadata=json.dumps({"source": "unit"}))
    train_script = load_train_module()
    config = {
        "data": {
            "name": "npz",
            "npz_path": str(path),
            "train_key": "h_train",
            "test_key": "h_test",
            "nc": 3,
            "nt": 5,
        }
    }
    h, meta = train_script.load_split(config, "test", seed=1)
    assert h.shape == (2, 2, 3, 5)
    assert meta["name"] == "npz"


def test_deepmimo_frequency_to_angular_delay_shapes():
    prep = load_prepare_deepmimo_module()
    h_freq = np.ones((3, 8, 4), dtype=np.complex64)
    h_ad = prep.frequency_to_angular_delay(h_freq, nc=4, nt=4, delay_fft=8)
    hf = prep.frequency_for_physical(h_freq, nt=4, n_eval=5)
    assert h_ad.shape == (3, 2, 4, 4)
    assert h_ad.dtype == np.float32
    assert hf.shape == (3, 4, 5)
    assert hf.dtype == np.complex64


def test_per_sample_complex_rms_scale_normalizes_each_channel():
    prep = load_prepare_deepmimo_module()
    h = np.ones((2, 2, 3, 4), dtype=np.float32)
    h[1] *= 10.0
    scale = prep.per_sample_complex_rms_scale(h)
    out = prep.apply_per_sample_scale(h, scale)
    power = out[:, 0] ** 2 + out[:, 1] ** 2
    assert np.allclose(np.sqrt(power.mean(axis=(1, 2))), np.ones(2), atol=1e-6)
    assert scale[1] > scale[0]
