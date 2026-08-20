import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcfhnet.utils.checkpoint import load_checkpoint, validate_checkpoint_compatibility
from qcfhnet.utils.config import load_config


def minimal_config(bitwidth: int = 4) -> dict:
    return {
        "experiment": {"name": "unit", "seed": 1},
        "data": {"name": "dummy", "nc": 4, "nt": 4},
        "model": {
            "name": "qcfhnet",
            "nc": 4,
            "nt": 4,
            "latent_dim": 4,
            "hidden_dim": 8,
            "num_wire_layers": 1,
            "coarse_grid": 2,
            "seed_channels": 4,
        },
        "quantization": {"enabled": True, "mode": "qat", "bitwidth": bitwidth, "init_scale": 1.0},
        "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3, "val_freq": 1},
    }


def write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_config_rejects_unknown_keys(tmp_path):
    config = minimal_config()
    config["quantization"]["bitwidht"] = 4
    with pytest.raises(ValueError, match="bitwidht"):
        load_config(write_config(tmp_path, config))


def test_config_rejects_unimplemented_quantization_mode(tmp_path):
    config = minimal_config()
    config["quantization"]["mode"] = "post_training"
    with pytest.raises(ValueError, match="only 'qat' is implemented"):
        load_config(write_config(tmp_path, config))


def test_config_rejects_unknown_upsampler(tmp_path):
    config = minimal_config()
    config["model"]["upsampler"] = "typo_cascade"
    with pytest.raises(ValueError, match="model.upsampler"):
        load_config(write_config(tmp_path, config))


def test_config_rejects_string_boolean(tmp_path):
    config = minimal_config()
    config["quantization"]["enabled"] = "false"
    with pytest.raises(ValueError, match="must be a boolean"):
        load_config(write_config(tmp_path, config))


def test_config_rejects_fractional_bitwidth(tmp_path):
    config = minimal_config()
    config["quantization"]["bitwidth"] = 4.5
    with pytest.raises(ValueError, match="must be an integer"):
        load_config(write_config(tmp_path, config))


def test_checkpoint_config_rejects_bitwidth_mismatch():
    checkpoint_config = minimal_config(bitwidth=4)
    runtime_config = minimal_config(bitwidth=8)
    payload = {"model_state": {}, "config": checkpoint_config}

    with pytest.raises(ValueError, match=r"quantization\.bitwidth"):
        validate_checkpoint_compatibility(runtime_config, payload)


def test_checkpoint_config_allows_local_path_changes():
    checkpoint_config = minimal_config()
    checkpoint_config["data"].update({"name": "cost2100", "scenario": "indoor", "train_path": "/remote/train.mat"})
    runtime_config = deepcopy(checkpoint_config)
    runtime_config["data"]["train_path"] = "data/COST2100/DATA_Htrainin.mat"

    assert validate_checkpoint_compatibility(runtime_config, {"model_state": {}, "config": checkpoint_config}) == checkpoint_config


def test_checkpoint_loader_uses_safe_weights_only_format(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": {}, "config": minimal_config()}, path)
    payload = load_checkpoint(path)
    assert payload["config"]["quantization"]["bitwidth"] == 4
