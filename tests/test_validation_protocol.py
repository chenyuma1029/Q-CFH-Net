import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dummy_train_validation_and_test_splits_are_distinct():
    train_script = load_module("train_script_for_dummy_split_test", ROOT / "scripts" / "train.py")
    config = {"data": {"name": "dummy", "train_samples": 4, "val_samples": 4, "test_samples": 4, "nc": 2, "nt": 2}}

    h_train, _ = train_script.load_split(config, "train", seed=7)
    h_val, _ = train_script.load_split(config, "val", seed=7)
    h_test, _ = train_script.load_split(config, "test", seed=7)

    assert not (h_train == h_val).all()
    assert not (h_val == h_test).all()
    assert not (h_train == h_test).all()


def test_train_uses_val_for_selection_and_test_for_final_metrics(tmp_path):
    config = {
        "experiment": {
            "name": "validation_protocol_dummy",
            "seed": 1,
            "output_dir": str(tmp_path / "results" / "validation_protocol_dummy"),
        },
        "data": {
            "name": "dummy",
            "train_samples": 8,
            "val_samples": 4,
            "test_samples": 4,
            "nc": 4,
            "nt": 4,
        },
        "model": {
            "name": "qcfhnet",
            "nc": 4,
            "nt": 4,
            "latent_dim": 4,
            "hidden_dim": 8,
            "num_wire_layers": 1,
            "coarse_grid": 2,
            "seed_channels": 4,
            "upsampler": "interp_conv",
        },
        "quantization": {"enabled": True, "bitwidth": 2},
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "eval_batch_size": 4,
            "lr": 1.0e-3,
            "loss": "mse",
            "val_freq": 1,
            "num_workers": 0,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path), "--device", "cpu"],
        cwd=ROOT,
        check=True,
    )

    result_path = tmp_path / "results" / "validation_protocol_dummy" / "results.json"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    assert results["selection_split"] == "val"
    assert results["selection_metric"] == "val_nmse_db_mean"
    assert results["test_not_used_for_selection"] is True
    assert "best_val_nmse_db_mean" in results
    assert results["nmse_db_mean"] == results["test_nmse_db_mean"]
    assert results["nmse_global_db"] == results["test_nmse_global_db"]
    assert results["cosine_mean"] == results["test_cosine_mean"]

    checkpoint = torch.load(
        ROOT / "checkpoints" / "validation_protocol_dummy" / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["selection_split"] == "val"
    assert checkpoint["selection_metric"] == "val_nmse_db_mean"
    assert "nmse_db_mean" in checkpoint["metrics"]


def test_selection_only_never_loads_or_evaluates_test_split(tmp_path):
    output_dir = tmp_path / "selection_only_result"
    config = {
        "experiment": {
            "name": "validation_only_dummy",
            "seed": 1,
            "output_dir": str(output_dir),
        },
        "data": {
            "name": "dummy",
            "train_samples": 8,
            "val_samples": 4,
            "test_path": "/this/path/must/not/be/accessed.mat",
            "nc": 4,
            "nt": 4,
        },
        "model": {
            "name": "qcfhnet",
            "nc": 4,
            "nt": 4,
            "latent_dim": 4,
            "hidden_dim": 8,
            "num_wire_layers": 1,
            "coarse_grid": 2,
            "seed_channels": 4,
            "upsampler": "interp_conv",
        },
        "quantization": {"enabled": True, "bitwidth": 2},
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "eval_batch_size": 4,
            "lr": 1.0e-3,
            "loss": "mse",
            "val_freq": 1,
            "num_workers": 0,
        },
    }
    cfg_path = tmp_path / "selection_only.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--config",
            str(cfg_path),
            "--device",
            "cpu",
            "--selection-only",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((output_dir / "selection_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert summary["selection_only"] is True
    assert summary["test_accessed"] is False
    assert summary["selection_split"] == "val"
    assert "test" not in metadata
    assert not (output_dir / "results.json").exists()


def test_experiment_name_override_updates_name_and_output_dir(tmp_path):
    original_output = tmp_path / "must_not_be_used"
    target_name = "validation_protocol_explicit_name"
    config = {
        "experiment": {
            "name": "old_name",
            "seed": 1,
            "output_dir": str(original_output),
        },
        "data": {
            "name": "dummy",
            "train_samples": 8,
            "val_samples": 4,
            "nc": 4,
            "nt": 4,
        },
        "model": {
            "name": "qcfhnet",
            "nc": 4,
            "nt": 4,
            "latent_dim": 4,
            "hidden_dim": 8,
            "num_wire_layers": 1,
            "coarse_grid": 2,
            "seed_channels": 4,
            "upsampler": "interp_conv",
        },
        "quantization": {"enabled": True, "bitwidth": 2},
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "eval_batch_size": 4,
            "lr": 1.0e-3,
            "loss": "mse",
            "val_freq": 1,
            "num_workers": 0,
        },
    }
    cfg_path = tmp_path / "explicit_name.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--config",
            str(cfg_path),
            "--device",
            "cpu",
            "--selection-only",
            "--experiment-name",
            target_name,
        ],
        cwd=tmp_path,
        check=True,
    )

    output_dir = ROOT / "results" / target_name
    summary = json.loads((output_dir / "selection_summary.json").read_text(encoding="utf-8"))
    saved_config = yaml.safe_load((output_dir / "config.yaml").read_text(encoding="utf-8"))
    assert summary["experiment"] == target_name
    assert saved_config["experiment"]["name"] == target_name
    assert saved_config["experiment"]["output_dir"] == f"results/{target_name}"
    assert not original_output.exists()
