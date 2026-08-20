from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


_ALLOWED_KEYS = {
    "experiment": {"name", "seed", "output_dir"},
    "data": {
        "name",
        "scenario",
        "train_path",
        "val_path",
        "test_path",
        "key",
        "nc",
        "nt",
        "normalization",
        "npz_path",
        "train_key",
        "val_key",
        "test_key",
        "train_samples",
        "val_samples",
        "test_samples",
        "train_samples_limit",
        "val_samples_limit",
        "test_samples_limit",
        "samples_limit",
    },
    "model": {
        "name",
        "split_real_imag",
        "nc",
        "nt",
        "latent_dim",
        "hidden_dim",
        "num_wire_layers",
        "num_layers",
        "coarse_grid",
        "seed_channels",
        "upsample_feat",
        "backbone",
        "use_hyperfilm",
        "upsampler",
    },
    "quantization": {"enabled", "mode", "bitwidth", "learn_scale", "init_scale", "symmetric", "input_clip"},
    "train": {
        "epochs",
        "batch_size",
        "eval_batch_size",
        "lr",
        "loss",
        "lambda_nmse",
        "weight_decay",
        "num_workers",
        "val_freq",
        "eta_min",
        "grad_clip",
    },
    "metrics": {"snr_db"},
}


def _require_positive(section: dict[str, Any], key: str, *, allow_zero: bool = False) -> None:
    if key not in section:
        return
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config value {key!r} must be numeric, got {type(value).__name__}")
    if (allow_zero and value < 0) or (not allow_zero and value <= 0):
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"Config value {key!r} must be {relation}, got {value}")


def _require_integer(section: dict[str, Any], key: str, *, allow_zero: bool = False) -> None:
    if key not in section:
        return
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Config value {key!r} must be an integer, got {type(value).__name__}")
    if (allow_zero and value < 0) or (not allow_zero and value <= 0):
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"Config value {key!r} must be {relation}, got {value}")


def _require_boolean(section: dict[str, Any], key: str) -> None:
    if key in section and not isinstance(section[key], bool):
        raise ValueError(f"Config value {key!r} must be a boolean, got {type(section[key]).__name__}")


def validate_config(config: dict[str, Any]) -> None:
    unknown_sections = sorted(set(config) - set(_ALLOWED_KEYS))
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {', '.join(unknown_sections)}")

    for section_name, allowed in _ALLOWED_KEYS.items():
        section = config.get(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"Config section {section_name!r} must be a mapping")
        unknown = sorted(set(section) - allowed)
        if unknown:
            raise ValueError(f"Unknown key(s) in {section_name}: {', '.join(unknown)}")

    data = config.get("data", {})
    data_name = str(data.get("name", "dummy")).lower()
    if data_name not in {"dummy", "cost2100", "npz", "deepmimo"}:
        raise ValueError(f"Unsupported data.name: {data_name}")
    if data_name in {"npz", "deepmimo"} and not data.get("npz_path"):
        raise ValueError(f"data.npz_path is required for data.name={data_name}")
    if data_name == "cost2100" and data.get("scenario") not in {"indoor", "outdoor"}:
        raise ValueError("COST2100 data.scenario must be 'indoor' or 'outdoor'")
    for key in (
        "nc",
        "nt",
        "train_samples",
        "val_samples",
        "test_samples",
        "train_samples_limit",
        "val_samples_limit",
        "test_samples_limit",
        "samples_limit",
    ):
        _require_integer(data, key)

    model = config.get("model", {})
    model_name = str(model.get("name", "qcfhnet")).lower()
    if model_name not in {"qcfhnet", "q-cfh-net", "qcfh", "cfhnet", "cfh", "original_cfhnet", "csi_inr_ff", "csi-inr-ff", "csiinrff"}:
        raise ValueError(f"Unsupported model.name: {model_name}")
    for key in ("nc", "nt", "latent_dim", "hidden_dim", "num_wire_layers", "num_layers", "coarse_grid", "seed_channels", "upsample_feat"):
        _require_integer(model, key)
    for key in ("split_real_imag", "use_hyperfilm"):
        _require_boolean(model, key)
    if str(model.get("backbone", "wire")).lower() not in {"wire", "siren", "fourier_mlp", "relu_mlp"}:
        raise ValueError(f"Unsupported model.backbone: {model.get('backbone')}")
    if str(model.get("upsampler", "interp_conv")).lower() not in {
        "interp_conv",
        "cnn",
        "bilinear_conv",
        "direct_interp",
        "bilinear_direct",
        "no_cnn_direct_interp",
        "deconv_cascade",
        "deconv_rect_cascade",
        "rect_deconv_cascade",
    }:
        raise ValueError(f"Unsupported model.upsampler: {model.get('upsampler')}")
    for key in ("nc", "nt"):
        if key in data and key in model and int(data[key]) != int(model[key]):
            raise ValueError(f"data.{key}={data[key]} does not match model.{key}={model[key]}")

    quant = config.get("quantization", {})
    mode = str(quant.get("mode", "qat")).lower()
    if mode != "qat":
        raise ValueError(f"Unsupported quantization.mode: {mode}; only 'qat' is implemented")
    _require_integer(quant, "bitwidth")
    if "bitwidth" in quant and quant["bitwidth"] < 2:
        raise ValueError("quantization.bitwidth must be an integer >= 2")
    for key in ("enabled", "learn_scale", "symmetric"):
        _require_boolean(quant, key)
    _require_positive(quant, "init_scale")
    if quant.get("input_clip", "tanh") not in {"tanh", "clamp", "none", None}:
        raise ValueError(f"Unsupported quantization.input_clip: {quant.get('input_clip')}")

    train = config.get("train", {})
    for key in ("epochs", "batch_size", "eval_batch_size", "val_freq"):
        _require_integer(train, key)
    _require_integer(train, "num_workers", allow_zero=True)
    for key in ("lr",):
        _require_positive(train, key)
    for key in ("weight_decay", "eta_min"):
        _require_positive(train, key, allow_zero=True)
    _require_positive(train, "grad_clip")
    if str(train.get("loss", "mse")).lower() not in {"mse", "standardized_mse", "nmse", "log_nmse", "mixed"}:
        raise ValueError(f"Unsupported train.loss: {train.get('loss')}")

    experiment = config.get("experiment", {})
    _require_integer(experiment, "seed", allow_zero=True)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    validate_config(data)
    return data


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def get_by_path(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
