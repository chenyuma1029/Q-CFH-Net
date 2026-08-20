from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _model_signature(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    data = config.get("data", {})
    raw_name = str(model.get("name", "qcfhnet")).lower()
    if raw_name in {"qcfhnet", "q-cfh-net", "qcfh"}:
        name = "qcfhnet"
    elif raw_name in {"cfhnet", "cfh", "original_cfhnet"}:
        name = "cfhnet"
    elif raw_name in {"csi_inr_ff", "csi-inr-ff", "csiinrff"}:
        name = "csi_inr_ff"
    else:
        name = raw_name
    return {
        "name": name,
        "nc": int(model.get("nc", data.get("nc", 32))),
        "nt": int(model.get("nt", data.get("nt", 32))),
        "latent_dim": int(model.get("latent_dim", 16)),
        "hidden_dim": int(model.get("hidden_dim", 256)),
        "num_layers": int(model.get("num_layers", model.get("num_wire_layers", 3))),
        "coarse_grid": int(model.get("coarse_grid", 2)),
        "seed_channels": int(model.get("seed_channels", model.get("upsample_feat", 32))),
        "backbone": str(model.get("backbone", "wire")).lower(),
        "use_hyperfilm": bool(model.get("use_hyperfilm", True)),
        "upsampler": str(model.get("upsampler", "interp_conv")).lower(),
        "split_real_imag": bool(model.get("split_real_imag", False)),
    }


def _quantization_signature(config: dict[str, Any]) -> dict[str, Any]:
    quant = config.get("quantization", {})
    return {
        "enabled": bool(quant.get("enabled", True)),
        "mode": str(quant.get("mode", "qat")).lower(),
        "bitwidth": int(quant.get("bitwidth", 4)),
        "learn_scale": bool(quant.get("learn_scale", True)),
        "init_scale": float(quant.get("init_scale", 1.0)),
        "symmetric": bool(quant.get("symmetric", True)),
        "input_clip": quant.get("input_clip", "tanh"),
    }


def _data_signature(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    return {
        "name": str(data.get("name", "dummy")).lower(),
        "scenario": str(data.get("scenario", "")).lower(),
        "nc": int(data.get("nc", config.get("model", {}).get("nc", 32))),
        "nt": int(data.get("nt", config.get("model", {}).get("nt", 32))),
        "normalization": data.get("normalization"),
    }


def validate_checkpoint_compatibility(
    runtime_config: dict[str, Any],
    checkpoint_payload: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_config = checkpoint_payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Checkpoint does not contain a valid config mapping")

    mismatches = []
    for section, signature_fn in (
        ("model", _model_signature),
        ("quantization", _quantization_signature),
        ("data", _data_signature),
    ):
        runtime_signature = signature_fn(runtime_config)
        checkpoint_signature = signature_fn(checkpoint_config)
        for key, runtime_value in runtime_signature.items():
            checkpoint_value = checkpoint_signature[key]
            if runtime_value != checkpoint_value:
                mismatches.append(
                    f"{section}.{key}: runtime={runtime_value!r}, checkpoint={checkpoint_value!r}"
                )
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Checkpoint/config mismatch: {details}")
    return checkpoint_config


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")
    if "model_state" not in payload or not isinstance(payload["model_state"], dict):
        raise ValueError(f"Checkpoint is missing a model_state mapping: {path}")
    return payload
