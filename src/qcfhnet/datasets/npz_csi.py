from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_npz_csi(
    path: str | Path,
    split_key: str,
    nc: int | None = None,
    nt: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if split_key not in data:
        available = list(data.keys())
        raise KeyError(f"Key {split_key!r} not found in {path}. Available keys: {available}")

    h = np.asarray(data[split_key], dtype=np.float32)
    if h.ndim != 4 or h.shape[1] != 2:
        raise ValueError(f"Expected [N,2,Nc,Nt] for {split_key!r}, got {h.shape}")
    if nc is not None and h.shape[2] != int(nc):
        raise ValueError(f"Expected Nc={nc}, got {h.shape[2]} for {split_key!r}")
    if nt is not None and h.shape[3] != int(nt):
        raise ValueError(f"Expected Nt={nt}, got {h.shape[3]} for {split_key!r}")

    metadata: dict[str, Any] = {
        "path": str(path),
        "split_key": split_key,
        "shape": list(h.shape),
        "nc": int(h.shape[2]),
        "nt": int(h.shape[3]),
    }
    if "metadata" in data:
        raw_meta = data["metadata"]
        try:
            metadata["source_metadata"] = json.loads(str(raw_meta.item()))
        except Exception:
            metadata["source_metadata"] = str(raw_meta)
    return np.ascontiguousarray(h), metadata
