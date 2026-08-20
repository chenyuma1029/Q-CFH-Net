from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def _normalize(x: np.ndarray, mode: str) -> np.ndarray:
    if mode in {"none", "identity", None}:
        return x
    if mode == "csinet_minus_0p5":
        return x - 0.5
    raise ValueError(f"Unsupported normalization: {mode}")


def _stats(x: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def load_cost2100_mat(
    path: str | Path,
    key: str = "HT",
    nc: int = 32,
    nt: int = 32,
    normalization: str = "csinet_minus_0p5",
) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    mat = sio.loadmat(path)
    if key not in mat:
        available = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Key {key!r} not found in {path}. Available keys: {available}")

    raw = np.asarray(mat[key], dtype=np.float32)
    raw_stats = _stats(raw)

    expected_flat = 2 * nc * nt
    if raw.ndim == 2 and raw.shape[1] == expected_flat:
        h = raw.reshape(raw.shape[0], 2, nc, nt)
    elif raw.ndim == 4 and raw.shape[1:] == (2, nc, nt):
        h = raw
    else:
        raise ValueError(
            f"Expected [N,{expected_flat}] or [N,2,{nc},{nt}], got {raw.shape}"
        )

    h = _normalize(h, normalization).astype(np.float32, copy=False)
    metadata = {
        "path": str(path),
        "key": key,
        "shape": list(h.shape),
        "nc": nc,
        "nt": nt,
        "normalization": normalization,
        "raw_stats": raw_stats,
        "normalized_stats": _stats(h),
    }
    return h, metadata


def make_dummy_csi(
    num_samples: int,
    nc: int = 32,
    nt: int = 32,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.normal(0.0, 0.25, size=(num_samples, 2, nc, nt)).astype(np.float32)
    return h


def load_npz(path: str | Path, split_key: str) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if split_key not in data:
            available = list(data.keys())
            raise KeyError(f"Key {split_key!r} not found in {path}. Available keys: {available}")
        h = np.array(data[split_key], dtype=np.float32, copy=True)
        source_metadata: Any = None
        if "metadata" in data:
            raw_meta = data["metadata"]
            try:
                source_metadata = json.loads(str(raw_meta.item()))
            except (json.JSONDecodeError, TypeError, ValueError):
                source_metadata = str(raw_meta)
    metadata: dict[str, Any] = {"path": str(path), "split_key": split_key, "shape": list(h.shape)}
    if source_metadata is not None:
        metadata["source_metadata"] = source_metadata
    return h, metadata


def load_cost2100_hf_all(
    path: str | Path,
    nt: int = 32,
    n_eval: int = 125,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    mat = sio.loadmat(path)
    if "HF_all" not in mat:
        available = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Key 'HF_all' not found in {path}. Available keys: {available}")

    raw = np.asarray(mat["HF_all"])
    if raw.ndim == 2 and raw.shape[1] == nt * n_eval:
        h = raw.reshape(raw.shape[0], nt, n_eval)
    elif raw.ndim == 3 and raw.shape[1:] == (nt, n_eval):
        h = raw
    else:
        raise ValueError(f"Expected [N,{nt * n_eval}] or [N,{nt},{n_eval}], got {raw.shape}")

    h = np.ascontiguousarray(h.astype(np.complex64, copy=False))
    metadata = {
        "path": str(path),
        "key": "HF_all",
        "shape": list(h.shape),
        "nt": nt,
        "n_eval": n_eval,
        "dtype": str(h.dtype),
    }
    return h, metadata


class Cost2100Dataset(Dataset):
    def __init__(self, h: np.ndarray, return_index: bool = False, scale: np.ndarray | None = None):
        if h.ndim != 4 or h.shape[1] != 2:
            raise ValueError(f"Expected [N,2,Nc,Nt], got {h.shape}")
        self.h = torch.from_numpy(h.astype(np.float32, copy=False))
        self.return_index = bool(return_index)
        self.scale = None
        if scale is not None:
            if scale.shape[0] != h.shape[0]:
                raise ValueError(f"Expected scale first dimension {h.shape[0]}, got {scale.shape}")
            self.scale = torch.from_numpy(scale.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.h.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {"h": self.h[idx]}
        if self.return_index:
            item["sample_id"] = torch.tensor(idx, dtype=torch.long)
        if self.scale is not None:
            item["target_scale"] = self.scale[idx]
        return item
