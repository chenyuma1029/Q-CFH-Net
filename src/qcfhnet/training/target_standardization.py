from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TargetStandardizer:
    mode: str = "none"
    fixed_scale: float = 1.0
    dataset_std: float | None = None
    eps: float = 1e-12

    @classmethod
    def from_train(
        cls,
        mode: str = "none",
        train_array: np.ndarray | None = None,
        fixed_scale: float = 1.0,
        eps: float = 1e-12,
    ) -> "TargetStandardizer":
        mode = str(mode or "none").lower()
        dataset_std = None
        if mode == "dataset_std":
            if train_array is None:
                raise ValueError("dataset_std target standardization requires train_array")
            dataset_std = float(np.std(train_array))
        return cls(mode=mode, fixed_scale=float(fixed_scale), dataset_std=dataset_std, eps=float(eps))

    def scale_np(self, x: np.ndarray) -> np.ndarray:
        if self.mode in {"none", "identity"}:
            return np.ones((x.shape[0], 1, 1, 1), dtype=np.float32)
        if self.mode == "sample_rms":
            scale = np.sqrt(np.mean(np.square(x), axis=(1, 2, 3), keepdims=True))
            return np.maximum(scale, self.eps).astype(np.float32)
        if self.mode == "dataset_std":
            return np.full((x.shape[0], 1, 1, 1), max(float(self.dataset_std or 0.0), self.eps), dtype=np.float32)
        if self.mode == "fixed_scale":
            return np.full((x.shape[0], 1, 1, 1), max(float(self.fixed_scale), self.eps), dtype=np.float32)
        raise ValueError(f"Unsupported target_standardization mode: {self.mode}")

    def transform_np(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = self.scale_np(x)
        return (x / scale).astype(np.float32, copy=False), scale.astype(np.float32, copy=False)

    def inverse_torch(self, x: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
        if self.mode in {"none", "identity"} or scale is None:
            return x
        return x * scale.to(device=x.device, dtype=x.dtype)
