from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformSTEQuantizer(nn.Module):
    def __init__(
        self,
        bitwidth: int = 4,
        enabled: bool = True,
        learn_scale: bool = True,
        init_scale: float = 1.0,
        symmetric: bool = True,
        input_clip: str = "tanh",
    ):
        super().__init__()
        if bitwidth < 2:
            raise ValueError("bitwidth must be >= 2")
        if init_scale <= 0:
            raise ValueError("init_scale must be > 0")
        self.bitwidth = int(bitwidth)
        self.enabled = bool(enabled)
        self.learn_scale = bool(learn_scale)
        self.symmetric = bool(symmetric)
        self.input_clip = input_clip

        init = torch.log(torch.expm1(torch.tensor(float(init_scale))))
        if learn_scale:
            self.log_scale = nn.Parameter(init.clone().detach())
        else:
            self.register_buffer("log_scale", init.clone().detach())

    @property
    def qmax(self) -> int:
        if self.symmetric:
            return 2 ** (self.bitwidth - 1) - 1
        return 2**self.bitwidth - 1

    @property
    def qmin(self) -> int:
        if self.symmetric:
            return -self.qmax
        return 0

    def scale(self) -> torch.Tensor:
        return F.softplus(self.log_scale).clamp_min(1e-6)

    def _bounded(self, z: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if self.input_clip == "tanh":
            return torch.tanh(z / scale) * scale
        elif self.input_clip == "clamp":
            return torch.clamp(z, -scale, scale)
        elif self.input_clip in {"none", None}:
            return z
        raise ValueError(f"Unsupported input_clip: {self.input_clip}")

    def encode_indices(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent values to signed integer quantization indices."""
        if not self.enabled:
            raise RuntimeError("Cannot encode integer indices when quantization is disabled")
        scale = self.scale()
        z_bounded = self._bounded(z, scale)
        if self.symmetric:
            normalized = torch.clamp(z_bounded / scale, -1.0, 1.0) * self.qmax
        else:
            normalized = torch.clamp((z_bounded / scale + 1.0) * 0.5, 0.0, 1.0) * self.qmax
        return torch.round(normalized).to(torch.int64)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Map integer quantization indices back to latent values."""
        if not self.enabled:
            raise RuntimeError("Cannot decode integer indices when quantization is disabled")
        if torch.any(indices < self.qmin) or torch.any(indices > self.qmax):
            raise ValueError(f"Quantization indices must be in [{self.qmin}, {self.qmax}]")
        indices_float = indices.to(dtype=self.scale().dtype)
        if self.symmetric:
            return indices_float / self.qmax * self.scale()
        return (indices_float / self.qmax * 2.0 - 1.0) * self.scale()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return z

        scale = self.scale()
        z_bounded = self._bounded(z, scale)

        if self.symmetric:
            normalized = torch.clamp(z_bounded / scale, -1.0, 1.0) * self.qmax
            rounded = normalized + (torch.round(normalized) - normalized).detach()
            return rounded / self.qmax * scale

        normalized = torch.clamp((z_bounded / scale + 1.0) * 0.5, 0.0, 1.0) * self.qmax
        rounded = normalized + (torch.round(normalized) - normalized).detach()
        return (rounded / self.qmax * 2.0 - 1.0) * scale

    def feedback_bits(self, latent_dim: int) -> int:
        if not self.enabled:
            return int(32 * latent_dim)
        return int(self.bitwidth * latent_dim)
