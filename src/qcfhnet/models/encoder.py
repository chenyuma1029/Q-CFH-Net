from __future__ import annotations

import torch
import torch.nn as nn


class EncoderMLP(nn.Module):
    def __init__(
        self,
        nc: int,
        nt: int,
        latent_dim: int,
        hidden_dim: int = 256,
        input_channels: int = 2,
    ):
        super().__init__()
        self.nc = int(nc)
        self.nt = int(nt)
        self.latent_dim = int(latent_dim)
        self.input_channels = int(input_channels)
        input_dim = self.input_channels * self.nc * self.nt
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h.flatten(1))

