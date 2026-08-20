from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierMLPBackbone(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_frequencies: int = 8,
    ):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        encoded_coord_dim = 2 + 2 * 2 * self.num_frequencies
        input_dim = encoded_coord_dim + latent_dim
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def encode_coords(self, coords: torch.Tensor) -> torch.Tensor:
        feats = [coords]
        for idx in range(self.num_frequencies):
            freq = 2.0**idx * math.pi
            feats.append(torch.sin(freq * coords))
            feats.append(torch.cos(freq * coords))
        return torch.cat(feats, dim=-1)

    def forward(self, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_rep = z.unsqueeze(1).expand(-1, coords.shape[1], -1)
        return self.net(torch.cat([self.encode_coords(coords), z_rep], dim=-1))

