from __future__ import annotations

import math

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega0: float = 30.0):
        super().__init__()
        self.omega0 = float(omega0)
        self.linear = nn.Linear(in_features, out_features)
        bound = 1.0 / in_features
        nn.init.uniform_(self.linear.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(x))


class SIRENBackbone(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        layers = [SineLayer(2 + hidden_dim, hidden_dim)]
        for _ in range(num_layers):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega0=math.sqrt(hidden_dim)))
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_feat = self.z_proj(z).unsqueeze(1).expand(-1, coords.shape[1], -1)
        return self.net(torch.cat([coords, z_feat], dim=-1))

