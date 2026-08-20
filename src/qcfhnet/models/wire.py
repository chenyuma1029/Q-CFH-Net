from __future__ import annotations

import torch
import torch.nn as nn

from .hyperfilm import HyperFiLM


class ComplexGaborLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega0: float = 10.0, sigma0: float = 10.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega = nn.Parameter(torch.rand(out_features) * omega0)
        self.scale = nn.Parameter(torch.rand(out_features) * sigma0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin = self.linear(x)
        freq = torch.exp(1j * self.omega * lin)
        envelope = torch.exp(-((self.scale * lin) ** 2))
        return freq * envelope


class WIREBackbone(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_hyperfilm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.use_hyperfilm = bool(use_hyperfilm)
        self.first = ComplexGaborLayer(2, hidden_dim)
        self.layers = nn.ModuleList([ComplexGaborLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.hyper = HyperFiLM(latent_dim, num_layers, hidden_dim) if use_hyperfilm else None
        self.out_dim = hidden_dim * 2

    def forward(self, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gammas = betas = None
        if self.hyper is not None:
            gammas, betas = self.hyper(z)

        x = self.first(coords)
        for idx, layer in enumerate(self.layers):
            if gammas is not None and betas is not None:
                g = gammas[:, idx, :].unsqueeze(1)
                b = betas[:, idx, :].unsqueeze(1)
                x = x * g + b
            x = layer(x.real)
        return torch.cat([x.real, x.imag], dim=-1)

