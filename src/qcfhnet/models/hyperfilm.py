from __future__ import annotations

import torch
import torch.nn as nn


class HyperFiLM(nn.Module):
    def __init__(self, latent_dim: int, num_layers: int, hidden_dim: int):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        total_params = self.num_layers * 2 * self.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, total_params),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(z)
        batch = z.shape[0]
        params = params.view(batch, self.num_layers, 2, self.hidden_dim)
        gammas = params[:, :, 0, :] + 1.0
        betas = params[:, :, 1, :]
        return gammas, betas

