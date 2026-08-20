from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import EncoderMLP
from .fourier_mlp import FourierMLPBackbone
from .siren import SIRENBackbone
from .upsampler import build_upsampler
from .wire import WIREBackbone


def make_coord_grid(grid_h: int, grid_w: int, device: torch.device | None = None) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, steps=grid_h, device=device)
    x = torch.linspace(-1.0, 1.0, steps=grid_w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)


class CFHDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        nc: int,
        nt: int,
        coarse_grid: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 3,
        seed_channels: int = 32,
        out_channels: int = 2,
        backbone: str = "wire",
        use_hyperfilm: bool = True,
        upsampler: str = "interp_conv",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.nc = int(nc)
        self.nt = int(nt)
        self.coarse_grid = int(coarse_grid)
        self.seed_channels = int(seed_channels)
        self.backbone_name = backbone
        self.upsampler_name = upsampler

        if backbone == "wire":
            self.backbone = WIREBackbone(latent_dim, hidden_dim, num_layers, use_hyperfilm)
        elif backbone == "siren":
            self.backbone = SIRENBackbone(latent_dim, hidden_dim, num_layers)
        elif backbone in {"fourier_mlp", "relu_mlp"}:
            self.backbone = FourierMLPBackbone(latent_dim, hidden_dim, num_layers)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.feature_proj = nn.Linear(self.backbone.out_dim, seed_channels)
        self.upsampler = build_upsampler(
            upsampler,
            in_channels=seed_channels,
            out_channels=out_channels,
            nc=nc,
            nt=nt,
            coarse_grid=coarse_grid,
        )
        self.upsampler_fallback_interpolate = bool(getattr(self.upsampler, "fallback_interpolate", False))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        coords = make_coord_grid(self.coarse_grid, self.coarse_grid, device=z.device)
        coords = coords.unsqueeze(0).expand(batch, -1, -1)
        feats = self.backbone(coords, z)
        seed = self.feature_proj(feats)
        seed_map = seed.permute(0, 2, 1).contiguous().view(
            batch, self.seed_channels, self.coarse_grid, self.coarse_grid
        )
        return self.upsampler(seed_map, (self.nc, self.nt))


class CFHNet(nn.Module):
    def __init__(
        self,
        nc: int = 32,
        nt: int = 32,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 3,
        coarse_grid: int = 2,
        seed_channels: int = 32,
        backbone: str = "wire",
        use_hyperfilm: bool = True,
        split_real_imag: bool = False,
        upsampler: str = "interp_conv",
    ):
        super().__init__()
        self.nc = int(nc)
        self.nt = int(nt)
        self.latent_dim = int(latent_dim)
        self.split_real_imag = bool(split_real_imag)

        if self.split_real_imag:
            self.encoder_real = EncoderMLP(nc, nt, latent_dim, hidden_dim, input_channels=1)
            self.encoder_imag = EncoderMLP(nc, nt, latent_dim, hidden_dim, input_channels=1)
            self.decoder_real = CFHDecoder(
                latent_dim, nc, nt, coarse_grid, hidden_dim, num_layers, seed_channels, 1, backbone, use_hyperfilm, upsampler
            )
            self.decoder_imag = CFHDecoder(
                latent_dim, nc, nt, coarse_grid, hidden_dim, num_layers, seed_channels, 1, backbone, use_hyperfilm, upsampler
            )
        else:
            self.encoder = EncoderMLP(nc, nt, latent_dim, hidden_dim, input_channels=2)
            self.decoder = CFHDecoder(
                latent_dim, nc, nt, coarse_grid, hidden_dim, num_layers, seed_channels, 2, backbone, use_hyperfilm, upsampler
            )

        self.upsampler_name = upsampler
        if self.split_real_imag:
            self.upsampler_fallback_interpolate = bool(
                self.decoder_real.upsampler_fallback_interpolate or self.decoder_imag.upsampler_fallback_interpolate
            )
        else:
            self.upsampler_fallback_interpolate = bool(self.decoder.upsampler_fallback_interpolate)

    @property
    def feedback_scalars(self) -> int:
        return self.latent_dim * 2 if self.split_real_imag else self.latent_dim

    def encode(self, h: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.split_real_imag:
            z_real = self.encoder_real(h[:, 0:1])
            z_imag = self.encoder_imag(h[:, 1:2])
            return z_real, z_imag
        return self.encoder(h)

    def decode(self, z: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.split_real_imag:
            z_real, z_imag = z
            out_real = self.decoder_real(z_real)
            out_imag = self.decoder_imag(z_imag)
            return torch.cat([out_real, out_imag], dim=1)
        return self.decoder(z)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(h)
        h_hat = self.decode(z)
        if isinstance(z, tuple):
            z_out = torch.cat(z, dim=1)
        else:
            z_out = z
        return {"h_hat": h_hat, "z": z_out, "z_tilde": z_out}
