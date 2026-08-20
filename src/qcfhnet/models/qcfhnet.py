from __future__ import annotations

import torch
import torch.nn as nn

from qcfhnet.quantization import UniformSTEQuantizer

from .cfhnet import CFHDecoder
from .encoder import EncoderMLP


class QCFHNet(nn.Module):
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
        upsampler: str = "interp_conv",
        split_real_imag: bool = False,
        quantization: dict | None = None,
    ):
        super().__init__()
        self.nc = int(nc)
        self.nt = int(nt)
        self.latent_dim = int(latent_dim)
        self.split_real_imag = bool(split_real_imag)
        quantization = quantization or {}

        def make_quantizer() -> UniformSTEQuantizer:
            return UniformSTEQuantizer(
                bitwidth=int(quantization.get("bitwidth", 4)),
                enabled=bool(quantization.get("enabled", True)),
                learn_scale=bool(quantization.get("learn_scale", True)),
                init_scale=float(quantization.get("init_scale", 1.0)),
                symmetric=bool(quantization.get("symmetric", True)),
                input_clip=quantization.get("input_clip", "tanh"),
            )

        if self.split_real_imag:
            self.encoder_real = EncoderMLP(nc, nt, latent_dim, hidden_dim, input_channels=1)
            self.encoder_imag = EncoderMLP(nc, nt, latent_dim, hidden_dim, input_channels=1)
            self.quantizer_real = make_quantizer()
            self.quantizer_imag = make_quantizer()
            self.decoder_real = CFHDecoder(
                latent_dim=latent_dim,
                nc=nc,
                nt=nt,
                coarse_grid=coarse_grid,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                seed_channels=seed_channels,
                out_channels=1,
                backbone=backbone,
                use_hyperfilm=use_hyperfilm,
                upsampler=upsampler,
            )
            self.decoder_imag = CFHDecoder(
                latent_dim=latent_dim,
                nc=nc,
                nt=nt,
                coarse_grid=coarse_grid,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                seed_channels=seed_channels,
                out_channels=1,
                backbone=backbone,
                use_hyperfilm=use_hyperfilm,
                upsampler=upsampler,
            )
        else:
            self.encoder = EncoderMLP(nc, nt, latent_dim, hidden_dim, input_channels=2)
            self.quantizer = make_quantizer()
            self.decoder = CFHDecoder(
                latent_dim=latent_dim,
                nc=nc,
                nt=nt,
                coarse_grid=coarse_grid,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                seed_channels=seed_channels,
                out_channels=2,
                backbone=backbone,
                use_hyperfilm=use_hyperfilm,
                upsampler=upsampler,
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

    @property
    def quantized_feedback_bits(self) -> int:
        if self.split_real_imag:
            return int(
                self.quantizer_real.feedback_bits(self.latent_dim)
                + self.quantizer_imag.feedback_bits(self.latent_dim)
            )
        return int(self.quantizer.feedback_bits(self.latent_dim))

    def encode(self, h: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.split_real_imag:
            return self.encoder_real(h[:, 0:1]), self.encoder_imag(h[:, 1:2])
        return self.encoder(h)

    def quantize(self, z: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.split_real_imag:
            z_real, z_imag = z
            return self.quantizer_real(z_real), self.quantizer_imag(z_imag)
        return self.quantizer(z)

    def decode(self, z_tilde: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.split_real_imag:
            z_real, z_imag = z_tilde
            out_real = self.decoder_real(z_real)
            out_imag = self.decoder_imag(z_imag)
            return torch.cat([out_real, out_imag], dim=1)
        return self.decoder(z_tilde)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(h)
        z_tilde = self.quantize(z)
        h_hat = self.decode(z_tilde)
        if self.split_real_imag:
            z_real, z_imag = z
            z_tilde_real, z_tilde_imag = z_tilde
            return {
                "h_hat": h_hat,
                "z": torch.cat([z_real, z_imag], dim=1),
                "z_tilde": torch.cat([z_tilde_real, z_tilde_imag], dim=1),
                "z_real": z_real,
                "z_imag": z_imag,
                "z_tilde_real": z_tilde_real,
                "z_tilde_imag": z_tilde_imag,
            }
        return {"h_hat": h_hat, "z": z, "z_tilde": z_tilde}
