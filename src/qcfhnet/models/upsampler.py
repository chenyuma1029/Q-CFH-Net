from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpConvUpsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        hidden_channels: int = 64,
        refinement_channels: int = 32,
    ):
        super().__init__()
        self.seed_refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        self.full_refine = nn.Sequential(
            nn.Conv2d(hidden_channels, refinement_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(refinement_channels),
            nn.ReLU(),
            nn.Conv2d(refinement_channels, refinement_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(refinement_channels),
            nn.ReLU(),
            nn.Conv2d(refinement_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, seed_map: torch.Tensor, output_shape: tuple[int, int]) -> torch.Tensor:
        x = self.seed_refine(seed_map)
        x = F.interpolate(x, size=output_shape, mode="bilinear", align_corners=False)
        return self.full_refine(x)


class DirectInterpolationUpsampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 2):
        super().__init__()
        if int(in_channels) != int(out_channels):
            raise ValueError(
                "direct_interp removes all CNN refinement, so seed_channels must match out_channels "
                f"(got seed_channels={in_channels}, out_channels={out_channels})."
            )
        self.fallback_interpolate = True

    def forward(self, seed_map: torch.Tensor, output_shape: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(seed_map, size=output_shape, mode="bilinear", align_corners=False)


class DeconvCascadeUpsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        nc: int = 32,
        nt: int = 32,
        coarse_grid: int = 2,
        hidden_channels: int = 64,
        refinement_channels: int = 32,
    ):
        super().__init__()
        self.nc = int(nc)
        self.nt = int(nt)
        self.coarse_grid = int(coarse_grid)
        self.fallback_interpolate = False

        blocks = []
        cur_h = self.coarse_grid
        cur_w = self.coarse_grid
        cur_channels = int(in_channels)
        target_hidden = int(hidden_channels)

        while cur_h < self.nc and cur_w < self.nt and cur_h * 2 <= self.nc and cur_w * 2 <= self.nt:
            blocks.extend(
                [
                    nn.ConvTranspose2d(cur_channels, target_hidden, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(target_hidden),
                    nn.ReLU(),
                ]
            )
            cur_channels = target_hidden
            cur_h *= 2
            cur_w *= 2

        if cur_h != self.nc or cur_w != self.nt:
            self.fallback_interpolate = True

        blocks.extend(
            [
                nn.Conv2d(cur_channels, refinement_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(refinement_channels),
                nn.ReLU(),
                nn.Conv2d(refinement_channels, out_channels, kernel_size=3, padding=1),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, seed_map: torch.Tensor, output_shape: tuple[int, int]) -> torch.Tensor:
        x = self.net(seed_map)
        if x.shape[-2:] != output_shape:
            x = F.interpolate(x, size=output_shape, mode="bilinear", align_corners=False)
        return x


class RectangularDeconvCascadeUpsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        nc: int = 32,
        nt: int = 32,
        coarse_grid: int = 2,
        hidden_channels: int = 64,
        refinement_channels: int = 32,
    ):
        super().__init__()
        self.nc = int(nc)
        self.nt = int(nt)
        self.coarse_grid = int(coarse_grid)
        self.fallback_interpolate = False

        blocks = []
        cur_h = self.coarse_grid
        cur_w = self.coarse_grid
        cur_channels = int(in_channels)
        target_hidden = int(hidden_channels)

        while cur_h < self.nc or cur_w < self.nt:
            stride_h = 2 if cur_h * 2 <= self.nc else 1
            stride_w = 2 if cur_w * 2 <= self.nt else 1
            if stride_h == 1 and stride_w == 1:
                break
            blocks.extend(
                [
                    nn.ConvTranspose2d(
                        cur_channels,
                        target_hidden,
                        kernel_size=(4 if stride_h == 2 else 3, 4 if stride_w == 2 else 3),
                        stride=(stride_h, stride_w),
                        padding=1,
                    ),
                    nn.BatchNorm2d(target_hidden),
                    nn.ReLU(),
                ]
            )
            cur_channels = target_hidden
            cur_h *= stride_h
            cur_w *= stride_w

        if cur_h != self.nc or cur_w != self.nt:
            self.fallback_interpolate = True

        blocks.extend(
            [
                nn.Conv2d(cur_channels, refinement_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(refinement_channels),
                nn.ReLU(),
                nn.Conv2d(refinement_channels, out_channels, kernel_size=3, padding=1),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, seed_map: torch.Tensor, output_shape: tuple[int, int]) -> torch.Tensor:
        x = self.net(seed_map)
        if x.shape[-2:] != output_shape:
            x = F.interpolate(x, size=output_shape, mode="bilinear", align_corners=False)
        return x


def build_upsampler(
    mode: str,
    in_channels: int,
    out_channels: int,
    nc: int,
    nt: int,
    coarse_grid: int,
) -> nn.Module:
    mode = mode.lower()
    if mode in {"interp_conv", "cnn", "bilinear_conv"}:
        return InterpConvUpsampler(in_channels, out_channels=out_channels)
    if mode in {"direct_interp", "bilinear_direct", "no_cnn_direct_interp"}:
        return DirectInterpolationUpsampler(in_channels, out_channels=out_channels)
    if mode == "deconv_cascade":
        return DeconvCascadeUpsampler(
            in_channels,
            out_channels=out_channels,
            nc=nc,
            nt=nt,
            coarse_grid=coarse_grid,
        )
    if mode in {"deconv_rect_cascade", "rect_deconv_cascade"}:
        return RectangularDeconvCascadeUpsampler(
            in_channels,
            out_channels=out_channels,
            nc=nc,
            nt=nt,
            coarse_grid=coarse_grid,
        )
    raise ValueError(f"Unsupported upsampler mode: {mode}")


CNNUpsampler = InterpConvUpsampler
