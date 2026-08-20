from __future__ import annotations

import torch


def _as_complex_per_subcarrier(h: torch.Tensor) -> torch.Tensor:
    if h.ndim != 4 or h.shape[1] != 2:
        raise ValueError(f"Expected [B,2,Nc,Nt], got {tuple(h.shape)}")
    return torch.complex(h[:, 0], h[:, 1])


def normalized_beamforming_gain(target: torch.Tensor, pred: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    h = _as_complex_per_subcarrier(target)
    hhat = _as_complex_per_subcarrier(pred)
    numerator = torch.abs(torch.sum(torch.conj(h) * hhat, dim=-1)) ** 2
    denom = torch.sum(torch.abs(h) ** 2, dim=-1) * torch.sum(torch.abs(hhat) ** 2, dim=-1)
    rho = numerator / denom.clamp_min(eps)
    return rho.mean(dim=1)


def achievable_rate_ratio(
    target: torch.Tensor,
    pred: torch.Tensor,
    snr_db: float = 10.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    h = _as_complex_per_subcarrier(target)
    hhat = _as_complex_per_subcarrier(pred)
    snr = 10.0 ** (snr_db / 10.0)

    v = hhat / torch.linalg.vector_norm(hhat, dim=-1, keepdim=True).clamp_min(eps)
    v_perfect = h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp_min(eps)

    eff = torch.abs(torch.sum(torch.conj(h) * v, dim=-1)) ** 2
    eff_perfect = torch.abs(torch.sum(torch.conj(h) * v_perfect, dim=-1)) ** 2
    rate = torch.log2(1.0 + snr * eff)
    rate_perfect = torch.log2(1.0 + snr * eff_perfect).clamp_min(eps)
    return (rate / rate_perfect).mean(dim=1)

