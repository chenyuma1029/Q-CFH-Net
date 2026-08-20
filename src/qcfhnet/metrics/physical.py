from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_to_complex_nt_nc(
    sparse: torch.Tensor,
    internal_layout: str = "b_2_nc_nt",
    already_centered: bool = True,
) -> torch.Tensor:
    if sparse.ndim != 4 or sparse.shape[1] != 2:
        raise ValueError(f"Expected [B,2,...], got {tuple(sparse.shape)}")

    real = sparse[:, 0]
    imag = sparse[:, 1]
    if not already_centered:
        real = real - 0.5
        imag = imag - 0.5

    if internal_layout == "b_2_nc_nt":
        real = real.transpose(1, 2)
        imag = imag.transpose(1, 2)
    elif internal_layout == "b_2_nt_nc":
        pass
    else:
        raise ValueError(f"Unsupported internal_layout: {internal_layout}")
    return torch.complex(real.contiguous(), imag.contiguous())


def sparse_to_frequency(
    sparse: torch.Tensor,
    internal_layout: str = "b_2_nc_nt",
    nc_fft: int = 257,
    n_eval: int = 125,
    already_centered: bool = True,
    antenna_domain: str = "spatial",
) -> torch.Tensor:
    sparse_complex = sparse_to_complex_nt_nc(
        sparse,
        internal_layout=internal_layout,
        already_centered=already_centered,
    )
    if antenna_domain == "spatial":
        pass
    elif antenna_domain == "angular_fftshift":
        sparse_complex = torch.fft.ifftshift(sparse_complex, dim=1)
        sparse_complex = torch.fft.ifft(sparse_complex, dim=1) * (float(sparse_complex.shape[1]) ** 0.5)
    else:
        raise ValueError(f"Unsupported antenna_domain: {antenna_domain}")
    nc_sparse = sparse_complex.shape[-1]
    if nc_sparse > nc_fft:
        raise ValueError(f"Sparse delay dimension {nc_sparse} exceeds nc_fft={nc_fft}")
    if n_eval > nc_fft:
        raise ValueError(f"n_eval={n_eval} exceeds nc_fft={nc_fft}")
    padded = F.pad(sparse_complex, (0, nc_fft - nc_sparse))
    return torch.fft.fft(padded, dim=-1)[..., :n_eval]


def physical_rho(freq_true: torch.Tensor, freq_pred: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if freq_true.shape != freq_pred.shape:
        raise ValueError(f"freq_true and freq_pred shape mismatch: {freq_true.shape} vs {freq_pred.shape}")
    numerator = torch.abs(torch.sum(torch.conj(freq_true) * freq_pred, dim=1))
    denom_true = torch.linalg.vector_norm(freq_true, dim=1)
    denom_pred = torch.linalg.vector_norm(freq_pred, dim=1)
    rho_per_subcarrier = numerator / (denom_true * denom_pred).clamp_min(eps)
    return rho_per_subcarrier.mean(dim=1)


def _achievable_rate_per_subcarrier(
    freq_true: torch.Tensor,
    beamformer_source: torch.Tensor,
    snr_db: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    if freq_true.shape != beamformer_source.shape:
        raise ValueError(
            f"freq_true and beamformer_source shape mismatch: {freq_true.shape} vs {beamformer_source.shape}"
        )
    snr = 10.0 ** (float(snr_db) / 10.0)
    beamformer = beamformer_source / torch.linalg.vector_norm(
        beamformer_source,
        dim=1,
        keepdim=True,
    ).clamp_min(eps)
    effective_gain = torch.abs(torch.sum(torch.conj(freq_true) * beamformer, dim=1)) ** 2
    return torch.log2(1.0 + snr * effective_gain)


def physical_achievable_rate(
    freq_true: torch.Tensor,
    beamformer_source: torch.Tensor,
    snr_db: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    return _achievable_rate_per_subcarrier(
        freq_true=freq_true,
        beamformer_source=beamformer_source,
        snr_db=snr_db,
        eps=eps,
    ).mean(dim=1)


def physical_rate_components(
    freq_true: torch.Tensor,
    freq_pred: torch.Tensor,
    snr_db: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if freq_true.shape != freq_pred.shape:
        raise ValueError(f"freq_true and freq_pred shape mismatch: {freq_true.shape} vs {freq_pred.shape}")
    rate_pred_per_subcarrier = _achievable_rate_per_subcarrier(freq_true, freq_pred, snr_db=snr_db, eps=eps)
    rate_oracle_per_subcarrier = _achievable_rate_per_subcarrier(freq_true, freq_true, snr_db=snr_db, eps=eps)
    rate_ratio_per_subcarrier = rate_pred_per_subcarrier / rate_oracle_per_subcarrier.clamp_min(eps)
    return (
        rate_pred_per_subcarrier.mean(dim=1),
        rate_oracle_per_subcarrier.mean(dim=1),
        rate_ratio_per_subcarrier.mean(dim=1),
    )


def physical_rate_ratio(
    freq_true: torch.Tensor,
    freq_pred: torch.Tensor,
    snr_db: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    _, _, ratio = physical_rate_components(freq_true, freq_pred, snr_db=snr_db, eps=eps)
    return ratio
