from __future__ import annotations

import torch


def _flatten_complex(h: torch.Tensor) -> torch.Tensor:
    if h.ndim < 3 or h.shape[1] != 2:
        raise ValueError(f"Expected [B,2,...], got {tuple(h.shape)}")
    real = h[:, 0].flatten(1)
    imag = h[:, 1].flatten(1)
    return torch.complex(real, imag)


def nmse_db(target: torch.Tensor, pred: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    target_c = _flatten_complex(target)
    pred_c = _flatten_complex(pred)
    error = torch.sum(torch.abs(target_c - pred_c) ** 2, dim=1)
    signal = torch.sum(torch.abs(target_c) ** 2, dim=1).clamp_min(eps)
    return 10.0 * torch.log10((error / signal).clamp_min(eps))


def nmse_global_db(target: torch.Tensor, pred: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    target_c = _flatten_complex(target)
    pred_c = _flatten_complex(pred)
    error = torch.sum(torch.abs(target_c - pred_c) ** 2)
    signal = torch.sum(torch.abs(target_c) ** 2).clamp_min(eps)
    return 10.0 * torch.log10((error / signal).clamp_min(eps))


def cosine_similarity(target: torch.Tensor, pred: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    target_c = _flatten_complex(target)
    pred_c = _flatten_complex(pred)
    numerator = torch.abs(torch.sum(torch.conj(target_c) * pred_c, dim=1))
    denominator = torch.linalg.vector_norm(target_c, dim=1) * torch.linalg.vector_norm(pred_c, dim=1)
    return numerator / denominator.clamp_min(eps)

