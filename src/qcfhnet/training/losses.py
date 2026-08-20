import torch
import torch.nn.functional as F


def reconstruction_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def reconstruction_nmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    error = (pred - target).flatten(start_dim=1)
    signal = target.flatten(start_dim=1)
    error_power = error.pow(2).sum(dim=1)
    signal_power = signal.pow(2).sum(dim=1).clamp_min(eps)
    return (error_power / signal_power).mean()


def mixed_mse_nmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_nmse: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    return reconstruction_mse(pred, target) + float(lambda_nmse) * reconstruction_nmse_loss(pred, target, eps=eps)


def log_nmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    nmse = reconstruction_nmse_loss(pred, target, eps=eps)
    return torch.log(nmse.clamp_min(eps))


def build_reconstruction_loss(name: str = "mse", lambda_nmse: float = 0.1):
    loss_name = name.lower()
    if loss_name in {"mse", "standardized_mse"}:
        return reconstruction_mse
    if loss_name == "nmse":
        return reconstruction_nmse_loss
    if loss_name == "log_nmse":
        return log_nmse_loss
    if loss_name == "mixed":
        return lambda pred, target: mixed_mse_nmse_loss(pred, target, lambda_nmse=lambda_nmse)
    raise ValueError(f"Unsupported reconstruction loss: {name}")
