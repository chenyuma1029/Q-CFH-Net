from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def count_named_parameter_groups(model: nn.Module) -> dict[str, int]:
    groups = {"ue_encoder": 0, "quantizer": 0, "bs_decoder": 0, "other": 0}
    for name, param in model.named_parameters():
        n = int(param.numel())
        if "quantizer" in name:
            groups["quantizer"] += n
        elif "encoder" in name:
            groups["ue_encoder"] += n
        elif "decoder" in name or "backbone" in name or "upsampler" in name or "feature_proj" in name:
            groups["bs_decoder"] += n
        else:
            groups["other"] += n
    groups["total"] = sum(groups.values())
    return groups


@dataclass
class FlopEstimate:
    flops: int
    unsupported_modules: dict[str, int]


def _conv_flops(module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> int:
    x = inputs[0]
    batch = int(x.shape[0])
    out_h = int(output.shape[-2])
    out_w = int(output.shape[-1])
    kernel_h, kernel_w = module.kernel_size
    in_per_group = int(module.in_channels // module.groups)
    return int(batch * module.out_channels * out_h * out_w * in_per_group * kernel_h * kernel_w * 2)


def _linear_flops(module: nn.Linear, inputs: tuple[Any, ...], output: torch.Tensor) -> int:
    return int(output.numel() * module.in_features * 2)


def _simple_elementwise_flops(output: torch.Tensor, multiplier: int = 1) -> int:
    return int(output.numel() * multiplier)


def estimate_forward_flops(fn: Callable[[], Any], module: nn.Module) -> FlopEstimate:
    total = 0
    unsupported: dict[str, int] = {}
    handles = []

    def hook(mod: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal total
        if isinstance(output, (tuple, list)):
            tensor_outputs = [v for v in output if isinstance(v, torch.Tensor)]
            out = tensor_outputs[0] if tensor_outputs else None
        else:
            out = output if isinstance(output, torch.Tensor) else None
        if out is None:
            return
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            total += _conv_flops(mod, inputs, out)
        elif isinstance(mod, nn.Linear):
            total += _linear_flops(mod, inputs, out)
        elif isinstance(mod, (nn.BatchNorm2d, nn.LayerNorm)):
            total += _simple_elementwise_flops(out, multiplier=2)
        elif isinstance(mod, (nn.ReLU, nn.PReLU, nn.Sigmoid, nn.Tanh, nn.Identity, nn.Dropout)):
            total += _simple_elementwise_flops(out)
        elif len(list(mod.children())) == 0:
            key = mod.__class__.__name__
            unsupported[key] = unsupported.get(key, 0) + 1

    for child in module.modules():
        handles.append(child.register_forward_hook(hook))
    try:
        fn()
    finally:
        for handle in handles:
            handle.remove()
    return FlopEstimate(flops=int(total), unsupported_modules=unsupported)


def detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(detach_tree(v) for v in value)
    if isinstance(value, list):
        return [detach_tree(v) for v in value]
    return value


def tensor_tree_numel(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, (tuple, list)):
        return sum(tensor_tree_numel(v) for v in value)
    return 0
