#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.insert(0, str(SRC))

from qcfhnet.models import build_model
from qcfhnet.utils.checkpoint import load_checkpoint
from qcfhnet.utils.complexity import (
    count_named_parameter_groups,
    detach_tree,
    estimate_forward_flops,
    tensor_tree_numel,
)
from qcfhnet.utils.config import load_config
from qcfhnet.utils.device import get_device


def summarize(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    p50 = ordered[int(0.50 * (len(ordered) - 1))]
    p90 = ordered[int(0.90 * (len(ordered) - 1))]
    return {"mean_ms": statistics.mean(times), "p50_ms": p50, "p90_ms": p90}


@torch.no_grad()
def benchmark(fn, device: torch.device, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return summarize(times)


@torch.no_grad()
def peak_memory_mb(fn, device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def try_fvcore_e2e(model: torch.nn.Module, x: torch.Tensor) -> dict:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    class HhatWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, inp):
            out = self.wrapped(inp)
            return out["h_hat"] if isinstance(out, dict) else out

    try:
        analysis = FlopCountAnalysis(HhatWrapper(model), x)
        return {
            "available": True,
            "e2e_flops": int(analysis.total()),
            "unsupported_ops": {str(k): int(v) for k, v in analysis.unsupported_ops().items()},
        }
    except Exception as exc:
        return {"available": True, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(args.device)
    model = build_model(config).to(device).eval()
    if args.checkpoint:
        payload = load_checkpoint(args.checkpoint, map_location=device)
        model.load_state_dict(payload["model_state"])
    nc = int(config.get("data", {}).get("nc", config.get("model", {}).get("nc", 32)))
    nt = int(config.get("data", {}).get("nt", config.get("model", {}).get("nt", 32)))
    x = torch.randn(1, 2, nc, nt, device=device)

    z_cache = detach_tree(model.encode(x))
    if hasattr(model, "quantize"):
        z_quantized_cache = detach_tree(model.quantize(z_cache))
        quantizer_fn = lambda: model.quantize(z_cache)
    else:
        z_quantized_cache = z_cache
        quantizer_fn = lambda: z_cache

    e2e_fn = lambda: model(x)
    encoder_fn = lambda: model.encode(x)
    decoder_fn = lambda: model.decode(z_quantized_cache)
    output = {
        "device": str(device),
        "batch_size": 1,
        "input_shape": [1, 2, nc, nt],
        "params": count_named_parameter_groups(model),
        "flops_backend": {
            "fvcore": try_fvcore_e2e(model, x),
            "fallback": "forward-hook estimate for Linear/Conv/ConvTranspose/Norm/activation modules",
        },
        "flops": {
            "ue_encoder": estimate_forward_flops(encoder_fn, model).flops,
            "quantizer": int(tensor_tree_numel(z_cache) * 6 if hasattr(model, "quantize") else 0),
            "bs_decoder": estimate_forward_flops(decoder_fn, model).flops,
            "e2e": estimate_forward_flops(e2e_fn, model).flops,
        },
        "unsupported_ops": {
            "ue_encoder": estimate_forward_flops(encoder_fn, model).unsupported_modules,
            "bs_decoder": estimate_forward_flops(decoder_fn, model).unsupported_modules,
            "e2e": estimate_forward_flops(e2e_fn, model).unsupported_modules,
        },
        "latency_ms": {
            "ue_encoder": benchmark(encoder_fn, device, args.warmup, args.repeat),
            "quantizer": benchmark(quantizer_fn, device, args.warmup, args.repeat),
            "bs_decoder": benchmark(decoder_fn, device, args.warmup, args.repeat),
            "e2e": benchmark(e2e_fn, device, args.warmup, args.repeat),
        },
        "peak_memory_mb": {
            "ue_encoder": peak_memory_mb(encoder_fn, device),
            "quantizer": peak_memory_mb(quantizer_fn, device),
            "bs_decoder": peak_memory_mb(decoder_fn, device),
            "e2e": peak_memory_mb(e2e_fn, device),
        },
    }
    text = json.dumps(output, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
