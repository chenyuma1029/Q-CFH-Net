#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalize(h: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
    h = h.astype(np.float32, copy=False)
    if mode == "none":
        return h, {"mode": mode}
    if mode == "per_dataset_standard":
        mean = h.mean(dtype=np.float64)
        std = h.std(dtype=np.float64)
        std = float(std if std > 1e-12 else 1.0)
        return ((h - float(mean)) / std).astype(np.float32), {"mode": mode, "mean": float(mean), "std": std}
    if mode == "minus_0p5":
        return (h - 0.5).astype(np.float32), {"mode": mode}
    raise ValueError(f"Unsupported normalization: {mode}")


def complex_rms_scale(h: np.ndarray) -> float:
    if h.ndim != 4 or h.shape[1] != 2:
        raise ValueError(f"Expected [N,2,Nc,Nt] two-channel complex tensor, got {h.shape}")
    power = np.square(h[:, 0], dtype=np.float64) + np.square(h[:, 1], dtype=np.float64)
    scale = float(np.sqrt(power.mean(dtype=np.float64)))
    return scale if scale > 1e-12 else 1.0


def per_sample_complex_rms_scale(h: np.ndarray) -> np.ndarray:
    if h.ndim != 4 or h.shape[1] != 2:
        raise ValueError(f"Expected [N,2,Nc,Nt] two-channel complex tensor, got {h.shape}")
    power = np.square(h[:, 0], dtype=np.float64) + np.square(h[:, 1], dtype=np.float64)
    scale = np.sqrt(power.mean(axis=(1, 2), dtype=np.float64)).astype(np.float32)
    return np.where(scale > 1e-12, scale, 1.0).astype(np.float32)


def apply_per_sample_scale(h: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (h / scale[:, None, None, None]).astype(np.float32)


def load_source_npz(
    path: Path,
    train_key: str,
    val_key: str,
    test_key: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        h_train = np.array(data[train_key], copy=True)
        h_val = np.array(data[val_key], copy=True) if val_key in data else None
        h_test = np.array(data[test_key], copy=True)
        source_metadata = None
        if "metadata" in data:
            raw_meta = data["metadata"]
            try:
                source_metadata = json.loads(str(raw_meta.item()))
            except (json.JSONDecodeError, TypeError, ValueError):
                source_metadata = str(raw_meta)
    metadata = {
        "source_npz": str(path),
        "source_train_key": train_key,
        "source_val_key": val_key if h_val is not None else "",
        "source_test_key": test_key,
    }
    if source_metadata is not None:
        metadata["source_metadata"] = source_metadata
    return h_train, h_val, h_test, metadata


def validate_h(name: str, h: np.ndarray, nc: int, nt: int) -> None:
    if h.ndim != 4 or h.shape[1:] != (2, nc, nt):
        raise ValueError(f"{name} expected [N,2,{nc},{nt}], got {h.shape}")


def as_complex_frequency(h: np.ndarray, layout: str) -> np.ndarray:
    if layout == "n_nc_nt":
        if h.ndim != 3:
            raise ValueError(f"Expected [N,Nc,Nt], got {h.shape}")
        return np.ascontiguousarray(h.astype(np.complex64, copy=False))
    if layout == "n_nt_nc":
        if h.ndim != 3:
            raise ValueError(f"Expected [N,Nt,Nc], got {h.shape}")
        return np.ascontiguousarray(np.transpose(h, (0, 2, 1)).astype(np.complex64, copy=False))
    if layout == "n_2_nc_nt":
        if h.ndim != 4 or h.shape[1] != 2:
            raise ValueError(f"Expected [N,2,Nc,Nt], got {h.shape}")
        return np.ascontiguousarray((h[:, 0] + 1j * h[:, 1]).astype(np.complex64, copy=False))
    if layout == "n_nc_nt_2":
        if h.ndim != 4 or h.shape[-1] != 2:
            raise ValueError(f"Expected [N,Nc,Nt,2], got {h.shape}")
        return np.ascontiguousarray((h[..., 0] + 1j * h[..., 1]).astype(np.complex64, copy=False))
    raise ValueError(f"Unsupported complex layout: {layout}")


def frequency_to_angular_delay(
    h_freq: np.ndarray,
    *,
    nc: int,
    nt: int,
    delay_fft: int | None = None,
    antenna_fft: bool = True,
    fftshift_antenna: bool = True,
) -> np.ndarray:
    if h_freq.ndim != 3:
        raise ValueError(f"Expected frequency-domain [N,Nfreq,Nt], got {h_freq.shape}")
    if h_freq.shape[2] < nt:
        raise ValueError(f"Source Nt={h_freq.shape[2]} is smaller than requested nt={nt}")

    h = np.ascontiguousarray(h_freq[:, :, :nt].astype(np.complex64, copy=False))
    fft_len = int(delay_fft) if delay_fft is not None else int(h.shape[1])
    delay = np.fft.ifft(h, n=fft_len, axis=1)
    if delay.shape[1] < nc:
        raise ValueError(f"Delay FFT length {delay.shape[1]} is smaller than requested nc={nc}")
    delay = delay[:, :nc, :]
    if antenna_fft:
        delay = np.fft.fft(delay, n=nt, axis=2) / np.sqrt(float(nt))
        if fftshift_antenna:
            delay = np.fft.fftshift(delay, axes=2)
    out = np.stack([delay.real, delay.imag], axis=1).astype(np.float32)
    return np.ascontiguousarray(out)


def frequency_for_physical(h_freq: np.ndarray, *, nt: int, n_eval: int) -> np.ndarray:
    if h_freq.ndim != 3:
        raise ValueError(f"Expected frequency-domain [N,Nfreq,Nt], got {h_freq.shape}")
    if h_freq.shape[1] < n_eval:
        raise ValueError(f"Source Nfreq={h_freq.shape[1]} is smaller than n_eval={n_eval}")
    if h_freq.shape[2] < nt:
        raise ValueError(f"Source Nt={h_freq.shape[2]} is smaller than nt={nt}")
    return np.ascontiguousarray(np.transpose(h_freq[:, :n_eval, :nt], (0, 2, 1)).astype(np.complex64, copy=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", default="")
    parser.add_argument("--source-train-key", default="h_train")
    parser.add_argument("--source-val-key", default="h_val")
    parser.add_argument("--source-test-key", default="h_test")
    parser.add_argument("--input-domain", default="sparse_2chan", choices=["sparse_2chan", "frequency_complex"])
    parser.add_argument(
        "--complex-layout",
        default="n_nc_nt",
        choices=["n_nc_nt", "n_nt_nc", "n_2_nc_nt", "n_nc_nt_2"],
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--nt", type=int, required=True)
    parser.add_argument("--nc", type=int, default=32)
    parser.add_argument("--delay-fft", type=int, default=None)
    parser.add_argument("--no-antenna-fft", action="store_true")
    parser.add_argument("--no-antenna-fftshift", action="store_true")
    parser.add_argument("--hf-eval-subcarriers", type=int, default=125)
    parser.add_argument("--no-save-hf", action="store_true")
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument(
        "--normalization",
        default="none",
        choices=["none", "minus_0p5", "per_dataset_standard", "train_rms_complex_magnitude", "per_sample_complex_rms"],
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not args.source_npz:
        raise RuntimeError(
            "DeepMIMO API generation is not configured in this repository yet. "
            "Provide --source-npz from a generated DeepMIMO/MATLAB export and this script will normalize and package it."
        )

    source_train, source_val, source_test, source_meta = load_source_npz(
        Path(args.source_npz),
        train_key=args.source_train_key,
        val_key=args.source_val_key,
        test_key=args.source_test_key,
    )

    hf_train = hf_val = hf_test = None
    conversion_meta = {"input_domain": args.input_domain}
    if args.input_domain == "frequency_complex":
        freq_train = as_complex_frequency(source_train, args.complex_layout)
        freq_val = as_complex_frequency(source_val, args.complex_layout) if source_val is not None else None
        freq_test = as_complex_frequency(source_test, args.complex_layout)
        h_train = frequency_to_angular_delay(
            freq_train,
            nc=args.nc,
            nt=args.nt,
            delay_fft=args.delay_fft,
            antenna_fft=not args.no_antenna_fft,
            fftshift_antenna=not args.no_antenna_fftshift,
        )
        h_val = (
            frequency_to_angular_delay(
                freq_val,
                nc=args.nc,
                nt=args.nt,
                delay_fft=args.delay_fft,
                antenna_fft=not args.no_antenna_fft,
                fftshift_antenna=not args.no_antenna_fftshift,
            )
            if freq_val is not None
            else None
        )
        h_test = frequency_to_angular_delay(
            freq_test,
            nc=args.nc,
            nt=args.nt,
            delay_fft=args.delay_fft,
            antenna_fft=not args.no_antenna_fft,
            fftshift_antenna=not args.no_antenna_fftshift,
        )
        if not args.no_save_hf:
            hf_train = frequency_for_physical(freq_train, nt=args.nt, n_eval=args.hf_eval_subcarriers)
            hf_val = (
                frequency_for_physical(freq_val, nt=args.nt, n_eval=args.hf_eval_subcarriers)
                if freq_val is not None
                else None
            )
            hf_test = frequency_for_physical(freq_test, nt=args.nt, n_eval=args.hf_eval_subcarriers)
        conversion_meta.update(
            {
                "complex_layout": args.complex_layout,
                "source_frequency_train_shape": list(freq_train.shape),
                "source_frequency_val_shape": list(freq_val.shape) if freq_val is not None else [],
                "source_frequency_test_shape": list(freq_test.shape),
                "delay_fft": args.delay_fft,
                "antenna_fft": not args.no_antenna_fft,
                "antenna_fftshift": not args.no_antenna_fftshift,
                "hf_eval_subcarriers": args.hf_eval_subcarriers,
                "saved_hf_keys": [
                    key
                    for key, value in (("hf_train", hf_train), ("hf_val", hf_val), ("hf_test", hf_test))
                    if value is not None
                ],
            }
        )
    else:
        h_train = np.asarray(source_train, dtype=np.float32)
        h_val = np.asarray(source_val, dtype=np.float32) if source_val is not None else None
        h_test = np.asarray(source_test, dtype=np.float32)

    validate_h("h_train", h_train, nc=args.nc, nt=args.nt)
    if h_val is not None:
        validate_h("h_val", h_val, nc=args.nc, nt=args.nt)
    validate_h("h_test", h_test, nc=args.nc, nt=args.nt)
    if args.train_samples is not None:
        h_train = np.ascontiguousarray(h_train[: args.train_samples])
        if hf_train is not None:
            hf_train = np.ascontiguousarray(hf_train[: args.train_samples])
    if args.train_samples is not None and h_val is not None:
        # Keep validation untouched by train-samples; this branch exists only to
        # make the intent explicit next to the train/test truncation code.
        h_val = np.ascontiguousarray(h_val)
    if args.test_samples is not None:
        h_test = np.ascontiguousarray(h_test[: args.test_samples])
        if hf_test is not None:
            hf_test = np.ascontiguousarray(hf_test[: args.test_samples])

    scale_train = scale_val = scale_test = None
    if args.normalization == "train_rms_complex_magnitude":
        scale = complex_rms_scale(h_train)
        h_train = (h_train / scale).astype(np.float32)
        if h_val is not None:
            h_val = (h_val / scale).astype(np.float32)
        h_test = (h_test / scale).astype(np.float32)
        norm_meta = {"mode": args.normalization, "scale": scale, "computed_from": "h_train"}
    elif args.normalization == "per_sample_complex_rms":
        scale_train = per_sample_complex_rms_scale(h_train)
        scale_val = per_sample_complex_rms_scale(h_val) if h_val is not None else None
        scale_test = per_sample_complex_rms_scale(h_test)
        h_train = apply_per_sample_scale(h_train, scale_train)
        if h_val is not None and scale_val is not None:
            h_val = apply_per_sample_scale(h_val, scale_val)
        h_test = apply_per_sample_scale(h_test, scale_test)
        norm_meta = {
            "mode": args.normalization,
            "computed_from": "each_sample_before_normalization",
            "saved_scale_keys": [
                key
                for key, value in (("scale_train", scale_train), ("scale_val", scale_val), ("scale_test", scale_test))
                if value is not None
            ],
        }
    else:
        _, norm_meta = normalize(h_train, args.normalization)
        if args.normalization == "per_dataset_standard":
            norm_meta["computed_from"] = "h_train"
    if args.normalization == "per_dataset_standard":
        h_train = ((h_train - norm_meta["mean"]) / norm_meta["std"]).astype(np.float32)
        if h_val is not None:
            h_val = ((h_val - norm_meta["mean"]) / norm_meta["std"]).astype(np.float32)
        h_test = ((h_test - norm_meta["mean"]) / norm_meta["std"]).astype(np.float32)
    elif args.normalization in {"none", "minus_0p5"}:
        h_train, _ = normalize(h_train, args.normalization)
        if h_val is not None:
            h_val, _ = normalize(h_val, args.normalization)
        h_test, _ = normalize(h_test, args.normalization)

    metadata = {
        "source": "DeepMIMO",
        "scenario": args.scenario,
        "nc": args.nc,
        "nt": args.nt,
        "train_shape": list(h_train.shape),
        "val_shape": list(h_val.shape) if h_val is not None else [],
        "test_shape": list(h_test.shape),
        "normalization": norm_meta,
        "conversion": conversion_meta,
        "internal_layout": "b_2_nc_nt",
        **source_meta,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "h_train": np.ascontiguousarray(h_train),
        "h_test": np.ascontiguousarray(h_test),
        "metadata": json.dumps(metadata, ensure_ascii=False),
    }
    if h_val is not None:
        payload["h_val"] = np.ascontiguousarray(h_val)
    if hf_train is not None and hf_test is not None:
        payload["hf_train"] = np.ascontiguousarray(hf_train)
        if hf_val is not None:
            payload["hf_val"] = np.ascontiguousarray(hf_val)
        payload["hf_test"] = np.ascontiguousarray(hf_test)
    if scale_train is not None and scale_test is not None:
        payload["scale_train"] = np.ascontiguousarray(scale_train)
        if scale_val is not None:
            payload["scale_val"] = np.ascontiguousarray(scale_val)
        payload["scale_test"] = np.ascontiguousarray(scale_test)
    np.savez_compressed(out, **payload)
    print(json.dumps({"out": str(out), "metadata": metadata}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
