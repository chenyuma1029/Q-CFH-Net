#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_channel_params(
    dm,
    nt: int,
    ofdm_subcarriers: int,
    selected_subcarriers: int,
    bandwidth_hz: float,
):
    return dm.ChannelParameters(
        bs_antenna={"shape": [int(nt), 1]},
        ue_antenna={"shape": [1, 1]},
        freq_domain=True,
        num_paths=25,
        ofdm={
            "subcarriers": int(ofdm_subcarriers),
            "selected_subcarriers": np.arange(int(selected_subcarriers)),
            "bandwidth": float(bandwidth_hz),
        },
    )


def compute_freq_split(
    dm,
    scenario: str,
    idxs: np.ndarray,
    nt: int,
    ofdm_subcarriers: int,
    selected_subcarriers: int,
    bandwidth_hz: float,
) -> np.ndarray:
    ds = dm.load(scenario, max_paths=25, compat_v3=True).trim(idxs=idxs)
    params = make_channel_params(
        dm,
        nt=nt,
        ofdm_subcarriers=ofdm_subcarriers,
        selected_subcarriers=selected_subcarriers,
        bandwidth_hz=bandwidth_hz,
    )
    ch = ds.compute_channels(params)
    if ch.ndim != 4 or ch.shape[1] != 1:
        raise ValueError(f"Expected channel [N,1,Nt,Nf], got {ch.shape}")
    # DeepMIMO returns [N, rx_ant=1, tx_ant=Nt, subcarriers]. Store [N,Nf,Nt].
    return np.ascontiguousarray(np.transpose(ch[:, 0, :, :], (0, 2, 1)).astype(np.complex64, copy=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="asu_campus_3p5")
    parser.add_argument("--nt", type=int, required=True)
    parser.add_argument("--n-subcarriers", type=int, default=125)
    parser.add_argument("--ofdm-subcarriers", type=int, default=512)
    parser.add_argument("--bandwidth-hz", type=float, default=100e6)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import deepmimo as dm

    base = dm.load(args.scenario, max_paths=25, compat_v3=True)
    active = np.asarray(base.get_idxs("active"), dtype=np.int64)
    needed = int(args.train_samples + args.val_samples + args.test_samples)
    if active.shape[0] < needed:
        raise ValueError(f"Only {active.shape[0]} active users available, need {needed}")
    rng = np.random.default_rng(args.seed)
    selected = rng.permutation(active)[:needed]
    train_idxs = np.sort(selected[: args.train_samples])
    val_idxs = np.sort(selected[args.train_samples : args.train_samples + args.val_samples])
    test_idxs = np.sort(selected[args.train_samples + args.val_samples :])

    if args.ofdm_subcarriers < args.n_subcarriers:
        raise ValueError("--ofdm-subcarriers must be >= --n-subcarriers")
    h_train = compute_freq_split(
        dm,
        args.scenario,
        train_idxs,
        args.nt,
        args.ofdm_subcarriers,
        args.n_subcarriers,
        args.bandwidth_hz,
    )
    h_val = compute_freq_split(
        dm,
        args.scenario,
        val_idxs,
        args.nt,
        args.ofdm_subcarriers,
        args.n_subcarriers,
        args.bandwidth_hz,
    )
    h_test = compute_freq_split(
        dm,
        args.scenario,
        test_idxs,
        args.nt,
        args.ofdm_subcarriers,
        args.n_subcarriers,
        args.bandwidth_hz,
    )

    metadata = {
        "source": "DeepMIMO v4",
        "scenario": args.scenario,
        "nt": int(args.nt),
        "n_subcarriers": int(args.n_subcarriers),
        "bandwidth_hz": float(args.bandwidth_hz),
        "seed": int(args.seed),
        "active_user_count": int(active.shape[0]),
        "train_indices_count": int(train_idxs.shape[0]),
        "val_indices_count": int(val_idxs.shape[0]),
        "test_indices_count": int(test_idxs.shape[0]),
        "frequency_layout": "n_nc_nt",
        "stored_frequency_shape": "[N,Nfreq,Nt]",
        "channel_params": {
            "bs_antenna_shape": [int(args.nt), 1],
            "ue_antenna_shape": [1, 1],
            "num_paths": 25,
            "freq_domain": True,
            "ofdm_subcarriers": int(args.ofdm_subcarriers),
            "selected_subcarriers": int(args.n_subcarriers),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        h_train=h_train,
        h_val=h_val,
        h_test=h_test,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    print(json.dumps({"out": str(out), "metadata": metadata, "shapes": {"h_train": list(h_train.shape), "h_val": list(h_val.shape), "h_test": list(h_test.shape)}}, indent=2))


if __name__ == "__main__":
    main()
