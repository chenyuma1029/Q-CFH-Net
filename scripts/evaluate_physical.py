#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.insert(0, str(SRC))

from qcfhnet.datasets import Cost2100Dataset, load_cost2100_hf_all
from qcfhnet.metrics import (
    physical_rate_components,
    physical_rate_components_paper_equation,
    physical_rate_ratio,
    physical_rate_ratio_paper_equation,
    physical_rho,
    physical_rho_paper_equation,
    sparse_to_frequency,
)
from qcfhnet.models import build_model
from qcfhnet.utils.checkpoint import load_checkpoint, validate_checkpoint_compatibility
from qcfhnet.utils.config import load_config, validate_config
from qcfhnet.utils.device import get_device

from train import load_split


def load_hf_all(path: str | Path, nt: int, n_eval: int, key: str) -> tuple[np.ndarray, dict]:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise KeyError(f"Key {key!r} not found in {path}. Available keys: {list(data.keys())}")
            h = np.array(data[key], copy=True)
            if h.ndim != 3 or h.shape[1:] != (nt, n_eval):
                raise ValueError(f"Expected {key} shape [N,{nt},{n_eval}], got {h.shape}")
            metadata = {
                "path": str(path),
                "key": key,
                "shape": list(h.shape),
                "nt": int(nt),
                "n_eval": int(n_eval),
                "dtype": str(h.dtype),
            }
            if "metadata" in data:
                raw_meta = data["metadata"]
                try:
                    metadata["source_metadata"] = json.loads(str(raw_meta.item()))
                except (json.JSONDecodeError, TypeError, ValueError):
                    metadata["source_metadata"] = str(raw_meta)
        return np.ascontiguousarray(h.astype(np.complex64, copy=False)), metadata
    return load_cost2100_hf_all(path, nt=nt, n_eval=n_eval)


def summarize(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = values.detach().cpu().float()
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(unbiased=False)),
    }


def validate_hf_pairing(
    config: dict,
    hf_path: str | Path,
    hf_metadata: dict,
    *,
    allow_unverified: bool,
) -> dict:
    data = config.get("data", {})
    data_name = str(data.get("name", "")).lower()
    scenario = str(data.get("scenario", "")).lower()
    pairing = {
        "data_name": data_name,
        "scenario": scenario,
        "hf_path": str(hf_path),
        "verified": False,
    }
    if allow_unverified:
        pairing["reason"] = "explicit --allow-unverified-hf-pair override"
        return pairing

    if data_name == "cost2100":
        expected = {
            "indoor": "data_htestfin_all.mat",
            "outdoor": "data_htestfout_all.mat",
        }.get(scenario)
        actual = Path(hf_path).name.lower()
        if expected is None or actual != expected:
            raise ValueError(
                f"Unverified COST2100 sparse/full-frequency pairing: scenario={scenario!r}, "
                f"expected file {expected!r}, got {Path(hf_path).name!r}. "
                "Use --allow-unverified-hf-pair only after checking sample alignment."
            )
        pairing.update({"verified": True, "method": "canonical COST2100 filename"})
        return pairing

    source_metadata = hf_metadata.get("source_metadata")
    source_scenario = ""
    if isinstance(source_metadata, dict):
        source_scenario = str(source_metadata.get("scenario", "")).lower()
    if not scenario or not source_scenario or scenario != source_scenario:
        raise ValueError(
            f"Unverified sparse/full-frequency pairing: config scenario={scenario!r}, "
            f"HF metadata scenario={source_scenario!r}. "
            "Use --allow-unverified-hf-pair only after checking sample alignment."
        )
    pairing.update({"verified": True, "method": "matching scenario metadata"})
    return pairing


@torch.no_grad()
def evaluate_physical(
    model: torch.nn.Module,
    loader: DataLoader,
    hf_all: np.ndarray,
    device: torch.device,
    layout: str,
    antenna_domain: str,
    nc_fft: int,
    n_eval: int,
    snr_db_list: list[float],
    pred_already_centered: bool,
    sparse_layout: str,
) -> dict:
    model.eval()
    rho_values = []
    paper_rho_values = []
    sparse_rho_values = []
    paper_sparse_rho_values = []
    rate_pred_by_snr = {float(snr): [] for snr in snr_db_list}
    rate_oracle_by_snr = {float(snr): [] for snr in snr_db_list}
    rate_ratio_by_snr = {float(snr): [] for snr in snr_db_list}
    paper_rate_ratio_by_snr = {float(snr): [] for snr in snr_db_list}
    sparse_rate_ratio_by_snr = {float(snr): [] for snr in snr_db_list}
    paper_sparse_rate_ratio_by_snr = {float(snr): [] for snr in snr_db_list}
    offset = 0

    for batch in loader:
        h = batch["h"].to(device)
        pred_sparse = model(h)["h_hat"]
        batch_size = int(h.shape[0])
        freq_true = torch.from_numpy(hf_all[offset : offset + batch_size]).to(device)
        freq_pred = sparse_to_frequency(
            pred_sparse,
            internal_layout=layout,
            nc_fft=nc_fft,
            n_eval=n_eval,
            already_centered=pred_already_centered,
            antenna_domain=antenna_domain,
        )
        rho_values.append(physical_rho(freq_true, freq_pred).detach().cpu())
        paper_rho_values.append(physical_rho_paper_equation(freq_true, freq_pred).detach().cpu())
        freq_sparse_oracle = sparse_to_frequency(
            h,
            internal_layout=sparse_layout,
            nc_fft=nc_fft,
            n_eval=n_eval,
            already_centered=True,
            antenna_domain=antenna_domain,
        )
        sparse_rho_values.append(physical_rho(freq_true, freq_sparse_oracle).detach().cpu())
        paper_sparse_rho_values.append(
            physical_rho_paper_equation(freq_true, freq_sparse_oracle).detach().cpu()
        )
        for snr in snr_db_list:
            pred_rate, oracle_rate, ratio = physical_rate_components(freq_true, freq_pred, snr_db=float(snr))
            _, _, paper_ratio = physical_rate_components_paper_equation(
                freq_true,
                freq_pred,
                snr_db=float(snr),
            )
            rate_pred_by_snr[float(snr)].append(pred_rate.detach().cpu())
            rate_oracle_by_snr[float(snr)].append(oracle_rate.detach().cpu())
            rate_ratio_by_snr[float(snr)].append(ratio.detach().cpu())
            paper_rate_ratio_by_snr[float(snr)].append(paper_ratio.detach().cpu())
            sparse_rate_ratio_by_snr[float(snr)].append(
                physical_rate_ratio(freq_true, freq_sparse_oracle, snr_db=float(snr)).detach().cpu()
            )
            paper_sparse_rate_ratio_by_snr[float(snr)].append(
                physical_rate_ratio_paper_equation(
                    freq_true,
                    freq_sparse_oracle,
                    snr_db=float(snr),
                ).detach().cpu()
            )
        offset += batch_size

    rho_all = torch.cat(rho_values)
    sparse_rho_all = torch.cat(sparse_rho_values)
    paper_rho_all = torch.cat(paper_rho_values)
    paper_sparse_rho_all = torch.cat(paper_sparse_rho_values)
    metrics = {
        **summarize(rho_all, "physical_rho"),
        **summarize(sparse_rho_all, "physical_oracle_sparse_rho"),
        **summarize(paper_rho_all, "paper_equation_rho"),
        **summarize(paper_sparse_rho_all, "paper_equation_oracle_sparse_rho"),
        "num_samples": int(rho_all.numel()),
        "internal_layout": layout,
        "sparse_oracle_layout": sparse_layout,
        "antenna_domain": antenna_domain,
        "nc_fft": int(nc_fft),
        "n_eval_subcarriers": int(n_eval),
        "prediction_already_centered": bool(pred_already_centered),
        "metric_definitions": {
            "physical_rho": "Published implementation: mean over subcarriers of the unsquared normalized complex correlation magnitude.",
            "physical_rate_ratio": "Published implementation: mean over subcarriers of the predicted-to-oracle rate ratio.",
            "paper_equation_rho": "Paper equation: mean over subcarriers of the squared normalized complex correlation magnitude.",
            "paper_equation_rate_ratio": "Paper equation: ratio of the subcarrier-averaged predicted and oracle rates.",
        },
    }
    for snr, ratio_chunks in rate_ratio_by_snr.items():
        metrics.update(summarize(torch.cat(rate_pred_by_snr[snr]), f"physical_rate_pred_snr{snr:g}"))
        metrics.update(summarize(torch.cat(rate_oracle_by_snr[snr]), f"physical_rate_oracle_snr{snr:g}"))
        metrics.update(summarize(torch.cat(ratio_chunks), f"physical_rate_ratio_snr{snr:g}"))
        metrics.update(
            summarize(
                torch.cat(paper_rate_ratio_by_snr[snr]),
                f"paper_equation_rate_ratio_snr{snr:g}",
            )
        )
        metrics.update(
            summarize(torch.cat(sparse_rate_ratio_by_snr[snr]), f"physical_oracle_sparse_rate_ratio_snr{snr:g}")
        )
        metrics.update(
            summarize(
                torch.cat(paper_sparse_rate_ratio_by_snr[snr]),
                f"paper_equation_oracle_sparse_rate_ratio_snr{snr:g}",
            )
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hf-all", required=True)
    parser.add_argument("--hf-key", default="hf_test")
    parser.add_argument("--layout", default="b_2_nt_nc", choices=["b_2_nc_nt", "b_2_nt_nc"])
    parser.add_argument("--sparse-layout", default="b_2_nt_nc", choices=["b_2_nc_nt", "b_2_nt_nc"])
    parser.add_argument("--antenna-domain", default="spatial", choices=["spatial", "angular_fftshift"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nc-fft", type=int, default=257)
    parser.add_argument("--n-eval", type=int, default=125)
    parser.add_argument("--snr-db-list", type=float, nargs="+", default=[0.0, 10.0, 20.0])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-test-samples", type=int, default=None)
    parser.add_argument("--prediction-raw-0p5", action="store_true")
    parser.add_argument(
        "--allow-unverified-hf-pair",
        action="store_true",
        help="Bypass scenario/file provenance checks after independently verifying sample alignment.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.limit_test_samples is not None:
        config.setdefault("data", {})["test_samples_limit"] = int(args.limit_test_samples)
    validate_config(config)

    exp_name = config.get("experiment", {}).get("name", Path(args.config).stem)
    checkpoint = args.checkpoint or str(ROOT / "checkpoints" / exp_name / "best.pt")
    device = get_device(args.device)

    payload = load_checkpoint(checkpoint, map_location=device)
    checkpoint_config = validate_checkpoint_compatibility(config, payload)
    model = build_model(checkpoint_config).to(device)
    model.load_state_dict(payload["model_state"])

    seed = int(config.get("experiment", {}).get("seed", 1))
    h_test, test_meta = load_split(config, "test", seed)
    hf_all, hf_meta = load_hf_all(
        args.hf_all,
        nt=int(config.get("data", {}).get("nt", 32)),
        n_eval=int(args.n_eval),
        key=args.hf_key,
    )
    pairing = validate_hf_pairing(
        config,
        args.hf_all,
        hf_meta,
        allow_unverified=bool(args.allow_unverified_hf_pair),
    )
    if args.limit_test_samples is None and hf_all.shape[0] != h_test.shape[0]:
        raise ValueError(
            f"HF_all has {hf_all.shape[0]} samples but the complete test split has {h_test.shape[0]}; "
            "refusing implicit truncation"
        )
    if args.limit_test_samples is not None and hf_all.shape[0] < h_test.shape[0]:
        raise ValueError(f"HF_all has {hf_all.shape[0]} samples but limited test split has {h_test.shape[0]}")
    hf_all = np.ascontiguousarray(hf_all[: h_test.shape[0]])

    batch_size = args.batch_size or int(config.get("train", {}).get("eval_batch_size", 64))
    loader = DataLoader(Cost2100Dataset(h_test), batch_size=batch_size, shuffle=False)
    metrics = evaluate_physical(
        model=model,
        loader=loader,
        hf_all=hf_all,
        device=device,
        layout=args.layout,
        antenna_domain=args.antenna_domain,
        nc_fft=int(args.nc_fft),
        n_eval=int(args.n_eval),
        snr_db_list=[float(v) for v in args.snr_db_list],
        pred_already_centered=not args.prediction_raw_0p5,
        sparse_layout=args.sparse_layout,
    )
    metrics.update(
        {
            "experiment": exp_name,
            "checkpoint": str(checkpoint),
            "config": str(args.config),
            "hf_all": str(args.hf_all),
            "test_metadata": test_meta,
            "hf_all_metadata": hf_meta,
            "hf_pairing": pairing,
        }
    )

    text = json.dumps(metrics, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
