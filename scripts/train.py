#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.insert(0, str(SRC))

from qcfhnet.datasets import Cost2100Dataset, load_cost2100_mat, load_npz_csi, make_dummy_csi
from qcfhnet.metrics import achievable_rate_ratio, cosine_similarity, nmse_db, nmse_global_db
from qcfhnet.models import build_model
from qcfhnet.training import build_reconstruction_loss
from qcfhnet.utils.checkpoint import load_checkpoint, save_checkpoint
from qcfhnet.utils.config import load_config, save_config
from qcfhnet.utils.device import get_device
from qcfhnet.utils.hardware import save_hardware
from qcfhnet.utils.logging import setup_logging
from qcfhnet.utils.seed import seed_everything


def load_split(config: dict, split: str, seed: int):
    data_cfg = config["data"]
    nc = int(data_cfg.get("nc", 32))
    nt = int(data_cfg.get("nt", 32))
    name = data_cfg.get("name", "dummy").lower()
    if name == "dummy":
        num = int(data_cfg.get(f"{split}_samples", 128 if split == "train" else 32))
        h = make_dummy_csi(num, nc=nc, nt=nt, seed=seed + (0 if split == "train" else 999))
        return h, {"name": "dummy", "split": split, "shape": list(h.shape)}
    if name == "cost2100":
        path = data_cfg[f"{split}_path"]
        h, meta = load_cost2100_mat(
            path=path,
            key=data_cfg.get("key", "HT"),
            nc=nc,
            nt=nt,
            normalization=data_cfg.get("normalization", "csinet_minus_0p5"),
        )
        limit = data_cfg.get(f"{split}_samples_limit", data_cfg.get("samples_limit"))
        if limit is not None:
            limit = int(limit)
            h = np.ascontiguousarray(h[:limit])
            meta["limited_samples"] = limit
            meta["shape_after_limit"] = list(h.shape)
        return h, meta
    if name in {"npz", "deepmimo"}:
        path = data_cfg["npz_path"]
        split_key = data_cfg.get(f"{split}_key", f"h_{split}")
        h, meta = load_npz_csi(path, split_key=split_key, nc=nc, nt=nt)
        limit = data_cfg.get(f"{split}_samples_limit", data_cfg.get("samples_limit"))
        if limit is not None:
            limit = int(limit)
            h = np.ascontiguousarray(h[:limit])
            meta["limited_samples"] = limit
            meta["shape_after_limit"] = list(h.shape)
        meta["name"] = name
        return h, meta
    raise ValueError(f"Unsupported dataset: {name}")


@torch.no_grad()
def evaluate(model, loader, device, snr_db: float = 10.0) -> dict:
    model.eval()
    forward_params = inspect.signature(model.forward).parameters
    accepts_sample_id = "sample_id" in forward_params
    nmse_values = []
    cosine_values = []
    rate_values = []
    targets = []
    preds = []
    for batch in loader:
        h = batch["h"].to(device)
        if accepts_sample_id and "sample_id" in batch:
            out = model(h, sample_id=batch["sample_id"].to(device))["h_hat"]
        else:
            out = model(h)["h_hat"]
        target_eval = h
        pred_eval = out
        if "target_scale" in batch:
            scale = batch["target_scale"].to(device)
            target_eval = h * scale
            pred_eval = out * scale
        nmse_values.append(nmse_db(target_eval, pred_eval).detach().cpu())
        cosine_values.append(cosine_similarity(target_eval, pred_eval).detach().cpu())
        rate_values.append(achievable_rate_ratio(target_eval, pred_eval, snr_db=snr_db).detach().cpu())
        targets.append(target_eval.detach().cpu())
        preds.append(pred_eval.detach().cpu())

    target_all = torch.cat(targets, dim=0)
    pred_all = torch.cat(preds, dim=0)
    nmse_all = torch.cat(nmse_values)
    cosine_all = torch.cat(cosine_values)
    rate_all = torch.cat(rate_values)
    return {
        "nmse_db_mean": float(nmse_all.mean()),
        "nmse_db_std": float(nmse_all.std(unbiased=False)),
        "nmse_global_db": float(nmse_global_db(target_all, pred_all)),
        "cosine_mean": float(cosine_all.mean()),
        "cosine_std": float(cosine_all.std(unbiased=False)),
        "rate_ratio_mean": float(rate_all.mean()),
        "rate_ratio_std": float(rate_all.std(unbiased=False)),
        "num_samples": int(target_all.shape[0]),
    }


def write_curve(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prefix_metrics(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def add_test_aliases(out: dict, metrics: dict) -> None:
    out.update(prefix_metrics(metrics, "test"))
    out["test_rate_ratio_mean_proxy"] = metrics.get("rate_ratio_mean")
    out["test_rate_ratio_std_proxy"] = metrics.get("rate_ratio_std")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--limit-test-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--experiment-name",
        default="",
        help="Override experiment.name and its results output directory.",
    )
    parser.add_argument("--experiment-suffix", default="")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Train/select on train+validation only; never load or evaluate the test split.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("train", {})["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        config.setdefault("train", {})["eval_batch_size"] = args.eval_batch_size
    if args.lr is not None:
        config.setdefault("train", {})["lr"] = args.lr
    if args.seed is not None:
        config.setdefault("experiment", {})["seed"] = args.seed
    if args.limit_train_samples is not None:
        config.setdefault("data", {})["train_samples_limit"] = args.limit_train_samples
    if args.limit_val_samples is not None:
        config.setdefault("data", {})["val_samples_limit"] = args.limit_val_samples
    if args.limit_test_samples is not None:
        config.setdefault("data", {})["test_samples_limit"] = args.limit_test_samples
    if args.experiment_name and args.experiment_suffix:
        parser.error("--experiment-name and --experiment-suffix are mutually exclusive")
    if args.experiment_name:
        exp = config.setdefault("experiment", {})
        exp["name"] = args.experiment_name
        exp["output_dir"] = f"results/{args.experiment_name}"
    elif args.experiment_suffix:
        exp = config.setdefault("experiment", {})
        base_name = exp.get("name", Path(args.config).stem)
        exp["name"] = f"{base_name}_{args.experiment_suffix}"
        exp["output_dir"] = f"results/{exp['name']}"

    exp_cfg = config.get("experiment", {})
    train_cfg = config.get("train", {})
    seed = int(exp_cfg.get("seed", 1))
    seed_everything(seed)

    exp_name = exp_cfg.get("name", Path(args.config).stem)
    result_dir = ROOT / exp_cfg.get("output_dir", f"results/{exp_name}")
    checkpoint_dir = ROOT / "checkpoints" / exp_name
    log_path = ROOT / "logs" / f"{exp_name}.log"
    logger = setup_logging(log_path)
    logger.info("Starting experiment: %s", exp_name)

    save_config(config, result_dir / "config.yaml")
    hardware = save_hardware(result_dir / "hardware.json")
    logger.info("Hardware: %s", hardware)

    h_train, train_meta = load_split(config, "train", seed)
    h_val, val_meta = load_split(config, "val", seed)
    dataset_metadata = {"train": train_meta, "val": val_meta}
    h_test = None
    if not args.selection_only:
        h_test, test_meta = load_split(config, "test", seed)
        dataset_metadata["test"] = test_meta
    with (result_dir / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=2)

    train_loader = DataLoader(
        Cost2100Dataset(h_train),
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    test_loader = None
    if h_test is not None:
        test_loader = DataLoader(
            Cost2100Dataset(h_test),
            batch_size=int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 64))),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
        )
    val_loader = DataLoader(
        Cost2100Dataset(h_val),
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 64))),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    device = get_device(args.device)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(train_cfg.get("epochs", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(train_cfg.get("eta_min", 1e-5)),
    )
    loss_name = str(train_cfg.get("loss", "mse")).lower()
    lambda_nmse = float(train_cfg.get("lambda_nmse", 0.1))
    loss_fn = build_reconstruction_loss(loss_name, lambda_nmse=lambda_nmse)
    logger.info("Training loss: %s lambda_nmse=%s", loss_name, lambda_nmse)

    best_val_nmse = float("inf")
    best_epoch = 0
    val_freq = max(1, int(train_cfg.get("val_freq", 1)))
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        telemetry: dict[str, float] = {}
        should_collect_telemetry = epoch == epochs or epoch % val_freq == 0
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs}",
            leave=False,
            disable=not sys.stderr.isatty(),
        )
        for batch_index, batch in enumerate(progress):
            h = batch["h"].to(device)
            optimizer.zero_grad(set_to_none=True)
            model_out = model(h)
            out = model_out["h_hat"]
            loss = loss_fn(out, h)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} batch={batch_index}")
            loss.backward()
            is_last_batch = batch_index + 1 == len(train_loader)
            if should_collect_telemetry and is_last_batch:
                grad_norms = [param.grad.detach().norm(2) for param in model.parameters() if param.grad is not None]
                if grad_norms:
                    telemetry["grad_norm_l2_last_batch"] = float(torch.stack(grad_norms).norm(2).cpu())
                quantizer_real = getattr(model, "quantizer_real", None)
                quantizer_imag = getattr(model, "quantizer_imag", None)
                quantizer = getattr(model, "quantizer", None)
                if quantizer_real is not None and quantizer_imag is not None:
                    scale_real = quantizer_real.scale().detach()
                    scale_imag = quantizer_imag.scale().detach()
                    telemetry["quantizer_scale_real"] = float(scale_real.cpu())
                    telemetry["quantizer_scale_imag"] = float(scale_imag.cpu())
                    telemetry["latent_abs_gt_scale_ratio_real"] = float(
                        (model_out["z_real"].detach().abs() > scale_real).float().mean().cpu()
                    )
                    telemetry["latent_abs_gt_scale_ratio_imag"] = float(
                        (model_out["z_imag"].detach().abs() > scale_imag).float().mean().cpu()
                    )
                elif quantizer is not None:
                    scale = quantizer.scale().detach()
                    telemetry["quantizer_scale"] = float(scale.cpu())
                    telemetry["latent_abs_gt_scale_ratio"] = float(
                        (model_out["z"].detach().abs() > scale).float().mean().cpu()
                    )
            grad_clip = train_cfg.get("grad_clip")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            loss_sum += float(loss.detach()) * h.shape[0]
            seen += h.shape[0]
            progress.set_postfix(loss=loss_sum / max(1, seen))
        scheduler.step()

        train_loss = loss_sum / max(1, seen)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": scheduler.get_last_lr()[0],
        }
        row.update(telemetry)
        should_eval = epoch == epochs or epoch % val_freq == 0
        if should_eval:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                snr_db=float(config.get("metrics", {}).get("snr_db", 10.0)),
            )
            row.update(prefix_metrics(val_metrics, "val"))
            logger.info(
                "epoch=%d train_loss=%.6g val_nmse=%.3f dB val_cosine=%.4f val_rate_ratio=%.4f",
                epoch,
                train_loss,
                val_metrics["nmse_db_mean"],
                val_metrics["cosine_mean"],
                val_metrics["rate_ratio_mean"],
            )
        else:
            val_metrics = {}
            logger.info("epoch=%d train_loss=%.6g", epoch, train_loss)
        curve.append(row)

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "metrics": val_metrics,
            "selection_split": "val",
            "selection_metric": "val_nmse_db_mean",
        }
        save_checkpoint(checkpoint_dir / "last.pt", payload)
        if val_metrics and val_metrics["nmse_db_mean"] < best_val_nmse:
            best_val_nmse = val_metrics["nmse_db_mean"]
            best_epoch = epoch
            save_checkpoint(checkpoint_dir / "best.pt", payload)

    best_path = checkpoint_dir / "best.pt"
    best_payload = None
    if best_path.exists():
        best_payload = load_checkpoint(best_path, map_location=device)
        model.load_state_dict(best_payload["model_state"])
    if args.selection_only:
        if best_payload is None:
            raise RuntimeError("Selection-only run produced no validation-selected checkpoint")
        final_metrics = {
            **prefix_metrics(best_payload.get("metrics", {}), "best_val"),
            "selection_only": True,
            "test_accessed": False,
        }
    else:
        if test_loader is None:
            raise RuntimeError("Formal run has no test loader")
        final_metrics = evaluate(
            model,
            test_loader,
            device,
            snr_db=float(config.get("metrics", {}).get("snr_db", 10.0)),
        )
        add_test_aliases(final_metrics, final_metrics)
    final_metrics.update(
        {
            "experiment": exp_name,
            "selection_split": "val",
            "selection_metric": "val_nmse_db_mean",
            "test_not_used_for_selection": True,
            "best_epoch": int(best_epoch),
            "best_val_nmse_db_mean": best_val_nmse,
            "best_nmse_db_mean": best_val_nmse,
            "feedback_scalars": int(getattr(model, "feedback_scalars", config.get("model", {}).get("latent_dim", 0))),
            "feedback_bits_float": int(32 * getattr(model, "feedback_scalars", config.get("model", {}).get("latent_dim", 0))),
            "upsampler": getattr(model, "upsampler_name", config.get("model", {}).get("upsampler", "interp_conv")),
            "upsampler_fallback_interpolate": bool(getattr(model, "upsampler_fallback_interpolate", False)),
            "split_real_imag": bool(getattr(model, "split_real_imag", config.get("model", {}).get("split_real_imag", False))),
            "loss": loss_name,
            "lambda_nmse": lambda_nmse,
        }
    )
    quantizer = getattr(model, "quantizer", None)
    quantizer_real = getattr(model, "quantizer_real", None)
    quantizer_imag = getattr(model, "quantizer_imag", None)
    if quantizer_real is not None and quantizer_imag is not None:
        final_metrics["feedback_bits_quantized"] = int(model.quantized_feedback_bits)
        final_metrics["quantizer_scale_real"] = float(quantizer_real.scale().detach().cpu())
        final_metrics["quantizer_scale_imag"] = float(quantizer_imag.scale().detach().cpu())
    elif quantizer is not None:
        final_metrics["feedback_bits_quantized"] = int(quantizer.feedback_bits(model.feedback_scalars))
        final_metrics["quantizer_scale"] = float(quantizer.scale().detach().cpu())

    result_dir.mkdir(parents=True, exist_ok=True)
    result_name = "selection_summary.json" if args.selection_only else "results.json"
    with (result_dir / result_name).open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    write_curve(result_dir / "train_curve.csv", curve)
    write_curve(result_dir / "metrics.csv", [final_metrics])
    logger.info("Finished experiment: %s", exp_name)


if __name__ == "__main__":
    main()
