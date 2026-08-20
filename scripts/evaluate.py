#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.insert(0, str(SRC))

from qcfhnet.datasets import Cost2100Dataset
from qcfhnet.models import build_model
from qcfhnet.utils.checkpoint import load_checkpoint
from qcfhnet.utils.config import load_config
from qcfhnet.utils.device import get_device

from train import evaluate, load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", Path(args.config).stem)
    checkpoint = args.checkpoint or str(ROOT / "checkpoints" / exp_name / "best.pt")
    device = get_device(args.device)

    model = build_model(config).to(device)
    payload = load_checkpoint(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"])

    h_eval, _ = load_split(config, args.split, int(config.get("experiment", {}).get("seed", 1)))
    loader = DataLoader(
        Cost2100Dataset(h_eval),
        batch_size=int(config.get("train", {}).get("eval_batch_size", 64)),
        shuffle=False,
    )
    metrics = evaluate(model, loader, device, snr_db=float(config.get("metrics", {}).get("snr_db", 10.0)))
    metrics["split"] = args.split
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
