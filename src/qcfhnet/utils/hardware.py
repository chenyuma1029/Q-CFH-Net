from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def collect_hardware() -> dict:
    gpus = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            gpus.append(torch.cuda.get_device_name(idx))
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": gpus,
        "num_gpus": len(gpus),
        "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_hardware(path: str | Path) -> dict:
    info = collect_hardware()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info

