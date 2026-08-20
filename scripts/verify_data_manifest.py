#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "COST2100")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "COST2100_MANIFEST.json")
    parser.add_argument("--size-only", action="store_true", help="Skip SHA256 computation.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = []
    for name, expected in manifest["files"].items():
        path = args.data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        actual_size = path.stat().st_size
        if actual_size != int(expected["bytes"]):
            raise ValueError(f"Size mismatch for {name}: expected {expected['bytes']}, got {actual_size}")
        row = {"file": name, "bytes": actual_size, "size": "PASS"}
        if not args.size_only:
            actual_hash = sha256(path)
            if actual_hash != expected["sha256"]:
                raise ValueError(f"SHA256 mismatch for {name}: expected {expected['sha256']}, got {actual_hash}")
            row["sha256"] = "PASS"
        checks.append(row)
    print(json.dumps({"status": "PASS", "data_dir": str(args.data_dir), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
