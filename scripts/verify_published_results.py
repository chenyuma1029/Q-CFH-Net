#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify(per_seed_path: Path, table_path: Path, tolerance: float) -> dict:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(per_seed_path):
        grouped[(row["scenario"].lower(), int(row["bits"]))].append(row)

    expected_groups = {("indoor", 4), ("outdoor", 4), ("indoor", 8), ("outdoor", 8)}
    if set(grouped) != expected_groups:
        raise ValueError(f"Unexpected per-seed groups: {sorted(grouped)}")

    summary = {}
    for key, rows in grouped.items():
        seeds = sorted(int(row["seed"]) for row in rows)
        if seeds != [1, 2, 3]:
            raise ValueError(f"Expected seeds 1/2/3 for {key}, got {seeds}")
        summary[key] = {
            metric: {
                "mean": float(np.mean([float(row[metric]) for row in rows])),
                "std_ddof0": float(np.std([float(row[metric]) for row in rows], ddof=0)),
            }
            for metric in ("nmse_db", "rho", "R10")
        }

    table_rows = {
        int(row["bits"]): row
        for row in read_rows(table_path)
        if row["method"] == "Q-CFH G2"
    }
    if set(table_rows) != {4, 8}:
        raise ValueError("Expected one Q-CFH G2 row for each of q4 and q8")

    checks = []
    table_metric_names = {"nmse_db": "nmse", "rho": "rho", "R10": "R10"}
    for (scenario, bits), metrics in sorted(summary.items()):
        table_row = table_rows[bits]
        for metric, values in metrics.items():
            table_metric = table_metric_names[metric]
            for statistic, table_suffix in (("mean", "mean"), ("std_ddof0", "std")):
                table_key = f"{scenario}_{table_metric}_{table_suffix}"
                expected = float(table_row[table_key])
                actual = values[statistic]
                difference = abs(actual - expected)
                checks.append(
                    {
                        "scenario": scenario,
                        "bits": bits,
                        "metric": metric,
                        "statistic": statistic,
                        "computed": actual,
                        "table": expected,
                        "difference": difference,
                    }
                )
                if difference > tolerance:
                    raise ValueError(
                        f"Published result mismatch for {scenario} q{bits} {metric} {statistic}: "
                        f"computed={actual}, table={expected}"
                    )

    return {
        "status": "PASS",
        "standard_deviation": "population (ddof=0)",
        "per_seed_csv": str(per_seed_path),
        "table_csv": str(table_path),
        "tolerance": tolerance,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-seed", type=Path, default=ROOT / "results" / "qcfh_cost2100_per_seed.csv")
    parser.add_argument("--table", type=Path, default=ROOT / "results" / "cost2100_table1.csv")
    parser.add_argument("--tolerance", type=float, default=5.1e-5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = verify(args.per_seed, args.table, tolerance=float(args.tolerance))
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
