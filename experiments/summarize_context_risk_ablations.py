#!/usr/bin/env python3
"""Summarize matched-compute risk components and label-free baselines.

Each ``--run`` is ``LABEL=real_evaluation.json``.  The script emits every
operating point plus the nondominated MASE/FLOPs frontier; it never chooses a
single GIFT-Eval-tuned deployment threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, Path(path)


def is_dominated(point: dict, candidates: list[dict]) -> bool:
    for other in candidates:
        no_worse = (
            other["flops_saved_pct"] >= point["flops_saved_pct"]
            and other["geomean_mase_ratio"] <= point["geomean_mase_ratio"]
        )
        strict = (
            other["flops_saved_pct"] > point["flops_saved_pct"]
            or other["geomean_mase_ratio"] < point["geomean_mase_ratio"]
        )
        if no_worse and strict:
            return True
    return False


def summarize(runs: list[tuple[str, Path]], output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for family, path in runs:
        report = json.loads(path.read_text())
        for method, metrics in report["aggregate"].items():
            if not (
                method.startswith("dense_")
                or method.startswith("fixed_w")
                or method.startswith("horizon_x")
                or method in {
                    "full_native", "expected_risk_argmin",
                    "risk_argmin_no_abstain",
                }
            ):
                continue
            rows.append({
                "model": report["model"],
                "family": family,
                "method": method,
                "geomean_mase_ratio": float(metrics["geomean_cell_mase_ratio"]),
                "mase_change_pct": float(
                    100.0 * (metrics["geomean_cell_mase_ratio"] - 1.0)),
                "flops_saved_pct": float(metrics["theoretical_flops_saved_pct"]),
                "harm5_pct": float(100.0 * metrics["instance_harm5_rate"]),
                "coverage_pct": float(100.0 * metrics["coverage"]),
                "n_cells": int(metrics["n_cells"]),
                "n_instances": int(metrics["n_instances"]),
            })
    if not rows:
        raise RuntimeError("No dense or baseline points found")

    learned = [row for row in rows if row["method"].startswith("dense_")]
    baseline = [
        row for row in rows
        if row["method"].startswith(("fixed_w", "horizon_x"))
        or row["method"] == "full_native"
    ]
    for row in rows:
        comparison = learned if row in learned else baseline
        row["on_family_pareto"] = not is_dominated(row, comparison)
    union = learned + baseline
    for row in rows:
        row["on_union_pareto"] = not is_dominated(row, union)

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "all_operating_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pareto_rows = [row for row in rows if row["on_union_pareto"]]
    with (output_dir / "pareto_operating_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pareto_rows)
    (output_dir / "summary.json").write_text(json.dumps({
        "n_points": len(rows),
        "n_union_pareto": len(pareto_rows),
        "families": sorted(set(row["family"] for row in rows)),
        "runs": {label: str(path) for label, path in runs},
    }, indent=2) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_labeled_path, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.run, args.output_dir)


if __name__ == "__main__":
    main()
