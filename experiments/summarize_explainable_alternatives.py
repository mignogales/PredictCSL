#!/usr/bin/env python3
"""Summarize validation-selected explainable candidates across models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(root: Path, output: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.glob("*/explainable_alternatives_screen.json")):
        report = json.loads(path.read_text())
        best = report["best_candidate"]
        pareto = best["validation_pareto"]
        row = {
            "model": report["model"],
            "slug": path.parent.name,
            "candidate": best["candidate"],
            "type": best["type"],
            "artifact_bytes": best["artifact_bytes"],
            "mean_context_saved_across_budgets_pct": pareto[
                "mean_context_saved_across_budgets_pct"],
        }
        row.update(pareto["context_saved_by_mase_budget_pct"])
        rows.append(row)
    if not rows:
        raise ValueError(f"No completed screens found under {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarize(args.root, args.output)
