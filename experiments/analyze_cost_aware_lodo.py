"""Leave-one-dataset-out calibration of cost-aware score tolerances."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _aggregate(rows: list[dict[str, str]]) -> tuple[float, float]:
    ratios = [float(row["normalized_mase"]) for row in rows]
    mase = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    selected = sum(float(row["selected_flops"]) for row in rows)
    native = sum(float(row["full_native_flops"]) for row in rows)
    return mase, 100.0 * (1.0 - selected / native)


def run(input_path: Path, output_path: Path, budgets: list[float]) -> None:
    with input_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if float(row["alpha"]) == 0.0:
            by_model[row["model"]].append(row)

    output_rows: list[dict[str, object]] = []
    for model, model_rows in sorted(by_model.items()):
        displays = sorted({row["dataset_display"] for row in model_rows})
        policies = sorted({
            row["fallback_policy"] for row in model_rows
            if row["fallback_policy"].startswith("score_tolerance_")
        })
        baseline_rows = [
            row for row in model_rows
            if row["fallback_policy"] == "feasible_backoff"
        ]
        baseline_mase, baseline_saving = _aggregate(baseline_rows)

        for budget in budgets:
            heldout_rows: list[dict[str, str]] = []
            choices: Counter[str] = Counter()
            for display in displays:
                train_base = [
                    row for row in baseline_rows
                    if row["dataset_display"] != display
                ]
                train_base_mase, _ = _aggregate(train_base)
                candidates: list[tuple[float, float, str]] = []
                for policy in policies:
                    policy_rows = [
                        row for row in model_rows
                        if row["fallback_policy"] == policy
                        and row["dataset_display"] != display
                    ]
                    mase, saving = _aggregate(policy_rows)
                    if mase <= train_base_mase * (1.0 + budget):
                        candidates.append((saving, -mase, policy))
                if not candidates:
                    policy = "score_tolerance_0"
                else:
                    policy = max(candidates)[2]
                choices[policy] += 1
                heldout_rows.extend(
                    row for row in model_rows
                    if row["fallback_policy"] == policy
                    and row["dataset_display"] == display
                )

            lodo_mase, lodo_saving = _aggregate(heldout_rows)
            output_rows.append({
                "model": model,
                "train_mase_budget_pct": 100.0 * budget,
                "alpha0_mase": baseline_mase,
                "lodo_mase": lodo_mase,
                "relative_mase_change_pct": 100.0 * (
                    lodo_mase / baseline_mase - 1.0),
                "alpha0_flops_saved_pct": baseline_saving,
                "lodo_flops_saved_pct": lodo_saving,
                "saving_change_pp": lodo_saving - baseline_saving,
                "selected_policy_counts": json.dumps(
                    dict(sorted(choices.items()))),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budgets", nargs="+", type=float,
        default=[0.001, 0.0025, 0.005],
        help="Allowed relative train MASE increases, expressed as fractions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output, args.budgets)
