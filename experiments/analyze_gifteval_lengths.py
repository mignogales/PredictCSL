"""Summarize GiftEval context lengths and predictor-gate coverage.

Run in an environment with GiftEval installed and ``GIFT_EVAL`` configured::

    python -m experiments.analyze_gifteval_lengths

The script only loads data; it performs no model inference.  It writes one row
per GiftEval cell to ``gifteval_length_summary.csv`` and a compact threshold
coverage table to ``gifteval_threshold_coverage.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from experiments import datasets_config


DEFAULT_THRESHOLDS = (256, 512, 768, 1024, 1536, 2048)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def summarize_lengths(
    lengths: Sequence[int], *, ge_name: str, term: str, display: str
) -> dict:
    """Return stable, CSV-friendly summary statistics for one dataset cell."""
    x = np.asarray(lengths, dtype=np.int64)
    if x.size == 0:
        raise ValueError(f"No valid contexts for {display}/{term}")
    return {
        "ge_name": ge_name,
        "dataset_display": display,
        "term": term,
        "n_instances": int(x.size),
        "min": int(x.min()),
        "p10": _percentile(x, 10),
        "p25": _percentile(x, 25),
        "median": _percentile(x, 50),
        "mean": float(x.mean()),
        "p75": _percentile(x, 75),
        "p90": _percentile(x, 90),
        "max": int(x.max()),
    }


def threshold_coverage(
    cells: Iterable[tuple[dict, np.ndarray]], thresholds: Sequence[int]
) -> pd.DataFrame:
    """Compute cell-level and instance-level coverage for candidate gates.

    ``cell_mean_gt`` matches the aggregate strategy's eligibility rule before
    model-specific context caps are applied. ``instance_gt`` answers the more
    natural row-wise question used by a per-instance policy.
    """
    materialized = list(cells)
    total_instances = sum(len(x) for _, x in materialized)
    rows = []
    for threshold in thresholds:
        eligible_cells = sum(float(x.mean()) > threshold for _, x in materialized)
        eligible_instances = sum(int((x > threshold).sum()) for _, x in materialized)
        rows.append({
            "threshold": int(threshold),
            "cells_mean_gt": int(eligible_cells),
            "cells_total": len(materialized),
            "cells_mean_gt_pct": 100.0 * eligible_cells / max(len(materialized), 1),
            "instances_gt": eligible_instances,
            "instances_total": total_instances,
            "instances_gt_pct": 100.0 * eligible_instances / max(total_instances, 1),
        })
    return pd.DataFrame(rows)


def _parse_thresholds(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(v.strip()) for v in value.split(",") if v.strip()}))
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("thresholds must be positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="logs/experiments/gifteval_length_analysis")
    parser.add_argument(
        "--thresholds", type=_parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated candidate lower gates (default: %(default)s)")
    parser.add_argument(
        "--catalog", action="store_true",
        help="Analyze the full catalog rather than only run=True cells")
    args = parser.parse_args()

    # GiftEval resolves its data root from this variable at construction time.
    load_dotenv()
    # Import after argument parsing so --help remains usable in lightweight/local
    # environments.
    from gift_eval.data import Dataset as GiftEvalDataset

    specs = datasets_config.catalog() if args.catalog else datasets_config.datasets_to_run()
    summaries = []
    cells: list[tuple[dict, np.ndarray]] = []
    failures = []
    for ge_name, term, display, to_univariate in specs:
        try:
            dataset = GiftEvalDataset(
                name=ge_name, term=term, to_univariate=to_univariate)
            horizon = int(dataset.prediction_length)
            lengths = np.asarray([
                len(test_input["target"])
                for test_input, test_label in dataset.test_data
                if len(test_label["target"]) >= horizon
            ], dtype=np.int64)
            summary = summarize_lengths(
                lengths, ge_name=ge_name, term=term, display=display)
            summary["horizon"] = horizon
            summary["frequency"] = str(dataset.freq)
            summaries.append(summary)
            cells.append((summary, lengths))
            print(
                f"{display:22s} {term:6s} n={len(lengths):6d} "
                f"min/median/mean/max={lengths.min()}/{np.median(lengths):.0f}/"
                f"{lengths.mean():.0f}/{lengths.max()}")
        except Exception as exc:  # catalog contains configs some versions reject
            failures.append({"ge_name": ge_name, "term": term, "error": str(exc)})
            print(f"SKIP {ge_name}/{term}: {exc}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(
        output_dir / "gifteval_length_summary.csv", index=False)
    coverage = threshold_coverage(cells, args.thresholds)
    coverage.to_csv(output_dir / "gifteval_threshold_coverage.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(
            output_dir / "gifteval_length_failures.csv", index=False)

    print("\nThreshold coverage")
    print(coverage.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\nWrote analysis to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
