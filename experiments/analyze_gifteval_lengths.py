"""Summarize GiftEval context lengths and predictor-gate coverage.

Run in an environment with GiftEval installed and ``GIFT_EVAL`` configured::

    python -m experiments.analyze_gifteval_lengths

The script only loads data; it performs no model inference. It writes one row
per GiftEval cell to ``gifteval_length_summary.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from experiments import datasets_config


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="logs/experiments/gifteval_length_analysis")
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
    if failures:
        pd.DataFrame(failures).to_csv(
            output_dir / "gifteval_length_failures.csv", index=False)

    print(f"\nWrote analysis to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
