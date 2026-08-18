#!/usr/bin/env python3
"""Evaluate a frozen policy with externally calibrated synthetic profiles."""

import argparse
import json
from pathlib import Path

import joblib

from experiments import calibrated_context_risk as base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--profiles-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-short", required=True)
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--cache-roots", nargs="+", required=True)
    parser.add_argument("--metric", default="mase_gluonts_real")
    parser.add_argument("--prediction-chunk", type=int, default=32768)
    parser.add_argument("--prediction-row-batch", type=int, default=256)
    parser.add_argument("--max-real-cells", type=int, default=0)
    parser.add_argument("--max-real-instances-per-cell", type=int, default=0)
    parser.add_argument("--fixed-baselines", action="store_true")
    parser.add_argument(
        "--fixed-windows", nargs="+", type=int,
        default=[32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192])
    parser.add_argument(
        "--horizon-multiples", nargs="+", type=float,
        default=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    original_load = joblib.load
    bundle = original_load(args.policy)
    calibration = json.loads(Path(args.profiles_json).read_text())
    bundle["profiles"] = calibration["profiles"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # The established evaluator obtains its bundle from output/policy.joblib.
    # Intercept only that exact load and leave all other joblib operations intact.
    expected = output / "policy.joblib"
    def load_with_override(path, *load_args, **load_kwargs):
        if Path(path) == expected:
            return bundle
        return original_load(path, *load_args, **load_kwargs)

    joblib.load = load_with_override
    try:
        base.evaluate_real(args)
    finally:
        joblib.load = original_load


if __name__ == "__main__":
    main()
