#!/usr/bin/env python3
"""Benchmark mask-safe TimesFM batching against the old exact-bucket path.

The benchmark uses identical model weights and inputs for both paths, checks
that every forecast is finite, and reports the maximum absolute forecast
difference. On the server, omit ``--random-weights`` to use the configured
TimesFM 2.5 checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from dotenv import load_dotenv

from experiments.timesfm_gifteval import (
    forecast_quantiles,
    load_model,
    model_context,
)


MODEL_ID = "google/timesfm-2.5-200m-pytorch"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


def _parse_lengths(value: str) -> list[int]:
    lengths = [int(part) for part in value.split(",") if part.strip()]
    if not lengths or min(lengths) < 1:
        raise argparse.ArgumentTypeError("lengths must be positive integers")
    return lengths


def _parse_nonnegative_ints(value: str) -> list[int]:
    values = [int(part) for part in value.split(",") if part.strip()]
    if not values or min(values) < 0:
        raise argparse.ArgumentTypeError("values must be nonnegative integers")
    return values


def _sync(model) -> None:
    if getattr(model.model, "device", torch.device("cpu")).type == "cuda":
        torch.cuda.synchronize()


def _timed(model, fn):
    calls = 0
    original_forecast = model.forecast

    def counted_forecast(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forecast(*args, **kwargs)

    model.forecast = counted_forecast
    _sync(model)
    started = time.perf_counter()
    try:
        output = fn()
        _sync(model)
        elapsed = time.perf_counter() - started
    finally:
        model.forecast = original_forecast
    return output, elapsed, calls


def _old_exact_bucket_forecast(
    model,
    contexts: list[np.ndarray],
    prediction_length: int,
    batch_size: int,
) -> np.ndarray:
    """Reproduce recipe-v4's eager exact patch-width bucketing."""
    patch = int(model.model.p)
    buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
    for row, raw_context in enumerate(contexts):
        context = model_context(raw_context)
        if context.size and np.isnan(context[0]):
            valid = np.flatnonzero(~np.isnan(context))
            if valid.size:
                context = context[int(valid[0]):]
        compiled = ((len(context) + patch - 1) // patch) * patch
        buckets.setdefault(compiled, []).append((row, context))

    outputs = np.empty(
        (len(contexts), 9, prediction_length), dtype=np.float32)
    for bucket in buckets.values():
        for start in range(0, len(bucket), batch_size):
            chunk = bucket[start:start + batch_size]
            rows = np.asarray([row for row, _ in chunk], dtype=np.int64)
            values = [context for _, context in chunk]
            outputs[rows] = forecast_quantiles(
                model,
                values,
                prediction_length,
                batch_size=len(values),
                safe_variable_length_batching=False,
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=_parse_lengths,
        default=_parse_lengths("64,96,65,95"),
        help="Comma-separated variable context lengths")
    parser.add_argument("--prediction-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--leading-nans",
        type=_parse_nonnegative_ints,
        help=("Comma-separated leading-NaN counts, one per context; exercises "
              "the missing-value preprocessing path"))
    parser.add_argument(
        "--leading-nan-ladder", type=int,
        help=("Generate one fixed-width context per 32-point effective length, "
              "from 32 through this value"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--random-weights", action="store_true",
        help="Instantiate the real architecture without loading a checkpoint")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.leading_nan_ladder is not None:
        if args.leading_nan_ladder < 32 or args.leading_nan_ladder % 32:
            parser.error("--leading-nan-ladder must be a positive multiple of 32")
        if args.leading_nans is not None:
            parser.error("--leading-nan-ladder and --leading-nans are exclusive")
        args.lengths = [
            args.leading_nan_ladder
            for _ in range(args.leading_nan_ladder // 32)
        ]
        args.leading_nans = list(range(
            args.leading_nan_ladder - 32, -1, -32))
    if args.leading_nans is not None:
        if len(args.leading_nans) != len(args.lengths):
            parser.error("--leading-nans must have one value per --lengths entry")
        if any(nans >= length for nans, length in zip(
                args.leading_nans, args.lengths)):
            parser.error("each leading-NaN count must be smaller than its length")

    if args.random_weights:
        from timesfm.timesfm_2p5.timesfm_2p5_torch import (
            TimesFM_2p5_200M_torch,
        )
        model = TimesFM_2p5_200M_torch(torch_compile=False)
        model.model.eval()
    else:
        model = load_model(MODEL_ID)

    rng = np.random.RandomState(args.seed)
    contexts = [
        rng.standard_normal(length).astype(np.float32)
        for length in args.lengths
    ]
    if args.leading_nans is not None:
        for context, nan_count in zip(contexts, args.leading_nans):
            context[:nan_count] = np.nan

    old_times = []
    new_times = []
    old_call_counts = []
    new_call_counts = []
    all_old_finite = True
    all_new_finite = True
    max_difference = 0.0
    with torch.inference_mode():
        for repeat in range(args.repeats):
            def run_old():
                return _old_exact_bucket_forecast(
                    model, contexts, args.prediction_length, args.batch_size)

            def run_new():
                return forecast_quantiles(
                    model, contexts, args.prediction_length,
                    batch_size=args.batch_size)

            if repeat % 2 == 0:
                old_forecasts, old_seconds, old_calls = _timed(model, run_old)
                new_forecasts, new_seconds, new_calls = _timed(model, run_new)
            else:
                new_forecasts, new_seconds, new_calls = _timed(model, run_new)
                old_forecasts, old_seconds, old_calls = _timed(model, run_old)

            old_times.append(old_seconds)
            new_times.append(new_seconds)
            old_call_counts.append(old_calls)
            new_call_counts.append(new_calls)
            all_old_finite &= bool(np.isfinite(old_forecasts).all())
            all_new_finite &= bool(np.isfinite(new_forecasts).all())
            max_difference = max(
                max_difference,
                float(np.max(np.abs(old_forecasts - new_forecasts))),
            )

    old_median = statistics.median(old_times)
    new_median = statistics.median(new_times)
    report = {
        "device": str(model.model.device),
        "random_weights": bool(args.random_weights),
        "series": len(contexts),
        "lengths": args.lengths,
        "leading_nans": args.leading_nans,
        "repeats": args.repeats,
        "old_model_batches": old_call_counts,
        "new_model_batches": new_call_counts,
        "old_seconds": old_times,
        "new_seconds": new_times,
        "old_median_seconds": old_median,
        "new_median_seconds": new_median,
        "median_speedup": old_median / new_median,
        "old_all_finite": all_old_finite,
        "new_all_finite": all_new_finite,
        "max_abs_forecast_difference": max_difference,
    }
    print(json.dumps(report, indent=2))
    if not report["old_all_finite"] or not report["new_all_finite"]:
        raise SystemExit("non-finite TimesFM forecast detected")


if __name__ == "__main__":
    main()
