#!/usr/bin/env python3
"""Measure counterfactual variable-grid backbone speed for fixed-grid TSFMs.

This diagnostic deliberately bypasses the public PatchTST-FM and TiRex2
padding layers.  The resulting outputs are not valid forecasts; only GPU
execution time is used.  Policy rollups therefore represent an optimistic
smart-batching upper bound, not an achievable GIFT-Eval result.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from experiments.test_window_ablation_gifteval_v5 import load_handle


METHODS = ("balanced", "efficiency", "max_efficiency")
MODEL_SPECS = {
    "PatchTST-FM-R1": ("ibm-research/patchtst-fm-r1", "patchtst_fm"),
    "TiRex2": ("NX-AI/TiRex-2-gifteval-zs", "tirex"),
}


def execution_shape(model: str, window: int, horizon: int) -> tuple[int, int]:
    """Return ``(tokens, repeated backbone calls)`` for the direct benchmark."""
    if model == "PatchTST-FM-R1":
        # The checkpoint requests at least 8 masked patches (128 points) and
        # limits the total context+forecast grid to 8192 points.
        forecast_grid = max(int(horizon), 128)
        total_points = min(8192, int(window) + forecast_grid)
        return math.ceil(total_points / 16), 1
    if model == "TiRex2":
        # Keep TiRex's complete 928-point future grid and sign-flip TTA, while
        # removing only the left context padding. Horizons >928 roll forward.
        tokens = math.ceil((int(window) + 928) / 32)
        return tokens, 2 * max(1, math.ceil(int(horizon) / 928))
    raise ValueError(f"Unsupported model: {model}")


def requested_shapes(histogram: pd.DataFrame, model: str) -> list[tuple[int, int]]:
    frame = histogram[histogram["method"].astype(str).isin(
        ("full_native",) + METHODS)]
    return sorted({
        execution_shape(model, int(row.window_size), int(row.horizon))
        for row in frame.itertuples(index=False)
    })


def _patchtst_forward(handle, tokens: int, batch_size: int) -> Callable[[], None]:
    patch = int(handle.config.d_patch)
    values = torch.randn(
        batch_size, tokens, patch, device="cuda", dtype=torch.float32)
    missing = torch.zeros_like(values)
    pad = torch.zeros(batch_size, tokens, device="cuda", dtype=torch.bool)

    def forward() -> None:
        with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16):
            output, _raw = handle.backbone.decode(
                values, mask=missing, t_pad_mask=pad)
            del output, _raw

    return forward


def _tirex_forward(handle, tokens: int, calls: int,
                   batch_size: int) -> Callable[[], None]:
    model = handle.model
    total_points = tokens * int(model.input_patch_size)
    future = int(model.future_len)
    context = max(1, total_points - future)
    past = torch.randn(batch_size, context, device="cuda")
    future_nan = torch.full(
        (batch_size, future), float("nan"), device="cuda")
    values = torch.cat((past, future_nan), dim=-1)
    flipped = torch.cat((-past, future_nan), dim=-1)
    group = torch.arange(batch_size, device="cuda", dtype=torch.float32)
    target = torch.ones(batch_size, device="cuda", dtype=torch.bool)
    batch = {"x": values, "group_vector": group, "target_mask": target}
    flipped_batch = {
        "x": flipped, "group_vector": group, "target_mask": target}

    def forward() -> None:
        with torch.inference_mode():
            for call in range(calls):
                output = model(batch if call % 2 == 0 else flipped_batch)
                del output

    return forward


def measure_shape(model: str, handle, tokens: int, calls: int,
                  requested_batch: int, warmup: int,
                  repeats: int) -> dict[str, float | int | str]:
    batch_size = int(requested_batch)
    while True:
        try:
            maker = _patchtst_forward if model == "PatchTST-FM-R1" else None
            forward = (maker(handle, tokens, batch_size) if maker else
                       _tirex_forward(handle, tokens, calls, batch_size))
            for _ in range(warmup):
                forward()
            torch.cuda.synchronize()
            samples = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                forward()
                end.record()
                torch.cuda.synchronize()
                samples.append(float(start.elapsed_time(end)))
            return {
                "model": model,
                "tokens": int(tokens),
                "backbone_calls": int(calls),
                "requested_batch_size": int(requested_batch),
                "effective_batch_size": int(batch_size),
                "warmup": int(warmup),
                "repeats": int(repeats),
                "median_ms": float(np.median(samples)),
                "mean_ms": float(np.mean(samples)),
                "std_ms": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
                "min_ms": float(np.min(samples)),
                "status": "complete",
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)


def rollup(histogram: pd.DataFrame, profile: pd.DataFrame,
           model: str) -> pd.DataFrame:
    profile_key = profile[[
        "tokens", "backbone_calls", "effective_batch_size", "median_ms"]]
    rows = []
    totals: dict[str, float] = {}
    for method in ("full_native",) + METHODS:
        frame = histogram[histogram["method"].astype(str) == method].copy()
        shapes = frame.apply(
            lambda row: execution_shape(
                model, int(row["window_size"]), int(row["horizon"])), axis=1)
        frame["tokens"] = [value[0] for value in shapes]
        frame["backbone_calls"] = [value[1] for value in shapes]
        counts = frame.groupby(
            ["tokens", "backbone_calls"], as_index=False)["n_instances"].sum()
        joined = counts.merge(
            profile_key, on=["tokens", "backbone_calls"],
            how="left", validate="one_to_one")
        if joined["median_ms"].isna().any():
            raise RuntimeError(f"Missing measured shapes for {model}/{method}")
        batches = np.ceil(
            joined["n_instances"].to_numpy(float)
            / joined["effective_batch_size"].to_numpy(float))
        totals[method] = float(np.sum(
            batches * joined["median_ms"].to_numpy(float)) / 1000.0)
    native = totals["full_native"]
    for method in METHODS:
        policy = totals[method]
        rows.append({
            "model": model,
            "method": method,
            "timing_scope": "counterfactual_unpadded_backbone",
            "native_s": native,
            "policy_s": policy,
            "speedup_x": native / max(policy, 1e-12),
            "time_saved_pct": 100.0 * (1.0 - policy / max(native, 1e-12)),
            "valid_forecast": False,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--histogram", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    histogram = pd.read_csv(args.histogram)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_dir / f"{args.model}_unpadded_profile.csv"
    rows = (pd.read_csv(profile_path).to_dict(orient="records")
            if profile_path.exists() else [])
    done = {
        (int(row["tokens"]), int(row["backbone_calls"])) for row in rows
        if row.get("status") == "complete"
    }
    model_id, family = MODEL_SPECS[args.model]
    handle = load_handle(family, model_id, "cuda")
    try:
        for tokens, calls in requested_shapes(histogram, args.model):
            if (tokens, calls) in done:
                continue
            row = measure_shape(
                args.model, handle, tokens, calls, args.batch_size,
                args.warmup, args.repeats)
            rows.append(row)
            pd.DataFrame(rows).sort_values(
                ["tokens", "backbone_calls"]).to_csv(profile_path, index=False)
            print(
                f"{args.model} tokens={tokens} calls={calls} "
                f"B={row['effective_batch_size']} median={row['median_ms']:.3f}ms",
                flush=True)
    finally:
        del handle
        torch.cuda.empty_cache()

    profile = pd.read_csv(profile_path)
    summary = rollup(histogram, profile, args.model)
    summary_path = args.output_dir / f"{args.model}_unpadded_policy_speedups.csv"
    summary.to_csv(summary_path, index=False)
    metadata = {
        "model": args.model,
        "schema_version": 1,
        "methodology": "counterfactual direct backbone with public fixed-grid padding bypassed",
        "valid_forecast": False,
        "requested_batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "n_shapes": int(len(profile)),
        "summary": summary.to_dict(orient="records"),
    }
    (args.output_dir / f"{args.model}_unpadded_policy_speedups.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
