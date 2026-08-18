#!/usr/bin/env python3
"""Evaluate the real/synthetic bounded selector on untouched GIFT-Eval tests."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gift_eval.data import Dataset as GiftEvalDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import datasets_config
from experiments.evaluate_instance_windows import (
    _cell_metric_path,
    _ground_tree,
    _load_vector,
    discover_cells,
)
from experiments.predict_context_length import MambaContextLength
from experiments.test_window_ablation_gifteval_v5 import (
    GiftEvalCache,
    _closest_horizon_idx,
    predict_curves_for_dataset,
)


def summarize(frame):
    native = frame["native_mase"].to_numpy(float)
    out = {}
    for method in ("raw", "gated", "oracle_three"):
        value = frame[f"{method}_mase"].to_numpy(float)
        valid = np.isfinite(value) & np.isfinite(native) & (native > 0) & (value > 0)
        ratio = value[valid] / native[valid]
        out[method] = {
            "n_cells": int(valid.sum()),
            "macro_mean_mase": float(np.mean(value[valid])),
            "macro_mean_relative_change_pct": float(100 * np.mean(ratio - 1)),
            "macro_geomean_relative_change_pct": float(
                100 * np.expm1(np.mean(np.log(ratio)))),
            "cell_win_rate": float(np.mean(ratio < 1)),
            "instance_native_action_rate": float(
                frame.loc[valid, f"{method}_native_rate"].mean()),
        }
    return out


def weighted_mean(values, counts):
    valid = np.isfinite(values) & np.isfinite(counts) & (counts > 0)
    return (float(np.average(values[valid], weights=counts[valid]))
            if valid.any() else float("nan"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", default="logs/experiments/window_ablation_gifteval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.005)
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["model_config"]
    windows = np.asarray(checkpoint["windows"], dtype=np.int64)
    horizons = list(checkpoint["horizons"])
    model = MambaContextLength(
        cfg["context_length"], cfg["patch_length"], cfg["d_model"],
        cfg["num_hidden_layers"], cfg["d_state"], cfg["d_conv"], cfg["expand"],
        cfg["dropout"], cfg["mask_ratio"], len(windows), len(horizons),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()

    specs = {(display, str(term)): (name, to_uni)
             for name, term, display, to_uni in datasets_config.datasets_to_run()}
    cells = discover_cells(args.ablation_root, ["Chronos2-Small"])
    ground = _ground_tree(args.ablation_root)
    rows = []
    for number, cell in enumerate(cells, 1):
        spec = specs.get((cell.dataset, str(cell.term)))
        if spec is None:
            continue
        ge_name, to_univariate = spec
        try:
            cache = GiftEvalCache(GiftEvalDataset(
                name=ge_name, term=cell.term, to_univariate=to_univariate), cell.dataset)
        except Exception as exc:
            print(f"skip {cell.dataset}/{cell.term}: {exc}", flush=True)
            continue
        n = cache.n_total
        with np.load(cell.anchor_npz) as anchor:
            expected = int(anchor["predicted_curves"].shape[0])
            aggregate_key = next(
                (key for key in ("real_curve_gluonts_real", "real_curve_gluonts",
                                 "real_curve") if key in anchor.files), None)
            aggregate_curve = (np.asarray(anchor[aggregate_key], dtype=np.float64)
                               if aggregate_key is not None
                               else np.full(len(windows), np.nan))
            finite_aggregate = np.flatnonzero(np.isfinite(aggregate_curve))
            aggregate_native = (float(aggregate_curve[finite_aggregate[-1]])
                                if finite_aggregate.size else float("nan"))
        if n != expected:
            print(f"skip {cell.dataset}/{cell.term}: n={n}, cache={expected}", flush=True)
            continue
        h_idx = _closest_horizon_idx(cache.horizon, horizons)
        prediction = predict_curves_for_dataset(
            model, cache, cfg["context_length"], h_idx, args.device,
            training_objective="risk", batch_size=128)
        error = np.full((n, len(windows)), np.nan)
        counts = np.zeros_like(error)
        for j, window in enumerate(windows):
            error[:, j], counts[:, j], _ = _load_vector(
                _cell_metric_path(ground, cell, int(window)), n, "mase_gluonts_real")
        native, native_counts, _ = _load_vector(
            _cell_metric_path(ground, cell, "full_native"), n, "mase_gluonts_real")
        # Chronos-2 Small's native cap is 8192. Older caches often omit a
        # separate full_native per-sample file because w8192 is identical.
        missing_native = (~np.isfinite(native) | ~np.isfinite(native_counts)
                          | (native_counts <= 0))
        eligible = np.isfinite(error) & np.isfinite(counts) & (counts > 0)
        grid_index = np.arange(len(windows))[None, :]
        last_available = np.where(eligible, grid_index, -1).max(axis=1)
        can_fill = missing_native & (last_available >= 0)
        native[can_fill] = error[np.arange(n)[can_fill], last_available[can_fill]]
        native_counts[can_fill] = counts[
            np.arange(n)[can_fill], last_available[can_fill]]
        raw_action = prediction.argmin(1)
        predicted_gain = prediction[:, -1] - prediction[np.arange(n), raw_action]
        gated_action = np.where(predicted_gain >= args.threshold, raw_action, len(windows))

        candidates = np.column_stack([error, native])
        candidate_counts = np.column_stack([counts, native_counts])
        candidates[~np.isfinite(candidates)] = np.inf
        oracle_action = candidates.argmin(1)
        methods = {"raw": raw_action, "gated": gated_action, "oracle_three": oracle_action}
        native_mase = weighted_mean(native, native_counts)
        if not np.isfinite(native_mase):
            native_mase = aggregate_native
        record = {"dataset": cell.dataset, "term": cell.term, "horizon": cache.horizon,
                  "n_instances": n, "native_mase": native_mase}
        for name, action in methods.items():
            # A missing cached action falls back to native, never another tuned action.
            chosen = candidates[np.arange(n), action]
            chosen_counts = candidate_counts[np.arange(n), action]
            missing = (~np.isfinite(chosen) | ~np.isfinite(chosen_counts)
                       | (chosen_counts <= 0))
            chosen[missing] = native[missing]
            chosen_counts[missing] = native_counts[missing]
            method_mase = weighted_mean(chosen, chosen_counts)
            record[f"{name}_mase"] = (
                method_mase if np.isfinite(method_mase) else native_mase)
            record[f"{name}_native_rate"] = float(np.mean(
                (action >= len(windows) - 1) | missing))
            record[f"{name}_counts"] = json.dumps(
                np.bincount(np.minimum(action, len(windows)), minlength=len(windows)+1).tolist())
        rows.append(record)
        print(f"[{number}/{len(cells)}] {cell.dataset}/{cell.term}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "cell_results.csv", index=False)
    report = {
        "checkpoint": args.checkpoint,
        "threshold_fixed_before_test": args.threshold,
        "windows": windows.tolist(),
        "horizons": horizons,
        "official_test_used_for_tuning": False,
        "summary": summarize(frame),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
