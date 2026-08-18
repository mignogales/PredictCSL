#!/usr/bin/env python3
"""Prepare synthetic-only dense score-component ablations for a frozen policy.

The output JSON files are consumed by ``evaluate_context_risk_profile_override``.
No GIFT-Eval inputs or outcomes are read here: score definitions and every
threshold are selected on the same held-out synthetic split used by the source
policy.  This makes the subsequent real-data comparison a final evaluation,
not a test-tuned threshold sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import calibrated_context_risk as risk
from experiments import distill_calibrated_context_risk as distill


def _dense_selected_family(candidates: list[dict], profiles: dict) -> dict:
    selected = profiles["balanced"]["config"]
    rows = [
        row for row in candidates
        if np.isclose(
            row["config"]["uncertainty_weight"],
            selected["uncertainty_weight"],
        )
        and np.isclose(
            row["config"]["harm_weight"], selected["harm_weight"])
    ]
    rows.sort(key=lambda row: float(row["config"]["threshold"]))
    return {f"dense_{index:03d}": row for index, row in enumerate(rows)}


def prepare(args: argparse.Namespace) -> dict:
    bundle = joblib.load(args.policy)
    data_args = argparse.Namespace(
        synthetic_dir=args.synthetic_dir,
        model_short=args.model_short,
        seed=args.seed,
        max_train_series=args.max_train_series,
        max_val_series=args.max_val_series,
        train_pairs=args.train_pairs,
    )
    data = distill._prepare_data(data_args)
    mean, std, harm = risk.predict_risk(
        bundle["regressor"], bundle["classifier"], data["val_features"],
        args.prediction_chunk,
    )
    shape = (len(data["validation_errors"]), len(data["windows"]))
    mean = mean.reshape(shape)[:, :-1]
    std = std.reshape(shape)[:, :-1]
    harm = harm.reshape(shape)[:, :-1]
    native_context = data["val_lengths"][data["task_series"]]

    # Keep score-component selection separate from threshold calibration even
    # for legacy frozen policies. Tasks are grouped by series, so no horizon of
    # one series can leak across the split.
    unique_series = np.unique(data["task_series"])
    split = max(1, len(unique_series) // 2)
    selection_series = set(unique_series[:split].tolist())
    selection_mask = np.asarray([
        int(value) in selection_series for value in data["task_series"]])
    calibration_mask = ~selection_mask

    variants = {
        "mean_only": ([0.0], [0.0]),
        "mean_plus_uncertainty": (args.uncertainty_weights, [0.0]),
        "mean_plus_harm": ([0.0], args.harm_weights),
        "full_score": (args.uncertainty_weights, args.harm_weights),
    }

    output_root = Path(args.output_root)
    summary = {
        "model": args.model_short,
        "policy": args.policy,
        "synthetic_dir": args.synthetic_dir,
        "seed": int(args.seed),
        "real_labels_used": False,
        "variants": {},
    }
    for name, (uncertainty_weights, harm_weights) in variants.items():
        selection_profiles, _ = risk._calibrate_profiles(
            mean[selection_mask], std[selection_mask], harm[selection_mask],
            data["validation_errors"][selection_mask], data["windows"],
            native_context[selection_mask], uncertainty_weights, harm_weights,
            args.dense_points,
        )
        selected = selection_profiles["balanced"]["config"]
        profiles, candidates = risk._calibrate_profiles(
            mean[calibration_mask], std[calibration_mask], harm[calibration_mask],
            data["validation_errors"][calibration_mask], data["windows"],
            native_context[calibration_mask],
            [float(selected["uncertainty_weight"])],
            [float(selected["harm_weight"])], args.dense_points,
        )
        all_profiles = dict(profiles)
        all_profiles.update(_dense_selected_family(candidates, profiles))
        payload = {
            "method": "synthetic-only calibrated risk-score component ablation",
            "variant": name,
            "model": args.model_short,
            "source_policy": args.policy,
            "real_labels_used_for_training_or_calibration": False,
            "profiles": all_profiles,
        }
        variant_dir = output_root / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        path = variant_dir / "synthetic_calibration.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        summary["variants"][name] = {
            "profiles_json": str(path),
            "selected_score": profiles["balanced"]["config"],
            "selection_series": int(selection_mask.sum() // len(data["horizons"])),
            "calibration_series": int(calibration_mask.sum() // len(data["horizons"])),
            "n_dense_points": sum(key.startswith("dense_") for key in all_profiles),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "ablation_profile_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--model-short", required=True)
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    # _prepare_data also materializes unused training-pair features. One row is
    # sufficient here because this script only needs the deterministic held-out
    # validation indices and grid.
    parser.add_argument("--max-train-series", type=int, default=1)
    parser.add_argument("--max-val-series", type=int, default=2000)
    parser.add_argument("--train-pairs", type=int, default=1)
    parser.add_argument("--dense-points", type=int, default=101)
    parser.add_argument("--prediction-chunk", type=int, default=32768)
    parser.add_argument(
        "--uncertainty-weights", nargs="+", type=float,
        default=[0.0, 0.5, 1.0, 1.645])
    parser.add_argument(
        "--harm-weights", nargs="+", type=float,
        default=[0.0, 0.1, 0.25, 0.5])
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
