#!/usr/bin/env python3
"""Freeze the completed context-interpretability evidence used in Section 4.

Run this script in an environment with pandas/numpy and access to the server
``logs/experiments/context_interpretability`` tree.  It deliberately exports
compact, auditable tables rather than copying the multi-million-row raw cells.
Every consumed source file is hashed into ``source_inventory.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PERTURBATIONS = ("block_mean", "matched_block", "permutation", "noise")
FULL_STATS = "attention_mask/full_history_stats"
TAIL_STATS = "attention_mask/tail_matched_stats"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(rows: list[dict], path: Path, columns: list[str] | None = None) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(
            [c for c in ("model", "metric", "lag", "strength", "context_length",
                         "perturbation", "lookback_start", "layer_index")
             if c in frame.columns],
            kind="stable",
        )
    frame.to_csv(path, index=False, float_format="%.10g")


def sufficient_context(curve: pd.DataFrame, tolerance: float = 0.05) -> int:
    best = float(curve["mean_loss"].min())
    return int(curve.loc[curve["mean_loss"] <= (1 + tolerance) * best,
                         "context_length"].min())


def collapse_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "sample_id", "perturbation_type", "context_length", "block_index",
        "lookback_start", "lookback_end", "severity",
    ]
    keys = [key for key in keys if key in frame.columns]
    return frame.groupby(keys, dropna=False, as_index=False).agg(
        loss_delta=("loss_delta", "mean"),
        prediction_distance=("prediction_distance", "mean"),
    )


def ratio_interval(frame: pd.DataFrame, boundary: int, seed: int,
                   draws: int = 2000) -> tuple[float, float, float]:
    """Sample-bootstrap ratio of distant to recent mean absolute effects."""
    data = frame.copy()
    data["region"] = np.where(data["lookback_start"] < boundary,
                              "recent", "distant")
    data["abs_delta"] = data["loss_delta"].abs()
    paired = data.groupby(["sample_id", "region"])["abs_delta"].mean().unstack()
    paired = paired.dropna(subset=["recent", "distant"])
    if paired.empty or float(paired["recent"].mean()) == 0:
        return math.nan, math.nan, math.nan
    estimate = float(paired["distant"].mean() / paired["recent"].mean())
    values = paired[["recent", "distant"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    boot = values[indices].mean(axis=1)
    ratios = boot[:, 1] / np.maximum(boot[:, 0], np.finfo(float).tiny)
    low, high = np.quantile(ratios, [0.025, 0.975])
    return estimate, float(low), float(high)


def rank_spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return math.nan
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def model_dirs(root: Path) -> list[Path]:
    return sorted(
        path for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("_") and path.name != "figures"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path,
        default=Path("logs/experiments/context_interpretability"),
    )
    parser.add_argument("--output", type=Path, default=Path("section4_frozen"))
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()

    root = args.input_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "source_json"
    raw.mkdir(exist_ok=True)

    inventory: list[dict] = []
    model_scope: list[dict] = []
    error_curves: list[dict] = []
    perturbation_ratios: list[dict] = []
    perturbation_profiles: list[dict] = []
    lag_controls: list[dict] = []
    lag_summary: list[dict] = []
    decomposition_summary: list[dict] = []
    decomposition_profiles: list[dict] = []
    internal_summary: list[dict] = []
    patching_profiles: list[dict] = []

    def consume(path: Path, role: str, model: str) -> None:
        inventory.append({
            "model": model,
            "role": role,
            "path_relative_to_input_root": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    for model_dir in model_dirs(root):
        model = model_dir.name
        meta_path = model_dir / "run_meta.json"
        hyp_path = model_dir / "hypotheses_report.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta_path.exists():
            consume(meta_path, "run metadata", model)
            shutil.copy2(meta_path, raw / f"{model}__run_meta.json")
        if hyp_path.exists():
            consume(hyp_path, "pre-specified hypothesis output", model)
            shutil.copy2(hyp_path, raw / f"{model}__hypotheses_report.json")

        exp1_files = sorted(
            (model_dir / "exp1_perturbation" / "synthetic").glob("w*/results.csv"),
            key=lambda path: int(path.parent.name[1:]),
        )
        exp1_samples = 0
        exp1_contexts = 0
        exp1_max = math.nan
        exp1_sufficient = math.nan
        if exp1_files:
            curve_rows: list[dict] = []
            for path in exp1_files:
                consume(path, "Exp1 raw cell", model)
                frame = pd.read_csv(path)
                context = int(frame["context_length"].iloc[0])
                clean = frame.groupby("sample_id", as_index=False)["clean_loss"].first()
                curve_rows.append({
                    "context_length": context,
                    "mean_loss": float(clean["clean_loss"].mean()),
                    "se_loss": float(clean["clean_loss"].std(ddof=1) / np.sqrt(len(clean))),
                    "n_samples": int(clean["sample_id"].nunique()),
                })
            curve = pd.DataFrame(curve_rows).sort_values("context_length")
            exp1_sufficient = sufficient_context(curve)
            exp1_contexts = len(curve)
            exp1_max = int(curve["context_length"].max())
            exp1_samples = int(curve.loc[
                curve["context_length"] == exp1_max, "n_samples"].iloc[0])
            for row in curve.to_dict("records"):
                error_curves.append({"model": model, **row,
                                     "sufficient_context_5pct": exp1_sufficient})

            max_path = next(path for path in exp1_files
                            if int(path.parent.name[1:]) == exp1_max)
            max_frame = collapse_seeds(pd.read_csv(max_path))
            for pidx, perturbation in enumerate(PERTURBATIONS):
                subset = max_frame[max_frame["perturbation_type"] == perturbation]
                if subset.empty:
                    continue
                estimate, low, high = ratio_interval(
                    subset, int(exp1_sufficient), seed=4200 + pidx,
                    draws=args.bootstrap_draws,
                )
                perturbation_ratios.append({
                    "model": model,
                    "context_length": exp1_max,
                    "sufficient_context_5pct": exp1_sufficient,
                    "perturbation": perturbation,
                    "distant_recent_ratio": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_samples": int(subset["sample_id"].nunique()),
                })
                if perturbation == "block_mean":
                    subset = subset.assign(abs_delta=subset["loss_delta"].abs())
                    grouped = subset.groupby("lookback_start")["abs_delta"].agg(
                        ["mean", "std", "count"]).reset_index()
                    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
                    scale = float(grouped["mean"].max())
                    for row in grouped.to_dict("records"):
                        perturbation_profiles.append({
                            "model": model,
                            "context_length": exp1_max,
                            "sufficient_context_5pct": exp1_sufficient,
                            "lookback_start": int(row["lookback_start"]),
                            "mean_abs_loss_delta": float(row["mean"]),
                            "se_abs_loss_delta": float(row["se"]),
                            "normalized_mean_abs_loss_delta": (
                                float(row["mean"]) / scale if scale else math.nan),
                            "n_rows": int(row["count"]),
                        })

        controls_path = model_dir / "exp4_synthetic_controls" / "controls_summary.json"
        h4_complete = False
        if controls_path.exists():
            consume(controls_path, "Exp4 controls summary", model)
            shutil.copy2(controls_path, raw / f"{model}__controls_summary.json")
            controls = json.loads(controls_path.read_text()).get("controls", [])
            core = [
                row for row in controls
                if row.get("design_role", "core") == "core"
                and row.get("spec", {}).get("family") == "B"
                and float(row.get("spec", {}).get("noise", 0.1)) == 0.1
            ]
            combos = {
                (int(row["spec"]["distant_lag"]), float(row["spec"]["strength"]))
                for row in core
            }
            expected = {(lag, strength) for lag in (32, 64, 128, 256)
                        for strength in (0.0, 0.5, 1.0)}
            h4_complete = expected.issubset(combos)
            for row in controls:
                spec = row.get("spec", {})
                lag_controls.append({
                    "model": model,
                    "family": spec.get("family"),
                    "design_role": row.get("design_role", "core"),
                    "local_kind": spec.get("local_kind"),
                    "lag": spec.get("distant_lag"),
                    "strength": spec.get("strength"),
                    "noise": spec.get("noise"),
                    "sufficient_context": row.get("sufficient_context"),
                    "oracle_relative_gain": row.get("oracle", {}).get("relative_gain"),
                    "oracle_distant_predictive": row.get("oracle", {}).get(
                        "distant_predictive"),
                    "config_broken": bool(row.get("config_broken")),
                    "limitation_flag": bool(row.get("limitation_flag")),
                })
            if h4_complete:
                complete_core = [row for row in core if not row.get("config_broken")]
                strong = [row for row in complete_core
                          if float(row["spec"]["strength"]) >= 0.5]
                lags = [int(row["spec"]["distant_lag"]) for row in strong]
                contexts = [int(row["sufficient_context"]) for row in strong]
                zero = [int(row["sufficient_context"]) for row in complete_core
                        if float(row["spec"]["strength"]) == 0]
                local = [int(row["sufficient_context"]) for row in controls
                         if row.get("design_role") == "local_control"]
                lag_summary.append({
                    "model": model,
                    "spearman_lag_vs_sufficient": rank_spearman(lags, contexts),
                    "fraction_reaching_lag": float(np.mean(
                        [context >= lag for context, lag in zip(contexts, lags)])),
                    "n_strong_cells": len(strong),
                    "zero_min_context": min(zero),
                    "zero_max_context": max(zero),
                    "local_min_context": min(local) if local else math.nan,
                    "local_max_context": max(local) if local else math.nan,
                    "n_broken_controls": int(sum(bool(row.get("config_broken"))
                                                  for row in controls)),
                })

        exp7_files = sorted(
            (model_dir / "exp7_context_decomposition" / "synthetic").glob(
                "w*/results.csv"),
            key=lambda path: int(path.parent.name[1:]),
        )
        exp7_complete = False
        exp7_max = math.nan
        exp7_samples = 0
        if exp7_files:
            path = exp7_files[-1]
            consume(path, "Exp7 largest-width cell", model)
            frame = pd.read_csv(path)
            exp7_max = int(frame["context_length"].max())
            frame = frame[frame["context_length"] == exp7_max]
            exp7_samples = int(frame["sample_id"].nunique())
            variants = set(frame["perturbation_type"].dropna())
            exp7_complete = FULL_STATS in variants and TAIL_STATS in variants
            for (metric, variant), group in frame.groupby(
                    ["metric", "perturbation_type"]):
                decomposition_summary.append({
                    "model": model,
                    "context_length": exp7_max,
                    "metric": metric,
                    "variant": variant,
                    "mean_loss_delta": float(group["loss_delta"].mean()),
                    "mean_abs_loss_delta": float(group["loss_delta"].abs().mean()),
                    "mean_prediction_distance": float(
                        group["prediction_distance"].mean()),
                    "n_samples": int(group["sample_id"].nunique()),
                    "n_rows": len(group),
                })
                prof = group.groupby("lookback_start", as_index=False).agg(
                    mean_loss_delta=("loss_delta", "mean"),
                    mean_abs_loss_delta=("loss_delta", lambda values: values.abs().mean()),
                    se_loss_delta=("loss_delta", lambda values: values.std(ddof=1) /
                                   np.sqrt(len(values))),
                    n_rows=("loss_delta", "size"),
                )
                for row in prof.to_dict("records"):
                    decomposition_profiles.append({
                        "model": model,
                        "context_length": exp7_max,
                        "metric": metric,
                        "variant": variant,
                        "visible_suffix": int(row["lookback_start"]),
                        **{key: row[key] for key in (
                            "mean_loss_delta", "mean_abs_loss_delta",
                            "se_loss_delta", "n_rows")},
                    })

        lens_path = model_dir / "exp3_forecast_lens" / "synthetic" / \
            "saturation_layers.json"
        lens_status = "unsupported_or_not_run"
        lens_layers = math.nan
        lens_contexts = 0
        lens_saturated = 0
        lens_early = 0
        if lens_path.exists():
            consume(lens_path, "Exp3 saturation summary", model)
            lens = json.loads(lens_path.read_text())
            lens_layers = len(lens.get("layers", []))
            saturation = lens.get("saturation_layer_by_context", {})
            lens_contexts = len(saturation)
            lens_saturated = sum(int(value) >= 0 for value in saturation.values())
            lens_early = sum(0 <= int(value) < lens_layers - 1
                             for value in saturation.values())
            lens_status = "complete"

        patch_files = sorted(
            (model_dir / "exp2_activation_patching" / "synthetic" / "full").glob(
                "w*/results.csv"),
            key=lambda path: int(path.parent.name[1:]),
        )
        patch_status = "unsupported_or_not_run"
        patch_max = math.nan
        patch_samples = 0
        if patch_files:
            path = patch_files[-1]
            consume(path, "Exp2 largest-width full cell", model)
            patch = pd.read_csv(path)
            patch = patch[patch["perturbation_type"].str.endswith(
                "/block_token", na=False)]
            patch_status = "complete"
            patch_max = int(patch["context_length"].max())
            patch_samples = int(patch["sample_id"].nunique())
            layer_order = list(dict.fromkeys(patch["layer"].astype(str)))
            for layer_index, layer in enumerate(layer_order):
                group = patch[patch["layer"] == layer]
                for region, selected in (
                    ("recent", group[group["lookback_start"] < exp1_sufficient]),
                    ("distant", group[group["lookback_start"] >= exp1_sufficient]),
                ):
                    if selected.empty:
                        continue
                    patching_profiles.append({
                        "model": model,
                        "context_length": patch_max,
                        "sufficient_context_5pct": exp1_sufficient,
                        "layer": layer,
                        "layer_index": layer_index,
                        "region": region,
                        "median_recovery_score": float(
                            selected["recovery_score"].median()),
                        "mean_recovery_score": float(
                            selected["recovery_score"].mean()),
                        "n_rows": len(selected),
                    })

        model_scope.append({
            "model": model,
            "checkpoint": meta.get("checkpoint"),
            "git_commit": meta.get("git_commit"),
            "seed": meta.get("seed"),
            "horizon": (meta.get("config") or {}).get("horizon"),
            "exp1_complete": bool(exp1_files),
            "exp1_n_contexts": exp1_contexts,
            "exp1_max_context": exp1_max,
            "exp1_n_samples_at_max": exp1_samples,
            "exp1_sufficient_context_5pct": exp1_sufficient,
            "exp4_complete_core_grid": h4_complete,
            "exp7_complete": exp7_complete,
            "exp7_full_context": exp7_max,
            "exp7_n_samples": exp7_samples,
            "activation_patching_status": patch_status,
            "activation_patching_full_context": patch_max,
            "activation_patching_n_samples": patch_samples,
            "forecast_lens_status": lens_status,
            "forecast_lens_n_layers": lens_layers,
            "forecast_lens_n_contexts": lens_contexts,
            "forecast_lens_n_saturated": lens_saturated,
            "forecast_lens_n_early_saturated": lens_early,
        })
        internal_summary.append({
            "model": model,
            "activation_patching_status": patch_status,
            "activation_patching_full_context": patch_max,
            "activation_patching_n_samples": patch_samples,
            "forecast_lens_status": lens_status,
            "forecast_lens_n_layers": lens_layers,
            "forecast_lens_n_contexts": lens_contexts,
            "forecast_lens_n_saturated": lens_saturated,
            "forecast_lens_n_early_saturated": lens_early,
        })

    write_csv(model_scope, out / "model_scope.csv")
    write_csv(error_curves, out / "exp1_error_curves.csv")
    write_csv(perturbation_ratios, out / "exp1_distant_recent_ratios.csv")
    write_csv(perturbation_profiles, out / "exp1_block_mean_profiles.csv")
    write_csv(lag_controls, out / "exp4_controls.csv")
    write_csv(lag_summary, out / "exp4_lag_tracking_summary.csv")
    write_csv(decomposition_summary, out / "exp7_decomposition_summary.csv")
    write_csv(decomposition_profiles, out / "exp7_decomposition_profiles.csv")
    write_csv(internal_summary, out / "internal_mechanism_scope.csv")
    write_csv(patching_profiles, out / "exp2_patching_profiles.csv")

    inventory = sorted(inventory, key=lambda row: (row["model"], row["path_relative_to_input_root"]))
    (out / "source_inventory.json").write_text(json.dumps({
        "input_root": str(root),
        "freeze_protocol": {
            "sufficient_context_tolerance": 0.05,
            "primary_metric": "MAE",
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_seed_base": 4200,
            "exp1_scope": "ordinary synthetic pool only; largest effective context for ratios",
            "exp4_scope": "complete 4 lag x 3 strength family-B core grid at noise 0.1",
            "exp7_scope": "largest completed full-width cell; MAE and MSE retained",
        },
        "files": inventory,
    }, indent=2, sort_keys=True) + "\n")

    generated = sorted(path for path in out.iterdir() if path.is_file())
    manifest = {
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in generated
        ]
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Frozen {len(model_scope)} model directories into {out}")


if __name__ == "__main__":
    main()
