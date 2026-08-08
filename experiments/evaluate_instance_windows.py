"""Evaluate context-window choices per GiftEval test instance.

Unlike ``compare_window_strategies_gifteval.py``, this module never averages the
predictor curves before choosing a window.  For every series i it:

1. masks windows that were not evaluated for i,
2. chooses W_i from that series' predictor scores,
3. retrieves that exact row/window's cached ``mase_gluonts_real``, and
4. aggregates selected ratios with their valid forecast counts (the exact
   GiftEval/GluonTS axis=None MASE rule).

All master predictor variants are evaluated together.  Full-native
context, an optional single globally fixed window, dataset-shared
predictor controls, and dataset/per-instance oracles are included as baselines.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from colorama import Fore
from tqdm import tqdm

from experiments import datasets_config


VARIANT_TREES: Dict[str, str] = {
    "cheap_curve": "general_v3",
    "cheap_classification": "general_v3_classification",
    "patchtst_risk": "general_v3_risk",
    "mamba_curve": "general_v4",
    "mamba_classification": "general_v4_classification",
}

# Instance-wise counterparts of the historical per-model Stage-4 reports.
# The master uses the two curve-regression variants below; the other evaluator
# variants remain available in the combined Phase-6 tables.
INSTANCE_STRATEGY_OUTPUTS: Dict[str, Tuple[str, str, str]] = {
    "cheap_curve": ("strategy_comparison_v3", "strategy_comparison_v3_instance",
                    "pred_cheap"),
    "mamba_curve": ("strategy_comparison_v4", "strategy_comparison_v4_instance",
                    "pred_mamba"),
}


@dataclass(frozen=True)
class Cell:
    model: str
    dataset: str
    term: str
    anchor_npz: str


def _cell_from_path(path: str, model: str) -> Optional[Cell]:
    name = os.path.basename(path)
    match = re.fullmatch(
        rf"compare_(.+)_t([^_]+)_{re.escape(model)}\.npz", name)
    if match is None:
        return None
    return Cell(model, match.group(1), match.group(2), path)


def discover_cells(ablation_root: str, models: Optional[Iterable[str]]) -> List[Cell]:
    allowed = set(models or [])
    active_dataset_cells = {
        (display, str(term))
        for _ge_name, term, display, _to_univariate
        in datasets_config.datasets_to_run()
    }
    found: Dict[Tuple[str, str, str], Cell] = {}
    for tree in VARIANT_TREES.values():
        pattern = os.path.join(
            ablation_root, tree, "models", "*",
            "compare_real_vs_predicted", "compare_*.npz")
        for path in glob.glob(pattern):
            model = path.split(os.sep)[-3]
            if allowed and model not in allowed:
                continue
            cell = _cell_from_path(path, model)
            # A resumed run may retain old derived NPZs for datasets that were
            # subsequently removed from the experiment cohort. The authoritative
            # run-set filter prevents such stale files from leaking back into
            # instance-level aggregates.
            if (cell is not None
                    and (cell.dataset, str(cell.term)) in active_dataset_cells):
                found.setdefault((model, cell.dataset, cell.term), cell)
    return [found[key] for key in sorted(found)]


def _load_vector(
    path: str, n: int, field: str = "mase_gluonts_real",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    out = np.full(n, np.nan, dtype=np.float64)
    counts_out = np.zeros(n, dtype=np.float64)
    if not os.path.isfile(path):
        return out, counts_out, {"source": "missing"}
    with np.load(path) as data:
        source = field
        if source not in data.files:
            if field == "mase_gluonts_real" and "mase_gluonts" in data.files:
                source = "mase_gluonts_proxy"
                values = np.asarray(data["mase_gluonts"], dtype=np.float64)
            else:
                return out, counts_out, {"source": "missing"}
        else:
            values = np.asarray(data[source], dtype=np.float64)
        if "served_index" in data.files:
            index = np.asarray(data["served_index"], dtype=np.int64)
        elif values.shape[0] == n:
            index = np.arange(n, dtype=np.int64)
        else:
            # Old skip-mode caches cannot be aligned safely.  Stage 3 now
            # backfills served_index without TSFM inference.
            return out, counts_out, {"unaligned": True, "source": source}
        if values.shape[0] != index.shape[0]:
            return out, counts_out, {"unaligned": True, "source": source}
        counts = (np.asarray(data["valid_count"], dtype=np.float64)
                  if "valid_count" in data.files
                  else np.ones(values.shape[0], dtype=np.float64))
        if counts.shape != values.shape:
            return out, counts_out, {"unaligned": True, "source": source}
        ok = (index >= 0) & (index < n)
        out[index[ok]] = values[ok]
        counts_out[index[ok]] = counts[ok]
        extra = {
            key: np.asarray(data[key])
            for key in ("effective_context",)
            if key in data.files
        }
        extra["source"] = source
    return out, counts_out, extra


def _ground_tree(ablation_root: str) -> Optional[str]:
    for tree in VARIANT_TREES.values():
        candidate = os.path.join(ablation_root, tree, "datasets")
        if os.path.isdir(candidate):
            return os.path.join(ablation_root, tree)
    return None


def _cell_metric_path(
    ground_tree: str, cell: Cell, window: object,
) -> str:
    return os.path.join(
        ground_tree, "datasets", cell.dataset, cell.model, f"t{cell.term}",
        f"w{window}", "per_sample_metrics.npz")


def _variant_prediction_path(
    ablation_root: str, tree: str, cell: Cell,
) -> str:
    return os.path.join(
        ablation_root, tree, "models", cell.model,
        "compare_real_vs_predicted",
        f"compare_{cell.dataset}_t{cell.term}_{cell.model}.npz")


def _choose_scores(
    scores: np.ndarray, errors: np.ndarray, windows: np.ndarray,
    native: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose each row's lowest-score feasible grid window.

    Rows with no cached grid window fall back to full-native context.
    """
    feasible = np.isfinite(errors) & np.isfinite(scores)
    masked = np.where(feasible, scores, np.inf)
    idx = np.argmin(masked, axis=1)
    has_grid = feasible.any(axis=1)
    rows = np.arange(errors.shape[0])
    selected = np.where(has_grid, errors[rows, idx], native)
    selected_w = np.where(has_grid, windows[idx], -1)
    return selected, selected_w, ~has_grid


def _choose_scores_with_native(
    scores: np.ndarray, errors: np.ndarray, windows: np.ndarray,
    native: np.ndarray, native_w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose per row while treating native-full as a real policy action.

    Synthetic labels at the largest requested window are generated with all
    genuine history (``min(window, real_length)``), so that output is the
    predictor's native/full score.  Reusing it here avoids forcing short real
    series onto the largest *grid* window below their non-grid native length.
    """
    # Prefer the real native action on an exact tie with the largest grid
    # output. For full-length rows they are normally the same workload; for
    # short rows only native can use the non-grid amount of genuine history.
    native_score = np.nextafter(scores[:, -1], -np.inf)
    candidate_scores = np.column_stack([scores, native_score])
    candidate_errors = np.column_stack([errors, native])
    feasible = np.isfinite(candidate_scores) & np.isfinite(candidate_errors)
    masked = np.where(feasible, candidate_scores, np.inf)
    idx = np.argmin(masked, axis=1)
    has_choice = feasible.any(axis=1)
    rows = np.arange(errors.shape[0])
    selected = np.where(has_choice, candidate_errors[rows, idx], native)
    selected_w = np.where(
        idx < windows.size,
        windows[np.minimum(idx, windows.size - 1)],
        native_w,
    )
    selected_native = (~has_choice) | (idx == windows.size)
    return selected, selected_w, selected_native


def _choose_capped_fixed(
    target: int, errors: np.ndarray, windows: np.ndarray, native: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use target W, capped to the largest feasible cached W <= target per row."""
    eligible = np.isfinite(errors) & (windows[None, :] <= int(target))
    rank = np.where(eligible, windows[None, :], -1)
    idx = np.argmax(rank, axis=1)
    has_grid = eligible.any(axis=1)
    rows = np.arange(errors.shape[0])
    selected = np.where(has_grid, errors[rows, idx], native)
    selected_w = np.where(has_grid, windows[idx], -1)
    return selected, selected_w, ~has_grid


def _record(
    cell: Cell, method: str, error: np.ndarray, window: np.ndarray,
    fallback: np.ndarray, method_kind: str, valid_count: np.ndarray,
    metric_source: str,
) -> dict:
    ok = np.isfinite(error) & np.isfinite(valid_count) & (valid_count > 0)
    w = window[(window >= 0) & ok]
    if method == "full_native" or method.startswith("period"):
        policy_scope = "instance_native"
    elif method_kind == "predictor_instance":
        policy_scope = "instance_native"
    elif method_kind == "predictor_instance_control":
        policy_scope = "instance_grid"
    elif method_kind in {"predictor_dataset", "oracle_control", "fixed"}:
        policy_scope = "dataset_shared_grid_capped_per_instance"
    elif method_kind == "oracle":
        policy_scope = "instance_native"
    else:
        policy_scope = "heuristic_capped_per_instance"
    return {
        "model": cell.model,
        "dataset_display": cell.dataset,
        "term": cell.term,
        "method": method,
        "method_kind": method_kind,
        "policy_scope": policy_scope,
        "mase_gluonts": (float(np.average(error[ok], weights=valid_count[ok]))
                          if ok.any() else float("nan")),
        "mase_metric_source": metric_source,
        "mase_metric_exact": metric_source not in {
            "mase_gluonts_proxy", "missing"},
        "n_instances": int(ok.sum()),
        "n_native_fallback": int(np.sum(fallback & ok)),
        "n_native_selected": int(np.sum(fallback & ok)),
        "window_mean": float(np.mean(w)) if w.size else float("nan"),
        "window_median": float(np.median(w)) if w.size else float("nan"),
    }


def _horizon_for_cell(ablation_root: str, cell: Cell) -> int:
    for tree in VARIANT_TREES.values():
        path = os.path.join(
            ablation_root, tree, "models", cell.model,
            "compare_real_vs_predicted", "compare_summary.csv")
        if not os.path.isfile(path):
            continue
        frame = pd.read_csv(path)
        row = frame[
            (frame["dataset_display"] == cell.dataset)
            & (frame["term"].astype(str) == str(cell.term))
        ]
        if not row.empty:
            return int(row.iloc[0]["horizon_real"])
    return 1


def evaluate_cell(
    cell: Cell, ablation_root: str, ground_tree: str,
    fixed_window: Optional[int] = None,
    mase_field: str = "mase_gluonts_real",
) -> Tuple[List[dict], dict]:
    with np.load(cell.anchor_npz) as anchor:
        windows = np.asarray(anchor["window_grid"], dtype=np.int64)
        n = int(np.asarray(anchor["predicted_curves"]).shape[0])
        curve_key = {
            "mase_gluonts": "real_curve_gluonts",
            "mase_gluonts_real": "real_curve_gluonts_real",
        }[mase_field]
        # Stage 3 is authoritative about which model/window cells are valid.
        # In particular, Moirai's usable context shrinks with the forecast
        # horizon, so the nominal 8192 grid action can be ineligible even when
        # an obsolete cache from before that constraint still exists on disk.
        # Never let such stale, unsupported caches enter (or block) Stage 6.
        valid_windows = (
            np.isfinite(np.asarray(anchor[curve_key], dtype=np.float64))
            if curve_key in anchor.files
            else np.ones(windows.shape, dtype=bool)
        )

    errors = np.full((n, windows.size), np.nan, dtype=np.float64)
    error_counts = np.zeros((n, windows.size), dtype=np.float64)
    metric_sources = set()
    unaligned: List[int] = []
    for j, window in enumerate(windows):
        if not valid_windows[j]:
            continue
        errors[:, j], error_counts[:, j], meta = _load_vector(
            _cell_metric_path(ground_tree, cell, int(window)), n, mase_field)
        metric_sources.add(meta.get("source", "missing"))
        if meta.get("unaligned"):
            unaligned.append(int(window))
    if unaligned:
        raise RuntimeError(
            f"{cell.model}/{cell.dataset}/t{cell.term}: old per-sample caches "
            f"lack served_index at windows {unaligned}. Re-run stage 3 once; "
            "cached forecasts will be backfilled without TSFM inference.")

    native, native_counts, native_meta = _load_vector(
        _cell_metric_path(ground_tree, cell, "full_native"), n, mase_field)
    metric_sources.add(native_meta.get("source", "missing"))
    if not np.isfinite(native).all():
        # Keep identical instance coverage even for older/no-native runs.
        fallback_grid = np.where(np.isfinite(errors), errors, np.inf)
        fallback_idx = np.argmin(fallback_grid, axis=1)
        rows = np.arange(n)
        best_available = fallback_grid[rows, fallback_idx]
        best_available[~np.isfinite(best_available)] = np.nan
        missing_native = ~np.isfinite(native)
        native = np.where(missing_native, best_available, native)
        native_counts = np.where(
            missing_native, error_counts[rows, fallback_idx], native_counts)
    native_w = np.asarray(
        native_meta.get("effective_context", np.full(n, -1)), dtype=np.int64)
    if native_w.shape != (n,):
        native_w = np.full(n, -1, dtype=np.int64)
    metric_source = ("mase_gluonts_proxy"
                     if "mase_gluonts_proxy" in metric_sources else mase_field)

    records: List[dict] = []
    audit: Dict[str, np.ndarray] = {
        "window_grid": windows,
        "grid_mase": errors,
        "native_mase": native,
        "native_valid_count": native_counts,
        "native_effective_context": native_w,
    }

    def counts_for_choice(
        chosen_w: np.ndarray, native_selected: np.ndarray,
    ) -> np.ndarray:
        counts = native_counts.copy()
        for j, window in enumerate(windows):
            use = (~native_selected) & (chosen_w == int(window))
            counts[use] = error_counts[use, j]
        return counts

    def add(method: str, values: np.ndarray, chosen_w: np.ndarray,
            fallback: np.ndarray, kind: str,
            selected_counts: Optional[np.ndarray] = None,
            selected_source: Optional[str] = None) -> None:
        counts_used = (native_counts if selected_counts is None
                       else selected_counts)
        records.append(_record(
            cell, method, values, chosen_w, fallback, kind,
            counts_used,
            metric_source if selected_source is None else selected_source))
        audit[f"{method}__mase"] = values
        audit[f"{method}__window"] = chosen_w
        audit[f"{method}__valid_count"] = counts_used

    add("full_native", native, native_w, np.zeros(n, dtype=bool), "baseline")

    if fixed_window is not None:
        values, chosen_w, fallback = _choose_capped_fixed(
            int(fixed_window), errors, windows, native)
        add(f"fixed_{int(fixed_window)}", values, chosen_w, fallback, "fixed",
            counts_for_choice(chosen_w, fallback))

    # Dataset-shared oracle: one window from the mean curve, then cap it per row.
    oracle_weights = np.where(
        np.isfinite(errors) & (native_counts[:, None] > 0),
        native_counts[:, None], 0.0)
    weight_per_window = oracle_weights.sum(axis=0)
    mean_error = np.divide(
        np.nansum(errors * oracle_weights, axis=0), weight_per_window,
        out=np.full(windows.shape, np.nan, dtype=np.float64),
        where=weight_per_window > 0,
    )
    shared_oracle_w = int(windows[int(np.nanargmin(mean_error))])
    values, chosen_w, fallback = _choose_capped_fixed(
        shared_oracle_w, errors, windows, native)
    add("oracle_dataset", values, chosen_w, fallback, "oracle_control",
        counts_for_choice(chosen_w, fallback))

    # Genuine per-instance oracle, including full-native as a candidate.
    candidates = np.column_stack([errors, native])
    oracle_idx = np.nanargmin(
        np.where(np.isfinite(candidates), candidates, np.inf), axis=1)
    rows = np.arange(n)
    oracle_values = candidates[rows, oracle_idx]
    oracle_w = np.where(oracle_idx < windows.size, windows[
        np.minimum(oracle_idx, windows.size - 1)], native_w)
    add("oracle_instance", oracle_values, oracle_w,
        oracle_idx == windows.size, "oracle",
        counts_for_choice(oracle_w, oracle_idx == windows.size))

    for variant, tree in VARIANT_TREES.items():
        path = _variant_prediction_path(ablation_root, tree, cell)
        if not os.path.isfile(path):
            continue
        with np.load(path) as data:
            grid = np.asarray(data["window_grid"], dtype=np.int64)
            scores = np.asarray(data["predicted_curves"], dtype=np.float64)
        if not np.array_equal(grid, windows) or scores.shape != errors.shape:
            print(Fore.YELLOW + f"Skip misaligned predictor artifact: {path}"
                  + Fore.RESET)
            continue
        values, chosen_w, fallback = _choose_scores(
            scores, errors, windows, native)
        add(f"{variant}_instance_grid", values, chosen_w, fallback,
            "predictor_instance_control", counts_for_choice(chosen_w, fallback))

        values, chosen_w, selected_native = _choose_scores_with_native(
            scores, errors, windows, native, native_w)
        add(f"{variant}_instance", values, chosen_w, selected_native,
            "predictor_instance",
            counts_for_choice(chosen_w, selected_native))
        audit[f"{variant}__scores"] = scores

        shared_w = int(windows[int(np.argmin(np.mean(scores, axis=0)))])
        values, chosen_w, fallback = _choose_capped_fixed(
            shared_w, errors, windows, native)
        add(f"{variant}_dataset", values, chosen_w, fallback,
            "predictor_dataset", counts_for_choice(chosen_w, fallback))

    return records, audit


def _geomean(values: np.ndarray) -> float:
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.exp(np.mean(np.log(values)))) if values.size else float("nan")


def _write_instance_strategy_reports(
    frame: pd.DataFrame, ablation_root: str, audit_dir: str,
) -> None:
    """Write v3/v4-shaped instance-policy reports beside Stage 4 outputs.

    ``comparison.csv`` retains Stage 4's baseline/oracle/normalizer columns but
    replaces the predictor action with the true per-instance policy aggregate.
    Timing/FLOPs fields for that action are cleared: a mixture of per-instance
    widths cannot honestly reuse the old one-window runtime or FLOPs estimate.
    ``instance_details.csv`` preserves every selected action and realised MASE.
    """
    from experiments.compare_window_strategies_gifteval import (
        DEFAULT_PATCH_SIZES, compute_summary_stats, theoretical_flops,
    )

    for variant, (base_subdir, output_subdir, alias) in (
            INSTANCE_STRATEGY_OUTPUTS.items()):
        method = f"{variant}_instance"
        selected = frame[frame["method"] == method]
        if selected.empty:
            continue
        for model, model_selected in selected.groupby("model", sort=True):
            base_path = os.path.join(
                ablation_root, model, base_subdir, "comparison.csv")
            if not os.path.isfile(base_path):
                print(Fore.YELLOW
                      + f"Skip instance strategy report for {model}/{variant}: "
                        f"missing {base_path}" + Fore.RESET)
                continue
            comparison = pd.read_csv(base_path)
            comparison["term"] = comparison["term"].astype(str)
            by_key = {
                (str(row.dataset_display), str(row.term)): row
                for row in model_selected.itertuples(index=False)
            }
            full_rows = frame[
                (frame["model"] == model) & (frame["method"] == "full_native")]
            oracle_rows = frame[
                (frame["model"] == model) & (frame["method"] == "oracle_instance")]
            full_by_key = {
                (str(row.dataset_display), str(row.term)): row
                for row in full_rows.itertuples(index=False)
            }
            oracle_by_key = {
                (str(row.dataset_display), str(row.term)): row
                for row in oracle_rows.itertuples(index=False)
            }

            keep = []
            for idx, row in comparison.iterrows():
                key = (str(row["dataset_display"]), str(row["term"]))
                pred = by_key.get(key)
                full = full_by_key.get(key)
                oracle = oracle_by_key.get(key)
                if pred is None or full is None:
                    continue
                keep.append(idx)
                full_mase = float(full.mase_gluonts)
                pred_mase = float(pred.mase_gluonts)
                best_mase = float(row.get("best_mase", float("nan")))
                audit_path = os.path.join(
                    audit_dir, f"{model}__{key[0]}__t{key[1]}.npz")
                pred_flops = float("nan")
                if os.path.isfile(audit_path):
                    with np.load(audit_path) as audit:
                        chosen_contexts = np.asarray(
                            audit[f"{method}__window"], dtype=np.int64)
                        native_contexts = np.asarray(
                            audit["native_effective_context"], dtype=np.int64)
                    chosen_contexts = np.where(
                        chosen_contexts > 0, chosen_contexts, native_contexts)
                    chosen_contexts = chosen_contexts[chosen_contexts > 0]
                    if chosen_contexts.size:
                        model_id = str(row.get("model", model))
                        horizon = int(row.get("horizon", 1))
                        pred_flops = float(np.mean([
                            theoretical_flops(
                                model_id, int(context), horizon,
                                DEFAULT_PATCH_SIZES)
                            for context in chosen_contexts
                        ]))
                full_flops = float(row.get("full_flops", float("nan")))
                complexity_ratio = (
                    pred_flops / full_flops
                    if np.isfinite(pred_flops) and full_flops > 0
                    else float("nan"))
                comparison.at[idx, "primary_strategy"] = f"{alias}_instance"
                comparison.at[idx, "full_mase"] = full_mase
                comparison.at[idx, "pred_mase"] = pred_mase
                comparison.at[idx, "pred_window"] = float(pred.window_mean)
                comparison.at[idx, "pred_window_median"] = float(pred.window_median)
                comparison.at[idx, "pred_policy_scope"] = "instance_native"
                comparison.at[idx, "pred_clamped"] = False
                comparison.at[idx, "pred_n_native_selected"] = int(
                    pred.n_native_selected)
                comparison.at[idx, "delta_pred_vs_full"] = pred_mase - full_mase
                comparison.at[idx, "delta_pred_vs_best"] = pred_mase - best_mase
                comparison.at[idx, "rel_gain_pred_over_full"] = (
                    (full_mase - pred_mase) / (abs(full_mase) + 1e-12))
                comparison.at[idx, "pred_flops"] = pred_flops
                comparison.at[idx, "complexity_ratio_pred_vs_full"] = (
                    complexity_ratio)
                if oracle is not None:
                    oracle_mase = float(oracle.mase_gluonts)
                    comparison.at[idx, "instance_oracle_mase"] = oracle_mase
                    comparison.at[idx, "delta_instance_oracle_vs_full"] = (
                        oracle_mase - full_mase)
                    comparison.at[idx, "rel_gain_instance_oracle_over_full"] = (
                        (full_mase - oracle_mase) / (abs(full_mase) + 1e-12))
                # Canonical alias used by cross-variant consumers.
                comparison.at[idx, f"{alias}_mase"] = pred_mase
                comparison.at[idx, f"{alias}_window"] = float(pred.window_mean)
                comparison.at[idx, f"{alias}_window_median"] = float(
                    pred.window_median)
                comparison.at[idx, f"{alias}_policy_scope"] = "instance_native"
                comparison.at[idx, f"delta_{alias}_vs_full"] = pred_mase - full_mase
                comparison.at[idx, f"delta_{alias}_vs_best"] = pred_mase - best_mase
                comparison.at[idx, f"rel_gain_{alias}_over_full"] = (
                    (full_mase - pred_mase) / (abs(full_mase) + 1e-12))
                comparison.at[idx, f"{alias}_flops"] = pred_flops
                comparison.at[idx, f"complexity_ratio_{alias}_vs_full"] = (
                    complexity_ratio)
                # Runtime still cannot be composed from the old dataset-wide
                # measurements: batching/kernel overhead is nonlinear. FLOPs,
                # in contrast, are estimated above per selected instance.
                for column in (
                    "pred_elapsed_s", "pred_elapsed_std_s",
                    "speedup_pred_vs_full",
                    f"{alias}_elapsed_s", f"{alias}_elapsed_std_s",
                    f"speedup_{alias}_vs_full",
                ):
                    comparison.at[idx, column] = float("nan")

            comparison = comparison.loc[keep].reset_index(drop=True)
            if comparison.empty:
                continue
            output_dir = os.path.join(ablation_root, model, output_subdir)
            os.makedirs(output_dir, exist_ok=True)
            comparison.to_csv(
                os.path.join(output_dir, "comparison.csv"), index=False)
            model_selected.to_csv(
                os.path.join(output_dir, "cell_results.csv"), index=False)

            details = []
            for cell in model_selected.itertuples(index=False):
                path = os.path.join(
                    audit_dir,
                    f"{model}__{cell.dataset_display}__t{cell.term}.npz")
                if not os.path.isfile(path):
                    continue
                with np.load(path) as data:
                    mase = np.asarray(data[f"{method}__mase"], dtype=float)
                    window = np.asarray(data[f"{method}__window"], dtype=int)
                    counts = np.asarray(
                        data[f"{method}__valid_count"], dtype=float)
                    full_mase = np.asarray(data["full_native__mase"], dtype=float)
                    native_context = np.asarray(
                        data["native_effective_context"], dtype=int)
                base_match = comparison[
                    (comparison["dataset_display"].astype(str)
                     == str(cell.dataset_display))
                    & (comparison["term"].astype(str) == str(cell.term))]
                model_id = (str(base_match.iloc[0].get("model", model))
                            if not base_match.empty else model)
                horizon = (int(base_match.iloc[0].get("horizon", 1))
                           if not base_match.empty else 1)
                valid = np.isfinite(mase) & np.isfinite(full_mase) & (counts > 0)
                for instance_index in np.flatnonzero(valid):
                    selected_context = int(window[instance_index])
                    if selected_context <= 0:
                        selected_context = int(native_context[instance_index])
                    selected_flops = theoretical_flops(
                        model_id, selected_context, horizon, DEFAULT_PATCH_SIZES)
                    native_flops = theoretical_flops(
                        model_id, int(native_context[instance_index]), horizon,
                        DEFAULT_PATCH_SIZES)
                    details.append({
                        "model": model,
                        "dataset_display": cell.dataset_display,
                        "term": cell.term,
                        "instance_index": int(instance_index),
                        "selected_window": int(window[instance_index]),
                        "native_context": int(native_context[instance_index]),
                        "selected_mase": float(mase[instance_index]),
                        "full_mase": float(full_mase[instance_index]),
                        "valid_count": float(counts[instance_index]),
                        "selected_flops_estimate": float(selected_flops),
                        "full_flops_estimate": float(native_flops),
                        "flops_saved_estimate": float(
                            native_flops - selected_flops),
                        "flops_saved_pct_estimate": float(
                            100 * (native_flops - selected_flops) / native_flops
                            if native_flops > 0 else float("nan")),
                    })
            pd.DataFrame(details).to_csv(
                os.path.join(output_dir, "instance_details.csv"), index=False)
            with open(os.path.join(output_dir, "summary_stats.json"), "w") as handle:
                json.dump(
                    compute_summary_stats(comparison), handle,
                    indent=2, allow_nan=True)
            print(Fore.GREEN
                  + f"Instance-wise {variant} report: {output_dir}"
                  + Fore.RESET)


def run(args: argparse.Namespace) -> None:
    ablation_root = os.path.normpath(args.ablation_root)
    ground_tree = _ground_tree(ablation_root)
    if ground_tree is None:
        raise SystemExit(f"No variant datasets tree found under {ablation_root}")
    cells = discover_cells(ablation_root, args.models)
    if not cells:
        raise SystemExit("No per-predictor comparison cells found.")

    os.makedirs(args.output_dir, exist_ok=True)
    audit_dir = os.path.join(args.output_dir, "cells")
    os.makedirs(audit_dir, exist_ok=True)
    all_records: List[dict] = []
    for cell in tqdm(cells, desc="Phase 6 instance evaluation", unit="cell"):
        records, audit = evaluate_cell(
            cell, ablation_root, ground_tree, args.fixed_window,
            args.mase_field)
        all_records.extend(records)
        np.savez_compressed(
            os.path.join(
                audit_dir, f"{cell.model}__{cell.dataset}__t{cell.term}.npz"),
            **audit)

    frame = pd.DataFrame(all_records)
    cell_keys = ["model", "dataset_display", "term"]
    full = frame[frame["method"] == "full_native"][
        cell_keys + ["mase_gluonts"]].rename(
            columns={"mase_gluonts": "full_native_mase_gluonts"})
    oracle = frame[frame["method"] == "oracle_instance"][
        cell_keys + ["mase_gluonts"]].rename(
            columns={"mase_gluonts": "oracle_instance_mase_gluonts"})
    frame = frame.merge(full, on=cell_keys, how="left")
    frame = frame.merge(oracle, on=cell_keys, how="left")
    frame["delta_vs_full"] = (
        frame["mase_gluonts"] - frame["full_native_mase_gluonts"])
    frame["regret_vs_instance_oracle"] = (
        frame["mase_gluonts"] - frame["oracle_instance_mase_gluonts"])
    frame.to_csv(os.path.join(args.output_dir, "cell_results.csv"), index=False)
    summary_rows = []
    for (method, kind), group in frame.groupby(
            ["method", "method_kind"], sort=False):
        values = group["mase_gluonts"].to_numpy(dtype=float)
        weights = group["n_instances"].to_numpy(dtype=float)
        valid = np.isfinite(values) & (weights > 0)
        summary_rows.append({
            "method": method,
            "method_kind": kind,
            "macro_mean_mase_gluonts": (
                float(np.mean(values[valid])) if valid.any() else float("nan")),
            "macro_geomean_mase_gluonts": _geomean(values),
            "instance_weighted_mase_gluonts": (
                float(np.average(values[valid], weights=weights[valid]))
                if valid.any() else float("nan")),
            "macro_mean_delta_vs_full": float(
                group.loc[valid, "delta_vs_full"].mean())
                if valid.any() else float("nan"),
            "macro_mean_regret_vs_instance_oracle": float(
                group.loc[valid, "regret_vs_instance_oracle"].mean())
                if valid.any() else float("nan"),
            "beats_full_rate": float(
                (group.loc[valid, "delta_vs_full"] < 0).mean())
                if valid.any() else float("nan"),
            "n_cells": int(valid.sum()),
            "n_instances": int(weights[valid].sum()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["method_kind", "macro_mean_mase_gluonts"])
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    _write_instance_strategy_reports(frame, ablation_root, audit_dir)

    predictor_summary = summary[
        summary["method_kind"].isin([
            "predictor_instance", "predictor_instance_control",
            "predictor_dataset"])]
    print(Fore.GREEN
          + f"Per-instance evaluation wrote {len(frame)} cell-method rows across "
            f"{len(cells)} cells to {args.output_dir}" + Fore.RESET)
    if not predictor_summary.empty:
        print(predictor_summary[
            ["method", "macro_mean_mase_gluonts", "n_cells"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fixed-window", type=int, default=None,
        help=("One global fixed context baseline, capped per instance. Omit until "
              "a value is selected from a separate development analysis."),
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--mase-field", choices=["mase_gluonts_real", "mase_gluonts"],
        default="mase_gluonts_real",
        help=("Per-instance metric stored by stage 3. The default reproduces "
              "the leaderboard GluonTS MASE; old caches may explicitly fall "
              "back to the proxy and are marked in the CSV."),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
