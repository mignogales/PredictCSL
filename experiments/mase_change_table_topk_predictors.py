#!/usr/bin/env python
"""
Top-K-predictor MASE-change table — a variant of ``mase_change_table_gluonts.png``.

The default stage-4 table (``plot_mase_change_table`` in
``compare_window_strategies_gifteval``) scores the *pred_window* strategy with the
SINGLE selected predictor (``best_model.pt`` = the trial that minimised the
selection metric). This script instead uses the **top-K trials** of the random
search: for each foundation model it ranks every trial by the same selection
metric (``SELECTION_METRIC`` -> ``val_regret`` by default), keeps the K best that
have a durable checkpoint, re-scores each one independently on the GiftEval cells,
and reports the **mean ± std** of their geomean MASE.

Why this is cheap (NO TSFM re-inference):
    Stage 3 already ran every TSFM at every grid window and cached the real
    error-vs-context curve per cell (``real_curve_gluonts_real`` in the
    ``compare_*.npz``). The only thing that differs between predictors is which
    window the predictor's curve argmin's to — a forward pass of a *small*
    predictor over the (cached) GiftEval contexts. So we reload the GiftEval
    inputs (like stage 3 does) and run the K predictors; the expensive labelled
    TSFM surface is read straight from disk.

Aggregation mirrors the leaderboard convention used everywhere else:
    * per (model, predictor k): geometric mean over the model's (dataset, term)
      cells of that predictor's chosen-window MASE (``strat_mase_k``), and the
      leaderboard-normalised twin (÷ same-definition Seasonal-Naive) over the
      cells that carry a baseline;
    * per model: MEAN ± STD across the K predictors of those geomeans;
    * TOTAL row: the same, aggregating each predictor over ALL cells of ALL
      models (unweighted geomean over cells), then mean ± std across K.

Usage (run on the SERVER, where logs/ and the GIFT_EVAL data live)::

    python -m experiments.mase_change_table_topk_predictors
    python -m experiments.mase_change_table_topk_predictors --k 5
    python -m experiments.mase_change_table_topk_predictors --models Moirai2-Small TimesFM2.5-200M
    python -m experiments.mase_change_table_topk_predictors --mase-metric mase_gluonts

Output: ``mase_change_table_topk{suffix}.png`` + ``mase_change_table_topk{suffix}.csv``
into ``--output-dir`` (default: the base run dir).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from colorama import Fore
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gift_eval.data import Dataset as GiftEvalDataset

from experiments import datasets_config
from experiments.predict_context_length import (
    SELECTION_METRIC, _METRIC_KEY, build_predictor,
)
from experiments.test_window_ablation_gifteval_v5 import (
    GiftEvalCache, _closest_horizon_idx, predict_curves_for_dataset,
)
from experiments.compare_window_strategies_gifteval import (
    CACHE_ROOT, DEFAULT_PATCH_SIZES, _MODEL_MARKERS, _geomean,
    _load_naive_baselines, _npz_filename, infer_model_family, theoretical_flops,
)

# Families whose compute does NOT actually vary with the window, so the FLOPs
# axis is a theoretical upper bound only (never sell compute savings for these):
#   tirex       — truncates AND NaN-left-pads every context to its 2048 train
#                 length internally; wall-clock is flat across windows.
#   patchtst_fm — fixed 8192 native context, no mask input; windows are realised
#                 by NaN-padding, so every forward pass costs the same.
# sundial is HALF-flat: it caps context at 2880, so compute varies below the cap
# but is flat above it (real-curve ties above 2880 make the argmin degenerate and
# the FLOPs there overstated) — kept solid on the plot, but read its x with care.
FLAT_COMPUTE_FAMILIES = {"tirex", "patchtst_fm"}

# Per-model predictor run dir layout (mirrors run_all.py's PREDICTOR_ROOT/<display>).
PREDICTOR_ROOT_DEFAULT = "logs/experiments/context_length_predictor"

# The three predictor variants, mirroring run_all.py / run_all_v3.py / run_all_v4.py.
# Each maps to (predictor-root suffix, ablation-tree basename, human label). The
# predictor root and the ablation run dir both carry the same _v3/_v4 suffix; the
# base "normal" (v1) predictor is the big PatchTST-Transformer, v3 the constrained
# cheap PatchTST, v4 the Mamba. build_predictor branches on the trial's cfg["arch"]
# so the Mamba checkpoints load exactly like the Transformer ones (needs mamba-ssm).
VARIANTS: Dict[str, Tuple[str, str, str]] = {
    "normal": ("",    "general",    "v1 PatchTST (full search)"),
    "cheap":  ("_v3", "general_v3", "v3 PatchTST (constrained/cheap)"),
    "mamba":  ("_v4", "general_v4", "v4 Mamba"),
}


# ==============================================================================
#  TOP-K TRIAL SELECTION
# ==============================================================================

def _trial_best_ckpt(predictor_dir: str, trial_idx: int) -> str:
    return os.path.join(predictor_dir, "trials", f"trial_{int(trial_idx):03d}_best.pt")


def _trial_cfg(predictor_dir: str, trial_idx: int) -> Optional[dict]:
    """The per-trial architecture config (``cfg`` inside trials/trial_NNN.json)."""
    p = os.path.join(predictor_dir, "trials", f"trial_{int(trial_idx):03d}.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f).get("cfg")
    except (OSError, json.JSONDecodeError):
        return None


def select_topk_trials(predictor_dir: str, k: int) -> Tuple[List[dict], dict]:
    """Rank trials by the selection metric and return the K best as
    ``[{trial_idx, metric}]`` (ascending metric = better) that have a durable
    checkpoint, plus the shared best_config.json (run-global constants:
    context_length / window_grid / …).

    Ranking column tracks ``SELECTION_METRIC`` exactly as the training run's
    ``_select_final`` does (``val_regret`` for the default 'regret') — so the
    rank-1 trial here is the SAME trial that was shipped as best_model.pt, which
    is what makes the per-cell agreement check against the stage-3 npz's
    ``predicted_mean`` a valid consistency test.
    """
    with open(os.path.join(predictor_dir, "best_config.json")) as f:
        best_cfg = json.load(f)

    summary_path = os.path.join(predictor_dir, "sweep_summary.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"No sweep_summary.csv in {predictor_dir}")
    df = pd.read_csv(summary_path)

    metric_key = _METRIC_KEY[SELECTION_METRIC]
    if metric_key not in df.columns:
        raise KeyError(f"{metric_key!r} not in {summary_path} columns {list(df.columns)}")

    # Keep only trials that did not fail, carry a finite metric, AND have a
    # durable per-trial checkpoint on disk (a resumed/failed trial may not).
    if "failed" in df.columns:
        df = df[df["failed"] != True]  # noqa: E712
    df = df[np.isfinite(pd.to_numeric(df[metric_key], errors="coerce"))]
    df = df.sort_values(metric_key, ascending=True)

    chosen: List[dict] = []
    for _, row in df.iterrows():
        ti = int(row["trial_idx"])
        if os.path.isfile(_trial_best_ckpt(predictor_dir, ti)) and _trial_cfg(predictor_dir, ti):
            chosen.append({"trial_idx": ti, "metric": float(row[metric_key]),
                           "metric_key": metric_key})
        if len(chosen) >= k:
            break

    # Sanity: the rank-1 trial should be the one best_config.json shipped. A
    # mismatch (ties, a search resumed after selection, or a hand-replaced
    # best_model.pt) makes the npz-agreement check advisory rather than exact.
    if chosen and "trial_idx" in best_cfg and int(best_cfg["trial_idx"]) != chosen[0]["trial_idx"]:
        print(Fore.YELLOW + f"  NOTE: rank-1 trial {chosen[0]['trial_idx']} != shipped "
              f"best_config trial {int(best_cfg['trial_idx'])} — npz agreement check "
              "will compare different predictors." + Fore.RESET)
    return chosen, best_cfg


def load_trial_predictor(predictor_dir: str, trial_idx: int, best_cfg: dict,
                         device: str) -> torch.nn.Module:
    """Build a predictor from (shared run constants + this trial's arch cfg) and
    load its durable checkpoint. Mirrors stage-3 ``load_predictor`` but for an
    arbitrary trial rather than the selected winner."""
    trial_cfg = _trial_cfg(predictor_dir, trial_idx) or {}
    # best_cfg supplies the run-global axes (n_windows, n_horizons, context_length,
    # window_grid, horizon_grid); trial_cfg overrides the per-trial architecture.
    merged = {**best_cfg, **trial_cfg}
    model = build_predictor(merged, merged["n_windows"], merged["n_horizons"])
    state = torch.load(_trial_best_ckpt(predictor_dir, trial_idx), map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ==============================================================================
#  PER-CELL SCORING
# ==============================================================================

def _real_curve(data, mase_metric: str) -> Optional[np.ndarray]:
    """Pick the requested gluonts curve from a compare_*.npz, with the same loud
    port stand-in / skip logic as ``load_strategy_records`` (never the legacy
    custom-mase ``real_curve``)."""
    curve_key = {"mase_gluonts": "real_curve_gluonts",
                 "mase_gluonts_real": "real_curve_gluonts_real"}[mase_metric]

    def _usable(key: str) -> bool:
        return key in data.files and bool(np.any(~np.isnan(data[key])))

    if _usable(curve_key):
        return data[curve_key]
    if mase_metric == "mase_gluonts_real" and _usable("real_curve_gluonts"):
        return data["real_curve_gluonts"]  # port stand-in
    return None


def score_model(model_short: str, run_dir: str, predictor_dir: str,
                display_lookup: Dict[str, Tuple[str, bool]],
                ge_cache: Dict[Tuple[str, str], Optional[GiftEvalCache]],
                naive_baselines: dict, mase_metric: str, k: int,
                device: str, batch_size: int) -> Tuple[List[dict], List[dict]]:
    """Return ``(records, trials_meta)``: one record per (dataset, term) cell for
    this model, holding the per-predictor chosen-window MASE/window/FLOPs
    (``pred_{mase,window,flops}_0..k-1``), plus full_mase and the seasonal-naive
    denominator; and the top-K trial metadata (idx + selection metric) for the
    per-trial breakdown. GiftEval caches are shared across models via
    ``ge_cache`` so each (dataset, term) is loaded at most once.

    Consistency check: the rank-1 trial is selected by the SAME criterion as the
    shipped best_model.pt, so its per-cell window choice must agree with the
    stage-3 npz's stored ``predicted_mean`` argmin — the agreement rate is
    printed per model, and any systematic disagreement means the checkpoint
    rebuild/inference path here diverges from stage 3 (a harness bug), whereas
    ~100% agreement certifies the top-K numbers as genuine predictor spread."""
    compare_dir = os.path.join(run_dir, "models", model_short, "compare_real_vs_predicted")
    summary_path = os.path.join(compare_dir, "compare_summary.csv")
    if not os.path.isfile(summary_path):
        print(Fore.YELLOW + f"  No compare_summary.csv for {model_short}, skipping." + Fore.RESET)
        return [], []

    trials_meta, best_cfg = select_topk_trials(predictor_dir, k)
    if not trials_meta:
        print(Fore.RED + f"  {model_short}: no valid trials with checkpoints in "
              f"{predictor_dir}." + Fore.RESET)
        return [], []
    trial_ids = [t["trial_idx"] for t in trials_meta]
    if len(trial_ids) < k:
        print(Fore.YELLOW + f"  {model_short}: only {len(trial_ids)} trials with "
              f"checkpoints (< k={k}); using those." + Fore.RESET)
    print(Fore.GREEN + f"  {model_short}: top-{len(trial_ids)} trials by "
          f"{_METRIC_KEY[SELECTION_METRIC]} -> "
          + ", ".join(f"{t['trial_idx']:03d}({t['metric']:.4f})" for t in trials_meta)
          + Fore.RESET)

    pred_ctx_len = int(best_cfg["context_length"])
    horizon_grid = list(best_cfg["horizon_grid"])
    predictors = [load_trial_predictor(predictor_dir, ti, best_cfg, device)
                  for ti in trial_ids]

    n_agree = n_checked = 0  # rank-1 vs shipped-npz argmin agreement

    summary = pd.read_csv(summary_path)
    records: List[dict] = []
    for _, row in summary.iterrows():
        dataset_display = str(row["dataset_display"])
        term = str(row["term"])
        n_instances = int(row["n_instances"])

        npz_path = os.path.join(compare_dir, _npz_filename(dataset_display, term, model_short))
        if not os.path.isfile(npz_path):
            print(Fore.YELLOW + f"  Missing .npz {dataset_display} t={term} {model_short}" + Fore.RESET)
            continue
        try:
            data = np.load(npz_path)
        except Exception as exc:  # noqa: BLE001
            print(Fore.YELLOW + f"  Error loading {npz_path}: {exc}" + Fore.RESET)
            continue

        real_curve = _real_curve(data, mase_metric)
        if real_curve is None:
            print(Fore.YELLOW + f"  Skip {dataset_display} t={term} {model_short}: no "
                  f"{mase_metric} curve — re-run stage 3 (cheap backfill)." + Fore.RESET)
            continue
        window_grid = np.asarray(data["window_grid"])
        valid_idx = np.where(~np.isnan(real_curve))[0]
        if valid_idx.size < 1:
            continue
        full_idx = int(valid_idx[-1])
        full_mase = float(real_curve[full_idx])

        # GiftEval contexts (shared across models). None => could not be built.
        if dataset_display not in display_lookup:
            print(Fore.YELLOW + f"  Skip {dataset_display}: not in datasets_config "
                  "run set (cannot recover ge_name)." + Fore.RESET)
            continue
        ge_name, to_univariate = display_lookup[dataset_display]
        ds_key = (ge_name, term)
        if ds_key not in ge_cache:
            try:
                ge_dataset = GiftEvalDataset(name=ge_name, term=term,
                                             to_univariate=to_univariate)
                ge_cache[ds_key] = GiftEvalCache(ge_dataset, dataset_display)
            except Exception as exc:  # noqa: BLE001
                print(Fore.RED + f"  SKIP {ge_name} term={term} ({dataset_display}): {exc}"
                      + Fore.RESET)
                ge_cache[ds_key] = None
        cache = ge_cache[ds_key]
        if cache is None:
            continue

        h_idx = _closest_horizon_idx(cache.horizon, horizon_grid)
        model_id = str(row["model"])
        horizon = int(row.get("horizon_real", cache.horizon))
        full_flops = theoretical_flops(
            model_id, int(window_grid[full_idx]), horizon, DEFAULT_PATCH_SIZES)

        # Curve-shape diagnostics: how much a PERFECT window choice could have
        # won on this cell (oracle headroom), and how flat/degenerate the real
        # curve is. Flat curves (high tie_frac, low cv) mean there was nothing
        # to win and the argmin label itself is noise — expected for capped
        # (Sundial >2880) or fixed-context NaN-padded (PatchTST-FM) models.
        rc_valid = real_curve[valid_idx].astype(float)
        best_local = int(np.argmin(rc_valid))
        best_mase = float(rc_valid[best_local])
        curve_tie_frac = float(np.mean(rc_valid <= best_mase * 1.01))
        curve_cv = float(np.std(rc_valid) / (np.mean(rc_valid) + 1e-12))

        rec = {
            "model_short": model_short, "model_family": infer_model_family(model_id),
            "dataset_display": dataset_display,
            "term": term, "n_instances": n_instances, "full_mase": full_mase,
            "best_mase": best_mase,
            "best_window": int(window_grid[valid_idx[best_local]]),
            "curve_tie_frac": curve_tie_frac, "curve_cv": curve_cv,
            "n_windows_valid": int(valid_idx.size),
            "full_window": int(window_grid[full_idx]), "full_flops": full_flops,
            "naive_mase": float(naive_baselines.get(f"{dataset_display}/t{term}", {})
                                .get(mase_metric, float("nan"))),
        }
        # The shipped predictor's per-cell choice, straight from the stage-3 npz —
        # the ground truth the rank-1 trial must reproduce.
        shipped_idx = None
        if "predicted_mean" in data.files:
            shipped_idx = int(np.argmin(data["predicted_mean"]))
            if np.isnan(real_curve[shipped_idx]):
                shipped_idx = full_idx
        for k_i, predictor in enumerate(predictors):
            pred_curves = predict_curves_for_dataset(
                predictor, cache, pred_ctx_len, h_idx, device, batch_size=batch_size)
            pred_mean = pred_curves.mean(axis=0)
            pred_idx = int(np.argmin(pred_mean))
            if np.isnan(real_curve[pred_idx]):  # window unavailable -> full (clamp)
                pred_idx = full_idx
            rec[f"pred_mase_{k_i}"] = float(real_curve[pred_idx])
            rec[f"pred_window_{k_i}"] = int(window_grid[pred_idx])
            rec[f"pred_flops_{k_i}"] = theoretical_flops(
                model_id, int(window_grid[pred_idx]), horizon, DEFAULT_PATCH_SIZES)
            if k_i == 0 and shipped_idx is not None:
                agree = (pred_idx == shipped_idx)
                rec["rank1_agrees_with_shipped"] = bool(agree)
                n_checked += 1
                n_agree += int(agree)
        records.append(rec)

    # Consistency verdict: rank-1 trial vs the shipped best_model.pt's choices.
    if n_checked:
        rate = 100.0 * n_agree / n_checked
        color = Fore.GREEN if rate >= 95.0 else Fore.RED
        print(color + f"  {model_short}: rank-1 vs shipped-npz window agreement "
              f"{n_agree}/{n_checked} ({rate:.1f}%)"
              + ("" if rate >= 95.0 else
                 "  <-- LOW: checkpoint rebuild/inference here diverges from "
                 "stage 3 (or best_model.pt is not the rank-1 trial) — treat the "
                 "top-K numbers for this model as suspect.")
              + Fore.RESET)

    # Free predictor VRAM before the next model.
    del predictors
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return records, trials_meta


# ==============================================================================
#  AGGREGATION
# ==============================================================================

def aggregate(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per-model (and TOTAL) rollup: full geomean MASE, and the mean ± std across
    the K predictors of each predictor's geomean-over-cells MASE. Normalised twins
    (÷ seasonal-naive) computed over the cells that carry a baseline."""
    pred_cols = [f"pred_mase_{i}" for i in range(k) if f"pred_mase_{i}" in df.columns]
    flops_cols = [f"pred_flops_{i}" for i in range(k) if f"pred_flops_{i}" in df.columns]

    def _agg(sub: pd.DataFrame, model_short: str) -> dict:
        full_vals = sub["full_mase"].values.astype(float)
        gm_full = _geomean(full_vals)
        # Oracle headroom: the ceiling on what ANY window strategy can win here.
        # A model with ~0 headroom cannot land above the zero line — its top-K
        # mean being negative is a property of its curves, not of the predictor.
        gm_best = (_geomean(sub["best_mase"].values.astype(float))
                   if "best_mase" in sub.columns else float("nan"))
        oracle_rel = (100.0 * (gm_full - gm_best) / gm_full
                      if pd.notna(gm_best) and gm_full > 0 else float("nan"))
        nm = sub["naive_mase"].values.astype(float)
        vmask = np.isfinite(nm) & (nm > 0)
        w = sub["n_instances"].values.astype(float)
        ff_vals = sub["full_flops"].values.astype(float)

        # Per-predictor aggregates -> K values -> mean ± std. Each predictor is
        # scored only on the cells it actually has (a model with fewer than K
        # checkpointed trials leaves NaN in the higher-rank columns; on the TOTAL
        # row those cells are masked out per column instead of poisoning the
        # geomean). The paired full-window aggregate uses the SAME mask so every
        # rel-drop is a like-for-like comparison.
        strat_k, rel_k, fsav_k, strat_k_norm = [], [], [], []
        for i, c in enumerate(pred_cols):
            v = sub[c].values.astype(float)
            m = np.isfinite(v)
            if not m.any():
                continue
            gm_i = _geomean(v[m])
            gm_full_i = _geomean(full_vals[m])
            strat_k.append(gm_i)
            rel_k.append(100.0 * (gm_full_i - gm_i) / gm_full_i
                         if gm_full_i > 0 else float("nan"))
            fc = f"pred_flops_{i}"
            if fc in sub.columns:
                fv = sub[fc].values.astype(float)
                mf = m & np.isfinite(fv) & np.isfinite(ff_vals)
                ff = float((ff_vals[mf] * w[mf]).sum())
                fsav_k.append(1.0 - float((fv[mf] * w[mf]).sum()) / ff
                              if mf.any() and ff > 0 else float("nan"))
            else:
                fsav_k.append(float("nan"))
            mn = m & vmask
            strat_k_norm.append(_geomean(v[mn] / nm[mn]) if mn.any() else float("nan"))
        strat_k = np.asarray(strat_k); rel_k = np.asarray(rel_k)
        fsav_k = np.asarray(fsav_k); strat_k_norm = np.asarray(strat_k_norm)

        # Normalised (leaderboard) full-window geomean over rows with a baseline.
        if vmask.any():
            gm_full_norm = _geomean(full_vals[vmask] / nm[vmask])
            n_norm = int(vmask.sum())
        else:
            gm_full_norm = float("nan")
            n_norm = 0

        strat_mean = float(np.mean(strat_k)) if strat_k.size else float("nan")
        strat_std = float(np.std(strat_k)) if strat_k.size else float("nan")
        has_norm_vals = strat_k_norm.size and np.isfinite(strat_k_norm).any()
        sn_mean = float(np.nanmean(strat_k_norm)) if has_norm_vals else float("nan")
        sn_std = float(np.nanstd(strat_k_norm)) if has_norm_vals else float("nan")
        return {
            "model_short": model_short, "n_rows": int(len(sub)),
            "n_predictors": int(strat_k.size),
            "geomean_full_mase": gm_full,
            "geomean_strategy_mase": strat_mean,
            "geomean_strategy_mase_std": strat_std,
            "mase_drop_vs_full": gm_full - strat_mean,
            "rel_mase_drop_pct": (100.0 * (gm_full - strat_mean) / gm_full
                                  if gm_full > 0 else float("nan")),
            # Curve-shape diagnostics (why this model can/can't win):
            "oracle_rel_mase_drop_pct": oracle_rel,
            "median_curve_tie_frac": (float(sub["curve_tie_frac"].median())
                                      if "curve_tie_frac" in sub.columns else float("nan")),
            "median_curve_cv": (float(sub["curve_cv"].median())
                                if "curve_cv" in sub.columns else float("nan")),
            "geomean_full_mase_norm": gm_full_norm,
            "geomean_strategy_mase_norm": sn_mean,
            "geomean_strategy_mase_norm_std": sn_std,
            "rel_mase_drop_pct_norm": (100.0 * (gm_full_norm - sn_mean) / gm_full_norm
                                       if pd.notna(gm_full_norm) and gm_full_norm > 0
                                       else float("nan")),
            "n_norm": n_norm,
            # Overview coordinates: mean ± std across the K predictors of the
            # per-predictor relative MASE drop (y) and FLOPs saved (x, as a fraction).
            "rel_mase_drop_pct_mean": (float(np.nanmean(rel_k))
                                       if rel_k.size and np.isfinite(rel_k).any()
                                       else float("nan")),
            "rel_mase_drop_pct_pstd": (float(np.nanstd(rel_k))
                                       if rel_k.size and np.isfinite(rel_k).any()
                                       else float("nan")),
            "pct_flops_saved_mean":  (float(np.nanmean(fsav_k))
                                      if fsav_k.size and np.isfinite(fsav_k).any()
                                      else float("nan")),
            "pct_flops_saved_pstd":  (float(np.nanstd(fsav_k))
                                      if fsav_k.size and np.isfinite(fsav_k).any()
                                      else float("nan")),
        }

    rows = [_agg(sub.reset_index(drop=True), ms)
            for ms, sub in df.groupby("model_short")]
    rows.append(_agg(df.reset_index(drop=True), "TOTAL"))
    return pd.DataFrame(rows)


def trial_breakdown(df: pd.DataFrame, trials_by_model: Dict[str, List[dict]],
                    k: int) -> pd.DataFrame:
    """Per-(model, rank) diagnostic: which of the K trials drags the mean.

    One row per (model_short, rank 0..K-1): the trial's selection metric
    (val_regret), its geomean MASE over the model's cells, the relative drop vs
    full, its FLOPs saved, and the distribution of windows it chose. A model
    whose top-K spans a wide val_regret range (rank-1 far better than rank-5)
    is a search that found ONE good config, not five — the top-K mean then
    understates the shipped predictor, which is a finding about selection
    stability, not a harness bug."""
    rows = []
    for model_short, sub in df.groupby("model_short"):
        sub = sub.reset_index(drop=True)
        gm_full = _geomean(sub["full_mase"].values.astype(float))
        w = sub["n_instances"].values.astype(float)
        full_f = float((sub["full_flops"].values.astype(float) * w).sum())
        metas = trials_by_model.get(model_short, [])
        for i in range(k):
            mc, wc, fc = f"pred_mase_{i}", f"pred_window_{i}", f"pred_flops_{i}"
            if mc not in sub.columns or sub[mc].isna().all():
                continue
            gm_i = _geomean(sub[mc].values.astype(float))
            meta = metas[i] if i < len(metas) else {}
            windows = sub[wc].dropna().values if wc in sub.columns else np.array([])
            fsave = (1.0 - float((sub[fc].values.astype(float) * w).sum()) / full_f
                     if fc in sub.columns and full_f > 0 else float("nan"))
            rows.append({
                "model_short": model_short, "rank": i,
                "trial_idx": meta.get("trial_idx"),
                "selection_metric": meta.get("metric"),
                "geomean_mase": gm_i,
                "rel_mase_drop_pct": (100.0 * (gm_full - gm_i) / gm_full
                                      if gm_full > 0 else float("nan")),
                "pct_flops_saved": fsave,
                "window_mean": float(windows.mean()) if windows.size else float("nan"),
                "window_median": float(np.median(windows)) if windows.size else float("nan"),
                "window_min": int(windows.min()) if windows.size else -1,
                "window_max": int(windows.max()) if windows.size else -1,
                "n_cells": int(len(sub)),
                "rank1_agreement_pct": (
                    100.0 * sub["rank1_agrees_with_shipped"].mean()
                    if i == 0 and "rank1_agrees_with_shipped" in sub.columns
                    and sub["rank1_agrees_with_shipped"].notna().any() else float("nan")),
            })
    return pd.DataFrame(rows)


# ==============================================================================
#  TABLE FIGURE
# ==============================================================================

def plot_topk_table(rollup: pd.DataFrame, out_path: str, k: int,
                    metric_label: str, variant_label: str = "") -> str:
    """Render the mean ± std MASE-change table (twin of ``plot_mase_change_table``,
    with the single predictor replaced by the top-K mean ± std)."""
    r = rollup.copy()
    r["__is_total"] = r["model_short"] == "TOTAL"
    r = r.sort_values(["__is_total", "model_short"]).reset_index(drop=True)

    has_norm = r["geomean_full_mase_norm"].notna().any()
    headers = ["Model", f"{metric_label} full\n(original)",
               f"{metric_label} top-{k}\n(mean ± std)", "Δ", "Δ% vs full\n(>0 better)"]
    if has_norm:
        headers += [f"{metric_label} full\nNORM (÷naive)",
                    f"{metric_label} top-{k}\nNORM (mean ± std)", "Δ% NORM\n(>0 better)"]

    def _tint(rel: float, tot: bool) -> str:
        base = "#F2F2F2" if tot else "#FFFFFF"
        if not pd.notna(rel):
            return base
        return "#E2EFDA" if rel > 0 else ("#FCE4E4" if rel < 0 else base)

    cell_text, cell_colors, is_total = [], [], []
    for _, row in r.iterrows():
        tot = bool(row["__is_total"])
        is_total.append(tot)
        base = "#F2F2F2" if tot else "#FFFFFF"
        rel = row["rel_mase_drop_pct"]
        delta = row["geomean_strategy_mase"] - row["geomean_full_mase"]
        txt = [
            row["model_short"],
            f"{row['geomean_full_mase']:.4f}",
            f"{row['geomean_strategy_mase']:.4f} ± {row['geomean_strategy_mase_std']:.4f}",
            f"{delta:+.4f}",
            f"{rel:+.2f}%" if pd.notna(rel) else "—",
        ]
        tint = _tint(rel, tot)
        cols = [base, base, base, tint, tint]
        if has_norm:
            fn = row["geomean_full_mase_norm"]
            sn = row["geomean_strategy_mase_norm"]
            sn_std = row["geomean_strategy_mase_norm_std"]
            reln = row["rel_mase_drop_pct_norm"]
            txt += [
                f"{fn:.4f}" if pd.notna(fn) else "—",
                (f"{sn:.4f} ± {sn_std:.4f}" if pd.notna(sn) else "—"),
                f"{reln:+.2f}%" if pd.notna(reln) else "—",
            ]
            cols += [base, base, _tint(reln, tot)]
        cell_text.append(txt)
        cell_colors.append(cols)

    n_rows = len(cell_text)
    fig_w = 15.5 if has_norm else 11.5
    fig, ax = plt.subplots(figsize=(fig_w, 1.1 + 0.36 * (n_rows + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    for (rr, _cc), cell in tbl.get_celld().items():
        if rr == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor("#4472C4")
        elif is_total[rr - 1]:
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#BFBFBF")

    vtag = f"  ·  {variant_label}" if variant_label else ""
    ax.set_title(f"MASE change per model  [{metric_label}]{vtag}\nfull window (original) "
                 f"vs the mean ± std over the TOP-{k} predictors",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_topk_overview(rollup: pd.DataFrame, out_path: str, k: int,
                       metric_label: str, variant_label: str = "",
                       flat_models: Optional[set] = None) -> str:
    """Compute-saved vs accuracy-change overview (twin of
    ``plot_model_strategy_overview``): one point per model for the TOP-K-predictor
    strategy, at (mean FLOPs saved, mean relative MASE drop) across the K
    predictors, with ± std error bars in both axes. Top-right = cheaper & better.

    ``flat_models`` are model_shorts whose compute does not actually vary with
    the window (TiRex, PatchTST-FM): their x-position is a theoretical proxy
    only — they get hollow markers + a † in the legend + a footnote."""
    from matplotlib.lines import Line2D

    flat_models = flat_models or set()
    r = rollup[rollup["model_short"] != "TOTAL"].dropna(
        subset=["pct_flops_saved_mean", "rel_mase_drop_pct_mean"]).copy()
    if r.empty:
        return ""

    models = sorted(r["model_short"].unique())
    marker_of = {m: _MODEL_MARKERS[i % len(_MODEL_MARKERS)] for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(11, 8))
    x = r["pct_flops_saved_mean"].values * 100.0
    y = r["rel_mase_drop_pct_mean"].values
    xerr = r["pct_flops_saved_pstd"].values * 100.0
    yerr = r["rel_mase_drop_pct_pstd"].values

    x_pad = max(2.0, 0.08 * (np.ptp(x) or 1.0)) + float(np.nanmax(xerr) if xerr.size else 0)
    y_pad = max(1e-3, 0.12 * (np.ptp(y) or 1.0)) + float(np.nanmax(yerr) if yerr.size else 0)
    x_lo, x_hi = x.min() - x_pad, x.max() + x_pad
    y_lo, y_hi = y.min() - y_pad, y.max() + y_pad
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)

    # Win quadrant (cheaper & better) + zero crosshair.
    ax.axhspan(0, max(0.0, y_hi), color="#70AD47", alpha=0.04)
    ax.axvspan(0, max(0.0, x_hi), color="#70AD47", alpha=0.04)
    ax.axhline(0, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax.axvline(0, color="gray", lw=1.2, ls="-.", alpha=0.7)

    color = "#C0504D"  # single strategy (top-K predictor mean)
    for xi, yi, xe, ye, m in zip(x, y, xerr, yerr, r["model_short"].values):
        ax.errorbar(xi, yi, xerr=xe, yerr=ye, fmt="none", ecolor=color,
                    elinewidth=1.0, alpha=0.45, capsize=3, zorder=2)
        if m in flat_models:
            # Hollow marker: the x-position is a theoretical proxy only.
            ax.scatter(xi, yi, marker=marker_of[m], s=180, facecolors="none",
                       edgecolors=color, linewidths=1.6, alpha=0.95, zorder=3)
        else:
            ax.scatter(xi, yi, marker=marker_of[m], s=180, c=color,
                       edgecolors="white", linewidths=0.9, alpha=0.95, zorder=3)

    for qx, qy, txt, ha in [
        (0.985, 0.985, "Cheaper & Better",  "right"),
        (0.015, 0.985, "Costlier & Better", "left"),
        (0.985, 0.015, "Cheaper & Worse",   "right"),
        (0.015, 0.015, "Costlier & Worse",  "left"),
    ]:
        ax.text(qx, qy, txt, transform=ax.transAxes, fontsize=8.5,
                ha=ha, va="top" if qy > 0.5 else "bottom",
                color="gray", style="italic", alpha=0.75)

    ax.set_xlabel("FLOPs saved vs full window  =  100·(1 − FLOPs_topK / FLOPs_full)   (%)",
                  fontsize=11)
    ax.set_ylabel(f"Relative {metric_label} change vs full  =  "
                  f"100·({metric_label}_full − {metric_label}_topK)/{metric_label}_full   (%, >0 = better)",
                  fontsize=11)
    vtag = f"  ·  {variant_label}" if variant_label else ""
    ax.set_title(f"Model Overview — Compute Saved vs Accuracy Change  [{metric_label}]{vtag}\n"
                 f"TOP-{k}-predictor mean ± std (error bars = spread across the {k} predictors)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)

    model_handles = [
        Line2D([0], [0], marker=marker_of[m], linestyle="none", markersize=9,
               markerfacecolor="none" if m in flat_models else "0.35",
               markeredgecolor="0.35" if m in flat_models else "white",
               label=m + (" †" if m in flat_models else ""))
        for m in models
    ]
    ax.legend(handles=model_handles, title="Model", fontsize=8.5, title_fontsize=10,
              loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)

    if flat_models & set(models):
        fig.text(0.01, 0.005,
                 "† hollow markers: compute does not actually vary with the window "
                 "(TiRex pads to 2048; PatchTST-FM fixed 8192) — the FLOPs axis is a "
                 "theoretical proxy only.",
                 fontsize=8, color="gray", style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ==============================================================================
#  MAIN
# ==============================================================================

def run_one_variant(variant: str, args: argparse.Namespace,
                    display_lookup: Dict[str, Tuple[str, bool]],
                    suffix: str, metric_label: str) -> None:
    """Build the top-K table for a single predictor variant (normal / cheap /
    mamba). Reads the variant's own ablation tree + predictor root; falls back to
    the base ``general/`` tree for the (predictor-independent) seasonal-naive
    baselines if the variant tree lacks its own."""
    ps, tree_name, vlabel = VARIANTS[variant]
    run_dir = os.path.normpath(os.path.join(args.ablation_root, tree_name))
    predictor_root = os.path.normpath(args.predictor_root_base + ps)
    out_dir = os.path.normpath(args.output_dir or run_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(Fore.MAGENTA + "\n" + "#" * 78
          + f"\n#  VARIANT: {variant}  ({vlabel})"
          + f"\n#  ablation: {run_dir}\n#  predictors: {predictor_root}\n"
          + "#" * 78 + Fore.RESET)

    models_root = os.path.join(run_dir, "models")
    if not os.path.isdir(models_root):
        print(Fore.YELLOW + f"  No models/ dir in {run_dir} — skip variant {variant}. "
              f"(Has run_all{ps or ''} been run?)" + Fore.RESET)
        return
    model_shorts = sorted(os.listdir(models_root))
    if args.models:
        model_shorts = [m for m in model_shorts if m in set(args.models)]

    # Seasonal-naive denominators come directly from the published GIFT-Eval CSV;
    # ``run_dir`` is accepted by the shared loader only for API compatibility.
    naive_baselines = _load_naive_baselines(run_dir)

    ge_cache: Dict[Tuple[str, str], Optional[GiftEvalCache]] = {}
    all_records: List[dict] = []
    trials_by_model: Dict[str, List[dict]] = {}
    for model_short in model_shorts:
        predictor_dir = os.path.join(predictor_root, model_short)
        if not os.path.isfile(os.path.join(predictor_dir, "best_config.json")):
            print(Fore.YELLOW + f"  {model_short}: no predictor run at {predictor_dir}, "
                  "skipping." + Fore.RESET)
            continue
        print(Fore.CYAN + f"\n=== [{variant}] {model_short} ===" + Fore.RESET)
        recs, trials_meta = score_model(
            model_short, run_dir, predictor_dir, display_lookup,
            ge_cache, naive_baselines, args.mase_metric, args.k,
            args.device, args.predictor_batch_size)
        if args.datasets:
            recs = [r for r in recs if r["dataset_display"] in set(args.datasets)]
        all_records.extend(recs)
        trials_by_model[model_short] = trials_meta

    if not all_records:
        print(Fore.YELLOW + f"  No records for variant {variant} — check the tree / "
              "predictor root / stage-3 curves." + Fore.RESET)
        return

    df = pd.DataFrame(all_records)
    rollup = aggregate(df, args.k)
    breakdown = trial_breakdown(df, trials_by_model, args.k)
    flat_models = set(df.loc[df["model_family"].isin(FLAT_COMPUTE_FAMILIES),
                             "model_short"].unique())

    base = f"mase_change_table_topk_{variant}{suffix}"
    csv_path = os.path.join(out_dir, base + ".csv")
    rollup.to_csv(csv_path, index=False)
    # Diagnostics: raw per-cell records (per-predictor MASE/window/agreement) and
    # the per-(model, rank) trial breakdown — enough to see which trial drags a
    # model's mean and whether the rank-1 trial reproduces the shipped predictor.
    cells_path = os.path.join(out_dir, f"topk_cells_{variant}{suffix}.csv")
    df.to_csv(cells_path, index=False)
    breakdown_path = os.path.join(out_dir, f"topk_trial_breakdown_{variant}{suffix}.csv")
    breakdown.to_csv(breakdown_path, index=False)

    png_path = plot_topk_table(rollup, os.path.join(out_dir, base + ".png"),
                               args.k, metric_label, variant_label=vlabel)
    overview_path = plot_topk_overview(
        rollup, os.path.join(out_dir, f"model_strategy_overview_topk_{variant}{suffix}.png"),
        args.k, metric_label, variant_label=vlabel, flat_models=flat_models)

    # Console: per-trial breakdown so the offender is visible without opening CSVs.
    print(Fore.CYAN + f"\n[{variant}] Per-trial breakdown "
          f"(rank | trial | {_METRIC_KEY[SELECTION_METRIC]} | geomean MASE | Δ% | window med):"
          + Fore.RESET)
    for model_short, sub in breakdown.groupby("model_short"):
        agree = sub.loc[sub["rank"] == 0, "rank1_agreement_pct"]
        agree_txt = (f"  rank-1 agreement {float(agree.iloc[0]):.0f}%"
                     if len(agree) and pd.notna(agree.iloc[0]) else "")
        print(f"  {model_short}{agree_txt}")
        for _, b in sub.sort_values("rank").iterrows():
            sm = (f"{b['selection_metric']:.4f}"
                  if pd.notna(b["selection_metric"]) else "—")
            print(f"    #{int(b['rank'])}  trial {b['trial_idx']}  "
                  f"{sm}  ->  {b['geomean_mase']:.4f}  "
                  f"({b['rel_mase_drop_pct']:+.2f}%)  w_med={b['window_median']:.0f}")

    # "Why can('t) this model win": oracle headroom bounds the y-axis from above;
    # a flat curve (high tie-frac, low CV) means the argmin labels the predictor
    # was trained on are themselves noise. Expect the outlier models (capped /
    # fixed-context / sampling-based) to separate from the pack HERE, not in the
    # predictor code.
    print(Fore.CYAN + f"\n[{variant}] Curve-shape diagnostics "
          "(oracle Δ% = max achievable by ANY window strategy; "
          "tie-frac = share of windows within 1% of the min; CV = curve spread):"
          + Fore.RESET)
    diag = rollup[rollup["model_short"] != "TOTAL"].sort_values("oracle_rel_mase_drop_pct")
    for _, d in diag.iterrows():
        won = d["rel_mase_drop_pct_mean"]
        flag = ("  <-- ~no headroom: predictor could only lose here"
                if pd.notna(d["oracle_rel_mase_drop_pct"])
                and d["oracle_rel_mase_drop_pct"] < 1.0 else "")
        print(f"  {d['model_short']:<22} oracle {d['oracle_rel_mase_drop_pct']:+6.2f}%   "
              f"top-{args.k} {won:+6.2f}% ± {d['rel_mase_drop_pct_pstd']:.2f}   "
              f"tie-frac {d['median_curve_tie_frac']:.2f}   "
              f"CV {d['median_curve_cv']:.3f}{flag}")

    tot = rollup[rollup["model_short"] == "TOTAL"].iloc[0]
    print(Fore.GREEN + f"\n[{variant}] Saved: {png_path}"
          + (f"\n              {overview_path}" if overview_path else "")
          + f"\n              {csv_path}\n              {breakdown_path}"
          + f"\n              {cells_path}" + Fore.RESET)
    print(Fore.GREEN + f"[{variant}] TOTAL  full {tot['geomean_full_mase']:.4f} -> "
          f"top-{args.k} {tot['geomean_strategy_mase']:.4f} ± "
          f"{tot['geomean_strategy_mase_std']:.4f}  ({tot['rel_mase_drop_pct']:+.2f}%)"
          + Fore.RESET)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", type=str, nargs="+", default=["normal"],
                   choices=list(VARIANTS.keys()) + ["all"],
                   help="Which predictor variant(s) to build the table for: "
                        "normal (v1), cheap (v3), mamba (v4), or 'all'.")
    p.add_argument("--ablation-root", type=str, default=CACHE_ROOT,
                   help=f"Root holding the general* ablation trees (default {CACHE_ROOT}).")
    p.add_argument("--predictor-root-base", type=str, default=PREDICTOR_ROOT_DEFAULT,
                   help="Base predictor root; the _v3/_v4 suffix is appended per "
                        f"variant (default {PREDICTOR_ROOT_DEFAULT}).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where to write tables + CSVs (default: each variant's own "
                        "ablation tree). Filenames carry the variant tag either way.")
    p.add_argument("--k", type=int, default=5, help="How many top predictors to average.")
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="Restrict to these model_short names (default: all present).")
    p.add_argument("--mase-metric", type=str, default="mase_gluonts_real",
                   choices=["mase_gluonts", "mase_gluonts_real"])
    p.add_argument("--datasets", type=str, nargs="+", default=None,
                   help="Restrict to these dataset_display names (default: all).")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--predictor-batch-size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    variants = list(VARIANTS.keys()) if "all" in args.variants else args.variants
    # De-dup while preserving order (normal, cheap, mamba).
    seen: set = set()
    variants = [v for v in variants if not (v in seen or seen.add(v))]

    suffix = {"mase_gluonts_real": "_gluonts_real", "mase_gluonts": "_gluonts"}[args.mase_metric]
    metric_label = {"mase_gluonts_real": "MASE (gluonts-real)",
                    "mase_gluonts": "MASE (gluonts)"}[args.mase_metric]

    # display -> (ge_name, to_univariate) so a cell's dataset can be reloaded.
    display_lookup: Dict[str, Tuple[str, bool]] = {
        display: (ge_name, to_univariate)
        for ge_name, _term, display, to_univariate in datasets_config.datasets_to_run()
    }

    for variant in variants:
        run_one_variant(variant, args, display_lookup, suffix, metric_label)


if __name__ == "__main__":
    main()
