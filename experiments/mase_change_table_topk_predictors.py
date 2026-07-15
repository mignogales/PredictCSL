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
    CACHE_ROOT, _geomean, _load_naive_baselines, _npz_filename,
)

# Per-model predictor run dir layout (mirrors run_all.py's PREDICTOR_ROOT/<display>).
PREDICTOR_ROOT_DEFAULT = "logs/experiments/context_length_predictor"
BASE_RUN_DIR_DEFAULT   = os.path.join(CACHE_ROOT, "general")


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


def select_topk_trials(predictor_dir: str, k: int) -> Tuple[List[int], dict]:
    """Rank trials by the selection metric and return the K best trial indices
    (ascending metric = better) that have a durable checkpoint, plus the shared
    best_config.json (run-global constants: context_length / window_grid / …).

    Ranking column tracks ``SELECTION_METRIC`` exactly as the training run's
    ``_select_final`` does (``val_regret`` for the default 'regret').
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

    chosen: List[int] = []
    for _, row in df.iterrows():
        ti = int(row["trial_idx"])
        if os.path.isfile(_trial_best_ckpt(predictor_dir, ti)) and _trial_cfg(predictor_dir, ti):
            chosen.append(ti)
        if len(chosen) >= k:
            break
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
                device: str, batch_size: int) -> List[dict]:
    """Return one record per (dataset, term) cell for this model, holding the
    per-predictor chosen-window MASE (``pred_mase_0..k-1``), plus full_mase and
    the seasonal-naive denominator. GiftEval caches are shared across models via
    ``ge_cache`` so each (dataset, term) is loaded at most once."""
    compare_dir = os.path.join(run_dir, "models", model_short, "compare_real_vs_predicted")
    summary_path = os.path.join(compare_dir, "compare_summary.csv")
    if not os.path.isfile(summary_path):
        print(Fore.YELLOW + f"  No compare_summary.csv for {model_short}, skipping." + Fore.RESET)
        return []

    trial_ids, best_cfg = select_topk_trials(predictor_dir, k)
    if not trial_ids:
        print(Fore.RED + f"  {model_short}: no valid trials with checkpoints in "
              f"{predictor_dir}." + Fore.RESET)
        return []
    if len(trial_ids) < k:
        print(Fore.YELLOW + f"  {model_short}: only {len(trial_ids)} trials with "
              f"checkpoints (< k={k}); using those." + Fore.RESET)
    print(Fore.GREEN + f"  {model_short}: top-{len(trial_ids)} trials by "
          f"{_METRIC_KEY[SELECTION_METRIC]} -> {trial_ids}" + Fore.RESET)

    pred_ctx_len = int(best_cfg["context_length"])
    horizon_grid = list(best_cfg["horizon_grid"])
    predictors = [load_trial_predictor(predictor_dir, ti, best_cfg, device)
                  for ti in trial_ids]

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

        rec = {
            "model_short": model_short, "dataset_display": dataset_display,
            "term": term, "n_instances": n_instances, "full_mase": full_mase,
            "naive_mase": float(naive_baselines.get(f"{dataset_display}/t{term}", {})
                                .get(mase_metric, float("nan"))),
        }
        for k_i, predictor in enumerate(predictors):
            pred_curves = predict_curves_for_dataset(
                predictor, cache, pred_ctx_len, h_idx, device, batch_size=batch_size)
            pred_mean = pred_curves.mean(axis=0)
            pred_idx = int(np.argmin(pred_mean))
            if np.isnan(real_curve[pred_idx]):  # window unavailable -> full (clamp)
                pred_idx = full_idx
            rec[f"pred_mase_{k_i}"] = float(real_curve[pred_idx])
        records.append(rec)

    # Free predictor VRAM before the next model.
    del predictors
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return records


# ==============================================================================
#  AGGREGATION
# ==============================================================================

def aggregate(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per-model (and TOTAL) rollup: full geomean MASE, and the mean ± std across
    the K predictors of each predictor's geomean-over-cells MASE. Normalised twins
    (÷ seasonal-naive) computed over the cells that carry a baseline."""
    pred_cols = [f"pred_mase_{i}" for i in range(k) if f"pred_mase_{i}" in df.columns]

    def _agg(sub: pd.DataFrame, model_short: str) -> dict:
        full_vals = sub["full_mase"].values.astype(float)
        gm_full = _geomean(full_vals)
        # Per-predictor geomean over cells -> K aggregates -> mean ± std.
        strat_k = np.array([_geomean(sub[c].values.astype(float)) for c in pred_cols])

        # Normalised (leaderboard) geomeans over rows with a valid baseline.
        nm = sub["naive_mase"].values.astype(float)
        vmask = np.isfinite(nm) & (nm > 0)
        if vmask.any():
            gm_full_norm = _geomean(full_vals[vmask] / nm[vmask])
            strat_k_norm = np.array([
                _geomean(sub[c].values.astype(float)[vmask] / nm[vmask]) for c in pred_cols])
            n_norm = int(vmask.sum())
        else:
            gm_full_norm = float("nan")
            strat_k_norm = np.full(len(pred_cols), float("nan"))
            n_norm = 0

        strat_mean = float(np.mean(strat_k))
        strat_std = float(np.std(strat_k))
        sn_mean = float(np.nanmean(strat_k_norm)) if n_norm else float("nan")
        sn_std = float(np.nanstd(strat_k_norm)) if n_norm else float("nan")
        return {
            "model_short": model_short, "n_rows": int(len(sub)),
            "n_predictors": len(pred_cols),
            "geomean_full_mase": gm_full,
            "geomean_strategy_mase": strat_mean,
            "geomean_strategy_mase_std": strat_std,
            "mase_drop_vs_full": gm_full - strat_mean,
            "rel_mase_drop_pct": (100.0 * (gm_full - strat_mean) / gm_full
                                  if gm_full > 0 else float("nan")),
            "geomean_full_mase_norm": gm_full_norm,
            "geomean_strategy_mase_norm": sn_mean,
            "geomean_strategy_mase_norm_std": sn_std,
            "rel_mase_drop_pct_norm": (100.0 * (gm_full_norm - sn_mean) / gm_full_norm
                                       if pd.notna(gm_full_norm) and gm_full_norm > 0
                                       else float("nan")),
            "n_norm": n_norm,
        }

    rows = [_agg(sub.reset_index(drop=True), ms)
            for ms, sub in df.groupby("model_short")]
    rows.append(_agg(df.reset_index(drop=True), "TOTAL"))
    return pd.DataFrame(rows)


# ==============================================================================
#  TABLE FIGURE
# ==============================================================================

def plot_topk_table(rollup: pd.DataFrame, out_path: str, k: int,
                    metric_label: str) -> str:
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

    ax.set_title(f"MASE change per model  [{metric_label}]  — full window (original) "
                 f"vs the mean ± std over the TOP-{k} predictors",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ==============================================================================
#  MAIN
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=str, default=BASE_RUN_DIR_DEFAULT,
                   help=f"Base ablation tree to read (default {BASE_RUN_DIR_DEFAULT}).")
    p.add_argument("--predictor-root", type=str, default=PREDICTOR_ROOT_DEFAULT,
                   help="Root holding per-model predictor runs (<root>/<model_short>).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where to write the table + CSV (default: --run-dir).")
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
    run_dir = os.path.normpath(args.run_dir)
    out_dir = os.path.normpath(args.output_dir or run_dir)
    os.makedirs(out_dir, exist_ok=True)

    suffix = {"mase_gluonts_real": "_gluonts_real", "mase_gluonts": "_gluonts"}[args.mase_metric]
    metric_label = {"mase_gluonts_real": "MASE (gluonts-real)",
                    "mase_gluonts": "MASE (gluonts)"}[args.mase_metric]

    models_root = os.path.join(run_dir, "models")
    if not os.path.isdir(models_root):
        raise SystemExit(f"No models/ dir in {run_dir}")
    model_shorts = sorted(os.listdir(models_root))
    if args.models:
        model_shorts = [m for m in model_shorts if m in set(args.models)]

    # display -> (ge_name, to_univariate) so a cell's dataset can be reloaded.
    display_lookup: Dict[str, Tuple[str, bool]] = {
        display: (ge_name, to_univariate)
        for ge_name, _term, display, to_univariate in datasets_config.datasets_to_run()
    }

    naive_baselines = _load_naive_baselines(run_dir)
    ge_cache: Dict[Tuple[str, str], Optional[GiftEvalCache]] = {}

    all_records: List[dict] = []
    for model_short in model_shorts:
        predictor_dir = os.path.join(args.predictor_root, model_short)
        if not os.path.isfile(os.path.join(predictor_dir, "best_config.json")):
            print(Fore.YELLOW + f"  {model_short}: no predictor run at {predictor_dir}, "
                  "skipping." + Fore.RESET)
            continue
        print(Fore.CYAN + f"\n=== {model_short} ===" + Fore.RESET)
        recs = score_model(model_short, run_dir, predictor_dir, display_lookup,
                           ge_cache, naive_baselines, args.mase_metric, args.k,
                           args.device, args.predictor_batch_size)
        if args.datasets:
            recs = [r for r in recs if r["dataset_display"] in set(args.datasets)]
        all_records.extend(recs)

    if not all_records:
        raise SystemExit("No records produced — check run dir / predictor root / stage-3 curves.")

    df = pd.DataFrame(all_records)
    rollup = aggregate(df, args.k)

    csv_path = os.path.join(out_dir, f"mase_change_table_topk{suffix}.csv")
    rollup.drop(columns=["__is_total"], errors="ignore").to_csv(csv_path, index=False)
    png_path = plot_topk_table(rollup, os.path.join(out_dir,
                               f"mase_change_table_topk{suffix}.png"), args.k, metric_label)

    tot = rollup[rollup["model_short"] == "TOTAL"].iloc[0]
    print(Fore.GREEN + f"\nSaved: {png_path}\n       {csv_path}" + Fore.RESET)
    print(Fore.GREEN + f"TOTAL  full {tot['geomean_full_mase']:.4f} -> top-{args.k} "
          f"{tot['geomean_strategy_mase']:.4f} ± {tot['geomean_strategy_mase_std']:.4f}  "
          f"({tot['rel_mase_drop_pct']:+.2f}%)" + Fore.RESET)


if __name__ == "__main__":
    main()
