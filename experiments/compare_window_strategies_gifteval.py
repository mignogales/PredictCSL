"""
compare_window_strategies_gifteval.py

Reads cached ablation results produced by test_window_ablation_gifteval_v5.py
and compares MASE + wall-clock time + theoretical complexity under three
context-selection strategies per (model, dataset, term):

  full_window   -- largest valid window in the ablation grid (maximum context)
  best_window   -- argmin of real MASE curve (oracle; best achievable from grid)
  pred_window   -- argmin of the predictor's mean curve (zero-shot recommendation)

No model inference is performed -- all numbers come from the v5 cache and the
compare_real_vs_predicted/*.npz files.

Complexity model (per model family)
------------------------------------
All current TSFMs use patch-based transformers, so the dominant complexity term
is quadratic in the number of context patches.  We model:

  FLOPs(C, H) ∝  n_ctx(C)² + n_ctx(C)·n_hor(H)

where n_ctx = ⌈C/P⌉, n_hor = ⌈H/P⌉, and P is the effective patch size.

For Moirai family the context + horizon tokens are processed jointly:

  FLOPs(C, H) ∝  (n_ctx(C) + n_hor(H))²

Patch sizes used (adjust via --patch-sizes JSON if needed):
  moirai        32  (Moirai2, frequency-adaptive but ~32 in practice)
  moirai_1_1    32  (explicit MOIRAI_1_1_PATCH_SIZE = 32 in v5 code)
  chronos2       1  (VQ token per timestep)
  chronos_bolt  32  (patch-based fast variant)
  timesfm       32  (PATCH_SIZE = 32 in v5 code)
  patchtst_fm   16  (typical PatchTST default)

Outputs (written to <run_dir>/strategy_comparison/)
----------------------------------------------------
  comparison.csv              per-row MASE, elapsed time, complexity per strategy
  summary_stats.json          aggregate stats (mean/median, win rates, speedups)
  bar_aggregate_mase.png      mean & median MASE per strategy
  bar_aggregate_time.png      mean elapsed time per strategy
  scatter_pred_vs_best.png    MASE(pred) vs MASE(best)
  scatter_pred_vs_full.png    MASE(pred) vs MASE(full)
  efficiency_frontier.png     MASE gain vs speedup scatter
  gain_histogram.png          distribution of relative MASE gain over full window
  gain_vs_best_histogram.png  distribution of regret vs oracle
  complexity_reduction.png    distribution of complexity ratio (pred vs full)
  per_dataset_bars.png        grouped bars (full / best / pred) per dataset
  window_choice_scatter.png   predictor window vs oracle window
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from colorama import Fore

CACHE_ROOT = "logs/experiments/window_ablation_gifteval"

# ==============================================================================
#  COMPLEXITY MODEL
# ==============================================================================

# Effective patch sizes per model family.
# Changing these does NOT affect MASE or time columns; only the theoretical
# complexity columns.  Override at runtime with --patch-sizes.
DEFAULT_PATCH_SIZES: Dict[str, int] = {
    "moirai":           32,
    "moirai_1_1":       32,
    "chronos2":          1,
    "chronos_bolt":     32,
    "timesfm":          32,
    "patchtst_fm":      16,
    "context_parroting": 1,
}

# Families that process context + horizon in a unified sequence
# (full self-attention over the concatenated sequence).
UNIFIED_SEQUENCE_FAMILIES = {"moirai", "moirai_1_1"}


def infer_model_family(model_id: str) -> str:
    m = model_id.lower()
    if "moirai-2" in m or "moirai2" in m:
        return "moirai"
    if "moirai" in m:
        return "moirai_1_1"
    if "chronos-2" in m or "chronos2" in m:
        return "chronos2"
    if "chronos" in m:
        return "chronos_bolt"
    if "timesfm" in m:
        return "timesfm"
    if "patchtst" in m:
        return "patchtst_fm"
    return "unknown"


def _n_patches(length: int, patch_size: int) -> int:
    return max(1, math.ceil(length / patch_size))


def theoretical_flops(
    model_id: str,
    context: int,
    horizon: int,
    patch_sizes: Dict[str, int],
) -> float:
    """
    Unnormalized FLOPs proxy.  Useful only as a ratio between two context sizes
    for the same (model, horizon).

    Formula:
      unified-sequence (Moirai):  (n_ctx + n_hor)^2
      encoder-decoder (rest):     n_ctx^2  +  n_ctx * n_hor
    """
    family = infer_model_family(model_id)
    P = patch_sizes.get(family, 1)
    n_ctx = _n_patches(context, P)
    n_hor = _n_patches(horizon, P)
    if family in UNIFIED_SEQUENCE_FAMILIES:
        return float((n_ctx + n_hor) ** 2)
    else:
        return float(n_ctx ** 2 + n_ctx * n_hor)


# ==============================================================================
#  CACHE PATHS (mirrors test_window_ablation_gifteval_v5.py)
# ==============================================================================

def _cache_dir(
    cache_root: str,
    dataset_display: str,
    model_short: str,
    term: str,
    window_size: int,
) -> str:
    return os.path.join(
        cache_root, dataset_display, model_short, f"t{term}", f"w{window_size}"
    )


def _load_elapsed(
    cache_root: str,
    dataset_display: str,
    model_short: str,
    term: str,
    window_size: int,
) -> float:
    """Return elapsed_seconds from metrics.json, or NaN if unavailable."""
    path = os.path.join(
        _cache_dir(cache_root, dataset_display, model_short, term, window_size),
        "metrics.json",
    )
    if not os.path.isfile(path):
        return float("nan")
    try:
        with open(path) as f:
            d = json.load(f)
        v = d.get("elapsed_seconds")
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")


# ==============================================================================
#  RUN DISCOVERY
# ==============================================================================

def find_latest_run(cache_root: str) -> str:
    """Most recent run dir under cache_root/runs that has compare_real_vs_predicted."""
    runs_dir = os.path.join(cache_root, "runs")
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"No runs/ directory under {cache_root}")
    candidates = []
    for name in os.listdir(runs_dir):
        rd = os.path.join(runs_dir, name)
        if not os.path.isdir(rd):
            continue
        if os.path.isdir(os.path.join(rd, "compare_real_vs_predicted")):
            candidates.append((os.path.getmtime(rd), rd))
    if not candidates:
        raise FileNotFoundError(
            f"No run with compare_real_vs_predicted/ under {runs_dir}"
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


# ==============================================================================
#  DATA LOADING
# ==============================================================================

def _npz_filename(dataset_display: str, term: str, model_short: str) -> str:
    return f"compare_{dataset_display}_t{term}_{model_short}.npz"


def load_strategy_records(
    run_dir: str,
    cache_root: str,
    patch_sizes: Dict[str, int],
) -> pd.DataFrame:
    """
    For each row in compare_summary.csv, load the paired .npz and derive:
      MASE / elapsed_seconds / theoretical_flops for full / best / pred windows.

    Notes on elapsed_seconds
    ------------------------
    The time is total wall-clock seconds for the inference pass at that window.
    The number of valid samples can differ across windows (instances whose
    context_length >= window_size); we do not renormalize here because that
    info is not stored in metrics.json.  Use speedup = elapsed_full / elapsed_pred
    as a relative indicator, keeping in mind that the full window may have
    fewer valid samples, which can make the speedup appear conservative.
    """
    compare_dir = os.path.join(run_dir, "compare_real_vs_predicted")
    summary_path = os.path.join(compare_dir, "compare_summary.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"compare_summary.csv not found in {compare_dir}")

    summary = pd.read_csv(summary_path)
    records: List[dict] = []

    for _, row in summary.iterrows():
        dataset_display = str(row["dataset_display"])
        term = str(row["term"])
        model_short = str(row["model_short"])
        model = str(row["model"])
        horizon = int(row["horizon_real"])
        n_instances = int(row["n_instances"])

        npz_path = os.path.join(
            compare_dir, _npz_filename(dataset_display, term, model_short)
        )
        if not os.path.isfile(npz_path):
            print(
                Fore.YELLOW
                + f"  Missing .npz  {dataset_display} t={term} {model_short}"
                + Fore.RESET
            )
            continue

        try:
            data = np.load(npz_path)
        except Exception as exc:
            print(Fore.YELLOW + f"  Error loading {npz_path}: {exc}" + Fore.RESET)
            continue

        window_grid: np.ndarray = data["window_grid"]
        real_curve: np.ndarray = data["real_curve"]      # raw MASE per window
        pred_mean: np.ndarray = data["predicted_mean"]   # z-scored curve for argmin

        valid = ~np.isnan(real_curve)
        if valid.sum() < 1:
            print(
                Fore.YELLOW
                + f"  Skip {dataset_display} t={term} {model_short}: no valid MASE"
                + Fore.RESET
            )
            continue

        valid_indices = np.where(valid)[0]

        # --- Strategy indices ------------------------------------------------
        full_idx = int(valid_indices[-1])
        best_idx = int(valid_indices[np.argmin(real_curve[valid_indices])])
        pred_idx = int(np.argmin(pred_mean))

        # If the predictor chose a window the dataset can't serve (not enough
        # input context for any instance), fall back to the largest valid window.
        pred_clamped = bool(np.isnan(real_curve[pred_idx]))
        if pred_clamped:
            pred_idx = full_idx
            print(
                Fore.YELLOW
                + f"  Clamp {dataset_display} t={term} {model_short}: "
                + f"pred_window={int(window_grid[int(np.argmin(pred_mean))])} "
                + f"unavailable -> using full_window={int(window_grid[full_idx])}"
                + Fore.RESET
            )

        full_w  = int(window_grid[full_idx])
        best_w  = int(window_grid[best_idx])
        pred_w  = int(window_grid[pred_idx])

        full_mase = float(real_curve[full_idx])
        best_mase = float(real_curve[best_idx])
        pred_mase = float(real_curve[pred_idx])

        # --- Elapsed time (from per-window metrics.json) ---------------------
        full_elapsed = _load_elapsed(cache_root, dataset_display, model_short, term, full_w)
        best_elapsed = _load_elapsed(cache_root, dataset_display, model_short, term, best_w)
        pred_elapsed = _load_elapsed(cache_root, dataset_display, model_short, term, pred_w)

        # Speedup: how much faster than full window (>1 = faster)
        speedup_pred = (
            full_elapsed / pred_elapsed
            if pred_elapsed > 0 and not math.isnan(pred_elapsed) and not math.isnan(full_elapsed)
            else float("nan")
        )
        speedup_best = (
            full_elapsed / best_elapsed
            if best_elapsed > 0 and not math.isnan(best_elapsed) and not math.isnan(full_elapsed)
            else float("nan")
        )

        # --- Theoretical complexity ------------------------------------------
        full_flops = theoretical_flops(model, full_w, horizon, patch_sizes)
        best_flops = theoretical_flops(model, best_w, horizon, patch_sizes)
        pred_flops = theoretical_flops(model, pred_w, horizon, patch_sizes)

        complexity_ratio_pred = pred_flops / full_flops
        complexity_ratio_best = best_flops / full_flops

        records.append({
            # identity
            "model":           model,
            "model_short":     model_short,
            "model_family":    infer_model_family(model),
            "dataset_display": dataset_display,
            "term":            term,
            "horizon":         horizon,
            "n_instances":     n_instances,
            # windows chosen
            "full_window":     full_w,
            "best_window":     best_w,
            "pred_window":     pred_w,
            "n_windows_valid": int(valid.sum()),
            "pred_clamped":    pred_clamped,   # True when pred window was unavailable
            # MASE values
            "full_mase":       full_mase,
            "best_mase":       best_mase,
            "pred_mase":       pred_mase,
            # MASE deltas
            "delta_pred_vs_full": pred_mase - full_mase,
            "delta_best_vs_full": best_mase - full_mase,
            "delta_pred_vs_best": pred_mase - best_mase,
            "rel_gain_pred_over_full": (
                (full_mase - pred_mase) / (abs(full_mase) + 1e-12)
            ),
            # elapsed wall-clock time (seconds; NaN if not cached)
            "full_elapsed_s":  full_elapsed,
            "best_elapsed_s":  best_elapsed,
            "pred_elapsed_s":  pred_elapsed,
            "speedup_pred_vs_full": speedup_pred,
            "speedup_best_vs_full": speedup_best,
            # theoretical complexity (unnormalized FLOPs proxy)
            "full_flops":      full_flops,
            "best_flops":      best_flops,
            "pred_flops":      pred_flops,
            "complexity_ratio_pred_vs_full": complexity_ratio_pred,
            "complexity_ratio_best_vs_full": complexity_ratio_best,
        })

    if not records:
        raise RuntimeError("No valid records found — check the run directory.")

    df = pd.DataFrame(records)
    n_combos = len(df.groupby(["model_short", "dataset_display", "term"]))
    print(
        Fore.GREEN
        + f"  Loaded {len(df)} records  |  {n_combos} (model,dataset,term) combos"
        + Fore.RESET
    )
    return df


# ==============================================================================
#  AGGREGATE STATS
# ==============================================================================

def compute_summary_stats(df: pd.DataFrame) -> dict:
    r = df.dropna(subset=["full_mase", "best_mase", "pred_mase"])
    stats: dict = {}

    for strategy in ("full_mase", "best_mase", "pred_mase"):
        vals = r[strategy].values
        stats[strategy] = {
            "mean":   float(vals.mean()),
            "median": float(np.median(vals)),
            "std":    float(vals.std()),
            "n":      int(len(vals)),
        }

    stats["pred_clamped_count"] = int(df["pred_clamped"].sum()) if "pred_clamped" in df.columns else 0
    pred_beats_full = int((r["pred_mase"] < r["full_mase"]).sum())
    stats["pred_beats_full_count"] = pred_beats_full
    stats["pred_beats_full_rate"]  = pred_beats_full / max(len(r), 1)
    stats["pred_beats_best_count"] = int((r["pred_mase"] < r["best_mase"]).sum())
    stats["total_rows"] = int(len(r))

    gain = r["rel_gain_pred_over_full"].values
    stats["rel_gain_pred_over_full"] = {
        "mean":         float(gain.mean()),
        "median":       float(np.median(gain)),
        "pct_positive": float((gain > 0).mean()),
    }
    regret = r["delta_pred_vs_best"].values
    stats["regret_pred_vs_best"] = {
        "mean":   float(regret.mean()),
        "median": float(np.median(regret)),
    }

    # Timing stats (drop NaN rows)
    t = df.dropna(subset=["full_elapsed_s", "pred_elapsed_s"])
    if not t.empty:
        sp = t["speedup_pred_vs_full"].dropna().values
        stats["speedup_pred_vs_full"] = {
            "mean":          float(sp.mean()) if sp.size else float("nan"),
            "median":        float(np.median(sp)) if sp.size else float("nan"),
            "pct_faster":    float((sp > 1).mean()) if sp.size else float("nan"),
        }
        stats["mean_full_elapsed_s"] = float(t["full_elapsed_s"].mean())
        stats["mean_pred_elapsed_s"] = float(t["pred_elapsed_s"].mean())
        stats["mean_best_elapsed_s"] = float(df["best_elapsed_s"].dropna().mean()) if not df["best_elapsed_s"].dropna().empty else float("nan")

    # Complexity stats
    cr = df["complexity_ratio_pred_vs_full"].dropna().values
    if cr.size:
        stats["complexity_ratio_pred_vs_full"] = {
            "mean":   float(cr.mean()),
            "median": float(np.median(cr)),
        }

    return stats


# ==============================================================================
#  PLOTS — MASE
# ==============================================================================

def plot_bar_aggregate_mase(df: pd.DataFrame, out_dir: str) -> str:
    r = df.dropna(subset=["full_mase", "best_mase", "pred_mase"])
    if r.empty:
        return ""
    labels  = ["Full Window", "Best Window\n(Oracle)", "Predictor\nWindow"]
    means   = [r["full_mase"].mean(),   r["best_mase"].mean(),   r["pred_mase"].mean()]
    medians = [r["full_mase"].median(), r["best_mase"].median(), r["pred_mase"].median()]

    x = np.arange(len(labels))
    w = 0.35
    c_m = ["#4472C4", "#ED7D31", "#70AD47"]
    c_d = ["#264FA0", "#A9511B", "#3E7327"]

    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - w / 2, means,   w, label="Mean MASE",   color=c_m, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w / 2, medians, w, label="Median MASE", color=c_d, alpha=0.85, edgecolor="white")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("MASE", fontsize=12)
    ax.set_title(f"MASE by Context Strategy  (n={len(r)} dataset-terms)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3); ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(out_dir, "bar_aggregate_mase.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_scatter(
    df: pd.DataFrame,
    x_col: str, y_col: str,
    xlabel: str, ylabel: str,
    title: str, fname: str,
    out_dir: str,
    diagonal: bool = True,
) -> str:
    rows = df.dropna(subset=[x_col, y_col])
    if rows.empty:
        return ""
    x, y = rows[x_col].values, rows[y_col].values
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, alpha=0.55, s=30, color="#4472C4",
               edgecolors="white", linewidths=0.5)
    if diagonal:
        lims = [min(x.min(), y.min()) * 0.95, max(x.max(), y.max()) * 1.05]
        ax.plot(lims, lims, "k--", lw=1.2, alpha=0.6, label="y = x")
        ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if diagonal:
        ax.legend(fontsize=10)
    if len(x) > 1:
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=11,
                va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax.grid(True, alpha=0.25); plt.tight_layout()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_gain_histogram(df: pd.DataFrame, out_dir: str) -> str:
    """Relative MASE gain of predictor vs full window."""
    r = df.dropna(subset=["rel_gain_pred_over_full"])
    if r.empty:
        return ""
    vals = r["rel_gain_pred_over_full"].values
    p_pos = 100 * (vals > 0).mean()
    clip  = np.percentile(np.abs(vals), 98)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vals[np.abs(vals) <= clip], bins=min(60, max(20, len(vals) // 5)),
            color="#70AD47", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", lw=1.5, ls="--", alpha=0.7, label="No change")
    ax.axvline(vals.mean(), color="red", lw=1.5, ls="-",
               label=f"Mean = {vals.mean():.4f}")
    ax.axvline(np.median(vals), color="purple", lw=1.5, ls=":",
               label=f"Median = {np.median(vals):.4f}")
    ax.set_xlabel("(MASE_full − MASE_pred) / |MASE_full|  (positive = predictor helps)", fontsize=11)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Relative MASE Gain: Predictor vs Full Window  ({p_pos:.1f}% of cases predictor is better)",
        fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25); plt.tight_layout()
    path = os.path.join(out_dir, "gain_histogram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_regret_histogram(df: pd.DataFrame, out_dir: str) -> str:
    """MASE regret of predictor vs oracle."""
    r = df.dropna(subset=["delta_pred_vs_best"])
    if r.empty:
        return ""
    vals = r["delta_pred_vs_best"].values
    p_zero = 100 * (vals <= 0).mean()
    clip   = np.percentile(np.abs(vals), 98)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vals[np.abs(vals) <= clip], bins=min(60, max(20, len(vals) // 5)),
            color="#ED7D31", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", lw=1.5, ls="--", alpha=0.7, label="Predictor = Oracle")
    ax.axvline(vals.mean(), color="red", lw=1.5, ls="-",
               label=f"Mean = {vals.mean():.4f}")
    ax.axvline(np.median(vals), color="purple", lw=1.5, ls=":",
               label=f"Median = {np.median(vals):.4f}")
    ax.set_xlabel("MASE_pred − MASE_best  (0 = predictor matches oracle)", fontsize=11)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Predictor Regret vs Oracle  ({p_zero:.1f}% at or below oracle)",
        fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25); plt.tight_layout()
    path = os.path.join(out_dir, "gain_vs_best_histogram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


# ==============================================================================
#  PLOTS — TIMING
# ==============================================================================

def plot_bar_aggregate_time(df: pd.DataFrame, out_dir: str) -> str:
    """Bar chart of mean wall-clock elapsed time per strategy."""
    t = df.dropna(subset=["full_elapsed_s", "best_elapsed_s", "pred_elapsed_s"])
    if t.empty:
        print(Fore.YELLOW + "  No timing data available (all NaN)." + Fore.RESET)
        return ""

    labels = ["Full Window", "Best Window\n(Oracle)", "Predictor\nWindow"]
    means  = [t["full_elapsed_s"].mean(), t["best_elapsed_s"].mean(), t["pred_elapsed_s"].mean()]
    stds   = [t["full_elapsed_s"].std(),  t["best_elapsed_s"].std(),  t["pred_elapsed_s"].std()]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    bars = ax.bar(x, means, width=0.5, color=colors, alpha=0.85,
                  edgecolor="white", yerr=stds, capsize=5, error_kw={"linewidth": 1.2})
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(stds) * 0.05,
                f"{b.get_height():.2f}s", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Elapsed Time (s)", fontsize=12)
    ax.set_title(f"Mean Wall-Clock Time per Context Strategy  (n={len(t)} dataset-terms)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(bottom=0)
    ax.text(0.98, 0.97,
            "Note: n_valid_samples may differ\nacross windows (larger windows\n"
            "may exclude short-context series)",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            color="gray", style="italic")
    plt.tight_layout()
    path = os.path.join(out_dir, "bar_aggregate_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_efficiency_frontier(df: pd.DataFrame, out_dir: str) -> str:
    """
    Scatter: MASE gain (pred vs full) vs speedup (full / pred time).
    Upper-right quadrant = predictor is both more accurate AND faster.
    """
    r = df.dropna(subset=["speedup_pred_vs_full", "delta_pred_vs_full"])
    if r.empty:
        print(Fore.YELLOW + "  No timing data for efficiency frontier." + Fore.RESET)
        return ""

    x = r["speedup_pred_vs_full"].values       # >1 means faster
    y = -r["delta_pred_vs_full"].values        # positive = pred has lower MASE (better)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(x, y, alpha=0.6, s=35, c=y, cmap="RdYlGn",
                    edgecolors="white", linewidths=0.4, vmin=-max(abs(y)), vmax=max(abs(y)))
    plt.colorbar(sc, ax=ax, label="MASE improvement (pred − full, flipped)")
    ax.axhline(0, color="gray", lw=1.0, ls="--", alpha=0.6, label="Same MASE")
    ax.axvline(1, color="gray", lw=1.0, ls="-.", alpha=0.6, label="Same time")
    ax.set_xlabel("Speedup  =  elapsed_full / elapsed_pred  (>1 means predictor is faster)",
                  fontsize=11)
    ax.set_ylabel("MASE Reduction  =  MASE_full − MASE_pred  (>0 means predictor is better)",
                  fontsize=11)
    ax.set_title("Efficiency Frontier: Accuracy vs Speed Tradeoff",
                 fontsize=13, fontweight="bold")

    # Quadrant labels
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xm, ym = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
    for qx, qy, label in [
        (xlim[1]*0.9, ylim[1]*0.9, "Faster + Better"),
        (xlim[0]*1.1, ylim[1]*0.9, "Slower + Better"),
        (xlim[1]*0.9, ylim[0]*1.1, "Faster + Worse"),
        (xlim[0]*1.1, ylim[0]*1.1, "Slower + Worse"),
    ]:
        pass  # skip labels to keep plot clean

    ax.legend(fontsize=9); ax.grid(True, alpha=0.2); plt.tight_layout()
    path = os.path.join(out_dir, "efficiency_frontier.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


# ==============================================================================
#  PLOTS — COMPLEXITY
# ==============================================================================

def plot_complexity_reduction(df: pd.DataFrame, out_dir: str) -> str:
    """
    Distribution of theoretical FLOPs ratio: pred vs full, and best vs full.
    Values < 1 mean the predictor uses less compute than the full window.
    """
    r = df.dropna(subset=["complexity_ratio_pred_vs_full", "complexity_ratio_best_vs_full"])
    if r.empty:
        return ""

    pred_ratios = r["complexity_ratio_pred_vs_full"].values
    best_ratios = r["complexity_ratio_best_vs_full"].values
    n_bins = min(60, max(20, len(pred_ratios) // 5))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, vals, label, color in [
        (axes[0], pred_ratios, "Predictor / Full", "#70AD47"),
        (axes[1], best_ratios, "Oracle Best / Full", "#ED7D31"),
    ]:
        clip = np.percentile(vals, 99)
        ax.hist(vals[vals <= clip], bins=n_bins, color=color,
                alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(1.0, color="black", lw=1.5, ls="--", alpha=0.7, label="No reduction")
        ax.axvline(float(np.median(vals)), color="purple", lw=1.5, ls=":",
                   label=f"Median = {np.median(vals):.3f}")
        ax.axvline(float(vals.mean()), color="red", lw=1.5, ls="-",
                   label=f"Mean = {vals.mean():.3f}")
        pct_cheaper = 100 * (vals < 1.0).mean()
        ax.set_xlabel(f"FLOPs Ratio: {label}  (<1 means cheaper)", fontsize=11)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{label}  ({pct_cheaper:.1f}% cheaper than full)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Theoretical FLOPs Ratio vs Full Window\n"
        "(FLOPs ∝ n_patches² for encoder + n_patches_ctx×n_patches_hor for decoder)",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "complexity_reduction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_complexity_vs_mase_gain(df: pd.DataFrame, out_dir: str) -> str:
    """
    Scatter: theoretical complexity reduction (1 - ratio) vs MASE gain.
    Shows whether context reduction also brings accuracy improvement.
    """
    r = df.dropna(subset=["complexity_ratio_pred_vs_full", "delta_pred_vs_full"])
    if r.empty:
        return ""

    x = 1 - r["complexity_ratio_pred_vs_full"].values   # >0 means cheaper
    y = -r["delta_pred_vs_full"].values                  # >0 means better accuracy

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, alpha=0.55, s=30, color="#4472C4",
               edgecolors="white", linewidths=0.5)
    ax.axhline(0, color="gray", lw=1.0, ls="--", alpha=0.6)
    ax.axvline(0, color="gray", lw=1.0, ls="-.", alpha=0.6)
    ax.set_xlabel("FLOPs Reduction  =  1 − (FLOPs_pred / FLOPs_full)  (>0 means cheaper)",
                  fontsize=11)
    ax.set_ylabel("MASE Improvement  =  MASE_full − MASE_pred  (>0 means better)",
                  fontsize=11)
    ax.set_title("Complexity Reduction vs Accuracy Gain (Predictor vs Full Window)",
                 fontsize=13, fontweight="bold")
    if len(x) > 1 and x.std() > 0 and y.std() > 0:
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=11,
                va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax.grid(True, alpha=0.25); plt.tight_layout()
    path = os.path.join(out_dir, "complexity_vs_mase_gain.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


# ==============================================================================
#  PLOTS — MISC
# ==============================================================================

def plot_per_dataset_bars(df: pd.DataFrame, out_dir: str) -> str:
    r = df.dropna(subset=["full_mase", "best_mase", "pred_mase"]).copy()
    if r.empty:
        return ""
    r["ds_label"] = r["dataset_display"] + "\n(t=" + r["term"] + ")"
    labels = r["ds_label"].tolist()
    n = len(labels)
    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(12, n * 0.55), 6))
    ax.bar(x - w, r["full_mase"].values, w, label="Full window",    color="#4472C4", alpha=0.85)
    ax.bar(x,     r["best_mase"].values, w, label="Best (oracle)",  color="#ED7D31", alpha=0.85)
    ax.bar(x + w, r["pred_mase"].values, w, label="Predictor",      color="#70AD47", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("MASE", fontsize=12)
    ax.set_title("MASE per Dataset: Full vs Best vs Predictor Window",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3); ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(out_dir, "per_dataset_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_window_choice_scatter(df: pd.DataFrame, out_dir: str) -> str:
    r = df.dropna(subset=["best_window", "pred_window"])
    if r.empty:
        return ""
    x, y = r["best_window"].values, r["pred_window"].values
    exact_match = (x == y).mean() * 100
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.5, s=30, color="#4472C4", edgecolors="white", linewidths=0.5)
    lims = [min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1.2, alpha=0.6, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("Oracle Best Window (argmin real MASE)", fontsize=12)
    ax.set_ylabel("Predictor Window (argmin predicted curve)", fontsize=12)
    ax.set_title(f"Window Choice: Predictor vs Oracle  ({exact_match:.1f}% exact match)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25, which="both"); plt.tight_layout()
    path = os.path.join(out_dir, "window_choice_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


# ==============================================================================
#  MAIN
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare MASE + time + complexity: full vs best vs predictor window."
    )
    p.add_argument("--cache-root", type=str, default=CACHE_ROOT)
    p.add_argument("--run-dir",    type=str, default=None,
                   help="Specific v5 run dir; auto-picks latest if omitted.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output dir; defaults to <run-dir>/strategy_comparison/.")
    p.add_argument(
        "--patch-sizes", type=str, default=None,
        help=(
            'JSON dict overriding patch sizes per model family.  '
            'Example: \'{"moirai": 16, "chronos2": 2}\''
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    patch_sizes = dict(DEFAULT_PATCH_SIZES)
    if args.patch_sizes:
        patch_sizes.update(json.loads(args.patch_sizes))
        print(Fore.CYAN + f"Patch sizes (after override): {patch_sizes}" + Fore.RESET)
    else:
        print(Fore.CYAN + f"Patch sizes: {patch_sizes}" + Fore.RESET)

    run_dir = args.run_dir or find_latest_run(args.cache_root)
    print(Fore.CYAN + f"Run directory: {run_dir}" + Fore.RESET)

    out_dir = args.output_dir or os.path.join(run_dir, "strategy_comparison")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load records -------------------------------------------------------
    df = load_strategy_records(run_dir, args.cache_root, patch_sizes)

    # ---- Save comparison CSV ------------------------------------------------
    csv_path = os.path.join(out_dir, "comparison.csv")
    df.to_csv(csv_path, index=False)
    print(Fore.GREEN + f"  Saved: {csv_path}" + Fore.RESET)

    # ---- Summary stats ------------------------------------------------------
    stats = compute_summary_stats(df)
    stats_path = os.path.join(out_dir, "summary_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(Fore.GREEN + f"  Saved: {stats_path}" + Fore.RESET)

    print(Fore.CYAN + "\n--- MASE summary ---" + Fore.RESET)
    for key, name in [("full_mase", "Full window  "),
                      ("best_mase", "Best (oracle)"),
                      ("pred_mase", "Predictor    ")]:
        s = stats[key]
        print(f"  {name}  mean={s['mean']:.4f}  median={s['median']:.4f}  std={s['std']:.4f}")
    print(f"  Pred beats full: {stats['pred_beats_full_count']}/{stats['total_rows']} "
          f"({100*stats['pred_beats_full_rate']:.1f}%)")
    if stats.get("pred_clamped_count", 0) > 0:
        print(
            Fore.YELLOW
            + f"  Pred clamped to full (insufficient context): "
            + f"{stats['pred_clamped_count']}/{stats['total_rows']}"
            + Fore.RESET
        )
    print(f"  Rel. gain pred/full:  mean={stats['rel_gain_pred_over_full']['mean']:.4f}  "
          f"median={stats['rel_gain_pred_over_full']['median']:.4f}")
    print(f"  Regret vs oracle:     mean={stats['regret_pred_vs_best']['mean']:.4f}  "
          f"median={stats['regret_pred_vs_best']['median']:.4f}")

    if "speedup_pred_vs_full" in stats:
        sp = stats["speedup_pred_vs_full"]
        print(Fore.CYAN + "\n--- Timing summary ---" + Fore.RESET)
        print(f"  Mean elapsed: full={stats['mean_full_elapsed_s']:.2f}s  "
              f"best={stats.get('mean_best_elapsed_s', float('nan')):.2f}s  "
              f"pred={stats['mean_pred_elapsed_s']:.2f}s")
        print(f"  Speedup pred/full:  mean={sp['mean']:.2f}x  median={sp['median']:.2f}x  "
              f"({100*sp['pct_faster']:.1f}% of cases faster)")

    if "complexity_ratio_pred_vs_full" in stats:
        cr = stats["complexity_ratio_pred_vs_full"]
        print(Fore.CYAN + "\n--- Complexity summary ---" + Fore.RESET)
        print(f"  FLOPs ratio pred/full:  mean={cr['mean']:.3f}  median={cr['median']:.3f}")

    # ---- Plots --------------------------------------------------------------
    print(Fore.CYAN + "\n--- Generating plots ---" + Fore.RESET)
    for path in [
        plot_bar_aggregate_mase(df, out_dir),
        plot_bar_aggregate_time(df, out_dir),
        plot_scatter(df, "best_mase", "pred_mase",
                     "MASE — Best (oracle)", "MASE — Predictor",
                     "Predictor MASE vs Oracle Best MASE",
                     "scatter_pred_vs_best.png", out_dir),
        plot_scatter(df, "full_mase", "pred_mase",
                     "MASE — Full Window", "MASE — Predictor",
                     "Predictor MASE vs Full-Window MASE",
                     "scatter_pred_vs_full.png", out_dir),
        plot_efficiency_frontier(df, out_dir),
        plot_gain_histogram(df, out_dir),
        plot_regret_histogram(df, out_dir),
        plot_complexity_reduction(df, out_dir),
        plot_complexity_vs_mase_gain(df, out_dir),
        plot_per_dataset_bars(df, out_dir),
        plot_window_choice_scatter(df, out_dir),
    ]:
        if path:
            print(Fore.GREEN + f"  {path}" + Fore.RESET)

    print(Fore.GREEN + f"\nDone.  Outputs: {out_dir}" + Fore.RESET)


if __name__ == "__main__":
    main()
