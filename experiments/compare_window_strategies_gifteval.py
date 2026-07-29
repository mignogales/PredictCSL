"""
compare_window_strategies_gifteval.py

Reads cached ablation results produced by test_window_ablation_gifteval_v5.py
and compares MASE + wall-clock time + theoretical complexity under three
context-selection strategies per (model, dataset, term):

  full_window   -- native full-context baseline (wfull_native) when present;
                   otherwise largest valid window in the ablation grid
  best_window   -- argmin of real MASE curve (oracle; best achievable from grid)
  pred_window   -- argmin of the predictor's mean curve (zero-shot recommendation)

No model inference is performed -- all numbers come from the v5 cache and the
compare_real_vs_predicted/*.npz files.

Complexity model (per model family)
------------------------------------
We estimate per-forecast transformer MACs, keeping BOTH cost terms rather than
just the quadratic attention term.  Per layer over a length-L sequence (hidden
d, feed-forward f, E active experts):

  projections  4·L·d²        feed-forward  2·L·d·f·E        (both linear in L)
  attention    2·L²·d                                       (quadratic in L)

  enc_dec (chronos2, chronos_bolt): encoder over n_ctx + decoder over n_hor
                                    with cross-attention to the context.
  unified (moirai, timesfm, patchtst_fm, sundial, timemoe): one stack over
                                    n_ctx + n_hor tokens.

where n_ctx = ⌈C/P⌉, n_hor = ⌈H/P⌉, P = effective patch size.  The linear terms
dominate when n_ctx < d (the usual regime here), so dropping them — as the old
n_ctx²-only proxy did — overstated context-shrink savings.  d_model / layer
count / d_ff / experts are taken from each model's HF config.json (see
MODEL_ARCH); patch sizes are overridable via --patch-sizes JSON.

Outputs (written to <run_dir>/models/<model_short>/strategy_comparison/)
----------------------------------------------------
  comparison.csv              per-row MASE, elapsed time, complexity per strategy
  summary_stats.json          aggregate stats (mean/median, win rates, speedups)
  flops_savings.csv           total FLOPs saved vs full + geomean MASE drop, per strategy
  time_savings.csv            total measured forward-pass time saved vs full + geomean
                              MASE drop, per strategy (Stage 6, twin of flops_savings)

FIGURES — by default only TWO are emitted, both on the primary --mase-metric
(default `mase_gluonts_real`, the leaderboard machinery):
  bar_aggregate_mase_gluonts.png             absolute MASE bars per strategy
  bar_aggregate_mase_gluonts_normalized.png  each cell ÷ its same-definition
                                             Seasonal-Naive, then aggregated — the
                                             leaderboard-faithful figure (=1.0 line)
Everything below only appears with --all-figures (note: in that mode the
`_gluonts` suffix reverts to meaning the PORT twin, plain = primary):

Run-level (at <run_dir>/, one figure for the whole run; via --rollup-only)
----------------------------------------------------
  flops_savings_all_models.csv   per-(model,strategy) savings + a grand TOTAL row
                                 (arith & geo means of ratio / %saved / abs saved)
  model_strategy_overview.png    single scatter: every (model × strategy) point,
                                 FLOPs saved (x) vs geomean MASE change (y)
  model_strategy_overview_gluonts[_real].png  twin scored on the OTHER gluonts
                                 metric (whichever isn't the primary --mase-metric)
  time_savings_all_models.csv    Stage 6 twin: per-(model,strategy) measured
                                 forward-pass time saved + grand TOTAL row
  model_strategy_overview_time.png  single scatter: every (model × strategy) point,
                                 wall-clock forward-pass time saved (x) vs geomean
                                 MASE change (y)
  bar_aggregate_mase.png      mean & median MASE per strategy (primary metric)
  bar_aggregate_mase_normalized.png  the primary MASE divided per cell by the
                              same-definition seasonal-naive (Seasonal Naive = 1.0)
                              — the leaderboard-faithful aggregation; with the
                              default `mase_gluonts_real` this is the figure that
                              lines up with the HF GiftEval board
  bar_aggregate_mase_gluonts[_real][_normalized].png  same pair on the other
                              gluonts metric (suffixed twins)

Exactly TWO MASE metrics exist in this comparison: `mase_gluonts` (numpy port of
the leaderboard definition) and `mase_gluonts_real` (gluonts' own
evaluate_forecasts machinery; the default). The legacy project `mase`
(pooled-training-naive, D->7/W->52 map) is no longer consumed here — cells whose
npz lacks the gluonts curves are skipped loudly, never silently mixed.
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
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from colorama import Fore
from experiments.gifteval_reference import (
    NORMALIZATION_REFERENCE, published_naive_by_display,
)

CACHE_ROOT = "logs/experiments/window_ablation_gifteval"
FULL_NATIVE_WINDOW = "full_native"
WindowKey = Union[int, str]

# ==============================================================================
#  PREDICTOR VARIANTS
# ==============================================================================
#
# The base run dir (``general/``) carries the v1 predictor's curve as the "pred"
# strategy.  The constrained/cheap (v3) and Mamba (v4) predictors write sibling
# ablation trees (``general_v3/`` / ``general_v4/``) that SHARE the same real
# MASE curves and window grid (the expensive GiftEval cells are symlinked — the
# grid is dataset-derived, not predictor-derived) and differ ONLY in their
# ``predicted_mean`` field.  So each variant's window choice can be evaluated on
# the *same* real curves and folded in as one extra strategy, exactly the way the
# ``period`` strategy is.
#
# Maps the ablation-tree suffix -> (strategy_key, display label, color).  The
# base run_dir's OWN predictor is always "pred"; any *other* tree present as a
# sibling is auto-discovered and added (no CLI flag, graceful when absent).
PRED_VARIANTS: Dict[str, Tuple[str, str, str]] = {
    "_v3": ("pred_cheap", "Predictor (cheap)", "#1F77B4"),
    "_v4": ("pred_mamba", "Predictor Mamba",   "#9C27B0"),
    "_v3_classification": (
        "pred_cheap_cls", "Predictor (cheap, cls)", "#17BECF"),
    "_v4_classification": (
        "pred_mamba_cls", "Predictor Mamba (cls)", "#E377C2"),
}


def discover_pred_variants(run_dir: str) -> List[Tuple[str, str, str, str]]:
    """Return ``(key, label, color, tree_dir)`` for every sibling predictor-variant
    ablation tree that exists, has a ``models/`` subdir, and differs from
    ``run_dir``.

    Robust to ``run_dir`` itself being a variant (e.g. ``.../general_v3``): the
    known suffix is stripped to recover the ``general`` stem before re-attaching
    each variant suffix, so the base ``general`` tree and the *other* variants are
    all discovered regardless of which tree drives the comparison.
    """
    run_dir = os.path.normpath(run_dir)
    parent = os.path.dirname(run_dir)
    base = os.path.basename(run_dir)

    # Recover the stem ("general") by peeling a known variant suffix if present.
    stem = base
    for suf in sorted(PRED_VARIANTS, key=len, reverse=True):
        if base.endswith(suf):
            stem = base[: -len(suf)]
            break

    found: List[Tuple[str, str, str, str]] = []
    for suf, (key, label, color) in PRED_VARIANTS.items():
        tree = os.path.join(parent, stem + suf)
        if os.path.normpath(tree) == run_dir:
            continue  # this is the base tree itself (its predictor is "pred")
        if os.path.isdir(os.path.join(tree, "models")):
            found.append((key, label, color, tree))
    return found


def present_pred_variants(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """Return ``(key, label, color)`` for each variant whose ``{key}_mase`` column
    is present in ``df`` and has at least one non-NaN value, in registry order.

    Single source of truth so every downstream consumer (stats, savings, plots,
    console) shows exactly the variants that were actually folded in.
    """
    out: List[Tuple[str, str, str]] = []
    for _suf, (key, label, color) in PRED_VARIANTS.items():
        col = f"{key}_mase"
        if col in df.columns and df[col].notna().any():
            out.append((key, label, color))
    return out

# ==============================================================================
#  COMPLEXITY MODEL
# ==============================================================================
#
# We estimate per-forecast transformer MACs (multiply-accumulates) so that the
# context-length savings reflect the *true* cost mix, not just the quadratic
# attention term.  A transformer layer over a sequence of L tokens (hidden size
# d, feed-forward size f) costs, per layer:
#
#     projections (Q,K,V,O):   4 * L * d^2          <- LINEAR in L
#     self-attention:          2 * L^2 * d          <- QUADRATIC in L
#     feed-forward (MLP):      2 * L * d * f * E     <- LINEAR in L (E = active experts)
#
# The linear (projection + FFN) terms DOMINATE whenever n_ctx < d, which is the
# regime most of these patch-based TSFMs actually run in (a few hundred patches
# vs d = 512..1280).  The previous proxy kept only the L^2 term and therefore
# overstated the savings from shrinking context.  Carrying d, the layer count,
# and f fixes both the within-model scaling AND makes cross-model numbers an
# approximate absolute MAC count (still a proxy: it ignores embeddings, norms,
# quantile heads, and exact attention-kernel constants).
#
# Architecture specs below are taken verbatim from each model's HF config.json.

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelArch:
    """Architecture spec used by the FLOPs proxy (from HF config.json)."""
    d_model: int
    d_ff: int
    patch_size: int
    seq_type: str            # "enc_dec" | "unified"
    n_enc_layers: int = 0    # encoder blocks (enc_dec) / total blocks (unified)
    n_dec_layers: int = 0    # decoder blocks (enc_dec only)
    experts_per_tok: int = 1 # active experts for MoE FFN (1 = dense)
    max_window: int = 8192   # largest context the model accepts (caps full_window);
                             # default = MAX_WINDOW grid top, overridden per family


# Per-family architecture.  Patch sizes here are the defaults; --patch-sizes can
# still override the patch field at runtime (see resolve_arch).
MODEL_ARCH: Dict[str, ModelArch] = {
    # T5-style encoder-decoder (encoder over context, decoder over horizon + cross-attn)
    "chronos2":     ModelArch(d_model=512,  d_ff=2048, patch_size=16, seq_type="enc_dec",
                              n_enc_layers=6,  n_dec_layers=6),
    "chronos_bolt": ModelArch(d_model=512,  d_ff=2048, patch_size=32, seq_type="enc_dec",
                              n_enc_layers=6,  n_dec_layers=6, max_window=2048),
    # Encoder over the [context; masked-horizon] sequence
    "moirai":       ModelArch(d_model=384,  d_ff=1024, patch_size=16, seq_type="unified",
                              n_enc_layers=6),
    "moirai_1_1":   ModelArch(d_model=384,  d_ff=1024, patch_size=32, seq_type="unified",
                              n_enc_layers=6),
    # Stacked decoder over context patches (intermediate_size == d_model for TimesFM-2.5)
    "timesfm":      ModelArch(d_model=1280, d_ff=1280, patch_size=32, seq_type="unified",
                              n_enc_layers=20, max_window=15360),
    # Encoder over a fixed 8192-ctx patch grid; d_ff not in config -> 4x expansion (estimate)
    "patchtst_fm":  ModelArch(d_model=1024, d_ff=4096, patch_size=16, seq_type="unified",
                              n_enc_layers=20),
    # Decoder-only patch LM (context capped at 2880; see SUNDIAL_MAX_CONTEXT)
    "sundial":      ModelArch(d_model=768,  d_ff=3072, patch_size=16, seq_type="unified",
                              n_enc_layers=12, max_window=2880),
    # Decoder-only MoE (top-2 of 8 experts active per token; ctx+horizon <= 4096)
    "timemoe":      ModelArch(d_model=768,  d_ff=3072, patch_size=1,  seq_type="unified",
                              n_enc_layers=12, experts_per_tok=2, max_window=4096),
    # Decoder-only probabilistic TSFM (Toto 2.0, quantile head). max_window
    # mirrors the TOTO_MAX_CONTEXT label cap. d_model/d_ff/patch/layers from
    # Toto-2.0-313m config.json (1024 / 2736 / 32 / 24) — re-confirm against the
    # installed checkpoint before trusting its FLOPs column.
    "toto":         ModelArch(d_model=1024, d_ff=2736, patch_size=32, seq_type="unified",
                              n_enc_layers=24, max_window=4096),
    # IBM Granite FlowState (r1.1). SSM encoder + functional-basis decoder, so its
    # *true* cost is LINEAR in context — the attention-based proxy below therefore
    # OVERSTATES it at long windows (keeps a spurious L^2 term). d_model/patch from
    # config.json (512 / 6); d_ff is unset there, estimated at 4*d_model. max_window
    # mirrors the FLOWSTATE_MAX_CONTEXT label cap. Treat its FLOPs column as a loose
    # upper bound, not a faithful SSM count.
    "flowstate":    ModelArch(d_model=512,  d_ff=2048, patch_size=6,  seq_type="unified",
                              n_enc_layers=6,  max_window=4096),
    # NX-AI TiRex (~35M). xLSTM (recurrent) blocks, so its *true* cost is LINEAR
    # in context — the attention proxy below OVERSTATES it at long windows (keeps
    # a spurious L^2 term), same caveat as FlowState. No external config.json
    # (arch is inside model.ckpt), so d_model/d_ff/patch/layers are ESTIMATES.
    # max_window = TiRex2's 8192 pretraining context (TiRex-1 was 2048), mirroring
    # the TIREX_MAX_CONTEXT label cap. Treat its FLOPs column as a loose upper
    # bound, and re-confirm the arch against the checkpoint.
    "tirex":        ModelArch(d_model=512,  d_ff=2048, patch_size=32, seq_type="unified",
                              n_enc_layers=12, max_window=8192),
}


# Patch sizes exposed for the --patch-sizes CLI override / printout.  Derived
# from MODEL_ARCH; overriding a family here changes only its FLOPs columns.
DEFAULT_PATCH_SIZES: Dict[str, int] = {
    fam: arch.patch_size for fam, arch in MODEL_ARCH.items()
}
DEFAULT_PATCH_SIZES["context_parroting"] = 1


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
    if "timemoe" in m or "time-moe" in m:
        return "timemoe"
    if "sundial" in m:
        return "sundial"
    if "toto" in m:
        return "toto"
    if "flowstate" in m:
        return "flowstate"
    if "tirex" in m:
        return "tirex"
    return "unknown"


def _n_patches(length: int, patch_size: int) -> int:
    return max(1, math.ceil(length / patch_size))


def _layer_macs(L: int, d: int, f: int, mem: int = 0, experts: int = 1) -> float:
    """Per-layer transformer MACs over a length-L sequence.

    ``mem`` > 0 adds cross-attention against a memory of that length (decoder
    cross-attn in encoder-decoder models).
    """
    proj      = 4.0 * L * d * d           # Q,K,V,O projections          (linear in L)
    self_attn = 2.0 * L * L * d           # QK^T + softmax·V             (quadratic)
    ffn       = 2.0 * L * d * f * experts # two MLP linears, MoE-scaled  (linear in L)
    cross = 0.0
    if mem > 0:
        # decoder Q/O projections + K,V over memory + the two attn matmuls
        cross = 2.0 * (L + mem) * d * d + 2.0 * L * mem * d
    return proj + self_attn + ffn + cross


def theoretical_flops(
    model_id: str,
    context: int,
    horizon: int,
    patch_sizes: Dict[str, int],
) -> float:
    """Approximate per-forecast transformer MACs.

    Carries d_model, layer count, feed-forward size, and (for MoE) active
    experts, so both the linear (projection + FFN) and quadratic (attention)
    cost terms are represented.  Comparable as a within-model ratio across
    context sizes, and — being a real MAC estimate — roughly comparable across
    models too (still a proxy: ignores embeddings/norms/heads and kernel
    constants).

      enc_dec  (chronos2, chronos_bolt): encoder over n_ctx + decoder over
                                         n_hor with cross-attn to the context.
      unified  (moirai, timesfm, patchtst_fm, sundial, timemoe): single stack
                                         over n_ctx + n_hor tokens.

    Families without a spec fall back to the legacy attention-only proxy.
    """
    family = infer_model_family(model_id)
    arch = MODEL_ARCH.get(family)
    if arch is None:
        # Legacy attention-only fallback for unknown families.
        P = patch_sizes.get(family, 1)
        n_ctx = _n_patches(context, P)
        n_hor = _n_patches(horizon, P)
        return float(n_ctx ** 2 + n_ctx * n_hor)

    P = patch_sizes.get(family, arch.patch_size)
    n_ctx = _n_patches(context, P)
    n_hor = _n_patches(horizon, P)
    d, f = arch.d_model, arch.d_ff

    if arch.seq_type == "enc_dec":
        enc = arch.n_enc_layers * _layer_macs(n_ctx, d, f)
        dec = arch.n_dec_layers * _layer_macs(n_hor, d, f, mem=n_ctx)
        return float(enc + dec)
    else:  # unified
        L = n_ctx + n_hor
        return float(arch.n_enc_layers *
                     _layer_macs(L, d, f, experts=arch.experts_per_tok))


# ==============================================================================
#  CACHE PATHS (mirrors test_window_ablation_gifteval_v5.py)
# ==============================================================================

def _cache_dir(
    cache_root: str,
    dataset_display: str,
    model_short: str,
    term: str,
    window_size: WindowKey,
) -> str:
    return os.path.join(
        cache_root, "datasets", dataset_display, model_short, f"t{term}", f"w{window_size}"
    )


def _load_elapsed(
    cache_root: str,
    dataset_display: str,
    model_short: str,
    term: str,
    window_size: WindowKey,
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


# When True (default), prefer the robust forward-pass timing produced by
# benchmark_window_timing_gifteval.py (mean over warmed-up repeats, in timing.json)
# over the single-shot elapsed_seconds, falling back per-cell when no timing.json
# exists. Toggled by --use-robust-timing / --no-use-robust-timing in main().
USE_ROBUST_TIMING = True


def _load_robust_elapsed(
    cache_root: str,
    dataset_display: str,
    model_short: str,
    term: str,
    window_size: WindowKey,
) -> Tuple[float, float]:
    """Return (mean_s, std_s) from the per-cell timing.json, or (NaN, NaN) when
    no robust timing has been recorded for this cell."""
    path = os.path.join(
        _cache_dir(cache_root, dataset_display, model_short, term, window_size),
        "timing.json",
    )
    if not os.path.isfile(path):
        return float("nan"), float("nan")
    try:
        with open(path) as f:
            d = json.load(f)
        mean = d.get("mean_s")
        std = d.get("std_s")
        return (
            float(mean) if mean is not None else float("nan"),
            float(std) if std is not None else float("nan"),
        )
    except Exception:
        return float("nan"), float("nan")


def _elapsed_and_std(
    cache_root: str,
    dataset_display: str,
    model_short: str,
    term: str,
    window_size: WindowKey,
) -> Tuple[float, float]:
    """Elapsed seconds + std for one cell. Uses the robust timing.json mean/std
    when enabled and present; otherwise the single-shot elapsed_seconds (std NaN).
    Per-cell fallback keeps the comparison working before the timing stage has run."""
    if USE_ROBUST_TIMING:
        mean, std = _load_robust_elapsed(
            cache_root, dataset_display, model_short, term, window_size)
        if not math.isnan(mean):
            return mean, std
    return _load_elapsed(cache_root, dataset_display, model_short, term, window_size), float("nan")


def _load_metrics(
    cache_root: str,
    dataset_display: str,
    model_short: str,
    term: str,
    window_size: WindowKey,
) -> Optional[dict]:
    path = os.path.join(
        _cache_dir(cache_root, dataset_display, model_short, term, window_size),
        "metrics.json",
    )
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _mean_theoretical_flops_for_contexts(
    model_id: str,
    contexts: np.ndarray,
    horizon: int,
    patch_sizes: Dict[str, int],
) -> float:
    vals = np.asarray(contexts, dtype=np.int64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return float("nan")
    uniq, counts = np.unique(vals, return_counts=True)
    total = 0.0
    for ctx, n in zip(uniq, counts):
        total += theoretical_flops(model_id, int(ctx), horizon, patch_sizes) * int(n)
    return float(total / vals.size)


def _full_native_flops(
    cache_root: str,
    dataset_display: str,
    model_short: str,
    term: str,
    model_id: str,
    horizon: int,
    patch_sizes: Dict[str, int],
    metrics: dict,
) -> float:
    npz_path = os.path.join(
        _cache_dir(cache_root, dataset_display, model_short, term, FULL_NATIVE_WINDOW),
        "per_sample_metrics.npz",
    )
    if os.path.isfile(npz_path):
        try:
            with np.load(npz_path) as d:
                if "effective_context" in d.files:
                    return _mean_theoretical_flops_for_contexts(
                        model_id, d["effective_context"], horizon, patch_sizes)
        except Exception:
            pass
    cap = metrics.get("_context_cap")
    if cap is not None:
        return theoretical_flops(model_id, int(cap), horizon, patch_sizes)
    return float("nan")


# ==============================================================================
#  RUN DISCOVERY
# ==============================================================================

def find_latest_run(cache_root: str) -> str:
    """Return the run directory.  With the current layout the run dir IS the
    cache_root (models/ and datasets/ live directly inside it)."""
    if not os.path.isdir(cache_root):
        raise FileNotFoundError(f"Cache root not found: {cache_root}")
    if os.path.isdir(os.path.join(cache_root, "models")):
        return cache_root
    raise FileNotFoundError(
        f"No models/ subdir found under {cache_root}"
    )


# ==============================================================================
#  DATA LOADING
# ==============================================================================

def _npz_filename(dataset_display: str, term: str, model_short: str) -> str:
    return f"compare_{dataset_display}_t{term}_{model_short}.npz"


def run_has_curve(run_dir: str, curve_key: str) -> bool:
    """True if any compare_*.npz in this run carries a populated ``curve_key``.
    Returns on the first populated curve found; older runs that never wrote it
    scan the trees and return False."""
    models_root = os.path.join(run_dir, "models")
    if not os.path.isdir(models_root):
        return False
    pattern = os.path.join(models_root, "*", "compare_real_vs_predicted", "*.npz")
    for npz in glob.iglob(pattern):
        try:
            with np.load(npz) as d:
                if curve_key in d.files and np.any(~np.isnan(d[curve_key])):
                    return True
        except Exception:
            continue
    return False


def run_has_gluonts_curve(run_dir: str) -> bool:
    """True if the ablation has computed the ported leaderboard MASE
    (``real_curve_gluonts`` / the `mase_gluonts` metric)."""
    return run_has_curve(run_dir, "real_curve_gluonts")


def run_has_gluonts_real_curve(run_dir: str) -> bool:
    """True if the ablation has computed the gluonts-machinery MASE
    (``real_curve_gluonts_real`` / the `mase_gluonts_real` metric)."""
    return run_has_curve(run_dir, "real_curve_gluonts_real")


def _load_period_record(
    compare_dir: str,
    dataset_display: str,
    term: str,
    model_short: str,
    period_multiple: int = 2,
) -> Optional[dict]:
    """Load the period_window_eval.py sidecar for this (dataset, term, model), if any.

    Returns the parsed JSON (period_mase / window stats / elapsed) or None when the
    period-window strategy was not evaluated -- in which case the period_* columns
    stay NaN and the comparison degrades gracefully to the original 3 strategies.
    """
    prefix = "period" if period_multiple == 2 else f"period{period_multiple}"
    path = os.path.join(
        compare_dir, f"{prefix}_{dataset_display}_t{term}_{model_short}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        print(Fore.YELLOW + f"  Could not read period sidecar {path}: {exc}" + Fore.RESET)
        return None


def _load_variant_pred_mean(
    tree_dir: str,
    dataset_display: str,
    term: str,
    model_short: str,
    expected_grid: np.ndarray,
) -> Optional[np.ndarray]:
    """Load a predictor-variant tree's ``predicted_mean`` for one (dataset, term,
    model) cell, or ``None`` when unavailable / grid-mismatched.

    The variant trees share the base run's window grid (symlinked cells), so a
    length mismatch means the cell is stale/incompatible and is skipped rather
    than silently misaligned.
    """
    npz_path = os.path.join(
        tree_dir, "models", model_short, "compare_real_vs_predicted",
        _npz_filename(dataset_display, term, model_short),
    )
    if not os.path.isfile(npz_path):
        return None
    try:
        data = np.load(npz_path)
        pred_mean = np.asarray(data["predicted_mean"])
        grid = np.asarray(data["window_grid"])
    except Exception as exc:
        print(Fore.YELLOW + f"  Could not read variant npz {npz_path}: {exc}" + Fore.RESET)
        return None
    if pred_mean.shape != expected_grid.shape or grid.shape != expected_grid.shape:
        print(Fore.YELLOW
              + f"  Variant grid mismatch {npz_path}: {pred_mean.shape} vs "
                f"{expected_grid.shape} — skipping this cell."
              + Fore.RESET)
        return None
    return pred_mean


def _load_naive_baselines(run_dir: str) -> Dict[str, dict]:
    """Published per-config Seasonal Naive MASE denominators.

    ``run_dir`` is retained for API compatibility with downstream scripts. Old
    ``naive_baselines.json`` files are deliberately ignored: the shipped
    GIFT-Eval CSV is the only normalization source.
    """
    del run_dir
    return published_naive_by_display()


def load_strategy_records(
    run_dir: str,
    cache_root: str,
    patch_sizes: Dict[str, int],
    mase_metric: str = "mase_gluonts_real",
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
    models_root = os.path.join(run_dir, "models")
    if not os.path.isdir(models_root):
        raise FileNotFoundError(f"No models/ dir found in {run_dir}")

    # Seasonal-naive denominators for the leaderboard-style normalised MASE. The
    # denominator field tracks the active metric so each MASE is divided by the
    # SAME-definition Seasonal-Naive (leaderboard convention). Exactly two metrics
    # exist: the numpy port and the gluonts-machinery one.
    if mase_metric not in ("mase_gluonts", "mase_gluonts_real"):
        raise ValueError(f"Unknown mase_metric {mase_metric!r}; expected "
                         "'mase_gluonts' or 'mase_gluonts_real'.")
    naive_baselines = _load_naive_baselines(run_dir)
    _naive_field = mase_metric

    # Sibling predictor-variant trees (v3 cheap, v4 Mamba, …) folded in as extra
    # strategies, evaluated on the SAME real curves as the base "pred".
    variants = discover_pred_variants(run_dir)
    if variants:
        print(Fore.CYAN
              + "  Predictor variants discovered: "
              + ", ".join(f"{key} ({os.path.basename(tree)})"
                          for key, _lbl, _clr, tree in variants)
              + Fore.RESET)

    records: List[dict] = []

    for model_short_dir in sorted(os.listdir(models_root)):
        compare_dir = os.path.join(models_root, model_short_dir, "compare_real_vs_predicted")
        summary_path = os.path.join(compare_dir, "compare_summary.csv")
        if not os.path.isfile(summary_path):
            print(Fore.YELLOW + f"  No compare_summary.csv for {model_short_dir}, skipping." + Fore.RESET)
            continue

        summary = pd.read_csv(summary_path)

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
            # Which MASE drives the comparison: the ported leaderboard
            # `mase_gluonts` (real_curve_gluonts) or the gluonts-machinery
            # `mase_gluonts_real` (real_curve_gluonts_real). NEVER fall back to the
            # legacy project `real_curve` (custom D->7/W->52 mase) — silently mixing
            # definitions is exactly what makes numbers stop lining up with the
            # leaderboard. A `_real` request may stand in with the port curve
            # (loudly); anything else missing skips the cell (loudly).
            _curve_key = {
                "mase_gluonts": "real_curve_gluonts",
                "mase_gluonts_real": "real_curve_gluonts_real",
            }[mase_metric]

            def _usable(key: str) -> bool:
                return key in data.files and bool(np.any(~np.isnan(data[key])))

            if _usable(_curve_key):
                real_curve = data[_curve_key]                # requested MASE per window
            elif mase_metric == "mase_gluonts_real" and _usable("real_curve_gluonts"):
                print(Fore.YELLOW
                      + f"  {dataset_display} t={term} {model_short}: no "
                        "real_curve_gluonts_real — standing in with the port curve "
                        "(re-run stage 3 to populate the machinery one)." + Fore.RESET)
                real_curve = data["real_curve_gluonts"]
            else:
                print(Fore.YELLOW
                      + f"  Skip {dataset_display} t={term} {model_short}: no "
                        f"{_curve_key} in {os.path.basename(npz_path)} — re-run "
                        "stage 3 (cheap backfill) to write the gluonts curves."
                      + Fore.RESET)
                continue
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
            full_elapsed_key: WindowKey = full_w
            full_baseline_source = "grid_largest_valid"
            full_effective_context_mean = float("nan")
            full_native_width_groups = float("nan")
            full_native_metrics = _load_metrics(
                cache_root, dataset_display, model_short, term, FULL_NATIVE_WINDOW)
            if full_native_metrics is not None:
                native_mase = full_native_metrics.get(mase_metric)
                native_standin = False
                native_ok = native_mase is not None and np.isfinite(float(native_mase))
                if not native_ok and mase_metric == "mase_gluonts_real":
                    native_mase = full_native_metrics.get("mase_gluonts")
                    native_standin = native_mase is not None
                if native_mase is not None and np.isfinite(float(native_mase)):
                    full_mase = float(native_mase)
                    full_w = int(full_native_metrics.get("_context_cap", full_w))
                    full_elapsed_key = FULL_NATIVE_WINDOW
                    full_baseline_source = (
                        "full_native_standin" if native_standin
                        else "full_native")
                    full_effective_context_mean = float(
                        full_native_metrics.get("_mean_effective_context", float("nan")))
                    full_native_width_groups = float(
                        full_native_metrics.get("_n_width_groups", float("nan")))

            # Published Seasonal Naive denominator for this official GiftEval
            # config. The local/reconstructed baseline is intentionally unused.
            naive_mase = float(
                naive_baselines.get(f"{dataset_display}/t{term}", {})
                .get(_naive_field, float("nan")))

            # --- Elapsed time (robust timing.json mean, else single-shot) --------
            full_elapsed, full_elapsed_std = _elapsed_and_std(
                cache_root, dataset_display, model_short, term, full_elapsed_key)
            best_elapsed, best_elapsed_std = _elapsed_and_std(
                cache_root, dataset_display, model_short, term, best_w)
            pred_elapsed, pred_elapsed_std = _elapsed_and_std(
                cache_root, dataset_display, model_short, term, pred_w)

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
            if full_baseline_source.startswith("full_native") and full_native_metrics is not None:
                full_flops = _full_native_flops(
                    cache_root, dataset_display, model_short, term,
                    model, horizon, patch_sizes, full_native_metrics)
                if not np.isfinite(full_flops):
                    full_flops = theoretical_flops(model, full_w, horizon, patch_sizes)
            else:
                full_flops = theoretical_flops(model, full_w, horizon, patch_sizes)
            best_flops = theoretical_flops(model, best_w, horizon, patch_sizes)
            pred_flops = theoretical_flops(model, pred_w, horizon, patch_sizes)

            complexity_ratio_pred = pred_flops / full_flops
            complexity_ratio_best = best_flops / full_flops

            # --- GiftEval-cadence period strategies (2xP and 3xP; off-grid) ------
            # Each multiplier has its own sidecar. Missing sidecars are NaN-filled
            # so old/partial runs still compare full/best/pred normally.
            def _period_values(period_multiple: int):
                prec = _load_period_record(
                    compare_dir, dataset_display, term, model_short,
                    period_multiple=period_multiple)
                if prec is None:
                    return (float("nan"), float("nan"), float("nan"),
                            float("nan"), 0, float("nan"), float("nan"))
                pw = float(prec.get("window_mean", float("nan")))
                pm_raw = (prec.get("all_metrics") or {}).get(mase_metric)
                pm = float(pm_raw) if pm_raw is not None else float("nan")
                pe = float(prec.get("period_elapsed_s", float("nan")))
                pwm = float(prec.get("window_median", float("nan")))
                pni = int(prec.get("n_instances", n_instances))
                pf = theoretical_flops(
                    model, max(1, int(round(pw))), horizon, patch_sizes)
                ps = (
                    full_elapsed / pe
                    if pe > 0 and not math.isnan(pe) and not math.isnan(full_elapsed)
                    else float("nan")
                )
                return pw, pm, pe, pwm, pni, pf, ps

            (period_w, period_mase, period_elapsed, period_w_med,
             period_n_inst, period_flops, speedup_period) = _period_values(2)
            (period3_w, period3_mase, period3_elapsed, period3_w_med,
             period3_n_inst, period3_flops, speedup_period3) = _period_values(3)

            # --- Predictor-variant strategies (v3 cheap, v4 Mamba, …) ------------
            # Each variant's curve picks a window; we score it on the SAME base
            # real_curve so every strategy shares identical ground truth. NaN-fill
            # the whole block when a variant cell is missing (graceful, like a
            # missing period sidecar).
            variant_cols: Dict[str, float] = {}
            for vkey, _vlbl, _vclr, vtree in variants:
                vpred = _load_variant_pred_mean(
                    vtree, dataset_display, term, model_short, window_grid)
                if vpred is None:
                    var_w = var_mase = var_elapsed = float("nan")
                    var_elapsed_std = float("nan")
                    var_flops = float("nan")
                    var_clamped = False
                    speedup_var = float("nan")
                    var_complexity = float("nan")
                else:
                    var_idx = int(np.argmin(vpred))
                    var_clamped = bool(np.isnan(real_curve[var_idx]))
                    if var_clamped:
                        var_idx = full_idx  # window unavailable -> fall back to full
                    var_w     = int(window_grid[var_idx])
                    var_mase  = float(real_curve[var_idx])
                    var_elapsed, var_elapsed_std = _elapsed_and_std(
                        cache_root, dataset_display, model_short, term, var_w)
                    var_flops = theoretical_flops(model, var_w, horizon, patch_sizes)
                    speedup_var = (
                        full_elapsed / var_elapsed
                        if var_elapsed > 0 and not math.isnan(var_elapsed)
                        and not math.isnan(full_elapsed)
                        else float("nan")
                    )
                    var_complexity = var_flops / full_flops if full_flops > 0 else float("nan")
                variant_cols.update({
                    f"{vkey}_window":   var_w,
                    f"{vkey}_mase":     var_mase,
                    f"{vkey}_clamped":  var_clamped,
                    f"delta_{vkey}_vs_full": var_mase - full_mase,
                    f"delta_{vkey}_vs_best": var_mase - best_mase,
                    f"delta_{vkey}_vs_pred": var_mase - pred_mase,
                    f"rel_gain_{vkey}_over_full": (
                        (full_mase - var_mase) / (abs(full_mase) + 1e-12)
                    ),
                    f"{vkey}_elapsed_s": var_elapsed,
                    f"{vkey}_elapsed_std_s": var_elapsed_std,
                    f"speedup_{vkey}_vs_full": speedup_var,
                    f"{vkey}_flops":    var_flops,
                    f"complexity_ratio_{vkey}_vs_full": var_complexity,
                })

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
            "full_baseline_source": full_baseline_source,
            "full_effective_context_mean": full_effective_context_mean,
            "full_native_width_groups": full_native_width_groups,
            "n_windows_valid": int(valid.sum()),
            "pred_clamped":    pred_clamped,   # True when pred window was unavailable
            # MASE values
            "full_mase":       full_mase,
            "best_mase":       best_mase,
            "pred_mase":       pred_mase,
            # Seasonal-naive baseline for this cell (denominator of the
            # leaderboard-style normalised MASE; NaN if unavailable).
            "naive_mase":      naive_mase,
            # MASE deltas
            "delta_pred_vs_full": pred_mase - full_mase,
            "delta_best_vs_full": best_mase - full_mase,
            "delta_pred_vs_best": pred_mase - best_mase,
            "rel_gain_pred_over_full": (
                (full_mase - pred_mase) / (abs(full_mase) + 1e-12)
            ),
            # elapsed wall-clock time (seconds; NaN if not cached). *_elapsed_std_s
            # is the std across robust-timing repeats (NaN for single-shot timing).
            "full_elapsed_s":  full_elapsed,
            "best_elapsed_s":  best_elapsed,
            "pred_elapsed_s":  pred_elapsed,
            "full_elapsed_std_s": full_elapsed_std,
            "best_elapsed_std_s": best_elapsed_std,
            "pred_elapsed_std_s": pred_elapsed_std,
            "speedup_pred_vs_full": speedup_pred,
            "speedup_best_vs_full": speedup_best,
            # theoretical complexity (unnormalized FLOPs proxy)
            "full_flops":      full_flops,
            "best_flops":      best_flops,
            "pred_flops":      pred_flops,
            "complexity_ratio_pred_vs_full": complexity_ratio_pred,
            "complexity_ratio_best_vs_full": complexity_ratio_best,
            # period-window strategy (per-series max(2*period, horizon); off-grid)
            "period_window":   period_w,        # mean per-series window (representative)
            "period_window_median": period_w_med,
            "period_n_instances":   period_n_inst,
            "period_mase":     period_mase,
            "delta_period_vs_full": period_mase - full_mase,
            "delta_period_vs_best": period_mase - best_mase,
            "delta_period_vs_pred": period_mase - pred_mase,
            "rel_gain_period_over_full": (
                (full_mase - period_mase) / (abs(full_mase) + 1e-12)
            ),
            "period_elapsed_s": period_elapsed,
            "period_elapsed_std_s": float("nan"),  # off-grid; single-shot only
            "speedup_period_vs_full": speedup_period,
            "period_flops":    period_flops,
            "complexity_ratio_period_vs_full": (
                period_flops / full_flops if full_flops > 0 else float("nan")
            ),
            # Same detected GiftEval cadence, but retain three complete cycles.
            "period3_window": period3_w,
            "period3_window_median": period3_w_med,
            "period3_n_instances": period3_n_inst,
            "period3_mase": period3_mase,
            "delta_period3_vs_full": period3_mase - full_mase,
            "delta_period3_vs_best": period3_mase - best_mase,
            "delta_period3_vs_pred": period3_mase - pred_mase,
            "rel_gain_period3_over_full": (
                (full_mase - period3_mase) / (abs(full_mase) + 1e-12)
            ),
            "period3_elapsed_s": period3_elapsed,
            "period3_elapsed_std_s": float("nan"),
            "speedup_period3_vs_full": speedup_period3,
            "period3_flops": period3_flops,
            "complexity_ratio_period3_vs_full": (
                period3_flops / full_flops if full_flops > 0 else float("nan")
            ),
            # predictor-variant strategies (v3 cheap, v4 Mamba, …); empty when none
            **variant_cols,
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

def _geomean(vals: np.ndarray) -> float:
    """GiftEval leaderboard geometric mean (same logic as the sanity check).

    Missing values are excluded, any negative value invalidates the aggregate,
    and a genuine zero makes the geometric mean exactly zero.  In particular we
    do not epsilon-clip: that would silently produce a different leaderboard
    number.
    """
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not arr.size or np.any(arr < 0.0):
        return float("nan")
    if np.any(arr == 0.0):
        return 0.0
    return float(np.exp(np.log(arr).mean()))


def _quadrature(vals: np.ndarray) -> float:
    """sqrt(sum of squares), ignoring NaNs — std of a sum of independent terms.
    Returns NaN when every entry is NaN (no robust timing recorded)."""
    arr = np.asarray(vals, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.sqrt(np.nansum(arr ** 2)))


def _wgeomean(vals: np.ndarray, weights: np.ndarray) -> float:
    """Weighted geometric mean: exp(sum(w*log(x)) / sum(w)). Clips to 1e-9.

    Combining per-group weighted geomeans with their group weights reproduces the
    global weighted geomean exactly (logs are additive), so the run-level rollup
    can aggregate the per-model MASE geomeans without the raw rows.
    """
    vals = np.asarray(vals, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    vals, w = vals[valid], w[valid]
    if not vals.size or np.any(vals < 0.0):
        return float("nan")
    if np.any(vals == 0.0):
        return 0.0
    return float(np.exp((w * np.log(vals)).sum() / w.sum()))


def compute_summary_stats(df: pd.DataFrame) -> dict:
    r = df.dropna(subset=["full_mase", "best_mase", "pred_mase"])
    stats: dict = {
        "headline_aggregation": {
            "formula": "geomean(cell_MASE / cell_seasonal_naive_MASE)",
            "cell_weighting": "unweighted",
            "missing_values": "drop",
            "zero_handling": "exact_zero",
            "reference": NORMALIZATION_REFERENCE,
        }
    }

    # Seasonal-naive normaliser present? Then also report the leaderboard-style
    # normalised geomean (geomean of MASE_strategy / MASE_seasonalnaive), over the
    # subset of rows that actually have a baseline.
    has_naive = "naive_mase" in r.columns and r["naive_mase"].notna().any()
    for strategy in ("full_mase", "best_mase", "pred_mase"):
        vals = r[strategy].values
        stats[strategy] = {
            "mean":    float(vals.mean()),
            "geomean": _geomean(vals),
            "median":  float(np.median(vals)),
            "std":     float(vals.std()),
            "n":       int(len(vals)),
        }
        if has_naive:
            m = r["naive_mase"].notna() & (r["naive_mase"] > 0)
            if m.any():
                ratio = (r.loc[m, strategy] / r.loc[m, "naive_mase"]).values
                stats[strategy]["geomean_norm"] = _geomean(ratio)
                stats[strategy]["n_norm"] = int(m.sum())

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

    # ---- Period-window strategy (only when sidecars were present) ------------
    if "period_mase" in df.columns:
        rp = df.dropna(subset=["full_mase", "best_mase", "period_mase"])
        if not rp.empty:
            vals = rp["period_mase"].values
            stats["period_mase"] = {
                "mean":    float(vals.mean()),
                "geomean": _geomean(vals),
                "median":  float(np.median(vals)),
                "std":     float(vals.std()),
                "n":       int(len(vals)),
            }
            stats["period_beats_full_count"] = int((rp["period_mase"] < rp["full_mase"]).sum())
            stats["period_beats_full_rate"]  = (
                int((rp["period_mase"] < rp["full_mase"]).sum()) / max(len(rp), 1))
            stats["period_beats_pred_count"] = int((rp["period_mase"] < rp["pred_mase"]).sum())
            gain_p = rp["rel_gain_period_over_full"].values
            stats["rel_gain_period_over_full"] = {
                "mean":         float(gain_p.mean()),
                "median":       float(np.median(gain_p)),
                "pct_positive": float((gain_p > 0).mean()),
            }
            regret_p = rp["delta_period_vs_best"].values
            stats["regret_period_vs_best"] = {
                "mean":   float(regret_p.mean()),
                "median": float(np.median(regret_p)),
            }
            crp = df["complexity_ratio_period_vs_full"].dropna().values
            if crp.size:
                stats["complexity_ratio_period_vs_full"] = {
                    "mean":   float(crp.mean()),
                    "median": float(np.median(crp)),
                }
            stats["mean_period_window"] = float(rp["period_window"].dropna().mean())

    if "period3_mase" in df.columns:
        rp3 = df.dropna(subset=["full_mase", "best_mase", "period3_mase"])
        if not rp3.empty:
            vals = rp3["period3_mase"].values
            stats["period3_mase"] = {
                "mean": float(vals.mean()),
                "geomean": _geomean(vals),
                "median": float(np.median(vals)),
                "std": float(vals.std()),
                "n": int(len(vals)),
            }
            beats = int((rp3["period3_mase"] < rp3["full_mase"]).sum())
            stats["period3_beats_full_count"] = beats
            stats["period3_beats_full_rate"] = beats / max(len(rp3), 1)
            stats["period3_beats_pred_count"] = int(
                (rp3["period3_mase"] < rp3["pred_mase"]).sum())
            gain = rp3["rel_gain_period3_over_full"].values
            stats["rel_gain_period3_over_full"] = {
                "mean": float(gain.mean()),
                "median": float(np.median(gain)),
                "pct_positive": float((gain > 0).mean()),
            }
            regret = rp3["delta_period3_vs_best"].values
            stats["regret_period3_vs_best"] = {
                "mean": float(regret.mean()),
                "median": float(np.median(regret)),
            }
            crp = df["complexity_ratio_period3_vs_full"].dropna().values
            if crp.size:
                stats["complexity_ratio_period3_vs_full"] = {
                    "mean": float(crp.mean()),
                    "median": float(np.median(crp)),
                }
            stats["mean_period3_window"] = float(
                rp3["period3_window"].dropna().mean())

    # ---- Predictor-variant strategies (v3 cheap, v4 Mamba, …) -----------------
    # Same sub-block the primary predictor / period strategies get, per variant.
    for vkey, _vlbl, _vclr in present_pred_variants(df):
        rv = df.dropna(subset=["full_mase", "best_mase", f"{vkey}_mase"])
        if rv.empty:
            continue
        vals = rv[f"{vkey}_mase"].values
        stats[f"{vkey}_mase"] = {
            "mean":    float(vals.mean()),
            "geomean": _geomean(vals),
            "median":  float(np.median(vals)),
            "std":     float(vals.std()),
            "n":       int(len(vals)),
        }
        stats[f"{vkey}_clamped_count"] = (
            int(df[f"{vkey}_clamped"].sum()) if f"{vkey}_clamped" in df.columns else 0)
        beats_full = int((rv[f"{vkey}_mase"] < rv["full_mase"]).sum())
        stats[f"{vkey}_beats_full_count"] = beats_full
        stats[f"{vkey}_beats_full_rate"]  = beats_full / max(len(rv), 1)
        stats[f"{vkey}_beats_pred_count"] = int((rv[f"{vkey}_mase"] < rv["pred_mase"]).sum())
        gain_v = rv[f"rel_gain_{vkey}_over_full"].values
        stats[f"rel_gain_{vkey}_over_full"] = {
            "mean":         float(gain_v.mean()),
            "median":       float(np.median(gain_v)),
            "pct_positive": float((gain_v > 0).mean()),
        }
        regret_v = rv[f"delta_{vkey}_vs_best"].values
        stats[f"regret_{vkey}_vs_best"] = {
            "mean":   float(regret_v.mean()),
            "median": float(np.median(regret_v)),
        }
        crv = df[f"complexity_ratio_{vkey}_vs_full"].dropna().values
        if crv.size:
            stats[f"complexity_ratio_{vkey}_vs_full"] = {
                "mean":   float(crv.mean()),
                "median": float(np.median(crv)),
            }
        spv = df[f"speedup_{vkey}_vs_full"].dropna().values
        if spv.size:
            stats[f"speedup_{vkey}_vs_full"] = {
                "mean":       float(spv.mean()),
                "median":     float(np.median(spv)),
                "pct_faster": float((spv > 1).mean()),
            }
        stats[f"mean_{vkey}_window"] = float(rv[f"{vkey}_window"].dropna().mean())

    return stats


# ==============================================================================
#  TOTAL FLOPs SAVINGS
# ==============================================================================

def compute_flops_savings(df: pd.DataFrame) -> pd.DataFrame:
    """Total theoretical FLOPs saved per strategy vs the full window, alongside
    the accompanying mean-MASE change.

    FLOPs and MASE use *different* aggregations on purpose:
    - FLOPs: each row stores the *per-forecast* proxy and aggregates ``n_instances``
      forecasts, so totals are instance-weighted sums — the physically correct
      "total compute" over the whole benchmark.
    - MASE: an *unweighted* geometric mean over the (dataset, term) rows — the
      M4/GiftEval convention, and identical to what ``plot_bar_aggregate_mase``
      reports (so ``geomean_full_mase`` matches the bar chart). Instance-weighting
      MASE would let a few high-window-count datasets dominate and pull the number
      far below the per-task average.

    ``mase_drop_vs_full`` (>0 = better) and its relative form ``rel_mase_drop_pct``
    = 100·(MASE_full − MASE_strategy)/MASE_full are both off these geomeans.
    ``mean_delta_vs_full`` keeps the (strategy − full) sign convention used
    elsewhere.  ``theoretical_flops`` is unnormalized (comparable only as a ratio
    within a model+horizon), so ``pct_flops_saved`` is the portable FLOPs number.
    """
    strat_specs = [("pred", "pred_flops", "pred_mase"),
                   ("best", "best_flops", "best_mase")]
    if "period_flops" in df.columns:
        strat_specs.append(("period", "period_flops", "period_mase"))
    for vkey, _vlbl, _vclr in present_pred_variants(df):
        if f"{vkey}_flops" in df.columns:
            strat_specs.append((vkey, f"{vkey}_flops", f"{vkey}_mase"))

    rows = []
    for name, flops_col, mase_col in strat_specs:
        sub = df.dropna(subset=["full_flops", flops_col, "full_mase", mase_col])
        if sub.empty:
            continue
        w = sub["n_instances"].values.astype(float)
        wsum = w.sum()
        full_f  = float((sub["full_flops"].values * w).sum())
        strat_f = float((sub[flops_col].values   * w).sum())
        saved   = full_f - strat_f
        # Unweighted geomean over (dataset, term) rows — matches the bar chart.
        gm_full_mase  = _geomean(sub["full_mase"].values)
        gm_strat_mase = _geomean(sub[mase_col].values)
        # Leaderboard-style NORMALISED geomean (MASE / seasonal-naive), over the
        # rows that have a valid baseline. Same row mask for full and strategy so
        # they stay comparable. NaN when no baseline is present (older runs / the
        # naive_mase column absent), which drops the norm columns from the table.
        gm_full_norm = gm_strat_norm = float("nan")
        n_norm = 0
        if "naive_mase" in sub.columns:
            nm = sub["naive_mase"].values.astype(float)
            valid = np.isfinite(nm) & (nm > 0)
            if valid.any():
                gm_full_norm  = _geomean(sub["full_mase"].values[valid] / nm[valid])
                gm_strat_norm = _geomean(sub[mase_col].values[valid]   / nm[valid])
                n_norm = int(valid.sum())
        rows.append({
            "strategy":             name,
            "n_rows":               int(len(sub)),
            "total_instances":      int(wsum),
            "total_full_flops":     full_f,
            "total_strategy_flops": strat_f,
            "flops_saved":          saved,
            "flops_ratio":          strat_f / full_f if full_f > 0 else float("nan"),
            "pct_flops_saved":      saved / full_f if full_f > 0 else float("nan"),
            "geomean_full_mase":     gm_full_mase,
            "geomean_strategy_mase": gm_strat_mase,
            "mase_drop_vs_full":     gm_full_mase - gm_strat_mase,  # >0 = better
            "rel_mase_drop_pct":     (100.0 * (gm_full_mase - gm_strat_mase) / gm_full_mase
                                      if gm_full_mase > 0 else float("nan")),
            "mean_delta_vs_full":    gm_strat_mase - gm_full_mase,  # (strat - full)
            # leaderboard-normalised twins (÷ seasonal-naive)
            "n_norm":                     n_norm,
            "geomean_full_mase_norm":     gm_full_norm,
            "geomean_strategy_mase_norm": gm_strat_norm,
            "rel_mase_drop_pct_norm":     (100.0 * (gm_full_norm - gm_strat_norm) / gm_full_norm
                                           if gm_full_norm > 0 else float("nan")),
        })
    return pd.DataFrame(rows)


def write_run_rollup(df: pd.DataFrame, out_dir: str,
                     plot_strategies: Optional[List[str]] = None,
                     suffix: str = "", metric_label: str = "MASE",
                     figures: bool = True) -> None:
    """Cross-model roll-up: per-(model, strategy) FLOPs savings + MASE change, a
    grand TOTAL row per strategy, the single overview figure, and the console
    summary.  Writes ``flops_savings_all_models{suffix}.csv`` and
    ``model_strategy_overview{suffix}.png`` into ``out_dir``.

    ``suffix`` / ``metric_label`` let each gluonts metric emit its own file set
    (e.g. ``model_strategy_overview_gluonts_real.png`` + CSV) without clobbering
    the other's.

    ``plot_strategies`` optionally restricts which strategies appear in the
    overview *figure* (the CSV and console totals always cover all strategies).

    Absolute FLOPs are an unnormalized proxy (comparable as a ratio within a
    model), so per-model rows carry the portable ``pct_flops_saved`` while the
    TOTAL row also sums the raw proxy FLOPs and reports both means.
    """
    rollup_rows = []
    for model_short, df_model in df.groupby("model_short"):
        fs = compute_flops_savings(df_model.reset_index(drop=True))
        if not fs.empty:
            fs.insert(0, "model_short", model_short)
            rollup_rows.append(fs)
    if not rollup_rows:
        print(Fore.YELLOW + "  No FLOPs/MASE records for the run-level roll-up." + Fore.RESET)
        return

    rollup = pd.concat(rollup_rows, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)

    # Single run-level overview figure: each (model, strategy) point. Skipped in
    # the default minimal-figures mode (the CSV + console totals always emit).
    if figures:
        overview_path = plot_model_strategy_overview(rollup, out_dir,
                                                     strategies=plot_strategies,
                                                     suffix=suffix, metric_label=metric_label)
        if overview_path:
            print(Fore.GREEN + f"\nSaved run-level overview: {overview_path}" + Fore.RESET)

    # Grand total per strategy across all models.
    totals = []
    for name, grp in rollup.groupby("strategy"):
        tot_full = grp["total_full_flops"].sum()
        tot_strat = grp["total_strategy_flops"].sum()
        inst = grp["total_instances"].sum()
        w_inst = grp["total_instances"].values.astype(float)
        # Per-model FLOPs ratios / fractional savings / absolute savings, aggregated
        # two ways.  Absolute FLOPs are unnormalized and incomparable across
        # families, so the geometric mean of the per-model ratio is the portable
        # aggregate; the arithmetic mean is provided alongside.
        ratios   = grp["flops_ratio"].dropna().values
        pct_sav  = grp["pct_flops_saved"].dropna().values
        abs_sav  = grp["flops_saved"].dropna().values
        # MASE: combine per-model *unweighted* geomeans with their row counts ->
        # exact global unweighted geometric-mean MASE over all (dataset, term) rows
        # (matches the bar chart's convention, just spanning every model).
        w_rows   = grp["n_rows"].values.astype(float)
        gm_full  = _wgeomean(grp["geomean_full_mase"].values,     w_rows)
        gm_strat = _wgeomean(grp["geomean_strategy_mase"].values, w_rows)
        # Same cross-model weighted-geomean aggregation for the leaderboard-
        # normalised twins, over the models that carry a baseline (weighted by the
        # per-model normalised row count). NaN if none do.
        gm_full_norm = gm_strat_norm = float("nan")
        if "geomean_full_mase_norm" in grp.columns:
            w_norm = grp.get("n_norm", pd.Series(0, index=grp.index)).values.astype(float)
            mnorm = (np.isfinite(grp["geomean_full_mase_norm"].values)
                     & np.isfinite(grp["geomean_strategy_mase_norm"].values)
                     & (w_norm > 0))
            if mnorm.any():
                gm_full_norm  = _wgeomean(grp["geomean_full_mase_norm"].values[mnorm],
                                          w_norm[mnorm])
                gm_strat_norm = _wgeomean(grp["geomean_strategy_mase_norm"].values[mnorm],
                                          w_norm[mnorm])
        totals.append({
            "model_short":          "TOTAL",
            "strategy":             name,
            "n_rows":               int(grp["n_rows"].sum()),
            "total_instances":      int(inst),
            "total_full_flops":     tot_full,
            "total_strategy_flops": tot_strat,
            "flops_saved":          tot_full - tot_strat,
            "flops_ratio":          tot_strat / tot_full if tot_full > 0 else float("nan"),
            "pct_flops_saved":      (tot_full - tot_strat) / tot_full if tot_full > 0 else float("nan"),
            # cross-model aggregation of the per-model FLOPs ratio
            "flops_ratio_amean":        float(ratios.mean()) if ratios.size else float("nan"),
            "flops_ratio_gmean":        _geomean(ratios) if ratios.size else float("nan"),
            # cross-model aggregation of the per-model fractional savings
            "pct_flops_saved_amean":    float(pct_sav.mean()) if pct_sav.size else float("nan"),
            "pct_flops_saved_gmean":    _geomean(pct_sav) if pct_sav.size else float("nan"),
            # cross-model aggregation of the per-model absolute FLOPs saved
            "flops_saved_amean":        float(abs_sav.mean()) if abs_sav.size else float("nan"),
            "flops_saved_gmean":        _geomean(abs_sav) if abs_sav.size else float("nan"),
            # MASE aggregated geometrically (unweighted, M4/GiftEval convention)
            "geomean_full_mase":     gm_full,
            "geomean_strategy_mase": gm_strat,
            "mase_drop_vs_full":     gm_full - gm_strat,
            "rel_mase_drop_pct":     (100.0 * (gm_full - gm_strat) / gm_full
                                      if gm_full > 0 else float("nan")),
            "mean_delta_vs_full":    gm_strat - gm_full,
            # leaderboard-normalised twins (÷ seasonal-naive)
            "geomean_full_mase_norm":     gm_full_norm,
            "geomean_strategy_mase_norm": gm_strat_norm,
            "rel_mase_drop_pct_norm":     (100.0 * (gm_full_norm - gm_strat_norm) / gm_full_norm
                                           if gm_full_norm > 0 else float("nan")),
        })
    rollup = pd.concat([rollup, pd.DataFrame(totals)], ignore_index=True)
    rollup_path = os.path.join(out_dir, f"flops_savings_all_models{suffix}.csv")
    rollup.to_csv(rollup_path, index=False, float_format="%.6g")
    print(Fore.GREEN + f"\nSaved run-level FLOPs savings: {rollup_path}" + Fore.RESET)

    # Companion table figure: original (full-window) MASE vs after-technique MASE
    # per (model, strategy), incl. the TOTAL rows just appended. Figure-gated too.
    if figures:
        table_path = plot_mase_change_table(rollup, out_dir, strategies=plot_strategies,
                                            suffix=suffix, metric_label=metric_label)
        if table_path:
            print(Fore.GREEN + f"Saved MASE change table: {table_path}" + Fore.RESET)
    print(Fore.CYAN + "\n--- Grand total FLOPs saved vs full window ---" + Fore.RESET)
    for t in totals:
        print(f"  {t['strategy']:<7}  pooled saved {100*t['pct_flops_saved']:5.1f}%  "
              f"({t['flops_saved']:.3g} FLOPs)")
        print(f"           per-model %saved  amean={100*t['pct_flops_saved_amean']:5.1f}%  "
              f"gmean={100*t['pct_flops_saved_gmean']:5.1f}%   |   "
              f"FLOPs ratio amean={t['flops_ratio_amean']:.3f}  gmean={t['flops_ratio_gmean']:.3f}")
        print(f"           abs FLOPs saved  amean={t['flops_saved_amean']:.3g}  "
              f"gmean={t['flops_saved_gmean']:.3g}")
        print(f"           geomean MASE {t['geomean_full_mase']:.4f} -> "
              f"{t['geomean_strategy_mase']:.4f}  (drop {t['mase_drop_vs_full']:+.4f}, "
              f"{t['rel_mase_drop_pct']:+.2f}%)")


# ==============================================================================
#  STAGE 6 — MEASURED WALL-CLOCK FORWARD-PASS TIME SAVED
# ==============================================================================
#  Twin of the FLOPs roll-up above, but using the *measured* wall-clock seconds
#  of the forward pass (``*_elapsed_s``, read from each window's metrics.json)
#  instead of the theoretical MAC proxy.  Same accuracy axis (geomean MASE), so
#  the resulting figure (``model_strategy_overview_time.png``) is a drop-in
#  companion to the FLOPs/MAC one — only the cost axis changes from "FLOPs saved"
#  to "wall-clock time saved".

def compute_time_savings(df: pd.DataFrame) -> pd.DataFrame:
    """Total measured wall-clock forward-pass time saved per strategy vs the full
    window, alongside the accompanying geomean-MASE change.

    Unlike the FLOPs proxy (per-forecast, instance-weighted), ``*_elapsed_s`` is
    already the *total* wall-clock for the whole (dataset, term) inference pass at
    that window, so the benchmark total is a plain sum of the per-row seconds — no
    instance weighting.  MASE is the same *unweighted* geometric mean over the
    (dataset, term) rows as ``compute_flops_savings`` uses, so ``rel_mase_drop_pct``
    is identical to the FLOPs figure's y-axis (only the cost axis differs).
    """
    strat_specs = [("pred", "pred_elapsed_s", "pred_mase"),
                   ("best", "best_elapsed_s", "best_mase")]
    if "period_elapsed_s" in df.columns:
        strat_specs.append(("period", "period_elapsed_s", "period_mase"))
    for vkey, _vlbl, _vclr in present_pred_variants(df):
        if f"{vkey}_elapsed_s" in df.columns:
            strat_specs.append((vkey, f"{vkey}_elapsed_s", f"{vkey}_mase"))

    rows = []
    for name, time_col, mase_col in strat_specs:
        sub = df.dropna(subset=["full_elapsed_s", time_col, "full_mase", mase_col])
        if sub.empty:
            continue
        full_t  = float(sub["full_elapsed_s"].values.sum())
        strat_t = float(sub[time_col].values.sum())
        saved   = full_t - strat_t
        # Per-cell robust-timing std (NaN for single-shot cells). The benchmark
        # total's std is the quadrature sum (independent timings); use it for the
        # error bar on the strategy time / time-saved axis.
        std_col = time_col.replace("_elapsed_s", "_elapsed_std_s")
        strat_std = _quadrature(sub[std_col].values) if std_col in sub.columns else float("nan")
        full_std  = _quadrature(sub["full_elapsed_std_s"].values) \
            if "full_elapsed_std_s" in sub.columns else float("nan")
        saved_std = float(np.sqrt(np.nansum([strat_std ** 2, full_std ** 2]))) \
            if not (math.isnan(strat_std) and math.isnan(full_std)) else float("nan")
        gm_full_mase  = _geomean(sub["full_mase"].values)
        gm_strat_mase = _geomean(sub[mase_col].values)
        rows.append({
            "strategy":              name,
            "n_rows":                int(len(sub)),
            "total_instances":       int(sub["n_instances"].sum()),
            "total_full_time_s":     full_t,
            "total_strategy_time_s": strat_t,
            "total_strategy_time_std_s": strat_std,
            "time_saved_s":          saved,
            "time_saved_std_s":      saved_std,
            "time_ratio":            strat_t / full_t if full_t > 0 else float("nan"),
            "pct_time_saved":        saved / full_t if full_t > 0 else float("nan"),
            "pct_time_saved_std":    (strat_std / full_t
                                      if full_t > 0 and not math.isnan(strat_std)
                                      else float("nan")),
            "geomean_full_mase":     gm_full_mase,
            "geomean_strategy_mase": gm_strat_mase,
            "mase_drop_vs_full":     gm_full_mase - gm_strat_mase,  # >0 = better
            "rel_mase_drop_pct":     (100.0 * (gm_full_mase - gm_strat_mase) / gm_full_mase
                                      if gm_full_mase > 0 else float("nan")),
            "mean_delta_vs_full":    gm_strat_mase - gm_full_mase,  # (strat - full)
        })
    return pd.DataFrame(rows)


def write_run_time_rollup(df: pd.DataFrame, out_dir: str,
                          plot_strategies: Optional[List[str]] = None,
                          figures: bool = True) -> None:
    """Cross-model roll-up of measured forward-pass time saved: per-(model,
    strategy) rows, a grand TOTAL per strategy, the overview figure, and the
    console summary.  Writes ``time_savings_all_models.csv`` and
    ``model_strategy_overview_time.png`` into ``out_dir``.  Twin of
    ``write_run_rollup`` for wall-clock time rather than theoretical FLOPs.

    ``plot_strategies`` optionally restricts which strategies appear in the
    overview *figure* (the CSV and console totals always cover all strategies).
    """
    rollup_rows = []
    for model_short, df_model in df.groupby("model_short"):
        ts = compute_time_savings(df_model.reset_index(drop=True))
        if not ts.empty:
            ts.insert(0, "model_short", model_short)
            rollup_rows.append(ts)
    if not rollup_rows:
        print(Fore.YELLOW + "  No timing/MASE records for the run-level time roll-up." + Fore.RESET)
        return

    rollup = pd.concat(rollup_rows, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)

    if figures:
        overview_path = plot_model_strategy_overview_time(rollup, out_dir,
                                                          strategies=plot_strategies)
        if overview_path:
            print(Fore.GREEN + f"\nSaved run-level time overview: {overview_path}" + Fore.RESET)

    totals = []
    for name, grp in rollup.groupby("strategy"):
        tot_full  = grp["total_full_time_s"].sum()
        tot_strat = grp["total_strategy_time_s"].sum()
        inst = grp["total_instances"].sum()
        # Per-model time ratios / fractional savings / absolute savings, aggregated
        # both ways; wall-clock seconds are comparable across families, so the
        # pooled sum *and* the per-model means are all meaningful.
        ratios  = grp["time_ratio"].dropna().values
        pct_sav = grp["pct_time_saved"].dropna().values
        abs_sav = grp["time_saved_s"].dropna().values
        # Pooled robust-timing std across models (quadrature of per-model totals).
        tot_strat_std = (_quadrature(grp["total_strategy_time_std_s"].values)
                         if "total_strategy_time_std_s" in grp.columns else float("nan"))
        w_rows   = grp["n_rows"].values.astype(float)
        gm_full  = _wgeomean(grp["geomean_full_mase"].values,     w_rows)
        gm_strat = _wgeomean(grp["geomean_strategy_mase"].values, w_rows)
        totals.append({
            "model_short":           "TOTAL",
            "strategy":              name,
            "n_rows":                int(grp["n_rows"].sum()),
            "total_instances":       int(inst),
            "total_full_time_s":     tot_full,
            "total_strategy_time_s": tot_strat,
            "total_strategy_time_std_s": tot_strat_std,
            "time_saved_s":          tot_full - tot_strat,
            "time_ratio":            tot_strat / tot_full if tot_full > 0 else float("nan"),
            "pct_time_saved":        (tot_full - tot_strat) / tot_full if tot_full > 0 else float("nan"),
            "pct_time_saved_std":    (tot_strat_std / tot_full
                                      if tot_full > 0 and not math.isnan(tot_strat_std)
                                      else float("nan")),
            "time_ratio_amean":      float(ratios.mean()) if ratios.size else float("nan"),
            "time_ratio_gmean":      _geomean(ratios) if ratios.size else float("nan"),
            "pct_time_saved_amean":  float(pct_sav.mean()) if pct_sav.size else float("nan"),
            "pct_time_saved_gmean":  _geomean(pct_sav) if pct_sav.size else float("nan"),
            "time_saved_amean":      float(abs_sav.mean()) if abs_sav.size else float("nan"),
            "time_saved_gmean":      _geomean(abs_sav) if abs_sav.size else float("nan"),
            "geomean_full_mase":     gm_full,
            "geomean_strategy_mase": gm_strat,
            "mase_drop_vs_full":     gm_full - gm_strat,
            "rel_mase_drop_pct":     (100.0 * (gm_full - gm_strat) / gm_full
                                      if gm_full > 0 else float("nan")),
            "mean_delta_vs_full":    gm_strat - gm_full,
        })
    rollup = pd.concat([rollup, pd.DataFrame(totals)], ignore_index=True)
    rollup_path = os.path.join(out_dir, "time_savings_all_models.csv")
    rollup.to_csv(rollup_path, index=False, float_format="%.6g")
    print(Fore.GREEN + f"\nSaved run-level time savings: {rollup_path}" + Fore.RESET)
    print(Fore.CYAN + "\n--- Grand total forward-pass time saved vs full window ---" + Fore.RESET)
    for t in totals:
        print(f"  {t['strategy']:<7}  pooled saved {100*t['pct_time_saved']:5.1f}%  "
              f"({t['time_saved_s']:.3g}s of {t['total_full_time_s']:.3g}s)")
        print(f"           per-model %saved  amean={100*t['pct_time_saved_amean']:5.1f}%  "
              f"gmean={100*t['pct_time_saved_gmean']:5.1f}%   |   "
              f"time ratio amean={t['time_ratio_amean']:.3f}  gmean={t['time_ratio_gmean']:.3f}")
        print(f"           abs time saved   amean={t['time_saved_amean']:.3g}s  "
              f"gmean={t['time_saved_gmean']:.3g}s")
        print(f"           geomean MASE {t['geomean_full_mase']:.4f} -> "
              f"{t['geomean_strategy_mase']:.4f}  (drop {t['mase_drop_vs_full']:+.4f}, "
              f"{t['rel_mase_drop_pct']:+.2f}%)")


# ==============================================================================
#  RELATIVE IMPROVEMENT TABLES
# ==============================================================================

# Maps the trailing frequency token in dataset_display to a canonical label.
_FREQ_SUFFIX_MAP: Dict[str, str] = {
    "10T": "10-min",
    "5T":  "5-min",
    "15T": "15-min",
    "H":   "Hourly",
    "D":   "Daily",
    "W":   "Weekly",
    "M":   "Monthly",
    "Q":   "Quarterly",
    # M4 full-word suffixes (e.g. "M4-Hourly")
    "Hourly":     "Hourly",
    "Daily":      "Daily",
    "Weekly":     "Weekly",
    "Monthly":    "Monthly",
    "Quarterly":  "Quarterly",
    "Yearly":     "Yearly",
}


def infer_freq_from_display(dataset_display: str) -> str:
    """
    Infer sampling frequency from dataset_display name.

    Checks the part after the last '-' first (e.g. 'ETTm1-H' → 'H'),
    then falls back to a keyword scan of the full name.
    Returns 'Unknown' when nothing matches.
    """
    parts = dataset_display.rsplit("-", 1)
    if len(parts) == 2:
        suffix = parts[1]
        if suffix in _FREQ_SUFFIX_MAP:
            return _FREQ_SUFFIX_MAP[suffix]
    # fallback: keyword scan (case-insensitive)
    name_lower = dataset_display.lower()
    for kw, label in [
        ("10t", "10-min"), ("5t", "5-min"), ("15t", "15-min"),
        ("hourly", "Hourly"), ("-h", "Hourly"),
        ("daily", "Daily"),   ("-d", "Daily"),
        ("weekly", "Weekly"), ("-w", "Weekly"),
        ("monthly", "Monthly"), ("-m", "Monthly"),
        ("quarterly", "Quarterly"),
        ("yearly", "Yearly"),
    ]:
        if kw in name_lower:
            return label
    return "Unknown"


# Canonical domain order for the by-domain plot/table.
DOMAIN_ORDER = ["Energy", "Nature", "Environment", "Health", "Transport", "Other"]

# Maps a dataset (by a keyword in its display name, freq-suffix-agnostic) to one of
# the analysis domains. EDIT HERE to re-bucket a dataset. Datasets that match no
# keyword fall into "Other" (web/cloudops/sales/finance — bizitobs, bitbrains,
# car_parts, hierarchical_sales, m4, restaurant). The Nature/Environment split
# follows: Nature = physical/biological processes (weather, river flow), Environment
# = environmental monitoring (air quality, rainfall/climate).
_DOMAIN_KEYWORDS: List[Tuple[str, str]] = [
    ("ett",             "Energy"),        # electricity transformer temperature
    ("electricity",     "Energy"),
    ("solar",           "Energy"),
    ("jenaweather",     "Nature"),
    ("saugeen",         "Nature"),        # river flow
    ("kddcup",          "Environment"),   # KDD Cup 2018 air quality
    ("temprain",        "Environment"),   # temperature_rain
    ("hospital",        "Health"),
    ("covid",           "Health"),
    ("usbirths",        "Health"),
    ("loopseattle",     "Transport"),
    ("sztaxi",          "Transport"),
    ("mdense",          "Transport"),     # M_DENSE road sensors
]


def infer_domain_from_display(dataset_display: str) -> str:
    """Map a dataset_display to an analysis domain (Energy / Nature / Environment /
    Health / Transport), falling back to 'Other'. Frequency-suffix-agnostic: the
    keyword scan ignores the trailing '-<freq>' token."""
    base = dataset_display.rsplit("-", 1)[0] if "-" in dataset_display else dataset_display
    key = base.lower().replace("_", "").replace("-", "")
    for kw, domain in _DOMAIN_KEYWORDS:
        if kw in key:
            return domain
    return "Other"


def compute_relative_improvement_tables(
    df: pd.DataFrame,
    out_dir: str,
    figures: bool = True,
) -> None:
    """
    Produces four CSV files and prints summaries:

    1. rel_improvement_individual.csv  — one row per (dataset, model, term)
    2. rel_improvement_by_dataset.csv  — mean over models/terms per dataset
    3. rel_improvement_by_frequency.csv — mean over datasets per frequency
    4. rel_improvement_by_horizon.csv  — mean over datasets per horizon value
    5. rel_improvement_by_domain.csv   — mean per domain (Energy/Nature/Environment/
                                          Health/Transport/Other), + rel_impr_by_domain.png

    Relative improvement (%) is defined as:
      rel_impr_oracle = (full_mase - best_mase) / full_mase * 100
      rel_impr_pred   = (full_mase - pred_mase) / full_mase * 100
    Positive = the alternative strategy achieves lower MASE than the full window.
    """
    r = df.dropna(subset=["full_mase", "best_mase", "pred_mase"]).copy()
    if r.empty:
        print(Fore.YELLOW + "  No valid rows for relative improvement tables." + Fore.RESET)
        return

    denom = r["full_mase"].clip(lower=1e-12)
    r["rel_impr_oracle_pct"] = (r["full_mase"] - r["best_mase"]) / denom * 100.0
    r["rel_impr_pred_pct"]   = (r["full_mase"] - r["pred_mase"]) / denom * 100.0
    r["freq"] = r["dataset_display"].apply(infer_freq_from_display)

    # Frequency display order
    freq_order = ["10-min", "5-min", "15-min", "Hourly", "Daily",
                  "Weekly", "Monthly", "Quarterly", "Yearly", "Unknown"]

    # ------------------------------------------------------------------
    # 1. Individual table
    # ------------------------------------------------------------------
    cols_ind = [
        "dataset_display", "freq", "model_short", "term", "horizon",
        "full_mase", "best_mase", "pred_mase",
        "rel_impr_oracle_pct", "rel_impr_pred_pct",
    ]
    ind = r[cols_ind].sort_values(["dataset_display", "model_short", "term"])
    ind_path = os.path.join(out_dir, "rel_improvement_individual.csv")
    ind.to_csv(ind_path, index=False, float_format="%.4f")
    print(Fore.GREEN + f"  Saved: {ind_path}" + Fore.RESET)

    # ------------------------------------------------------------------
    # 2. By dataset
    # ------------------------------------------------------------------
    ds_agg = (
        r.groupby(["dataset_display", "freq"], sort=True)
        .agg(
            n_rows=("full_mase", "count"),
            full_mase_mean=("full_mase", "mean"),
            best_mase_mean=("best_mase", "mean"),
            pred_mase_mean=("pred_mase", "mean"),
            rel_impr_oracle_mean=("rel_impr_oracle_pct", "mean"),
            rel_impr_pred_mean=("rel_impr_pred_pct", "mean"),
            rel_impr_oracle_median=("rel_impr_oracle_pct", "median"),
            rel_impr_pred_median=("rel_impr_pred_pct", "median"),
        )
        .reset_index()
    )
    ds_path = os.path.join(out_dir, "rel_improvement_by_dataset.csv")
    ds_agg.to_csv(ds_path, index=False, float_format="%.4f")
    print(Fore.GREEN + f"  Saved: {ds_path}" + Fore.RESET)

    # ------------------------------------------------------------------
    # 3. By frequency
    # ------------------------------------------------------------------
    freq_agg = (
        r.groupby("freq", sort=False)
        .agg(
            n_rows=("full_mase", "count"),
            full_mase_mean=("full_mase", "mean"),
            best_mase_mean=("best_mase", "mean"),
            pred_mase_mean=("pred_mase", "mean"),
            rel_impr_oracle_mean=("rel_impr_oracle_pct", "mean"),
            rel_impr_pred_mean=("rel_impr_pred_pct", "mean"),
            rel_impr_oracle_median=("rel_impr_oracle_pct", "median"),
            rel_impr_pred_median=("rel_impr_pred_pct", "median"),
        )
        .reset_index()
    )
    # Sort by canonical frequency order
    freq_agg["_order"] = freq_agg["freq"].apply(
        lambda f: freq_order.index(f) if f in freq_order else len(freq_order)
    )
    freq_agg = freq_agg.sort_values("_order").drop(columns="_order")
    freq_path = os.path.join(out_dir, "rel_improvement_by_frequency.csv")
    freq_agg.to_csv(freq_path, index=False, float_format="%.4f")
    print(Fore.GREEN + f"  Saved: {freq_path}" + Fore.RESET)

    # ------------------------------------------------------------------
    # 4. By horizon
    # ------------------------------------------------------------------
    hor_agg = (
        r.groupby(["term", "horizon"], sort=False)
        .agg(
            n_rows=("full_mase", "count"),
            full_mase_mean=("full_mase", "mean"),
            best_mase_mean=("best_mase", "mean"),
            pred_mase_mean=("pred_mase", "mean"),
            rel_impr_oracle_mean=("rel_impr_oracle_pct", "mean"),
            rel_impr_pred_mean=("rel_impr_pred_pct", "mean"),
            rel_impr_oracle_median=("rel_impr_oracle_pct", "median"),
            rel_impr_pred_median=("rel_impr_pred_pct", "median"),
        )
        .reset_index()
    )
    term_order = {"short": 0, "medium": 1, "long": 2}
    hor_agg["_order"] = hor_agg["term"].apply(lambda t: term_order.get(t, 99))
    hor_agg = hor_agg.sort_values(["_order", "horizon"]).drop(columns="_order")
    hor_path = os.path.join(out_dir, "rel_improvement_by_horizon.csv")
    hor_agg.to_csv(hor_path, index=False, float_format="%.4f")
    print(Fore.GREEN + f"  Saved: {hor_path}" + Fore.RESET)

    # ------------------------------------------------------------------
    # 5. By domain (Energy / Nature / Environment / Health / Transport / Other)
    # ------------------------------------------------------------------
    r["domain"] = r["dataset_display"].apply(infer_domain_from_display)
    domain_agg = (
        r.groupby("domain", sort=False)
        .agg(
            n_rows=("full_mase", "count"),
            full_mase_mean=("full_mase", "mean"),
            best_mase_mean=("best_mase", "mean"),
            pred_mase_mean=("pred_mase", "mean"),
            rel_impr_oracle_mean=("rel_impr_oracle_pct", "mean"),
            rel_impr_pred_mean=("rel_impr_pred_pct", "mean"),
            rel_impr_oracle_median=("rel_impr_oracle_pct", "median"),
            rel_impr_pred_median=("rel_impr_pred_pct", "median"),
        )
        .reset_index()
    )
    domain_agg["_order"] = domain_agg["domain"].apply(
        lambda d: DOMAIN_ORDER.index(d) if d in DOMAIN_ORDER else len(DOMAIN_ORDER)
    )
    domain_agg = domain_agg.sort_values("_order").drop(columns="_order")
    domain_path = os.path.join(out_dir, "rel_improvement_by_domain.csv")
    domain_agg.to_csv(domain_path, index=False, float_format="%.4f")
    print(Fore.GREEN + f"  Saved: {domain_path}" + Fore.RESET)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    def _fmt_table(frame: pd.DataFrame, title: str) -> None:
        print(Fore.CYAN + f"\n{'='*70}" + Fore.RESET)
        print(Fore.CYAN + f"  {title}" + Fore.RESET)
        print(Fore.CYAN + f"{'='*70}" + Fore.RESET)
        with pd.option_context(
            "display.max_rows", 200,
            "display.width", 160,
            "display.float_format", "{:.2f}".format,
        ):
            print(frame.to_string(index=False))

    _fmt_table(
        ds_agg[["dataset_display", "freq", "n_rows",
                "full_mase_mean", "best_mase_mean", "pred_mase_mean",
                "rel_impr_oracle_mean", "rel_impr_pred_mean"]].rename(columns={
            "full_mase_mean": "full",
            "best_mase_mean": "oracle",
            "pred_mase_mean": "pred",
            "rel_impr_oracle_mean": "Δoracle%",
            "rel_impr_pred_mean":   "Δpred%",
        }),
        "Relative improvement vs full window — by dataset (mean over models/terms)",
    )

    _fmt_table(
        freq_agg[["freq", "n_rows",
                  "full_mase_mean", "best_mase_mean", "pred_mase_mean",
                  "rel_impr_oracle_mean", "rel_impr_pred_mean"]].rename(columns={
            "full_mase_mean": "full",
            "best_mase_mean": "oracle",
            "pred_mase_mean": "pred",
            "rel_impr_oracle_mean": "Δoracle%",
            "rel_impr_pred_mean":   "Δpred%",
        }),
        "Relative improvement vs full window — by sampling frequency",
    )

    _fmt_table(
        hor_agg[["term", "horizon", "n_rows",
                 "full_mase_mean", "best_mase_mean", "pred_mase_mean",
                 "rel_impr_oracle_mean", "rel_impr_pred_mean"]].rename(columns={
            "full_mase_mean": "full",
            "best_mase_mean": "oracle",
            "pred_mase_mean": "pred",
            "rel_impr_oracle_mean": "Δoracle%",
            "rel_impr_pred_mean":   "Δpred%",
        }),
        "Relative improvement vs full window — by horizon",
    )

    _fmt_table(
        domain_agg[["domain", "n_rows",
                    "full_mase_mean", "best_mase_mean", "pred_mase_mean",
                    "rel_impr_oracle_mean", "rel_impr_pred_mean"]].rename(columns={
            "full_mase_mean": "full",
            "best_mase_mean": "oracle",
            "pred_mase_mean": "pred",
            "rel_impr_oracle_mean": "Δoracle%",
            "rel_impr_pred_mean":   "Δpred%",
        }),
        "Relative improvement vs full window — by domain",
    )

    # ------------------------------------------------------------------
    # Plots (skipped in the default minimal-figures mode; CSVs above always
    # written)
    # ------------------------------------------------------------------
    if figures:
        _plot_rel_impr_by_dataset(ds_agg, out_dir)
        _plot_rel_impr_by_frequency(freq_agg, out_dir)
        _plot_rel_impr_by_horizon(hor_agg, out_dir)
        _plot_rel_impr_by_domain(domain_agg, out_dir)
        _plot_rel_impr_pred_vs_oracle(ds_agg, out_dir)


def _plot_rel_impr_by_dataset(ds_agg: pd.DataFrame, out_dir: str) -> str:
    """
    Horizontal grouped bar chart: Δoracle% and Δpred% per dataset.
    Sorted by Δoracle% descending so the most-improvable datasets appear first.
    """
    data = ds_agg.sort_values("rel_impr_oracle_mean", ascending=True).copy()
    if data.empty:
        return ""

    n = len(data)
    bar_h = 0.35
    fig_h = max(6, n * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    y = np.arange(n)
    oracle_vals = data["rel_impr_oracle_mean"].values
    pred_vals   = data["rel_impr_pred_mean"].values
    labels      = data["dataset_display"].values

    bars_oracle = ax.barh(y + bar_h / 2, oracle_vals, bar_h,
                          label="Oracle (best window)", color="#ED7D31", alpha=0.85, edgecolor="white")
    bars_pred   = ax.barh(y - bar_h / 2, pred_vals,   bar_h,
                          label="Predictor",            color="#70AD47", alpha=0.85, edgecolor="white")

    # Value annotations
    x_range = max(np.abs(oracle_vals).max(), np.abs(pred_vals).max(), 1e-3)
    pad = x_range * 0.015
    for bar, val in zip(list(bars_oracle) + list(bars_pred),
                        list(oracle_vals) + list(pred_vals)):
        w = bar.get_width()
        xpos = w + pad if w >= 0 else w - pad
        ha   = "left"   if w >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha=ha, fontsize=6.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_xlabel("Relative MASE improvement over full window  (positive = better than full)",
                  fontsize=10)
    ax.set_title(
        "Relative MASE Improvement vs Full Window — per dataset\n"
        "(mean over all models and forecast terms; sorted by oracle improvement)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out_dir, "rel_impr_by_dataset.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(Fore.GREEN + f"  Saved: {path}" + Fore.RESET)
    return path


def _plot_rel_impr_by_frequency(freq_agg: pd.DataFrame, out_dir: str) -> str:
    """
    Vertical grouped bar chart: Δoracle% and Δpred% per sampling frequency.
    Also overlays median as a dot for robustness.
    """
    data = freq_agg.copy()
    if data.empty:
        return ""

    n = len(data)
    x = np.arange(n)
    w = 0.32
    fig, ax = plt.subplots(figsize=(max(8, n * 1.1 + 2), 6))

    oracle_mean = data["rel_impr_oracle_mean"].values
    pred_mean   = data["rel_impr_pred_mean"].values
    oracle_med  = data["rel_impr_oracle_median"].values
    pred_med    = data["rel_impr_pred_median"].values
    freq_labels = data["freq"].values

    b1 = ax.bar(x - w / 2, oracle_mean, w, label="Oracle — mean",
                color="#ED7D31", alpha=0.82, edgecolor="white")
    b2 = ax.bar(x + w / 2, pred_mean,   w, label="Predictor — mean",
                color="#70AD47", alpha=0.82, edgecolor="white")

    # Median dots
    ax.scatter(x - w / 2, oracle_med, s=45, color="#A84000", zorder=4,
               label="Oracle — median", marker="D")
    ax.scatter(x + w / 2, pred_med,   s=45, color="#2A6E10", zorder=4,
               label="Predictor — median", marker="D")

    # Annotate mean bars
    y_rng = max(np.abs(oracle_mean).max(), np.abs(pred_mean).max(), 1e-3)
    pad   = y_rng * 0.03
    for b, val in zip(list(b1) + list(b2), list(oracle_mean) + list(pred_mean)):
        h = b.get_height()
        ypos = h + pad if h >= 0 else h - pad
        va   = "bottom" if h >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, ypos,
                f"{val:.1f}%", ha="center", va=va, fontsize=8.5, color="#222222")

    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(freq_labels, fontsize=10)
    ax.set_ylabel("Relative MASE improvement over full window (%)", fontsize=10)
    ax.set_title(
        "Relative MASE Improvement vs Full Window — by sampling frequency\n"
        "(mean ± median across all datasets / models / terms)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out_dir, "rel_impr_by_frequency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(Fore.GREEN + f"  Saved: {path}" + Fore.RESET)
    return path


def _plot_rel_impr_by_domain(domain_agg: pd.DataFrame, out_dir: str) -> str:
    """
    Vertical grouped bar chart: Δoracle% and Δpred% per domain (Energy / Nature /
    Environment / Health / Transport / Other). Median overlaid as a dot. Twin of
    ``_plot_rel_impr_by_frequency`` — mean ± median across datasets/models/terms.
    """
    data = domain_agg.copy()
    if data.empty:
        return ""

    n = len(data)
    x = np.arange(n)
    w = 0.32
    fig, ax = plt.subplots(figsize=(max(8, n * 1.3 + 2), 6))

    oracle_mean = data["rel_impr_oracle_mean"].values
    pred_mean   = data["rel_impr_pred_mean"].values
    oracle_med  = data["rel_impr_oracle_median"].values
    pred_med    = data["rel_impr_pred_median"].values
    domain_labels = [f"{d}\n(n={int(nr)})" for d, nr in
                     zip(data["domain"].values, data["n_rows"].values)]

    b1 = ax.bar(x - w / 2, oracle_mean, w, label="Oracle — mean",
                color="#ED7D31", alpha=0.82, edgecolor="white")
    b2 = ax.bar(x + w / 2, pred_mean,   w, label="Predictor — mean",
                color="#70AD47", alpha=0.82, edgecolor="white")

    # Median dots
    ax.scatter(x - w / 2, oracle_med, s=45, color="#A84000", zorder=4,
               label="Oracle — median", marker="D")
    ax.scatter(x + w / 2, pred_med,   s=45, color="#2A6E10", zorder=4,
               label="Predictor — median", marker="D")

    # Annotate mean bars
    y_rng = max(np.abs(oracle_mean).max(), np.abs(pred_mean).max(), 1e-3)
    pad   = y_rng * 0.03
    for b, val in zip(list(b1) + list(b2), list(oracle_mean) + list(pred_mean)):
        h = b.get_height()
        ypos = h + pad if h >= 0 else h - pad
        va   = "bottom" if h >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, ypos,
                f"{val:.1f}%", ha="center", va=va, fontsize=8.5, color="#222222")

    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels, fontsize=10)
    ax.set_ylabel("Relative MASE improvement over full window (%)", fontsize=10)
    ax.set_title(
        "Relative MASE Improvement vs Full Window — by domain\n"
        "(mean ± median across all datasets / models / terms)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out_dir, "rel_impr_by_domain.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(Fore.GREEN + f"  Saved: {path}" + Fore.RESET)
    return path


def _plot_rel_impr_by_horizon(hor_agg: pd.DataFrame, out_dir: str) -> str:
    """
    Two-panel figure:
      Left:  grouped bars by term (short / medium / long), mean + median overlay.
      Right: scatter of mean Δ% vs horizon value for oracle and predictor.
    """
    data = hor_agg.copy()
    if data.empty:
        return ""

    fig, (ax_bar, ax_sc) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: grouped bars by term ----------------------------------------
    term_data = (
        data.groupby("term", sort=False)
        .agg(
            oracle_mean=("rel_impr_oracle_mean", "mean"),
            pred_mean=("rel_impr_pred_mean",   "mean"),
            oracle_med=("rel_impr_oracle_median", "mean"),
            pred_med=("rel_impr_pred_median",   "mean"),
        )
        .reset_index()
    )
    term_order = {"short": 0, "medium": 1, "long": 2}
    term_data["_ord"] = term_data["term"].map(term_order).fillna(99)
    term_data = term_data.sort_values("_ord").drop(columns="_ord")

    n_t = len(term_data)
    xt  = np.arange(n_t)
    wt  = 0.32
    b1 = ax_bar.bar(xt - wt / 2, term_data["oracle_mean"].values, wt,
                    label="Oracle — mean", color="#ED7D31", alpha=0.82, edgecolor="white")
    b2 = ax_bar.bar(xt + wt / 2, term_data["pred_mean"].values,   wt,
                    label="Predictor — mean", color="#70AD47", alpha=0.82, edgecolor="white")
    ax_bar.scatter(xt - wt / 2, term_data["oracle_med"].values, s=50,
                   color="#A84000", zorder=4, label="Oracle — median", marker="D")
    ax_bar.scatter(xt + wt / 2, term_data["pred_med"].values,   s=50,
                   color="#2A6E10", zorder=4, label="Predictor — median", marker="D")

    y_rng = max(
        np.abs(term_data["oracle_mean"].values).max(),
        np.abs(term_data["pred_mean"].values).max(), 1e-3,
    )
    pad = y_rng * 0.04
    for b, val in zip(list(b1) + list(b2),
                      list(term_data["oracle_mean"].values) + list(term_data["pred_mean"].values)):
        h = b.get_height()
        yp = h + pad if h >= 0 else h - pad
        ax_bar.text(b.get_x() + b.get_width() / 2, yp,
                    f"{val:.1f}%", ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=9, color="#222222")

    ax_bar.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax_bar.set_xticks(xt)
    ax_bar.set_xticklabels(term_data["term"].values, fontsize=11)
    ax_bar.set_ylabel("Relative MASE improvement (%)", fontsize=10)
    ax_bar.set_title("By forecast term", fontsize=11, fontweight="bold")
    ax_bar.legend(fontsize=8, ncol=2)
    ax_bar.grid(axis="y", alpha=0.25)

    # ---- Right: scatter Δ% vs actual horizon value -------------------------
    ax_sc.scatter(data["horizon"], data["rel_impr_oracle_mean"],
                  s=60, color="#ED7D31", alpha=0.80, edgecolors="white", linewidths=0.5,
                  label="Oracle")
    ax_sc.scatter(data["horizon"], data["rel_impr_pred_mean"],
                  s=60, color="#70AD47", alpha=0.80, edgecolors="white", linewidths=0.5,
                  marker="^", label="Predictor")
    ax_sc.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax_sc.set_xlabel("Horizon (steps)", fontsize=10)
    ax_sc.set_ylabel("Mean relative MASE improvement (%)", fontsize=10)
    ax_sc.set_title("By horizon length", fontsize=11, fontweight="bold")
    ax_sc.legend(fontsize=9)
    ax_sc.grid(True, alpha=0.25)

    fig.suptitle(
        "Relative MASE Improvement vs Full Window — by forecast horizon",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "rel_impr_by_horizon.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(Fore.GREEN + f"  Saved: {path}" + Fore.RESET)
    return path


def _plot_rel_impr_pred_vs_oracle(ds_agg: pd.DataFrame, out_dir: str) -> str:
    """
    Scatter: Δpred% (y) vs Δoracle% (x) coloured by frequency, one point per
    dataset.  Shows how well the predictor tracks the oracle improvement signal
    across datasets.  Points above the diagonal mean the predictor captures more
    than the oracle (unusual); points below are the typical regret cases.
    """
    data = ds_agg.dropna(subset=["rel_impr_oracle_mean", "rel_impr_pred_mean"]).copy()
    if data.empty:
        return ""

    freqs = data["freq"].unique().tolist()
    cmap  = plt.get_cmap("tab10")
    freq_colors = {f: cmap(i % 10) for i, f in enumerate(sorted(freqs))}

    fig, ax = plt.subplots(figsize=(8, 7))
    for freq, grp in data.groupby("freq"):
        ax.scatter(
            grp["rel_impr_oracle_mean"], grp["rel_impr_pred_mean"],
            s=65, alpha=0.85, edgecolors="white", linewidths=0.5,
            color=freq_colors[freq], label=freq, zorder=3,
        )
        for _, row in grp.iterrows():
            ax.annotate(
                row["dataset_display"],
                (row["rel_impr_oracle_mean"], row["rel_impr_pred_mean"]),
                fontsize=5.5, alpha=0.7,
                xytext=(3, 2), textcoords="offset points",
            )

    all_vals = np.concatenate([
        data["rel_impr_oracle_mean"].values,
        data["rel_impr_pred_mean"].values,
    ])
    lo = float(np.nanmin(all_vals)) * 1.1 - 1
    hi = float(np.nanmax(all_vals)) * 1.1 + 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.55, label="y = x  (predictor = oracle)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Oracle Δ% = (MASE_full − MASE_oracle) / MASE_full × 100", fontsize=10)
    ax.set_ylabel("Predictor Δ% = (MASE_full − MASE_pred) / MASE_full × 100", fontsize=10)
    ax.set_title(
        "Predictor vs Oracle MASE Improvement over Full Window\n"
        "(per dataset, coloured by frequency; above diagonal = predictor beats oracle)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, title="Frequency", title_fontsize=8,
              loc="upper left", framealpha=0.8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out_dir, "rel_impr_pred_vs_oracle.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(Fore.GREEN + f"  Saved: {path}" + Fore.RESET)
    return path


# ==============================================================================
#  PLOTS — MASE
# ==============================================================================

def plot_bar_aggregate_mase(df: pd.DataFrame, out_dir: str,
                            metric_label: str = "MASE",
                            fname: str = "bar_aggregate_mase.png",
                            normalize: bool = False,
                            extra_strategies: bool = True) -> str:
    r = df.dropna(subset=["full_mase", "best_mase", "pred_mase"]).copy()
    if r.empty:
        return ""

    # Leaderboard-style normalisation: divide every strategy's MASE by this cell's
    # seasonal-naive baseline before aggregating, so bars read relative to Seasonal
    # Naive (=1.0), exactly like the HF GiftEval board. Needs the `naive_mase`
    # column; restrict to rows that carry a usable baseline.
    if normalize:
        if "naive_mase" not in r.columns:
            return ""
        r = r[r["naive_mase"].notna() & (r["naive_mase"] > 0)].copy()
        if r.empty:
            return ""

    def _vals(col: str) -> np.ndarray:
        # Read the divisor live from the current `r` — period/variant appends below
        # further filter `r`, so capturing the array up front would misalign it.
        v = r[col].values
        return v / r["naive_mase"].values if normalize else v

    strategies = ["full_mase", "best_mase", "pred_mase"]
    labels = ["Full Window", "Best Window\n(Oracle)", "Predictor\nWindow"]
    # Per-strategy bar colors (geom = saturated, median = lighter). Base three are
    # fixed; period + variants append their registry color for both shades.
    c_geom   = ["#264FA0", "#A9511B", "#3E7327"]
    c_median = ["#7BA9D8", "#F0A86A", "#9ED67A"]
    # Append the period-window strategy when sidecars supplied it (restricted to
    # the rows that actually have a period_mase, so bars stay comparable).
    # ``extra_strategies=False`` (the minimal-figures mode) keeps the bar on the
    # core full/best/pred trio: each extra strategy dropna's `r` to ITS coverage,
    # so partial period/v3/v4 trees would silently shrink n below the run's true
    # cell count (e.g. 95 -> 80) — exactly the kind of headline distortion the
    # minimal figures exist to avoid.
    if extra_strategies and "period_mase" in r.columns and r["period_mase"].notna().any():
        r = r.dropna(subset=["period_mase"]).copy()
        strategies.append("period_mase")
        labels.append("Period\n(2×Period)")
        c_geom.append("#6A1B9A"); c_median.append("#C39BD3")
    # Append each auto-discovered predictor variant (v3 cheap, v4 Mamba, …),
    # restricted to rows that have its MASE so bars stay comparable.
    if extra_strategies:
        for vkey, vlbl, vclr in present_pred_variants(r):
            r = r.dropna(subset=[f"{vkey}_mase"]).copy()
            strategies.append(f"{vkey}_mase")
            labels.append(vlbl.replace(" ", "\n"))
            c_geom.append(vclr); c_median.append(vclr)
    c_geom   = c_geom[:len(labels)]
    c_median = c_median[:len(labels)]

    # Geometric mean — exp(mean(log)) — what M4/OWA use; robust to outlier spikes
    gmeans  = [_geomean(_vals(s)) for s in strategies]
    # Median — nonparametric sanity check
    medians = [float(np.median(_vals(s))) for s in strategies]

    # Identify datasets driving spikes (top 5 % by full_mase)
    thresh_95 = float(np.percentile(r["full_mase"].values, 95))
    outliers = r[r["full_mase"] > thresh_95][["dataset_display", "model_short", "term", "full_mase"]]
    n_outliers = len(outliers)

    x = np.arange(len(labels))
    wb = 0.30

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - wb / 2, gmeans,  wb, label="Geometric Mean ★", color=c_geom,   alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + wb / 2, medians, wb, label="Median",           color=c_median, alpha=0.85, edgecolor="white")

    y_top = max(max(gmeans), max(medians))
    for b in list(b1) + list(b2):
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + y_top * 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Seasonal-Naive reference (=1.0) for the normalised view — below the line
    # beats the leaderboard baseline, above loses to it.
    if normalize:
        ax.axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.7,
                   label="Seasonal Naive (=1.0)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"{metric_label} by Context Strategy  (n={len(r)} dataset-terms)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_scatter(
    df: pd.DataFrame,
    x_col: str, y_col: str,
    xlabel: str, ylabel: str,
    title: str, fname: str,
    out_dir: str,
    diagonal: bool = True,
    clip_pct: Optional[float] = None,
) -> str:
    rows = df.dropna(subset=[x_col, y_col])
    if rows.empty:
        return ""
    x_raw, y_raw = rows[x_col].values, rows[y_col].values

    n_clipped = 0
    if clip_pct is not None:
        combined = np.concatenate([x_raw, y_raw])
        lo = float(np.percentile(combined, 100 - clip_pct))
        hi = float(np.percentile(combined, clip_pct))
        mask = (x_raw <= hi) & (x_raw >= lo) & (y_raw <= hi) & (y_raw >= lo)
        n_clipped = int((~mask).sum())
        x, y = x_raw[mask], y_raw[mask]
    else:
        x, y = x_raw, y_raw

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, alpha=0.7, s=35, color="#4472C4",
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
        label = f"r = {corr:.3f}"
        if n_clipped > 0:
            label += f"\n({n_clipped} pts clipped at {clip_pct:.0f}th pct)"
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=10,
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

    x_raw = r["speedup_pred_vs_full"].values       # >1 means faster
    y_raw = -r["delta_pred_vs_full"].values        # positive = pred has lower MASE (better)

    # Clip to [1st, 99th] percentile on y to prevent MASE spikes from squashing the plot
    y_lo = float(np.percentile(y_raw, 1))
    y_hi = float(np.percentile(y_raw, 99))
    x_hi = float(np.percentile(x_raw, 99))
    mask = (y_raw >= y_lo) & (y_raw <= y_hi) & (x_raw <= x_hi)
    n_clipped = int((~mask).sum())
    x, y = x_raw[mask], y_raw[mask]

    sym = max(abs(y.min()), abs(y.max()))
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(x, y, alpha=0.85, s=55, c=y, cmap="RdYlGn",
                    edgecolors="white", linewidths=0.4, vmin=-sym, vmax=sym)
    plt.colorbar(sc, ax=ax, label="MASE improvement  MASE_full − MASE_pred")
    ax.axhline(0, color="gray", lw=1.2, ls="--", alpha=0.7, label="Same MASE")
    ax.axvline(1, color="gray", lw=1.2, ls="-.", alpha=0.7, label="Same time")

    # Quadrant annotations
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for qx, qy, qtxt, qa in [
        (0.97, 0.97, "Faster & Better", "right"),
        (0.03, 0.97, "Slower & Better", "left"),
        (0.97, 0.03, "Faster & Worse",  "right"),
        (0.03, 0.03, "Slower & Worse",  "left"),
    ]:
        ax.text(qx, qy, qtxt, transform=ax.transAxes, fontsize=8,
                ha=qa, va="top" if qy > 0.5 else "bottom",
                color="gray", style="italic", alpha=0.7)

    if n_clipped > 0:
        ax.text(0.5, 0.01, f"{n_clipped} extreme points clipped (outside 1–99th pct)",
                transform=ax.transAxes, fontsize=8, ha="center", va="bottom",
                color="gray", style="italic")

    ax.set_xlabel("Speedup  =  elapsed_full / elapsed_pred  (>1 means predictor is faster)",
                  fontsize=11)
    ax.set_ylabel("MASE Reduction  =  MASE_full − MASE_pred  (>0 means predictor is better)",
                  fontsize=11)
    ax.set_title("Efficiency Frontier: Accuracy vs Speed Tradeoff",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25); plt.tight_layout()
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

    n_bins = min(60, max(20, len(r) // 5))

    # Panels in display order. The period panel is appended only when the
    # sidecars supplied period FLOPs ratios (else the figure stays at 2 panels).
    panels = [
        (r["complexity_ratio_pred_vs_full"].values, "Predictor / Full", "#70AD47"),
        (r["complexity_ratio_best_vs_full"].values, "Oracle Best / Full", "#ED7D31"),
    ]
    if "complexity_ratio_period_vs_full" in df.columns:
        period_ratios = df["complexity_ratio_period_vs_full"].dropna().values
        if period_ratios.size:
            panels.append((period_ratios, "Period (2×P) / Full", "#6A1B9A"))
    # One panel per auto-discovered predictor variant (v3 cheap, v4 Mamba, …).
    for vkey, vlbl, vclr in present_pred_variants(df):
        col = f"complexity_ratio_{vkey}_vs_full"
        if col in df.columns:
            v_ratios = df[col].dropna().values
            if v_ratios.size:
                panels.append((v_ratios, f"{vlbl} / Full", vclr))

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), sharey=False)
    if len(panels) == 1:
        axes = [axes]

    for ax, (vals, label, color) in zip(axes, panels):
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

    x_raw = 1 - r["complexity_ratio_pred_vs_full"].values   # >0 means cheaper
    y_raw = -r["delta_pred_vs_full"].values                  # >0 means better accuracy

    # Clip y to 1–99th percentile: high-MASE outliers stretch the axis to ±500
    # and collapse all meaningful variation into a thin band near zero
    y_lo = float(np.percentile(y_raw, 1))
    y_hi = float(np.percentile(y_raw, 99))
    mask = (y_raw >= y_lo) & (y_raw <= y_hi)
    n_clipped = int((~mask).sum())
    x, y = x_raw[mask], y_raw[mask]

    # Symmetric y-axis so the zero line sits at center
    y_sym = max(abs(y.min()), abs(y.max())) * 1.05

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(x, y, alpha=0.80, s=45, c=y, cmap="RdYlGn",
                    edgecolors="white", linewidths=0.4,
                    vmin=-y_sym, vmax=y_sym)
    plt.colorbar(sc, ax=ax, label="MASE Improvement  (MASE_full − MASE_pred)")
    ax.axhline(0, color="gray", lw=1.2, ls="--", alpha=0.7, label="No accuracy change")
    ax.axvline(0, color="gray", lw=1.2, ls="-.", alpha=0.7, label="No complexity change")
    ax.set_ylim(-y_sym, y_sym)

    # Quadrant labels
    for qx, qy, qtxt, ha in [
        (0.97, 0.97, "Cheaper & Better",  "right"),
        (0.03, 0.97, "Costlier & Better", "left"),
        (0.97, 0.03, "Cheaper & Worse",   "right"),
        (0.03, 0.03, "Costlier & Worse",  "left"),
    ]:
        ax.text(qx, qy, qtxt, transform=ax.transAxes, fontsize=8,
                ha=ha, va="top" if qy > 0.5 else "bottom",
                color="gray", style="italic", alpha=0.7)

    if n_clipped > 0:
        ax.text(0.5, 0.01, f"{n_clipped} extreme points clipped (outside 1–99th pct on y)",
                transform=ax.transAxes, fontsize=8, ha="center", va="bottom",
                color="gray", style="italic")

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
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out_dir, "complexity_vs_mase_gain.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


# ==============================================================================
#  PLOT — RUN-LEVEL MODEL x STRATEGY OVERVIEW
# ==============================================================================

# Strategy display names + palette, shared with the other complexity plots.
_STRATEGY_STYLE = {
    "pred":       ("Predictor",         "#70AD47"),
    # predictor variants (auto-discovered sibling trees); see PRED_VARIANTS
    "pred_cheap": ("Predictor (cheap)", "#1F77B4"),
    "pred_mamba": ("Predictor Mamba",   "#9C27B0"),
    "best":       ("Oracle best",       "#ED7D31"),
    "period":     ("Period (2×P)",      "#6A1B9A"),
}
_MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]


def plot_mase_change_table(rollup: pd.DataFrame, out_dir: str,
                           strategies: Optional[List[str]] = None,
                           suffix: str = "", metric_label: str = "MASE") -> str:
    """Render a table figure of the MASE change per (model, strategy): the
    original full-window MASE and the MASE after each technique, side by side.

    One row per (model, strategy). Columns: Model, Strategy, ``{metric} full``
    (original), ``{metric} strategy`` (after), ``Δ`` (strategy − full, <0 = better)
    and ``Δ%`` (= ``rel_mase_drop_pct``, >0 = better). The Δ% cell is tinted green
    when the technique lowered the MASE and red when it raised it. The grand TOTAL
    rows (geomean over all models) are kept and shown in bold at the bottom.

    ``strategies`` optionally restricts which strategies appear (a subset of
    ``_STRATEGY_STYLE`` keys); ``suffix`` / ``metric_label`` mirror
    ``write_run_rollup`` so a gluonts / gluonts-real pass writes its own file
    (``mase_change_table_gluonts_real.png``) without clobbering the default one.
    """
    r = rollup.dropna(subset=["geomean_full_mase", "geomean_strategy_mase"]).copy()
    if strategies:
        r = r[r["strategy"].isin(strategies)].copy()
    if r.empty:
        return ""

    # Stable ordering: models alphabetically with TOTAL last, strategies in the
    # canonical palette order.
    strat_rank = {s: i for i, s in enumerate(_STRATEGY_STYLE)}
    r["__s"] = r["strategy"].map(lambda s: strat_rank.get(s, len(strat_rank)))
    r["__m"] = r["model_short"].map(lambda m: (m == "TOTAL", m))
    r = r.sort_values(["__m", "__s"]).reset_index(drop=True)

    def _slabel(s: str) -> str:
        return _STRATEGY_STYLE.get(s, (s, ""))[0]

    def _tint(rel: float, tot: bool) -> str:
        base = "#F2F2F2" if tot else "#FFFFFF"
        if not pd.notna(rel):
            return base
        return "#E2EFDA" if rel > 0 else ("#FCE4E4" if rel < 0 else base)

    # The leaderboard-normalised (÷ seasonal-naive) twin columns are shown only
    # when a baseline was available (non-NaN norm), so runs without
    # a published Seasonal Naive row keep the plain absolute-MASE table.
    has_norm = ("geomean_full_mase_norm" in r.columns
                and r["geomean_full_mase_norm"].notna().any())

    headers = ["Model", "Strategy", f"{metric_label} full\n(original)",
               f"{metric_label} strategy\n(after)", "Δ", "Δ% vs full\n(>0 better)"]
    if has_norm:
        headers += [f"{metric_label} full\nNORM (÷naive)",
                    f"{metric_label} strategy\nNORM (÷naive)", "Δ% NORM\n(>0 better)"]

    cell_text, cell_colors, is_total = [], [], []
    for _, row in r.iterrows():
        tot = row["model_short"] == "TOTAL"
        is_total.append(tot)
        base = "#F2F2F2" if tot else "#FFFFFF"
        rel = row.get("rel_mase_drop_pct", float("nan"))
        delta = row["geomean_strategy_mase"] - row["geomean_full_mase"]
        txt = [
            row["model_short"],
            _slabel(row["strategy"]),
            f"{row['geomean_full_mase']:.4f}",
            f"{row['geomean_strategy_mase']:.4f}",
            f"{delta:+.4f}",
            f"{rel:+.2f}%" if pd.notna(rel) else "—",
        ]
        tint = _tint(rel, tot)
        cols = [base, base, base, base, tint, tint]
        if has_norm:
            fn = row.get("geomean_full_mase_norm", float("nan"))
            sn = row.get("geomean_strategy_mase_norm", float("nan"))
            reln = row.get("rel_mase_drop_pct_norm", float("nan"))
            txt += [
                f"{fn:.4f}" if pd.notna(fn) else "—",
                f"{sn:.4f}" if pd.notna(sn) else "—",
                f"{reln:+.2f}%" if pd.notna(reln) else "—",
            ]
            tn = _tint(reln, tot)
            cols += [base, base, tn]
        cell_text.append(txt)
        cell_colors.append(cols)

    n_rows = len(cell_text)
    fig_h = 1.1 + 0.36 * (n_rows + 1)
    fig_w = 15.5 if has_norm else 11
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)

    # Header styling + bold TOTAL rows.
    for (rr, _cc), cell in tbl.get_celld().items():
        if rr == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor("#4472C4")
        elif is_total[rr - 1]:
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#BFBFBF")

    ax.set_title(f"MASE change per model × strategy  [{metric_label}]  — "
                 "full window (original) vs after the technique",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f"mase_change_table{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_model_strategy_overview(rollup: pd.DataFrame, out_dir: str,
                                 strategies: Optional[List[str]] = None,
                                 suffix: str = "", metric_label: str = "MASE") -> str:
    """Single run-level figure: every (model, strategy) as one point, trading off
    FLOPs saved (x) against the *relative* geomean MASE change (y).

    x = pct_flops_saved (>0 = cheaper than full); y = rel_mase_drop_pct =
    100·(MASE_full − MASE_strategy)/MASE_full (>0 = lower MASE than full, i.e.
    better).  Colour encodes strategy, marker shape encodes model, and a faint
    line links each model's strategies so a family's spread is readable at a
    glance.  The top-right quadrant — cheaper *and* better — is the win region.

    ``strategies`` optionally restricts which strategies are plotted (a subset of
    ``_STRATEGY_STYLE`` keys, e.g. ``["pred", "best"]``); ``None`` plots all.
    """
    from matplotlib.lines import Line2D

    r = rollup.dropna(subset=["pct_flops_saved", "rel_mase_drop_pct"]).copy()
    if strategies:
        r = r[r["strategy"].isin(strategies)].copy()
    if r.empty:
        return ""

    models = sorted(r["model_short"].unique())
    marker_of = {m: _MODEL_MARKERS[i % len(_MODEL_MARKERS)] for i, m in enumerate(models)}
    # Strategy plot order (and any unexpected strategy falls back to grey).
    strat_order = [s for s in _STRATEGY_STYLE if s in set(r["strategy"])]

    fig, ax = plt.subplots(figsize=(11, 8))

    x_all = r["pct_flops_saved"].values * 100.0
    y_all = r["rel_mase_drop_pct"].values
    x_pad = max(2.0, 0.05 * (np.ptp(x_all) or 1.0))
    y_pad = max(1e-3, 0.10 * (np.ptp(y_all) or 1.0))
    x_lo, x_hi = x_all.min() - x_pad, x_all.max() + x_pad
    y_lo, y_hi = y_all.min() - y_pad, y_all.max() + y_pad
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)

    # Win quadrant (cheaper & better) + zero crosshair. Clamp the shaded spans to
    # the visible positive region so they collapse (not invert) if every point is
    # costlier or worse than full.
    ax.axhspan(0, max(0.0, y_hi), color="#70AD47", alpha=0.04)
    ax.axvspan(0, max(0.0, x_hi), color="#70AD47", alpha=0.04)
    ax.axhline(0, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax.axvline(0, color="gray", lw=1.2, ls="-.", alpha=0.7)

    # Faint connector tracing each model across its strategies (sorted by x).
    for m in models:
        sub = r[r["model_short"] == m].copy()
        sub["__o"] = sub["strategy"].map(lambda s: strat_order.index(s)
                                         if s in strat_order else len(strat_order))
        sub = sub.sort_values("__o")
        if len(sub) > 1:
            ax.plot(sub["pct_flops_saved"] * 100.0, sub["rel_mase_drop_pct"],
                    color="gray", lw=0.8, alpha=0.30, zorder=1)

    # Markers: colour = strategy, shape = model.
    for strat in strat_order:
        _, color = _STRATEGY_STYLE[strat]
        sub = r[r["strategy"] == strat]
        for _, row in sub.iterrows():
            ax.scatter(row["pct_flops_saved"] * 100.0, row["rel_mase_drop_pct"],
                       marker=marker_of[row["model_short"]], s=170, c=color,
                       edgecolors="white", linewidths=0.8, alpha=0.92, zorder=3)

    # Quadrant captions.
    for qx, qy, txt, ha in [
        (0.985, 0.985, "Cheaper & Better",  "right"),
        (0.015, 0.985, "Costlier & Better", "left"),
        (0.985, 0.015, "Cheaper & Worse",   "right"),
        (0.015, 0.015, "Costlier & Worse",  "left"),
    ]:
        ax.text(qx, qy, txt, transform=ax.transAxes, fontsize=8.5,
                ha=ha, va="top" if qy > 0.5 else "bottom",
                color="gray", style="italic", alpha=0.75)

    ax.set_xlabel("FLOPs saved vs full window  =  100·(1 − FLOPs_strategy / FLOPs_full)   (%)",
                  fontsize=11)
    ax.set_ylabel(f"Relative {metric_label} change vs full  =  "
                  f"100·({metric_label}_full − {metric_label}_strategy)/{metric_label}_full   (%, >0 = better)",
                  fontsize=11)
    ax.set_title(f"Model × Strategy Overview — Compute Saved vs Accuracy Change  [{metric_label}]",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)

    # Two legends: strategy (colour) kept inside, model (shape) outside right.
    strat_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=_STRATEGY_STYLE[s][1], markeredgecolor="white",
               label=_STRATEGY_STYLE[s][0])
        for s in strat_order
    ]
    model_handles = [
        Line2D([0], [0], marker=marker_of[m], linestyle="none", markersize=9,
               markerfacecolor="0.35", markeredgecolor="white", label=m)
        for m in models
    ]
    leg1 = ax.legend(handles=strat_handles, title="Strategy", fontsize=9,
                     title_fontsize=10, loc="upper left", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Model", fontsize=8.5, title_fontsize=10,
              loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(out_dir, f"model_strategy_overview{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_model_strategy_overview_time(rollup: pd.DataFrame, out_dir: str,
                                      strategies: Optional[List[str]] = None) -> str:
    """Stage-6 twin of ``plot_model_strategy_overview``: every (model, strategy)
    as one point, trading measured wall-clock forward-pass time saved (x) against
    the relative geomean MASE change (y).

    x = pct_time_saved (>0 = faster than full); y = rel_mase_drop_pct =
    100·(MASE_full − MASE_strategy)/MASE_full (>0 = better).  Colour = strategy,
    marker = model.  The top-right quadrant — faster *and* better — is the win
    region.  Identical layout to the FLOPs figure so the two can sit side by side;
    only the cost axis differs (measured seconds vs theoretical MACs).

    ``strategies`` optionally restricts which strategies are plotted (a subset of
    ``_STRATEGY_STYLE`` keys, e.g. ``["pred", "best"]``); ``None`` plots all.
    """
    from matplotlib.lines import Line2D

    r = rollup.dropna(subset=["pct_time_saved", "rel_mase_drop_pct"]).copy()
    if strategies:
        r = r[r["strategy"].isin(strategies)].copy()
    if r.empty:
        return ""

    models = sorted(r["model_short"].unique())
    marker_of = {m: _MODEL_MARKERS[i % len(_MODEL_MARKERS)] for i, m in enumerate(models)}
    strat_order = [s for s in _STRATEGY_STYLE if s in set(r["strategy"])]

    fig, ax = plt.subplots(figsize=(11, 8))

    x_all = r["pct_time_saved"].values * 100.0
    y_all = r["rel_mase_drop_pct"].values
    x_pad = max(2.0, 0.05 * (np.ptp(x_all) or 1.0))
    y_pad = max(1e-3, 0.10 * (np.ptp(y_all) or 1.0))
    x_lo, x_hi = x_all.min() - x_pad, x_all.max() + x_pad
    y_lo, y_hi = y_all.min() - y_pad, y_all.max() + y_pad
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)

    # Win quadrant (faster & better) + zero crosshair.
    ax.axhspan(0, max(0.0, y_hi), color="#70AD47", alpha=0.04)
    ax.axvspan(0, max(0.0, x_hi), color="#70AD47", alpha=0.04)
    ax.axhline(0, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax.axvline(0, color="gray", lw=1.2, ls="-.", alpha=0.7)

    # Faint connector tracing each model across its strategies (sorted by x).
    for m in models:
        sub = r[r["model_short"] == m].copy()
        sub["__o"] = sub["strategy"].map(lambda s: strat_order.index(s)
                                         if s in strat_order else len(strat_order))
        sub = sub.sort_values("__o")
        if len(sub) > 1:
            ax.plot(sub["pct_time_saved"] * 100.0, sub["rel_mase_drop_pct"],
                    color="gray", lw=0.8, alpha=0.30, zorder=1)

    # Markers: colour = strategy, shape = model. When robust-timing std is
    # available (pct_time_saved_std), draw it as a horizontal error bar on the
    # time-saved axis — the MASE (y) axis has no timing uncertainty.
    has_xerr = "pct_time_saved_std" in r.columns
    for strat in strat_order:
        _, color = _STRATEGY_STYLE[strat]
        sub = r[r["strategy"] == strat]
        for _, row in sub.iterrows():
            x = row["pct_time_saved"] * 100.0
            y = row["rel_mase_drop_pct"]
            if has_xerr:
                xerr = row["pct_time_saved_std"]
                if pd.notna(xerr) and xerr > 0:
                    ax.errorbar(x, y, xerr=xerr * 100.0, fmt="none", ecolor=color,
                                elinewidth=1.0, capsize=3, alpha=0.55, zorder=2)
            ax.scatter(x, y, marker=marker_of[row["model_short"]], s=170, c=color,
                       edgecolors="white", linewidths=0.8, alpha=0.92, zorder=3)

    # Quadrant captions.
    for qx, qy, txt, ha in [
        (0.985, 0.985, "Faster & Better",  "right"),
        (0.015, 0.985, "Slower & Better",  "left"),
        (0.985, 0.015, "Faster & Worse",   "right"),
        (0.015, 0.015, "Slower & Worse",   "left"),
    ]:
        ax.text(qx, qy, txt, transform=ax.transAxes, fontsize=8.5,
                ha=ha, va="top" if qy > 0.5 else "bottom",
                color="gray", style="italic", alpha=0.75)

    ax.set_xlabel("Forward-pass time saved vs full window  =  100·(1 − time_strategy / time_full)   (%)",
                  fontsize=11)
    ax.set_ylabel("Relative MASE change vs full  =  100·(MASE_full − MASE_strategy)/MASE_full   (%, >0 = better)",
                  fontsize=11)
    ax.set_title("Model × Strategy Overview — Forward-Pass Time Saved vs Accuracy Change",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)

    strat_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=_STRATEGY_STYLE[s][1], markeredgecolor="white",
               label=_STRATEGY_STYLE[s][0])
        for s in strat_order
    ]
    model_handles = [
        Line2D([0], [0], marker=marker_of[m], linestyle="none", markersize=9,
               markerfacecolor="0.35", markeredgecolor="white", label=m)
        for m in models
    ]
    leg1 = ax.legend(handles=strat_handles, title="Strategy", fontsize=9,
                     title_fontsize=10, loc="upper left", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Model", fontsize=8.5, title_fontsize=10,
              loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(out_dir, "model_strategy_overview_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


# ==============================================================================
#  PLOTS — MISC
# ==============================================================================

def plot_per_dataset_bars(df: pd.DataFrame, out_dir: str) -> List[str]:
    r = df.dropna(subset=["full_mase", "best_mase", "pred_mase"]).copy()
    if r.empty:
        return []

    sub_dir = os.path.join(out_dir, "per_dataset_bars")
    os.makedirs(sub_dir, exist_ok=True)

    # Strategies to draw, in plot order. Period is appended only when the
    # sidecars supplied at least one period_mase, so the figure degrades to the
    # original three strategies when stage 5 was never run.
    strat_cols = ["full_mase", "best_mase", "pred_mase"]
    strat_lbls = ["Full window", "Best (oracle)", "Predictor"]
    strat_cols_clrs = ["#4472C4", "#ED7D31", "#70AD47"]
    has_period = "period_mase" in r.columns and r["period_mase"].notna().any()
    if has_period:
        strat_cols.append("period_mase")
        strat_lbls.append("Period (2×P)")
        strat_cols_clrs.append("#6A1B9A")
    # Auto-discovered predictor variants (v3 cheap, v4 Mamba, …).
    for vkey, vlbl, vclr in present_pred_variants(r):
        strat_cols.append(f"{vkey}_mase")
        strat_lbls.append(vlbl)
        strat_cols_clrs.append(vclr)

    paths: List[str] = []
    for dataset_name, grp in r.groupby("dataset_display", sort=True):
        grp = grp.sort_values(["model_short", "term"]).reset_index(drop=True)

        # Clip at this dataset's own 99th percentile (over the strategies shown,
        # ignoring NaNs so missing period cells don't poison the percentile).
        ds_vals = grp[strat_cols].values.ravel()
        clip_hi = float(np.nanpercentile(ds_vals, 99))

        # x-axis label: model + term
        grp["bar_label"] = grp["model_short"] + "\nt=" + grp["term"]
        labels = grp["bar_label"].tolist()
        n = len(labels)
        x = np.arange(n)
        k = len(strat_cols)
        w = 0.8 / k                       # total cluster width ~0.8
        # Symmetric offsets centred on each x tick: e.g. k=4 -> [-1.5w..1.5w].
        offsets = [(i - (k - 1) / 2.0) * w for i in range(k)]

        fig, ax = plt.subplots(figsize=(max(6, n * 1.2 + 2), 5))
        for col, lbl, clr, off in zip(strat_cols, strat_lbls, strat_cols_clrs, offsets):
            ax.bar(x + off, grp[col].clip(upper=clip_hi).values, w,
                   label=lbl, color=clr, alpha=0.85, edgecolor="white")

        # Annotate bars: ▲ + true value when clipped, else the value.
        for xi in range(n):
            for col, off in zip(strat_cols, offsets):
                raw = grp[col].values[xi]
                if np.isnan(raw):
                    continue
                if raw > clip_hi:
                    ax.text(xi + off, clip_hi * 1.01, f"▲{raw:.2f}",
                            ha="center", va="bottom", fontsize=6, color="darkred", rotation=90)
                else:
                    ax.text(xi + off, raw + clip_hi * 0.01, f"{raw:.2f}",
                            ha="center", va="bottom", fontsize=6, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("MASE", fontsize=11)
        ax.set_title(f"{dataset_name}  —  MASE by context strategy",
                     fontsize=12, fontweight="bold")
        ax.set_ylim(bottom=0, top=clip_hi * 1.15)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.99, 0.99,
                f"y clipped at global 99th-pct ({clip_hi:.2f});  ▲ = true value above clip",
                transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
                color="gray", style="italic")
        plt.tight_layout()

        # Sanitise filename: replace / and spaces
        safe_name = str(dataset_name).replace("/", "_").replace(" ", "_")
        path = os.path.join(sub_dir, f"{safe_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(path)

    return paths


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
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="Restrict comparison to these model_short names "
                        "(default: every model present in the run dir).")
    p.add_argument("--mase-metric",
                   choices=["mase_gluonts", "mase_gluonts_real"], default="mase_gluonts_real",
                   help="Which MASE drives the strategy comparison + all flops/time "
                        "savings outputs. Exactly two exist: `mase_gluonts` (numpy "
                        "port of the leaderboard definition) and `mase_gluonts_real` "
                        "(the actual gluonts evaluate_forecasts machinery — the "
                        "default and the leaderboard-faithful one). The other metric "
                        "is always emitted alongside as suffixed twin files. Use a "
                        "distinct --output-dir per metric to avoid overwriting.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output dir (default: models/<model>/strategy_comparison/). "
                        "Only safe with a single --models entry, else models clobber each other.")
    p.add_argument(
        "--patch-sizes", type=str, default=None,
        help=(
            'JSON dict overriding patch sizes per model family.  '
            'Example: \'{"moirai": 16, "chronos2": 2}\''
        ),
    )
    p.add_argument(
        "--plot-strategies", type=str, nargs="+", default=None,
        choices=list(_STRATEGY_STYLE.keys()),
        help="Restrict the cross-model overview figures (model_strategy_overview"
             "[_time].png) to these strategies (default: all present). The CSVs "
             "and console totals are unaffected. E.g. --plot-strategies pred best.",
    )
    p.add_argument(
        "--use-robust-timing", action=argparse.BooleanOptionalAction, default=True,
        help="Prefer the robust forward-pass timing (timing.json mean/std from "
             "benchmark_window_timing_gifteval) over single-shot elapsed_seconds, "
             "falling back per-cell when absent. Adds *_elapsed_std_s columns and "
             "error bars to the time overview. Use --no-use-robust-timing to force "
             "single-shot timing.",
    )
    p.add_argument(
        "--rollup-only", action="store_true",
        help="Skip the per-model outputs and emit only the cross-model overview "
             "figure + flops_savings_all_models.csv (covering every model in the "
             "run dir). Used for the final aggregation pass after per-model runs.",
    )
    p.add_argument(
        "--all-figures", action=argparse.BooleanOptionalAction, default=False,
        help="Emit the full historical figure set (metric twins, scatters, "
             "histograms, per-dataset bars, rollup overviews/tables). Default "
             "emits ONLY bar_aggregate_mase_gluonts[_normalized].png on the "
             "primary metric; every CSV / summary_stats.json is unaffected.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global USE_ROBUST_TIMING
    USE_ROBUST_TIMING = args.use_robust_timing
    print(Fore.CYAN
          + f"Timing source: {'robust (timing.json mean/std, single-shot fallback)' if USE_ROBUST_TIMING else 'single-shot elapsed_seconds'}"
          + Fore.RESET)

    patch_sizes = dict(DEFAULT_PATCH_SIZES)
    if args.patch_sizes:
        patch_sizes.update(json.loads(args.patch_sizes))
        print(Fore.CYAN + f"Patch sizes (after override): {patch_sizes}" + Fore.RESET)
    else:
        print(Fore.CYAN + f"Patch sizes: {patch_sizes}" + Fore.RESET)

    run_dir = args.run_dir or find_latest_run(args.cache_root)
    print(Fore.CYAN + f"Run directory: {run_dir}" + Fore.RESET)

    print(Fore.CYAN + f"MASE metric: {args.mase_metric}" + Fore.RESET)

    # ---- Load all records ---------------------------------------------------
    df = load_strategy_records(run_dir, args.cache_root, patch_sizes,
                               mase_metric=args.mase_metric)

    if args.models:
        wanted = set(args.models)
        available = sorted(df["model_short"].unique())
        df = df[df["model_short"].isin(wanted)].reset_index(drop=True)
        if df.empty:
            raise SystemExit(
                f"No records for models {sorted(wanted)} in {run_dir}. "
                f"Available: {available}"
            )

    # ---- Twin pass on the OTHER gluonts metric --------------------------------
    # Exactly two metrics exist (mase_gluonts port / mase_gluonts_real machinery);
    # whichever isn't the primary is ALSO scored on every strategy (its own
    # oracle-best window) so its suffixed twin plots/CSVs get emitted beside the
    # primary ones. No extra inference — a second pass over the cached npz.
    df_g: Optional[pd.DataFrame] = None
    if args.mase_metric != "mase_gluonts" and run_has_gluonts_curve(run_dir):
        df_g = load_strategy_records(run_dir, args.cache_root, patch_sizes,
                                     mase_metric="mase_gluonts")
        if args.models:
            df_g = df_g[df_g["model_short"].isin(set(args.models))].reset_index(drop=True)
        print(Fore.CYAN + "Gluonts MASE available: bar plot uses mase_gluonts; "
              "adding model_strategy_overview_gluonts." + Fore.RESET)

    # Same idea for the ACTUAL gluonts-machinery MASE (`mase_gluonts_real`): when
    # it isn't already the primary metric, score every strategy on it too so the
    # *_gluonts_real twin plots/CSVs get emitted beside the others.
    df_gr: Optional[pd.DataFrame] = None
    if args.mase_metric != "mase_gluonts_real" and run_has_gluonts_real_curve(run_dir):
        df_gr = load_strategy_records(run_dir, args.cache_root, patch_sizes,
                                      mase_metric="mase_gluonts_real")
        if args.models:
            df_gr = df_gr[df_gr["model_short"].isin(set(args.models))].reset_index(drop=True)
        print(Fore.CYAN + "Gluonts-real MASE available: adding "
              "bar_aggregate_mase_gluonts_real + model_strategy_overview_gluonts_real."
              + Fore.RESET)

    def _run_outputs(df_subset: pd.DataFrame, out_dir: str,
                     df_g_subset: Optional[pd.DataFrame] = None,
                     df_gr_subset: Optional[pd.DataFrame] = None) -> None:
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, "comparison.csv")
        df_subset.to_csv(csv_path, index=False)
        print(Fore.GREEN + f"  Saved: {csv_path}" + Fore.RESET)

        stats = compute_summary_stats(df_subset)
        stats_path = os.path.join(out_dir, "summary_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(Fore.GREEN + f"  Saved: {stats_path}" + Fore.RESET)

        print(Fore.CYAN + "\n--- MASE summary ---" + Fore.RESET)
        mase_keys = [("full_mase", "Full window  "),
                     ("best_mase", "Best (oracle)"),
                     ("pred_mase", "Predictor    ")]
        if "period_mase" in stats:
            mase_keys.append(("period_mase", "Period (2xP) "))
        if "period3_mase" in stats:
            mase_keys.append(("period3_mase", "Period (3xP) "))
        # Auto-discovered predictor variants (v3 cheap, v4 Mamba, …).
        for vkey, vlbl, _vclr in present_pred_variants(df_subset):
            if f"{vkey}_mase" in stats:
                mase_keys.append((f"{vkey}_mase", f"{vlbl:<13}"[:13]))
        for key, name in mase_keys:
            s = stats[key]
            print(f"  {name}  mean={s['mean']:.4f}  geomean={s['geomean']:.4f}  "
                  f"median={s['median']:.4f}  std={s['std']:.4f}")
        if "period_mase" in stats:
            print(f"  Period beats full: {stats['period_beats_full_count']}/{stats['total_rows']} "
                  f"({100*stats['period_beats_full_rate']:.1f}%)  |  "
                  f"regret vs oracle: mean={stats['regret_period_vs_best']['mean']:.4f}  "
                  f"mean window={stats.get('mean_period_window', float('nan')):.0f}")
        if "period3_mase" in stats:
            print(f"  Period 3x beats full: "
                  f"{stats['period3_beats_full_count']}/{stats['total_rows']} "
                  f"({100*stats['period3_beats_full_rate']:.1f}%)  |  "
                  f"regret vs oracle: mean="
                  f"{stats['regret_period3_vs_best']['mean']:.4f}  "
                  f"mean window={stats.get('mean_period3_window', float('nan')):.0f}")
        print(f"  Pred beats full: {stats['pred_beats_full_count']}/{stats['total_rows']} "
              f"({100*stats['pred_beats_full_rate']:.1f}%)")
        for vkey, vlbl, _vclr in present_pred_variants(df_subset):
            if f"{vkey}_beats_full_count" in stats:
                print(f"  {vlbl} beats full: {stats[f'{vkey}_beats_full_count']}/{stats['total_rows']} "
                      f"({100*stats[f'{vkey}_beats_full_rate']:.1f}%)  |  "
                      f"regret vs oracle: mean={stats[f'regret_{vkey}_vs_best']['mean']:.4f}  "
                      f"mean window={stats.get(f'mean_{vkey}_window', float('nan')):.0f}")
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

        # Headline aggregations on the FULL-window strategy for the primary
        # metric, side by side: the absolute geomean and the seasonal-naive-
        # NORMALISED geomean — the latter is the leaderboard aggregation (each
        # cell ÷ its Seasonal-Naive MASE of the SAME definition, then geomean),
        # i.e. the number that lines up with the HF GiftEval board when the
        # primary metric is `mase_gluonts_real`.
        full_s = stats.get("full_mase", {})
        print(Fore.CYAN + f"\n--- Aggregate {args.mase_metric} (full window) ---"
              + Fore.RESET)
        print(f"  absolute geomean   : {full_s.get('geomean', float('nan')):.4f}")
        if "geomean_norm" in full_s:
            print(f"  NORMALISED geomean : {full_s['geomean_norm']:.4f}"
                  f"  <- leaderboard aggregation (÷ seasonal-naive, "
                  f"n={full_s.get('n_norm', 0)})")
        else:
            print(Fore.YELLOW + "  NORMALISED geomean: no published Seasonal "
                  "Naive row found for the selected configs"
                  + Fore.RESET)

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

        flops_sav = compute_flops_savings(df_subset)
        if not flops_sav.empty:
            fs_path = os.path.join(out_dir, "flops_savings.csv")
            flops_sav.to_csv(fs_path, index=False, float_format="%.6g")
            print(Fore.GREEN + f"  Saved: {fs_path}" + Fore.RESET)
            print(Fore.CYAN + "\n--- Total FLOPs saved vs full window ---" + Fore.RESET)
            for _, fr in flops_sav.iterrows():
                print(f"  {fr['strategy']:<7}  saved {100*fr['pct_flops_saved']:5.1f}%  "
                      f"({fr['flops_saved']:.3g} of {fr['total_full_flops']:.3g} FLOPs)  |  "
                      f"geomean MASE {fr['geomean_full_mase']:.4f} -> {fr['geomean_strategy_mase']:.4f}  "
                      f"({fr['rel_mase_drop_pct']:+.2f}%)")

        # Stage 6: measured wall-clock forward-pass time saved (twin of FLOPs).
        time_sav = compute_time_savings(df_subset)
        if not time_sav.empty:
            ts_path = os.path.join(out_dir, "time_savings.csv")
            time_sav.to_csv(ts_path, index=False, float_format="%.6g")
            print(Fore.GREEN + f"  Saved: {ts_path}" + Fore.RESET)
            print(Fore.CYAN + "\n--- Total forward-pass time saved vs full window ---" + Fore.RESET)
            for _, fr in time_sav.iterrows():
                print(f"  {fr['strategy']:<7}  saved {100*fr['pct_time_saved']:5.1f}%  "
                      f"({fr['time_saved_s']:.3g}s of {fr['total_full_time_s']:.3g}s)  |  "
                      f"geomean MASE {fr['geomean_full_mase']:.4f} -> {fr['geomean_strategy_mase']:.4f}  "
                      f"({fr['rel_mase_drop_pct']:+.2f}%)")

        print(Fore.CYAN + "\n--- Relative improvement tables ---" + Fore.RESET)
        compute_relative_improvement_tables(df_subset, out_dir,
                                            figures=args.all_figures)

        print(Fore.CYAN + "\n--- Generating plots ---" + Fore.RESET)
        metric_lbls = {"mase_gluonts": "MASE (gluonts)",
                       "mase_gluonts_real": "MASE (gluonts-real)"}
        primary_lbl = metric_lbls[args.mase_metric]

        # ---- Minimal figure set (the default) --------------------------------
        # Exactly TWO figures, both on the PRIMARY metric (default
        # `mase_gluonts_real` = gluonts' own machinery, the leaderboard
        # definition): the absolute bars and the leaderboard-faithful
        # seasonal-naive-NORMALISED bars (each cell ÷ its same-definition
        # Seasonal-Naive; Seasonal Naive = 1.0). Fixed filenames regardless of
        # --mase-metric — the figure title carries the exact metric. Everything
        # else (twins, scatters, histograms, per-dataset bars) is behind
        # --all-figures; CSVs and summary_stats.json are always written.
        if not args.all_figures:
            # extra_strategies=False: core full/best/pred only, so partial
            # period/v3/v4 coverage can't dropna the bar below the run's true n.
            minimal_paths = [plot_bar_aggregate_mase(
                df_subset, out_dir, metric_label=primary_lbl,
                fname="bar_aggregate_mase_gluonts.png",
                extra_strategies=False)]
            if "naive_mase" in df_subset.columns and df_subset["naive_mase"].notna().any():
                minimal_paths.append(plot_bar_aggregate_mase(
                    df_subset, out_dir,
                    metric_label=f"{primary_lbl}, norm. vs seasonal-naive",
                    fname="bar_aggregate_mase_gluonts_normalized.png",
                    normalize=True, extra_strategies=False))
            for path in minimal_paths:
                if path:
                    print(Fore.GREEN + f"  {path}" + Fore.RESET)
            print(Fore.GREEN + f"\nDone.  Outputs: {out_dir}" + Fore.RESET)
            return

        # ---- --all-figures: the full historical set ---------------------------
        # Primary metric: absolute bars + the leaderboard-faithful NORMALISED bars
        # (each cell ÷ its same-definition Seasonal-Naive; Seasonal Naive = 1.0).
        # With the default `mase_gluonts_real`, the normalised figure is the one
        # that lines up with the HF GiftEval board.
        bar_mase_path = plot_bar_aggregate_mase(df_subset, out_dir,
                                                metric_label=primary_lbl)
        bar_mase_norm_path = ""
        if "naive_mase" in df_subset.columns and df_subset["naive_mase"].notna().any():
            bar_mase_norm_path = plot_bar_aggregate_mase(
                df_subset, out_dir,
                metric_label=f"{primary_lbl}, norm. vs seasonal-naive",
                fname="bar_aggregate_mase_normalized.png",
                normalize=True)
        # Twin files on the OTHER gluonts metric (absolute + normalised), suffixed
        # so the two metrics never clobber each other.
        bar_mase_gluonts_path = bar_mase_gluonts_norm_path = ""
        if df_g_subset is not None and not df_g_subset.empty:
            bar_mase_gluonts_path = plot_bar_aggregate_mase(
                df_g_subset, out_dir, metric_label="MASE (gluonts)",
                fname="bar_aggregate_mase_gluonts.png")
            if "naive_mase" in df_g_subset.columns and df_g_subset["naive_mase"].notna().any():
                bar_mase_gluonts_norm_path = plot_bar_aggregate_mase(
                    df_g_subset, out_dir,
                    metric_label="MASE (gluonts), norm. vs seasonal-naive",
                    fname="bar_aggregate_mase_gluonts_normalized.png",
                    normalize=True)
        bar_mase_gluonts_real_path = bar_mase_gluonts_real_norm_path = ""
        if df_gr_subset is not None and not df_gr_subset.empty:
            bar_mase_gluonts_real_path = plot_bar_aggregate_mase(
                df_gr_subset, out_dir, metric_label="MASE (gluonts-real)",
                fname="bar_aggregate_mase_gluonts_real.png")
            if "naive_mase" in df_gr_subset.columns and df_gr_subset["naive_mase"].notna().any():
                bar_mase_gluonts_real_norm_path = plot_bar_aggregate_mase(
                    df_gr_subset, out_dir,
                    metric_label="MASE (gluonts-real), norm. vs seasonal-naive",
                    fname="bar_aggregate_mase_gluonts_real_normalized.png",
                    normalize=True)
        single_paths = [
            bar_mase_path,
            bar_mase_norm_path,
            bar_mase_gluonts_path,
            bar_mase_gluonts_norm_path,
            bar_mase_gluonts_real_path,
            bar_mase_gluonts_real_norm_path,
            plot_bar_aggregate_time(df_subset, out_dir),
            plot_scatter(df_subset, "best_mase", "pred_mase",
                         "MASE — Best (oracle)", "MASE — Predictor",
                         "Predictor MASE vs Oracle Best MASE",
                         "scatter_pred_vs_best.png", out_dir, clip_pct=99),
            plot_scatter(df_subset, "full_mase", "pred_mase",
                         "MASE — Full Window", "MASE — Predictor",
                         "Predictor MASE vs Full-Window MASE",
                         "scatter_pred_vs_full.png", out_dir, clip_pct=99),
            plot_efficiency_frontier(df_subset, out_dir),
            plot_gain_histogram(df_subset, out_dir),
            plot_regret_histogram(df_subset, out_dir),
            plot_complexity_reduction(df_subset, out_dir),
            plot_complexity_vs_mase_gain(df_subset, out_dir),
            plot_window_choice_scatter(df_subset, out_dir),
        ]
        per_dataset_paths = plot_per_dataset_bars(df_subset, out_dir)
        for path in single_paths + per_dataset_paths:
            if path:
                print(Fore.GREEN + f"  {path}" + Fore.RESET)

        print(Fore.GREEN + f"\nDone.  Outputs: {out_dir}" + Fore.RESET)

    # ---- Rollup-only mode: skip per-model outputs, emit the cross-model -------
    # overview + grand-total CSV from every model in the run dir, then return.
    if getattr(args, "rollup_only", False):
        out = args.output_dir or run_dir
        _primary_lbl = {"mase_gluonts": "MASE (gluonts)",
                        "mase_gluonts_real": "MASE (gluonts-real)"}[args.mase_metric]
        write_run_rollup(df, out, plot_strategies=args.plot_strategies,
                         metric_label=_primary_lbl, figures=args.all_figures)
        write_run_time_rollup(df, out, plot_strategies=args.plot_strategies,
                              figures=args.all_figures)
        # Twin overview on the other gluonts metric alongside the primary one.
        if df_g is not None:
            write_run_rollup(df_g, out, plot_strategies=args.plot_strategies,
                             suffix="_gluonts", metric_label="MASE (gluonts)",
                             figures=args.all_figures)
        if df_gr is not None:
            write_run_rollup(df_gr, out, plot_strategies=args.plot_strategies,
                             suffix="_gluonts_real", metric_label="MASE (gluonts-real)",
                             figures=args.all_figures)
        return

    # ---- Per-model outputs --------------------------------------------------
    for model_short, df_model in df.groupby("model_short"):
        print(Fore.CYAN + f"\n{'='*78}\n  MODEL: {model_short}\n{'='*78}" + Fore.RESET)
        model_out_dir = (
            args.output_dir
            or os.path.join(run_dir, "models", model_short, "strategy_comparison")
        )
        df_g_model = (df_g[df_g["model_short"] == model_short].reset_index(drop=True)
                      if df_g is not None else None)
        df_gr_model = (df_gr[df_gr["model_short"] == model_short].reset_index(drop=True)
                       if df_gr is not None else None)
        _run_outputs(df_model.reset_index(drop=True), model_out_dir,
                     df_g_model, df_gr_model)

    # ---- Run-level roll-up across models -------------------------------------
    # Only meaningful with >1 model in scope. run_all drives stage 4 one model at
    # a time (--models X), so the cross-model figure comes from a dedicated final
    # --rollup-only pass; here we emit it only for multi-model (e.g. standalone)
    # invocations so single-model runs don't drop a stray one-point overview.
    if df["model_short"].nunique() > 1:
        out = args.output_dir or run_dir
        _primary_lbl = {"mase_gluonts": "MASE (gluonts)",
                        "mase_gluonts_real": "MASE (gluonts-real)"}[args.mase_metric]
        write_run_rollup(df, out, plot_strategies=args.plot_strategies,
                         metric_label=_primary_lbl, figures=args.all_figures)
        write_run_time_rollup(df, out, plot_strategies=args.plot_strategies,
                              figures=args.all_figures)
        if df_g is not None:
            write_run_rollup(df_g, out, plot_strategies=args.plot_strategies,
                             suffix="_gluonts", metric_label="MASE (gluonts)",
                             figures=args.all_figures)
        if df_gr is not None:
            write_run_rollup(df_gr, out, plot_strategies=args.plot_strategies,
                             suffix="_gluonts_real", metric_label="MASE (gluonts-real)",
                             figures=args.all_figures)


if __name__ == "__main__":
    main()
