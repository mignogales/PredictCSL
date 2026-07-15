"""
Figure generation (spec §14) — everything renders from SAVED result files
(schema rows / npz curves), never from in-memory experiment state, so figures
can be regenerated without re-running any model.

`generate_all(run_dir)` walks one run tree and emits every figure whose inputs
exist, into ``<run_dir>/figures/``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from context_interpretability.analysis import aggregate as agg
from context_interpretability.metrics.statistics import bootstrap_ci, spearman
from context_interpretability.schema import load_results

DPI = 130


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"[figures] wrote {path}")
    return path


def _heat(ax, piv: pd.DataFrame, xlabel: str, ylabel: str, title: str,
          cmap: str = "magma", center_zero: bool = False):
    vals = piv.to_numpy(dtype=float)
    kw = {}
    if center_zero and np.isfinite(vals).any():
        vmax = np.nanpercentile(np.abs(vals), 98) or 1.0
        kw = {"vmin": -vmax, "vmax": vmax, "cmap": "coolwarm"}
    else:
        kw = {"cmap": cmap}
    im = ax.imshow(vals, aspect="auto", origin="lower",
                   interpolation="nearest", **kw)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([str(c) for c in piv.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([str(i) for i in piv.index], fontsize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax)


# -- 1. context length vs forecasting error --------------------------------------

def fig_error_curve(clean_cache_dir: str, out_dir: str, tag: str) -> None:
    Ws, means = [], []
    if not os.path.isdir(clean_cache_dir):
        return
    for f in sorted(os.listdir(clean_cache_dir)):
        if f.startswith("clean_w") and f.endswith(".npz"):
            z = np.load(os.path.join(clean_cache_dir, f))
            Ws.append(int(f[len("clean_w"):-len(".npz")]))
            means.append(float(np.nanmean(z["loss"])))
    if not Ws:
        return
    order = np.argsort(Ws)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.array(Ws)[order], np.array(means)[order], "o-")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length")
    ax.set_ylabel("mean loss")
    ax.set_title(f"context length vs forecast error — {tag}")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, f"01_error_vs_context_{tag}.png")


# -- 2. attention-masking effect vs lookback --------------------------------------

def fig_masking_effect(df: pd.DataFrame, out_dir: str) -> None:
    d = df[df["method"] == "attention_masking"]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for W, g in agg.collapse_seeds(d).groupby("context_length"):
        prof = g.groupby("lookback_start")["loss_delta"].mean()
        ax.plot(prof.index, prof.values, "o-", label=f"W={W}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("visible span L (timesteps) — everything older is masked")
    ax.set_ylabel("Δ loss (masked − clean)")
    ax.set_title("attention masking: effect vs visible span")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "02_masking_effect_vs_lookback.png")


# -- 3+6. block-effect heatmaps (perturbation Δloss / pred-distance / IG) ---------

def fig_block_heatmaps(df: pd.DataFrame, out_dir: str) -> None:
    for method, value, name, center in [
        ("perturbation", "loss_delta", "03a_perturbation_heatmap_dloss", True),
        ("perturbation", "prediction_distance",
         "03b_perturbation_heatmap_preddist", False),
        ("integrated_gradients", "attribution_score",
         "06_ig_attribution_heatmap", False),
    ]:
        d = df[df["method"] == method]
        if d.empty:
            continue
        piv = agg.heatmap_matrix(d, value=value)
        if piv.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5.5))
        _heat(ax, piv, "context length", "perturbed block lookback (timesteps)",
              f"{method}: mean {value}", center_zero=center)
        _save(fig, out_dir, f"{name}.png")


def fig_perturbation_profiles(df: pd.DataFrame, out_dir: str) -> None:
    """Per-context line plots: effect vs lookback, one curve per perturbation
    type (spec §4.4)."""
    d = df[df["method"] == "perturbation"]
    if d.empty:
        return
    ctxs = sorted(d["context_length"].unique())
    k = len(ctxs)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 3.6), squeeze=False)
    for ax, W in zip(axes[0], ctxs):
        g = agg.collapse_seeds(d[d["context_length"] == W])
        for ptype, gg in g.groupby("perturbation_type"):
            prof = gg.groupby("lookback_start")["loss_delta"].mean()
            ax.plot(prof.index, prof.values, "o-", ms=3, label=str(ptype))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("symlog", base=2, linthresh=32)
        ax.set_xlabel("lookback")
        ax.set_title(f"W={W}", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("mean Δ loss")
    axes[0][0].legend(fontsize=6)
    _save(fig, out_dir, "03c_perturbation_profiles.png")


# -- 4. activation-patching heatmap ------------------------------------------------

def fig_patching_heatmaps(df: pd.DataFrame, out_dir: str) -> None:
    d = df[df["method"] == "activation_patching"]
    if d.empty:
        return
    piv = agg.layer_block_matrix(d, "recovery_score")
    if not piv.empty:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        _heat(ax, piv, "corrupted block lookback", "layer",
              "activation patching: recovery score", cmap="viridis")
        _save(fig, out_dir, "04_patching_recovery_heatmap.png")
    for W in sorted(d["context_length"].unique()):     # context-conditioned
        pw = agg.layer_block_matrix(d, "recovery_score", context_length=W)
        if pw.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5.5))
        _heat(ax, pw, "corrupted block lookback", "layer",
              f"patching recovery — W={W}", cmap="viridis")
        _save(fig, out_dir, f"04_patching_recovery_w{W}.png")


# -- 5. forecast-lens heatmaps ------------------------------------------------------

def fig_lens_heatmaps(df: pd.DataFrame, out_dir: str) -> None:
    mats = agg.lens_matrices(df)
    for key, piv in mats.items():
        if piv.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5.5))
        _heat(ax, piv, "context length", "layer", f"forecast lens: {key}")
        _save(fig, out_dir, f"05_lens_{key}.png")


# -- 7+8. synthetic control sweeps ---------------------------------------------------

def fig_control_sweeps(controls_summary_path: str, out_dir: str) -> None:
    if not os.path.exists(controls_summary_path):
        return
    with open(controls_summary_path) as f:
        summ = json.load(f)
    rows = [c for c in summ.get("controls", [])
            if c["spec"]["family"] == "B" and not c.get("config_broken")]
    if not rows:
        return
    strengths = sorted({c["spec"]["strength"] for c in rows})
    smax = strengths[-1]

    # 7: sufficient context vs true lag d (max strength, per noise)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nz in sorted({c["spec"]["noise"] for c in rows}):
        pts = sorted((c["spec"]["distant_lag"], c["sufficient_context"])
                     for c in rows
                     if c["spec"]["strength"] == smax
                     and c["spec"]["noise"] == nz)
        if pts:
            ax.plot(*zip(*pts), "o-", label=f"noise={nz}")
    lims = [c["spec"]["distant_lag"] for c in rows]
    ax.plot(sorted(set(lims)), sorted(set(lims)), "k--", alpha=0.5,
            label="suff = d")
    ax.set_xlabel("true dependency lag d")
    ax.set_ylabel("estimated sufficient context")
    ax.set_title("synthetic lag sweep (strength = max)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "07_sufficient_context_vs_lag.png")

    # 8: dependency-strength sweep
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d_lag in sorted({c["spec"]["distant_lag"] for c in rows}):
        pts = sorted((c["spec"]["strength"], c["sufficient_context"])
                     for c in rows if c["spec"]["distant_lag"] == d_lag)
        if pts:
            ax.plot(*zip(*pts), "o-", label=f"d={d_lag}")
    ax.set_xlabel("dependency strength")
    ax.set_ylabel("estimated sufficient context")
    ax.set_title("synthetic dependency-strength sweep")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "08_sufficient_context_vs_strength.png")


# -- 9. cross-method comparison -------------------------------------------------------

CROSS_METHODS = {
    "attention_masking": ("loss_delta", False),
    "perturbation": ("loss_delta", False),
    "activation_patching": ("recovery_score", True),   # high = influential
    "integrated_gradients": ("attribution_score", True),
}


def cross_method_table(df: pd.DataFrame, context_length: int,
                       top_k: int = 5) -> Optional[pd.DataFrame]:
    profiles: Dict[str, pd.Series] = {}
    for m, (val, _hi) in CROSS_METHODS.items():
        p = agg.method_block_profile(df, m, val, context_length)
        if not p.empty:
            profiles[m] = p
    if len(profiles) < 2:
        return None
    idx = sorted(set().union(*[set(p.index) for p in profiles.values()]))
    mat = pd.DataFrame({m: p.reindex(idx) for m, p in profiles.items()},
                       index=idx)
    methods = list(mat.columns)
    rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            va, vb = mat[a].to_numpy(), mat[b].to_numpy()
            ka = set(mat[a].abs().nlargest(top_k).index)
            kb = set(mat[b].abs().nlargest(top_k).index)
            rows.append({"context_length": context_length,
                         "method_a": a, "method_b": b,
                         "spearman": spearman(np.abs(va), np.abs(vb)),
                         "topk_overlap": len(ka & kb) / max(1, top_k)})
    return pd.DataFrame(rows)


def fig_cross_method(df: pd.DataFrame, out_dir: str, top_k: int = 5) -> None:
    tabs = []
    for W in sorted(df["context_length"].dropna().unique()):
        t = cross_method_table(df, int(W), top_k)
        if t is not None:
            tabs.append(t)
    if not tabs:
        return
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(os.path.join(out_dir, "09_cross_method_agreement.csv"),
               index=False)
    piv = tab.pivot_table(index=["method_a", "method_b"],
                          columns="context_length", values="spearman")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _heat(ax, piv.rename(index=lambda t: " vs ".join(t) if isinstance(t, tuple)
                         else t),
          "context length", "method pair",
          "cross-method Spearman (|block effect| profiles)",
          cmap="viridis")
    _save(fig, out_dir, "09_cross_method_spearman.png")


# -- 10. instance-level sufficient-context distribution --------------------------------

def fig_instance_sufficient_context(clean_cache_dir: str, out_dir: str,
                                    tag: str, tolerance: float = 0.05) -> None:
    if not os.path.isdir(clean_cache_dir):
        return
    Ws, losses = [], []
    for f in sorted(os.listdir(clean_cache_dir)):
        if f.startswith("clean_w") and f.endswith(".npz"):
            z = np.load(os.path.join(clean_cache_dir, f))
            Ws.append(int(f[len("clean_w"):-len(".npz")]))
            losses.append(z["loss"])
    if len(Ws) < 3:
        return
    order = np.argsort(Ws)
    mat = np.stack([losses[i] for i in order], axis=1)
    windows = [Ws[i] for i in order]
    suff = agg.instance_sufficient_context(mat, windows, tolerance)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.array(windows, dtype=float)
    ax.hist(suff, bins=np.concatenate([bins, [bins[-1] * 2]]),
            edgecolor="k", alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("per-instance sufficient context (timesteps)")
    ax.set_ylabel("instances")
    ax.set_title(f"instance-level sufficient context — {tag} "
                 f"(tol={tolerance})")
    _save(fig, out_dir, f"10_instance_sufficient_context_{tag}.png")


# -- driver ----------------------------------------------------------------------------

def generate_all(run_dir: str, tolerance: float = 0.05) -> None:
    out_dir = os.path.join(run_dir, "figures")
    df = load_results(run_dir)
    if not df.empty:
        fig_masking_effect(df, out_dir)
        fig_block_heatmaps(df, out_dir)
        fig_perturbation_profiles(df, out_dir)
        fig_patching_heatmaps(df, out_dir)
        fig_lens_heatmaps(df, out_dir)
        fig_cross_method(df, out_dir)
    # error curves + instance distributions from every clean cache found
    for dirpath, dirs, _files in os.walk(run_dir):
        for d in dirs:
            if d.startswith("clean_cache"):
                tag = os.path.relpath(dirpath, run_dir).replace(os.sep, "_")
                fig_error_curve(os.path.join(dirpath, d), out_dir, tag)
                fig_instance_sufficient_context(
                    os.path.join(dirpath, d), out_dir, tag, tolerance)
    ctrl = os.path.join(run_dir, "exp4_synthetic_controls",
                        "controls_summary.json")
    fig_control_sweeps(ctrl, out_dir)
