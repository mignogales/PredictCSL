"""
Figure generation (spec §14) — everything renders from SAVED result files
(schema rows / npz curves), never from in-memory experiment state, so figures
can be regenerated without re-running any model.

`generate_all(run_dir)` walks one run tree and emits every figure whose inputs
exist, into ``<run_dir>/figures/``.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from context_interpretability.analysis import aggregate as agg
from context_interpretability.metrics.statistics import bootstrap_ci, spearman
from context_interpretability.schema import load_results

DPI = 130


def _save(fig, out_dir: str, name: str,
          tight_rect=None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout(rect=tight_rect)
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


def _column_normalize_signed(piv: pd.DataFrame,
                             signed_log: bool = False) -> pd.DataFrame:
    """Normalize each context-length column without discarding direction.

    The denominator is the largest absolute effect in that column.  Optional
    signed-log compression is applied first, so negative improvements remain
    negative instead of being folded together with harmful perturbations.
    """
    vals = piv.to_numpy(dtype=float, copy=True)
    if signed_log:
        vals = np.sign(vals) * np.log1p(np.abs(vals))
    scales = np.nanmax(np.abs(vals), axis=0)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    return pd.DataFrame(vals / scales, index=piv.index, columns=piv.columns)


def _normalize_profile(prof: pd.Series) -> tuple[pd.Series, float]:
    """Scale a profile by its largest absolute value, preserving its sign."""
    values = prof.to_numpy(dtype=float)
    finite = np.isfinite(values)
    scale = float(np.max(np.abs(values[finite]))) if finite.any() else 0.0
    if scale == 0.0:
        return prof.copy(), scale
    return prof / scale, scale


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

def _sliced_loss_delta_profiles(df: pd.DataFrame,
                                masking: pd.DataFrame) -> Dict[int, pd.Series]:
    """Loss change from actually slicing to L, relative to each full W.

    Clean loss is repeated across intervention rows, so first reduce it to one
    value per sample and context.  Pairing by sample keeps the sliced baseline
    comparable to masking's ``masked(L) - clean(W)`` effect and avoids weighting
    samples by the number of experiment rows they happen to have.
    """
    keys = ["model", "dataset", "sample_id", "context_length"]
    required = set(keys + ["clean_loss"])
    if masking.empty or not required.issubset(df.columns):
        return {}
    clean = (df[keys + ["clean_loss"]]
             .dropna(subset=["clean_loss"])
             .groupby(keys, as_index=False, dropna=False)["clean_loss"].mean())
    profiles: Dict[int, pd.Series] = {}
    for W, group in masking.groupby("context_length"):
        full = clean[clean["context_length"] == W].rename(
            columns={"clean_loss": "full_loss"})
        points = {}
        for L in sorted(group["lookback_start"].dropna().unique()):
            sliced = clean[clean["context_length"] == L].rename(
                columns={"clean_loss": "sliced_loss"})
            paired = sliced.merge(
                full, on=["model", "dataset", "sample_id"], how="inner")
            if not paired.empty:
                points[L] = float(
                    (paired["sliced_loss"] - paired["full_loss"]).mean())
        if points:
            profiles[int(W)] = pd.Series(points, dtype=float)
    return profiles


def _masking_colors(contexts) -> Dict[int, object]:
    """Stable, non-repeating color map for full-context lengths W."""
    windows = [int(w) for w in sorted(set(contexts))]
    cmap = plt.get_cmap("turbo", max(1, len(windows)))
    return {w: cmap(i) for i, w in enumerate(windows)}


def _masking_legend_handles(colors: Dict[int, object]):
    """Independent legend handles for color (W) and shape (intervention)."""
    color_handles = [
        Line2D([0], [0], color=color, lw=2.2, label=f"W={W}")
        for W, color in colors.items()
    ]
    shape_handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="-",
               label="attention masked"),
        Line2D([0], [0], color="black", marker="s", linestyle="--",
               label="input sliced"),
    ]
    return color_handles, shape_handles


def _add_masking_legends(ax, colors: Dict[int, object]) -> None:
    color_handles, shape_handles = _masking_legend_handles(colors)
    color_legend = ax.legend(
        handles=color_handles, title="full context W (color)",
        fontsize=7, title_fontsize=8, ncol=2, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.add_artist(color_legend)
    ax.legend(
        handles=shape_handles, title="intervention (shape / line)",
        fontsize=7, title_fontsize=8, loc="lower left",
        bbox_to_anchor=(1.01, 0.0), borderaxespad=0)


def _plot_masking_effect(ax, df: pd.DataFrame, title: str,
                         colors: Optional[Dict[int, object]] = None,
                         add_legends: bool = True) -> bool:
    """Draw masking and paired slicing curves on ``ax``."""
    d = df[df["method"] == "attention_masking"]
    if d.empty:
        return False
    collapsed = agg.collapse_seeds(d)
    sliced_profiles = _sliced_loss_delta_profiles(df, collapsed)
    if colors is None:
        colors = _masking_colors(collapsed["context_length"].unique())
    for W, g in collapsed.groupby("context_length"):
        color = colors[int(W)]
        prof = g.groupby("lookback_start")["loss_delta"].mean()
        ax.plot(prof.index, prof.values, "o-", color=color)
        sliced = sliced_profiles.get(int(W))
        if sliced is not None:
            ax.plot(sliced.index, sliced.values, "s--", color=color)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("visible span L (timesteps) — everything older is masked")
    ax.set_ylabel("Δ loss vs full-W clean input")
    ax.set_title(title)
    if add_legends:
        _add_masking_legends(ax, colors)
    ax.grid(True, alpha=0.3)
    return True


def fig_masking_effect(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not _plot_masking_effect(
            ax, df, "attention masking vs input slicing"):
        plt.close(fig)
        return
    _save(fig, out_dir, "02_masking_effect_vs_lookback.png")


def fig_masking_effect_all_models(model_frames: Dict[str, pd.DataFrame],
                                  out_dir: str) -> None:
    """One attention-masking/slicing overview containing every model."""
    frames = {
        model: frame for model, frame in model_frames.items()
        if not frame.empty
        and (frame["method"] == "attention_masking").any()
    }
    if not frames:
        return
    all_windows = sorted({
        int(W)
        for frame in frames.values()
        for W in frame.loc[
            frame["method"] == "attention_masking",
            "context_length"].dropna().unique()
    })
    colors = _masking_colors(all_windows)
    ncols = min(3, len(frames))
    nrows = math.ceil(len(frames) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 4.8 * nrows), squeeze=False)
    for ax, (model, frame) in zip(axes.flat, frames.items()):
        _plot_masking_effect(
            ax, frame, model, colors=colors, add_legends=False)
    for ax in axes.flat[len(frames):]:
        ax.set_visible(False)
    color_handles, shape_handles = _masking_legend_handles(colors)
    fig.legend(
        handles=color_handles, title="full context W (color)",
        ncol=min(8, len(color_handles)), loc="upper center",
        bbox_to_anchor=(0.5, 0.96), fontsize=7, title_fontsize=8)
    fig.legend(
        handles=shape_handles, title="intervention (shape / line)",
        ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.89),
        fontsize=7, title_fontsize=8)
    fig.suptitle(
        "attention masking vs input slicing — all models",
        fontsize=14, y=0.995)
    _save(
        fig, out_dir, "02_masking_effect_vs_lookback_all_models.png",
        tight_rect=(0, 0, 1, 0.84))


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
        # Comparing raw columns mostly reveals scale changes with prediction /
        # context length.  Normalize within each column so the temporal profile
        # is comparable, using a signed log first to retain beneficial (<0) and
        # harmful (>0) effects while making small effects visible.
        piv = _column_normalize_signed(piv, signed_log=True)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        _heat(ax, piv, "context length", "perturbed block lookback (timesteps)",
              f"{method}: signed-log effect / column max |effect|",
              center_zero=True if center else False)
        _save(fig, out_dir, f"{name}.png")


def fig_perturbation_profiles(df: pd.DataFrame, out_dir: str,
                              dataset: Optional[str] = None,
                              log_y: bool = False) -> None:
    """Per-context normalized effect profiles, one per perturbation type.

    Each curve is divided by its own maximum absolute effect so a
    high-magnitude perturbation cannot hide the shapes of the other curves.
    The original scale used for normalization is retained in the legend.
    Context grids of 13--15 retain the largest 12 panels in a 3x4 comparison;
    grids of 16 or more retain the largest 16 in a 4x4 comparison. ``log_y``
    uses a symmetric log because loss deltas can legitimately be negative.
    """
    d = df[df["method"] == "perturbation"]
    if dataset is not None:
        d = d[d["dataset"] == dataset]
    if d.empty:
        return
    ctxs, nrows, ncols = _perturbation_grid_layout(
        d["context_length"].unique())
    if not ctxs:
        return
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.6 * nrows), squeeze=False)
    for ax, W in zip(axes.flat, ctxs):
        g = agg.collapse_seeds(d[d["context_length"] == W])
        for ptype, gg in g.groupby("perturbation_type"):
            prof = gg.groupby("lookback_start")["loss_delta"].mean()
            normalized, max_abs = _normalize_profile(prof)
            xvals = np.maximum(prof.index.to_numpy(dtype=float), 1.0)
            ax.plot(xvals, normalized.values, "o-", ms=3,
                    label=f"{ptype} (max |Δ|={max_abs:.3g})")
        ax.axhline(0, color="k", lw=0.5)
        # Lookback is a non-negative distance.  A plain log axis avoids the
        # meaningless 2**-k ticks produced by symlog around zero; put the first
        # block at x=1 solely for plotting and label that tick as zero.
        ax.set_xscale("log", base=2)
        ax.set_xlim(left=1)
        if log_y:
            ax.set_yscale("symlog", base=10, linthresh=1e-2)
        ax.set_xlabel("lookback")
        ax.set_title(f"W={W}", fontsize=9)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(ctxs):]:
        ax.set_visible(False)
    ylabel = "normalized mean Δ loss (per perturbation type)"
    if log_y:
        ylabel += " — symlog"
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(ylabel)
    suffix = "" if dataset is None else "_" + "".join(
        c if c.isalnum() or c in "-_" else "_" for c in str(dataset))
    scale_suffix = "_log_y" if log_y else ""
    title = "03c_perturbation_profiles" + suffix + scale_suffix + ".png"
    _save(fig, out_dir, title)


def _perturbation_grid_layout(contexts) -> tuple[List[int], int, int]:
    """Choose a 3x4 or 4x4 grid, discarding only the smallest contexts.

    Up to 15 available contexts use the requested 3x4 layout and retain the
    largest 12.  Sixteen or more use a 4x4 layout and retain the largest 16.
    Smaller test/model grids keep all their contexts in a 3x4 canvas.
    """
    available = [int(w) for w in sorted(set(contexts))]
    if len(available) >= 16:
        return available[-16:], 4, 4
    if len(available) > 12:
        return available[-12:], 3, 4
    return available, 3, 4


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
        fig_perturbation_profiles(df, out_dir, log_y=True)
        # The pooled plot is useful for the headline result, but can conceal
        # periodic/AR/trend and real-world differences.  Persist one profile per
        # dataset as well; synthetic families already encoded as distinct
        # dataset names are consequently kept separate.
        for dataset in sorted(df.loc[df["method"] == "perturbation",
                                     "dataset"].dropna().unique()):
            fig_perturbation_profiles(df, out_dir, str(dataset))
            fig_perturbation_profiles(
                df, out_dir, str(dataset), log_y=True)
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


def generate_all_models(output_root: str, models: List[str]) -> None:
    """Generate figures whose comparison spans multiple model run trees."""
    frames = {
        model: load_results(os.path.join(output_root, model))
        for model in models
        if os.path.isdir(os.path.join(output_root, model))
    }
    fig_masking_effect_all_models(
        frames, os.path.join(output_root, "figures"))
