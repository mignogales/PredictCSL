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
import re
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

from context_interpretability.analysis import aggregate as agg
from context_interpretability.metrics.statistics import bootstrap_ci, spearman
from context_interpretability.schema import load_results

DPI = 130


def _save(fig, out_dir: str, name: str,
          tight_rect=None, use_tight_layout: bool = True) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    if use_tight_layout:
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
    df = df[df["method"] == "attention_masking"]
    keys = ["model", "dataset", "sample_id", "context_length", "metric"]
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
                full, on=["model", "dataset", "sample_id", "metric"],
                how="inner")
            if not paired.empty:
                points[L] = float(
                    (paired["sliced_loss"] - paired["full_loss"]).mean())
        if points:
            profiles[int(W)] = pd.Series(points, dtype=float)
    return profiles


def _sliced_loss_profiles(df: pd.DataFrame,
                          masking: pd.DataFrame) -> Dict[int, pd.Series]:
    """Absolute loss from actual input slicing, paired to each full-W cohort."""
    df = df[df["method"] == "attention_masking"]
    keys = ["model", "dataset", "sample_id", "context_length", "metric"]
    required = set(keys + ["clean_loss"])
    if masking.empty or not required.issubset(df.columns):
        return {}
    clean = (df[keys + ["clean_loss"]]
             .dropna(subset=["clean_loss"])
             .groupby(keys, as_index=False, dropna=False)["clean_loss"].mean())
    profiles: Dict[int, pd.Series] = {}
    for W, group in masking.groupby("context_length"):
        cohort = (clean[clean["context_length"] == W]
                  [["model", "dataset", "sample_id", "metric"]]
                  .drop_duplicates())
        points = {}
        for L in sorted(group["lookback_start"].dropna().unique()):
            sliced = clean[clean["context_length"] == L].rename(
                columns={"clean_loss": "sliced_loss"})
            paired = sliced.merge(
                cohort, on=["model", "dataset", "sample_id", "metric"],
                how="inner")
            if not paired.empty:
                points[L] = float(paired["sliced_loss"].mean())
        if points:
            profiles[int(W)] = pd.Series(points, dtype=float)
    return profiles


def _masking_colors(contexts) -> Dict[int, object]:
    """Stable, non-repeating color map for full-context lengths W."""
    windows = [int(w) for w in sorted(set(contexts))]
    cmap = plt.get_cmap("turbo", max(1, len(windows)))
    return {w: cmap(i) for i, w in enumerate(windows)}


def _masking_legend_handles(
        colors: Dict[int, object], masked_label: str = "attention masked"):
    """Independent legend handles for color (W) and shape (intervention)."""
    color_handles = [
        Line2D([0], [0], color=color, lw=2.2, label=f"W={W}")
        for W, color in colors.items()
    ]
    shape_handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="-",
               label=masked_label),
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
                         add_legends: bool = True,
                         absolute_loss: bool = False,
                         dataset: Optional[str] = None,
                         metric: Optional[str] = None,
                         largest_full_context_only: bool = False) -> bool:
    """Draw masking and paired slicing curves on ``ax``."""
    scoped = df[df["method"] == "attention_masking"]
    if dataset is not None:
        scoped = scoped[scoped["dataset"].astype(str) == str(dataset)]
    if metric is not None:
        scoped = scoped[scoped["metric"].astype(str) == str(metric)]
    if scoped.empty:
        return False
    datasets = scoped["dataset"].astype(str).unique()
    metrics = scoped["metric"].astype(str).unique()
    if len(datasets) != 1 or len(metrics) != 1:
        raise ValueError(
            "masking plots require exactly one dataset and one metric; got "
            f"datasets={datasets.tolist()}, metrics={metrics.tolist()}")
    masking = scoped
    if largest_full_context_only:
        masking = masking[
            masking["context_length"] == masking["context_length"].max()]
    collapsed = agg.collapse_seeds(masking)
    sliced_profiles = (
        _sliced_loss_profiles(scoped, collapsed) if absolute_loss
        else _sliced_loss_delta_profiles(scoped, collapsed)
    )
    masked_value = "intervened_loss" if absolute_loss else "loss_delta"
    if colors is None:
        colors = _masking_colors(collapsed["context_length"].unique())
    for W, g in collapsed.groupby("context_length"):
        color = colors[int(W)]
        prof = g.groupby("lookback_start")[masked_value].mean()
        ax.plot(prof.index, prof.values, "o-", color=color)
        sliced = sliced_profiles.get(int(W))
        if sliced is not None:
            ax.plot(sliced.index, sliced.values, "s--", color=color)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("visible span L (timesteps) — everything older is masked")
    metric_label = str(metrics[0]).upper()
    ax.set_ylabel(
        f"mean {metric_label}" if absolute_loss
        else f"Δ {metric_label} vs full-W clean input")
    ax.set_title(title)
    if add_legends:
        _add_masking_legends(ax, colors)
    ax.grid(True, alpha=0.3)
    return True


def fig_masking_effect(df: pd.DataFrame, out_dir: str) -> None:
    d = df[df["method"] == "attention_masking"]
    if d.empty:
        return
    datasets = sorted(d["dataset"].astype(str).unique())
    # The shared tree contains Exp4 control rows under the same method name.
    # Never pool them with the ordinary evaluation dataset.
    if "synthetic" in datasets:
        datasets = ["synthetic"]
    first = True
    for dataset in datasets:
        metrics = sorted(d.loc[d["dataset"].astype(str) == dataset,
                               "metric"].astype(str).unique())
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            if not _plot_masking_effect(
                    ax, df,
                    f"attention masking vs input slicing — {dataset} — {metric.upper()}",
                    dataset=dataset, metric=metric):
                plt.close(fig)
                continue
            suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{dataset}_{metric}")
            _save(fig, out_dir,
                  "02_masking_effect_vs_lookback.png" if first else
                  f"02_masking_effect_vs_lookback_{suffix}.png")
            first = False


def fig_masking_loss(df: pd.DataFrame, out_dir: str) -> None:
    """Absolute-loss companion to the delta-loss masking comparison."""
    d = df[df["method"] == "attention_masking"]
    if d.empty:
        return
    datasets = sorted(d["dataset"].astype(str).unique())
    if "synthetic" in datasets:
        datasets = ["synthetic"]
    first = True
    for dataset in datasets:
        metrics = sorted(d.loc[d["dataset"].astype(str) == dataset,
                               "metric"].astype(str).unique())
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            if not _plot_masking_effect(
                    ax, df,
                    f"attention masking vs input slicing — {dataset} — absolute {metric.upper()}",
                    absolute_loss=True, dataset=dataset, metric=metric):
                plt.close(fig)
                continue
            suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{dataset}_{metric}")
            _save(fig, out_dir,
                  "02b_masking_loss_vs_lookback.png" if first else
                  f"02b_masking_loss_vs_lookback_{suffix}.png")
            first = False


def fig_masking_effect_all_models(model_frames: Dict[str, pd.DataFrame],
                                  out_dir: str,
                                  absolute_loss: bool = False,
                                  dataset: Optional[str] = None,
                                  largest_full_context_only: bool = True) -> None:
    """One attention-masking/slicing overview containing every model."""
    candidates = {
        model: frame for model, frame in model_frames.items()
        if not frame.empty and (frame["method"] == "attention_masking").any()
    }
    if not candidates:
        return
    if dataset is None:
        common_datasets = None
        for frame in candidates.values():
            available = set(frame.loc[
                frame["method"] == "attention_masking", "dataset"].astype(str))
            common_datasets = (available if common_datasets is None else
                               common_datasets & available)
        if not common_datasets:
            return
        dataset = ("synthetic" if "synthetic" in common_datasets else
                   sorted(common_datasets)[0])
    frames = {
        model: frame for model, frame in candidates.items()
        if not frame.empty
        and ((frame["method"] == "attention_masking")
             & (frame["dataset"].astype(str) == dataset)).any()
    }
    if not frames:
        return
    common_metrics = None
    for frame in frames.values():
        available = set(frame.loc[
            (frame["method"] == "attention_masking")
            & (frame["dataset"].astype(str) == dataset), "metric"].astype(str))
        common_metrics = available if common_metrics is None else common_metrics & available
    if not common_metrics:
        return
    metric = "mse" if "mse" in common_metrics else "mae"
    if largest_full_context_only:
        all_windows = sorted({int(frame.loc[
            (frame["method"] == "attention_masking")
            & (frame["dataset"].astype(str) == dataset)
            & (frame["metric"].astype(str) == metric), "context_length"].max())
            for frame in frames.values()})
    else:
        all_windows = sorted({
            int(W) for frame in frames.values()
            for W in frame.loc[
                (frame["method"] == "attention_masking")
                & (frame["dataset"].astype(str) == dataset)
                & (frame["metric"].astype(str) == metric),
                "context_length"].dropna().unique()
        })
    colors = _masking_colors(all_windows)
    ncols = min(3, len(frames))
    nrows = math.ceil(len(frames) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 4.8 * nrows), squeeze=False)
    for ax, (model, frame) in zip(axes.flat, frames.items()):
        _plot_masking_effect(
            ax, frame, model, colors=colors, add_legends=False,
            absolute_loss=absolute_loss, dataset=dataset, metric=metric,
            largest_full_context_only=largest_full_context_only)
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
    comparison = (f"absolute {metric.upper()}" if absolute_loss
                  else f"Δ {metric.upper()}")
    context_scope = ("largest full context per model"
                     if largest_full_context_only else
                     "all available full input sizes")
    fig.suptitle(
        f"attention masking vs input slicing — {dataset} — "
        f"{context_scope} ({comparison})",
        fontsize=14, y=0.995)
    filename = (
        "02b_masking_loss_vs_lookback_all_models.png"
        if absolute_loss
        else "02_masking_effect_vs_lookback_all_models.png"
    )
    if not largest_full_context_only:
        filename = filename.replace(".png", "_all_input_sizes.png")
    _save(
        fig, out_dir, filename,
        tight_rect=(0, 0, 1, 0.84))


def _plot_tail_stats_masking(ax, df: pd.DataFrame, title: str,
                             colors: Dict[int, object], dataset: str,
                             metric: str,
                             largest_full_context_only: bool) -> bool:
    variant = "attention_mask/tail_matched_stats"
    d = df[(df["method"] == "context_decomposition")
           & (df["perturbation_type"] == variant)
           & (df["dataset"].astype(str) == dataset)
           & (df["metric"].astype(str) == metric)]
    if d.empty:
        return False
    if largest_full_context_only:
        d = d[d["context_length"] == d["context_length"].max()]
    for W, g in d.groupby("context_length"):
        color = colors[int(W)]
        masked = g.groupby("lookback_start")["intervened_loss"].mean()
        sliced = g.groupby("lookback_start")["clean_loss"].mean()
        ax.plot(masked.index, masked.values, "o-", color=color)
        ax.plot(sliced.index, sliced.values, "s--", color=color)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("visible span L (timesteps) — everything older is masked")
    ax.set_ylabel(f"mean {metric.upper()}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return True


def fig_tail_stats_masking_all_models(
        model_frames: Dict[str, pd.DataFrame], out_dir: str,
        largest_full_context_only: bool = True) -> None:
    """Main-style masking/slicing plot using tail-matched preprocessing."""
    variant = "attention_mask/tail_matched_stats"
    frames = {
        model: frame for model, frame in model_frames.items()
        if not frame.empty and ((frame["method"] == "context_decomposition")
                                & (frame["perturbation_type"] == variant)).any()
    }
    if not frames:
        return
    common_datasets = None
    for frame in frames.values():
        available = set(frame.loc[
            (frame["method"] == "context_decomposition")
            & (frame["perturbation_type"] == variant), "dataset"].astype(str))
        common_datasets = (available if common_datasets is None else
                           common_datasets & available)
    if not common_datasets:
        return
    dataset = ("kernelsynth_chronos"
               if "kernelsynth_chronos" in common_datasets
               else sorted(common_datasets)[0])
    metric = "mse"
    frames = {
        model: frame for model, frame in frames.items()
        if ((frame["method"] == "context_decomposition")
            & (frame["perturbation_type"] == variant)
            & (frame["dataset"].astype(str) == dataset)
            & (frame["metric"].astype(str) == metric)).any()
    }
    if not frames:
        return
    if largest_full_context_only:
        windows = sorted({int(frame.loc[
            (frame["method"] == "context_decomposition")
            & (frame["perturbation_type"] == variant)
            & (frame["dataset"].astype(str) == dataset)
            & (frame["metric"].astype(str) == metric),
            "context_length"].max()) for frame in frames.values()})
    else:
        windows = sorted({int(W) for frame in frames.values() for W in
            frame.loc[(frame["method"] == "context_decomposition")
                      & (frame["perturbation_type"] == variant)
                      & (frame["dataset"].astype(str) == dataset)
                      & (frame["metric"].astype(str) == metric),
                      "context_length"].unique()})
    colors = _masking_colors(windows)
    ncols = min(3, len(frames))
    nrows = math.ceil(len(frames) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 4.8 * nrows), squeeze=False)
    for ax, (model, frame) in zip(axes.flat, frames.items()):
        _plot_tail_stats_masking(
            ax, frame, model, colors, dataset, metric,
            largest_full_context_only)
    for ax in axes.flat[len(frames):]:
        ax.set_visible(False)
    color_handles, shape_handles = _masking_legend_handles(
        colors, masked_label="attention masked + tail stats")
    fig.legend(handles=color_handles, title="full context W (color)",
               ncol=min(8, len(color_handles)), loc="upper center",
               bbox_to_anchor=(0.5, 0.96), fontsize=7, title_fontsize=8)
    fig.legend(handles=shape_handles, title="intervention (shape / line)",
               ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.89),
               fontsize=7, title_fontsize=8)
    scope = ("largest full context per model" if largest_full_context_only
             else "all available full input sizes")
    fig.suptitle(
        f"tail-stat attention masking vs input slicing — {dataset} — "
        f"{scope} (absolute MSE)", fontsize=14, y=0.995)
    filename = "02c_tail_stats_masking_loss_all_models.png"
    if not largest_full_context_only:
        filename = filename.replace(".png", "_all_input_sizes.png")
    _save(fig, out_dir, filename, tight_rect=(0, 0, 1, 0.84))


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


def _sparse_axis_labels(ax, values, axis: str, max_ticks: int = 10) -> None:
    """Label a dense categorical heatmap without producing an unreadable wall."""
    n = len(values)
    if n <= max_ticks:
        positions = np.arange(n)
    else:
        positions = np.unique(np.linspace(0, n - 1, max_ticks).round().astype(int))
    labels = [str(values[i]) for i in positions]
    if axis == "x":
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=7)


def fig_perturbation_heatmaps_by_type(df: pd.DataFrame,
                                      out_dir: str) -> None:
    """Raw Exp1 block sweeps: one triangular heatmap per perturbation type.

    Columns fix the input length W; rows fix the perturbed block's lookback.
    Each cell is the sample-mean absolute loss change, after stochastic seeds
    have first been averaged within sample. Unlike the legacy headline heatmap,
    this view never pools intervention types or normalizes columns.
    """
    d = df[df["method"] == "perturbation"]
    if d.empty:
        return
    collapsed = agg.collapse_seeds(d)
    collapsed = collapsed.assign(abs_loss_delta=collapsed["loss_delta"].abs())
    matrices = {}
    positives = []
    for ptype, group in collapsed.groupby("perturbation_type"):
        piv = group.pivot_table(
            index="lookback_start", columns="context_length",
            values="abs_loss_delta", aggfunc="mean")
        if not piv.empty:
            matrices[str(ptype)] = piv
            vals = piv.to_numpy(dtype=float)
            positives.extend(vals[np.isfinite(vals) & (vals > 0)].tolist())
    if not matrices or not positives:
        return
    vmin = max(float(np.nanpercentile(positives, 2)), np.finfo(float).tiny)
    vmax = max(float(np.nanpercentile(positives, 98)), vmin * 1.01)
    names = sorted(matrices)
    ncols = 2
    nrows = int(math.ceil(len(names) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.2 * ncols, 5.2 * nrows), squeeze=False)
    image = None
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")
    for ax, name in zip(axes.flat, names):
        piv = matrices[name]
        image = ax.imshow(
            np.ma.masked_invalid(piv.to_numpy(dtype=float)), aspect="auto",
            origin="lower", interpolation="nearest", cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax))
        _sparse_axis_labels(ax, list(piv.columns), "x", max_ticks=12)
        _sparse_axis_labels(ax, list(piv.index), "y", max_ticks=11)
        ax.set_xlabel("fixed input length W (timesteps)")
        ax.set_ylabel("perturbed-block lookback (timesteps)")
        suffix = " (all configured severities)" if name == "noise" else ""
        ax.set_title(f"{name}{suffix}", fontsize=10)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.subplots_adjust(
        left=0.07, right=0.88, bottom=0.08, top=0.91,
        hspace=0.30, wspace=0.20)
    if image is not None:
        cax = fig.add_axes([0.91, 0.20, 0.018, 0.60])
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label("mean |loss(perturbed block) − loss(clean)|")
    fig.suptitle(
        "Exp1 block-by-block temporal perturbation — raw effect by input length",
        fontsize=14)
    _save(
        fig, out_dir, "03d_perturbation_heatmaps_by_type_abs_dloss.png",
        use_tight_layout=False)


def fig_perturbation_block_sweep_raw(df: pd.DataFrame, out_dir: str,
                                     dataset: Optional[str] = None) -> None:
    """One panel per W; every point is one block intervention on raw scale."""
    d = df[(df["method"] == "perturbation")
           & (df["perturbation_type"] != "noise")]
    if dataset is not None:
        d = d[d["dataset"] == dataset]
    if d.empty:
        return
    ctxs, nrows, ncols = _perturbation_grid_layout(
        d["context_length"].unique())
    if not ctxs:
        return
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.6 * nrows), squeeze=False,
        sharey=True)
    for ax, W in zip(axes.flat, ctxs):
        g = agg.collapse_seeds(d[d["context_length"] == W])
        for ptype, gg in g.groupby("perturbation_type"):
            prof = gg.groupby("lookback_start")["loss_delta"].apply(
                lambda s: float(np.nanmean(np.abs(s.to_numpy(dtype=float)))))
            prof = prof[prof > 0]
            if not prof.empty:
                xvals = np.maximum(prof.index.to_numpy(dtype=float), 1.0)
                ax.plot(xvals, prof.values, "o-", ms=3, label=str(ptype))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(left=1)
        ax.set_xlabel("block lookback (0 = most recent)")
        ax.set_title(f"fixed input W={W}", fontsize=9)
        ax.legend(fontsize=6)
        ax.grid(True, which="both", alpha=0.25)
    for ax in axes.flat[len(ctxs):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("mean |Δ loss| (raw units)")
    suffix = "" if dataset is None else "_" + "".join(
        c if c.isalnum() or c in "-_" else "_" for c in str(dataset))
    fig.suptitle(
        "Exp1: fix input length, perturb one block at a time, measure error change",
        fontsize=14)
    _save(
        fig, out_dir, f"03e_perturbation_block_sweep_raw{suffix}.png",
        tight_rect=(0, 0, 1, 0.97))


def fig_perturbation_test_schematic(out_dir: str) -> None:
    """Data-independent visual definition of the Exp1 intervention."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), height_ratios=[1, 1.05])
    for ax in axes:
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 2.2)
        ax.axis("off")
    colors = {"clean": "#9ecae1", "perturbed": "#f28e2b",
              "edge": "#334155", "forecast": "#2e74b5"}
    for row, (ax, perturbed) in enumerate(zip(axes, [False, True])):
        for i in range(10):
            fill = colors["clean"]
            hatch = None
            if perturbed and i == 6:
                fill, hatch = colors["perturbed"], "///"
            ax.add_patch(Rectangle(
                (0.6 + i * 0.72, 0.9), 0.65, 0.55, facecolor=fill,
                edgecolor=colors["edge"], linewidth=0.8, hatch=hatch))
        ax.text(0.6, 1.68, "older history", fontsize=9, ha="left")
        ax.text(7.75, 1.68, "forecast origin", fontsize=9, ha="right")
        ax.annotate("lookback = 0", xy=(7.75, 0.84), xytext=(7.75, 0.42),
                    ha="right", fontsize=8,
                    arrowprops={"arrowstyle": "->", "color": colors["edge"]})
        ax.add_patch(FancyArrowPatch(
            (8.1, 1.18), (9.5, 1.18), arrowstyle="-|>", mutation_scale=14,
            color=colors["edge"], linewidth=1.2))
        ax.text(8.8, 1.38, "same model", ha="center", fontsize=8)
        ax.add_patch(Rectangle(
            (9.65, 0.85), 1.5, 0.68, facecolor="#dbeafe",
            edgecolor=colors["forecast"], linewidth=1.2))
        ax.text(10.4, 1.19, "forecast", ha="center", va="center", fontsize=9)
        loss = "E_clean" if not perturbed else "E_block"
        ax.text(11.45, 1.18, f"→  {loss}", fontsize=11, va="center")
        label = "A. clean input" if not perturbed else (
            "B. replace exactly one block; width, positions, and all other blocks stay fixed")
        ax.text(0.1, 2.02, label, fontsize=10, weight="bold")
        if perturbed:
            ax.annotate(
                "selected block", xy=(0.6 + 6 * 0.72 + 0.32, 1.47),
                xytext=(5.0, 1.92), ha="center", fontsize=8,
                arrowprops={"arrowstyle": "->", "color": colors["perturbed"]})
            ax.text(11.45, 0.68, "Δerror = E_block − E_clean", fontsize=10,
                    color="#9a3412")
    fig.suptitle(
        "Temporal perturbation test: repeat B for every block and every input length W",
        fontsize=14, weight="bold")
    _save(fig, out_dir, "00_exp1_temporal_perturbation_test.png",
          tight_rect=(0, 0, 1, 0.94))


def fig_long_lag_control_schematic(out_dir: str) -> None:
    """Data-independent visual definition of the long-lag Exp4 control."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 8)
    ax.axis("off")
    ax.plot([5, 93], [3.7, 3.7], color="#334155", lw=1.5)
    ax.scatter([90], [3.7], s=90, color="#2e74b5", zorder=3)
    ax.text(90, 4.2, "target y(t)", ha="center", weight="bold")
    ax.add_patch(Rectangle((82, 3.2), 7, 1.0, facecolor="#bfdbfe",
                           edgecolor="#2e74b5"))
    ax.text(85.5, 2.75, "local lags 1…8", ha="center", fontsize=9)
    ax.scatter([40], [3.7], s=90, color="#f28e2b", zorder=3)
    ax.text(40, 4.2, "planted x(t−d)", ha="center", weight="bold",
            color="#9a3412")
    ax.annotate("verified incremental signal", xy=(40, 3.85), xytext=(57, 6.4),
                ha="center", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#f28e2b"})
    ax.add_patch(FancyArrowPatch(
        (41, 4.0), (88, 4.0), arrowstyle="-|>", connectionstyle="arc3,rad=-.22",
        mutation_scale=14, color="#f28e2b", lw=1.6))
    ax.plot([55, 90], [1.65, 1.65], lw=5, color="#94a3b8")
    ax.text(72.5, 1.0, "short W < d: distant cause excluded", ha="center")
    ax.plot([20, 90], [0.35, 0.35], lw=5, color="#2e74b5")
    ax.text(55, -0.3, "long W ≥ d: distant cause available", ha="center")
    ax.text(5, 7.25, "1  Oracle check", weight="bold", color="#0b2545")
    ax.text(5, 6.72, "recent-only ridge vs recent + lag-d ridge", fontsize=9)
    ax.text(38, 7.25, "2  Context curve", weight="bold", color="#0b2545")
    ax.text(38, 6.72, "does sufficient W grow with d?", fontsize=9)
    ax.text(70, 7.25, "3  Block intervention", weight="bold", color="#0b2545")
    ax.text(70, 6.72, "does sensitivity peak near lag d?", fontsize=9)
    fig.suptitle(
        "Long-dependency control: distinguish adaptive reach from inability to use history",
        fontsize=14, weight="bold")
    _save(fig, out_dir, "00_exp4_long_dependency_test.png",
          tight_rect=(0, 0.03, 1, 0.94))


def fig_perturbation_profiles(df: pd.DataFrame, out_dir: str,
                              dataset: Optional[str] = None,
                              log_y: bool = False) -> None:
    """Per-context normalized effect profiles for informative perturbations.

    Each curve is divided by its own maximum absolute effect so a
    high-magnitude perturbation cannot hide the shapes of the other curves.
    Additive-noise controls are intentionally omitted from these profile
    figures; their rows remain available in the saved results for analysis.
    The original scale used for normalization is retained in the legend.
    Context grids of 13--15 retain the largest 12 panels in a 3x4 comparison;
    grids of 16 or more retain the largest 16 in a 4x4 comparison. ``log_y``
    uses a symmetric log because loss deltas can legitimately be negative.
    """
    d = df[
        (df["method"] == "perturbation")
        & (df["perturbation_type"] != "noise")
    ]
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


# -- 11. predictor curve-contrast saliency --------------------------------------

def fig_predictor_contrast_saliency(df: pd.DataFrame, out_dir: str) -> None:
    """Signed block attribution for E_hat(long)-E_hat(short).

    Positive cells push the predictor toward higher long-window error; negative
    cells push it toward lower long-window error. Rows are lookback distances,
    columns retain the baseline and window-pair label.
    """
    d = df[df["method"] == "predictor_contrast_saliency"]
    if d.empty:
        return
    piv = d.pivot_table(
        index="lookback_start", columns="perturbation_type",
        values="attribution_score", aggfunc="mean").sort_index()
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(piv.columns)), 6))
    _heat(
        ax, piv.T, "predictor-patch lookback (timesteps)",
        "baseline / curve contrast",
        "predictor saliency: signed IG for E_hat(long) - E_hat(short)",
        center_zero=True)
    _save(fig, out_dir, "11_predictor_contrast_saliency.png")


# -- 12. masking/slicing/normalization decomposition -----------------------------

def fig_context_decomposition(df: pd.DataFrame, out_dir: str) -> None:
    d = df[df["method"] == "context_decomposition"]
    if d.empty:
        return
    full_name = "attention_mask/full_history_stats"
    tail_name = "attention_mask/tail_matched_stats"
    combinations = d[["dataset", "metric"]].dropna().drop_duplicates()
    first_by_metric = set()
    for dataset, metric in combinations.sort_values(
            ["dataset", "metric"]).itertuples(index=False, name=None):
        dm = d[(d["dataset"] == dataset) & (d["metric"] == metric)]
        contexts = sorted(int(v) for v in dm["context_length"].unique())
        colors = _masking_colors(contexts)
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        for W, g in dm.groupby("context_length"):
            color = colors[int(W)]
            grouped = g.groupby(
                ["lookback_start", "perturbation_type"], as_index=False
            )[["clean_loss", "intervened_loss", "loss_delta"]].mean()
            sliced = (grouped.groupby("lookback_start")["clean_loss"]
                      .mean().sort_index())
            axes[0].plot(sliced.index, sliced.values, "s--", color=color,
                         label=f"slice, W={int(W)}")
            deltas = {}
            for variant, marker, label in [
                (full_name, "o", "mask/full stats"),
                (tail_name, "^", "mask/tail stats"),
            ]:
                v = grouped[grouped["perturbation_type"] == variant]
                if v.empty:
                    continue
                prof = v.set_index("lookback_start").sort_index()
                axes[0].plot(
                    prof.index, prof["intervened_loss"], marker + "-",
                    color=color, alpha=0.85,
                    label=f"{label}, W={int(W)}")
                axes[1].plot(
                    prof.index, prof["loss_delta"], marker + "-",
                    color=color, alpha=0.85,
                    label=f"{label}, W={int(W)}")
                deltas[variant] = prof["loss_delta"]
            if full_name in deltas and tail_name in deltas:
                common = deltas[full_name].index.intersection(
                    deltas[tail_name].index)
                norm_component = (
                    deltas[full_name].loc[common]
                    - deltas[tail_name].loc[common])
                axes[2].plot(common, norm_component, "d-", color=color,
                             label=f"W={int(W)}")

        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("visible suffix L")
            ax.axhline(0, color="black", lw=0.6, alpha=0.6)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6)
        axes[0].set_ylabel(f"mean {metric.upper()}")
        axes[0].set_title("absolute forecast error")
        axes[1].set_ylabel(f"Δ {metric.upper()} vs slicing")
        axes[1].set_title("masking minus slicing")
        axes[2].set_ylabel(f"Δ {metric.upper()}")
        axes[2].set_title("normalization proxy: full-stats mask − tail-stats mask")
        fig.suptitle(
            f"Context restriction decomposition — {dataset} — {metric.upper()}",
            fontsize=13)
        suffix = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", f"{dataset}_{metric}")
        filename = (f"12_context_decomposition_{metric}.png"
                    if metric not in first_by_metric else
                    f"12_context_decomposition_{suffix}.png")
        first_by_metric.add(metric)
        _save(fig, out_dir, filename,
              tight_rect=(0, 0, 1, 0.94))


def fig_normalization_stat_ablation(df: pd.DataFrame, out_dir: str) -> None:
    """Mean-only versus scale-only direct normalization interventions."""
    variants = [
        ("attention_mask/full_history_stats", "mask/full stats", "o-"),
        ("attention_mask/tail_mean_full_scale", "tail mean only", "^-"),
        ("attention_mask/full_mean_tail_scale", "tail scale only", "v-"),
        ("attention_mask/tail_mean_tail_scale_direct", "tail mean + scale", "D-"),
    ]
    d = df[(df["method"] == "context_decomposition") &
           df["perturbation_type"].isin([v[0] for v in variants])]
    if d.empty or not d["perturbation_type"].str.contains(
            "tail_mean_tail_scale_direct", regex=False).any():
        return
    first_by_metric = set()
    for (dataset, metric), dm in d.groupby(["dataset", "metric"]):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        for stat, ax in (("mean", axes[0]), ("median", axes[1])):
            sliced = dm.groupby("lookback_start")["clean_loss"].agg(stat)
            ax.plot(sliced.index, sliced.values, "s--", lw=2,
                    label="input sliced")
            for name, label, style in variants:
                v = dm[dm["perturbation_type"] == name]
                if v.empty:
                    continue
                prof = v.groupby("lookback_start")["intervened_loss"].agg(stat)
                ax.plot(prof.index, prof.values, style, lw=1.7, label=label)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("visible suffix L")
            ax.set_ylabel(f"{stat} {str(metric).upper()} (log scale)")
            ax.set_title(f"{stat} across instances")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle(
            f"Direct normalization-statistic ablation — {dataset} — "
            f"{str(metric).upper()}", fontsize=13)
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{dataset}_{metric}")
        filename = (f"12b_normalization_stat_ablation_{metric}.png"
                    if metric not in first_by_metric else
                    f"12b_normalization_stat_ablation_{suffix}.png")
        first_by_metric.add(metric)
        _save(fig, out_dir, filename, tight_rect=(0, 0, 1, 0.94))


# -- 13. direct TSFM loss-contrast saliency -------------------------------------

def fig_tsfm_contrast_saliency(df: pd.DataFrame, out_dir: str) -> None:
    d = df[df["method"] == "tsfm_loss_contrast_saliency"].copy()
    if d.empty:
        return
    d["contrast"] = d["metric"].astype(str).str.upper() + " | " + \
        d["perturbation_type"].astype(str)
    piv = d.pivot_table(
        index="lookback_start", columns="contrast",
        values="attribution_score", aggfunc="mean").sort_index()
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(piv.columns)), 6))
    _heat(
        ax, piv.T, "long-input lookback (timesteps)",
        "loss / baseline / context pair",
        "TSFM saliency: signed IG for loss(long) - loss(short)",
        center_zero=True)
    _save(fig, out_dir, "13_tsfm_loss_contrast_saliency.png")


# -- driver ----------------------------------------------------------------------------

def generate_all(run_dir: str, tolerance: float = 0.05) -> None:
    out_dir = os.path.join(run_dir, "figures")
    fig_perturbation_test_schematic(out_dir)
    fig_long_lag_control_schematic(out_dir)
    df = load_results(run_dir)
    if not df.empty:
        fig_masking_effect(df, out_dir)
        fig_masking_loss(df, out_dir)
        fig_block_heatmaps(df, out_dir)
        # The shared run tree also contains Exp4's many nested perturbation
        # datasets. Keep the exact W-by-block Exp1 view scoped to the ordinary
        # evaluation pool; control datasets have their own per-dataset plots.
        dataset_names = df["dataset"].astype(str)
        exp1_rows = df[~dataset_names.str.match(r"^fam[ABC]_")]
        fig_perturbation_heatmaps_by_type(exp1_rows, out_dir)
        fig_perturbation_block_sweep_raw(exp1_rows, out_dir)
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
        fig_predictor_contrast_saliency(df, out_dir)
        fig_context_decomposition(df, out_dir)
        fig_normalization_stat_ablation(df, out_dir)
        fig_tsfm_contrast_saliency(df, out_dir)
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


def fig_control_sweeps_all_models(output_root: str, models: List[str],
                                  out_dir: str) -> None:
    """Compare long-lag sufficient-context tracking across model families."""
    series = {}
    all_lags = set()
    for model in models:
        path = os.path.join(
            output_root, model, "exp4_synthetic_controls",
            "controls_summary.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = json.load(f).get("controls", [])
        core = [r for r in rows if r.get("design_role") == "core"
                and not r.get("config_broken")]
        if not core:
            continue
        max_strength = max(float(r["spec"]["strength"]) for r in core)
        points = sorted(
            (int(r["spec"]["distant_lag"]), int(r["sufficient_context"]))
            for r in core
            if float(r["spec"]["strength"]) == max_strength)
        if points:
            series[model] = points
            all_lags.update(x for x, _y in points)
    if not series:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, points in series.items():
        ax.plot(*zip(*points), "o-", label=model)
    ref = sorted(all_lags)
    ax.plot(ref, ref, "k--", alpha=0.5, label="sufficient context = lag")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("planted dependency lag d (timesteps)")
    ax.set_ylabel("estimated 5% sufficient context (timesteps)")
    ax.set_title("Long-lag control — adaptive context reach across models")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    _save(fig, out_dir, "07_sufficient_context_vs_lag_all_models.png")


def fig_perturbation_recency_all_models(frames: Dict[str, pd.DataFrame],
                                        out_dir: str) -> None:
    """Compare the block-mean Exp1 profile at each model's largest input W."""
    curves = {}
    for model, frame in frames.items():
        if frame.empty:
            continue
        dataset_names = frame["dataset"].astype(str)
        d = frame[(frame["method"] == "perturbation")
                  & (frame["perturbation_type"] == "block_mean")
                  & ~dataset_names.str.match(r"^fam[ABC]_")]
        if d.empty:
            continue
        W = int(d["context_length"].max())
        g = agg.collapse_seeds(d[d["context_length"] == W])
        prof = g.groupby("lookback_start")["loss_delta"].apply(
            lambda s: float(np.nanmean(np.abs(s.to_numpy(dtype=float)))))
        scale = float(prof.max()) if not prof.empty else 0.0
        if scale > 0:
            curves[model] = (W, prof / scale)
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, (W, prof) in curves.items():
        xvals = np.maximum(prof.index.to_numpy(dtype=float), 1.0)
        ax.plot(xvals, prof.values, "o-", ms=3, label=f"{model} (W={W})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("perturbed-block lookback (timesteps; 0 plotted at 1)")
    ax.set_ylabel("mean |Δ loss| / model-profile maximum")
    ax.set_title("Exp1 block-mean sensitivity at each model's largest input")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    _save(fig, out_dir, "03e_perturbation_recency_all_models.png")


def generate_all_models(output_root: str, models: List[str]) -> None:
    """Generate figures whose comparison spans multiple model run trees."""
    frames = {
        model: load_results(os.path.join(output_root, model))
        for model in models
        if os.path.isdir(os.path.join(output_root, model))
    }
    fig_masking_effect_all_models(
        frames, os.path.join(output_root, "figures"))
    fig_masking_effect_all_models(
        frames, os.path.join(output_root, "figures"), absolute_loss=True)
    fig_masking_effect_all_models(
        frames, os.path.join(output_root, "figures"),
        largest_full_context_only=False)
    fig_masking_effect_all_models(
        frames, os.path.join(output_root, "figures"), absolute_loss=True,
        largest_full_context_only=False)
    fig_tail_stats_masking_all_models(
        frames, os.path.join(output_root, "figures"))
    fig_tail_stats_masking_all_models(
        frames, os.path.join(output_root, "figures"),
        largest_full_context_only=False)
    fig_perturbation_recency_all_models(
        frames, os.path.join(output_root, "figures"))
    fig_control_sweeps_all_models(
        output_root, models, os.path.join(output_root, "figures"))
