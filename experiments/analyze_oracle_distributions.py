"""Analyze per-instance oracle context distributions from cached GiftEval cells.

This is a post-processing experiment: it NEVER runs a TSFM or a context
predictor.  It scans the ``general/datasets`` cache produced by
``test_window_ablation_gifteval_v5.py``, aligns every numeric context window and
the native/full action by ``served_index``, and selects the lowest realised
per-instance MASE.

The analysis answers two separate reproducibility questions:

1. Cross-model consistency: for the same GiftEval series, do different TSFMs
   prefer similar fractions of the available history?
2. Intra-dataset consistency: is a dataset's oracle-window distribution stable
   under repeated random half-splits, or is it diffuse / instance-specific?

Outputs
-------
``instance_oracles.csv.gz``
    One row per model x dataset x test instance.
``cell_summary.csv``
    Distribution summaries (mode, quantiles, entropy, native-selection rate).
``cross_model_cell_agreement.csv``
    Pairwise agreement for every model pair within each dataset/term.
``model_pair_summary.csv``
    Cross-dataset aggregation of the preceding table.
``intra_dataset_stability.csv``
    Bootstrap half-split Jensen-Shannon distance for each model/dataset cell.
``figures/*.png``
    Compact cross-model and intra-dataset overview plots.

Example
-------
python -m experiments.analyze_oracle_distributions \
  --run-dir logs/experiments/master_recompute/window_ablation_gifteval/general
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import datasets_config


WINDOW_RE = re.compile(r"w([0-9]+)$")
# Fixed bins make distributions comparable when model context grids/caps differ.
FRACTION_EDGES = np.asarray(
    [0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1.0 + 1e-12],
    dtype=np.float64,
)


def _resolve_datasets_root(run_dir: str) -> Path:
    root = Path(run_dir)
    if root.name == "datasets" and root.is_dir():
        return root
    candidate = root / "datasets"
    if candidate.is_dir():
        return candidate
    raise SystemExit(
        f"No datasets cache found at {candidate}. Pass the ablation general/ "
        "directory (or its datasets/ child)."
    )


def _metric_vector(
    path: Path, n: int, metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    values = np.full(n, np.nan, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    effective = np.full(n, np.nan, dtype=np.float64)
    with np.load(path) as data:
        source = metric
        if source not in data.files:
            if metric == "mase_gluonts_real" and "mase_gluonts" in data.files:
                source = "mase_gluonts"
            else:
                return values, counts, effective, "missing"
        raw = np.asarray(data[source], dtype=np.float64)
        if "served_index" not in data.files:
            raise RuntimeError(
                f"{path} lacks served_index. Re-run Stage 3 once to backfill "
                "alignment without TSFM inference."
            )
        index = np.asarray(data["served_index"], dtype=np.int64)
        if raw.shape != index.shape:
            raise RuntimeError(f"Misaligned values/served_index in {path}")
        raw_counts = (
            np.asarray(data["valid_count"], dtype=np.float64)
            if "valid_count" in data.files
            else np.ones(raw.shape, dtype=np.float64)
        )
        if raw_counts.shape != raw.shape:
            raise RuntimeError(f"Misaligned valid_count in {path}")
        ok = (index >= 0) & (index < n)
        values[index[ok]] = raw[ok]
        counts[index[ok]] = raw_counts[ok]
        if "effective_context" in data.files:
            raw_effective = np.asarray(data["effective_context"], dtype=np.float64)
            if raw_effective.shape != raw.shape:
                raise RuntimeError(f"Misaligned effective_context in {path}")
            effective[index[ok]] = raw_effective[ok]
    return values, counts, effective, source


def _cell_size(paths: Sequence[Path]) -> int:
    largest = -1
    for path in paths:
        with np.load(path) as data:
            if "served_index" not in data.files:
                raise RuntimeError(
                    f"{path} lacks served_index. Re-run Stage 3 once to backfill it."
                )
            index = np.asarray(data["served_index"], dtype=np.int64)
            if index.size:
                largest = max(largest, int(index.max()))
    return largest + 1


def select_oracles(
    errors: np.ndarray,
    requested_windows: np.ndarray,
    effective_contexts: np.ndarray,
    native_error: np.ndarray,
    native_context: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Select the minimum-error action per row, matching existing tie policy."""
    candidates = np.column_stack([errors, native_error])
    finite = np.isfinite(candidates)
    has_action = finite.any(axis=1)
    indices = np.argmin(np.where(finite, candidates, np.inf), axis=1)
    rows = np.arange(candidates.shape[0])
    selected_error = candidates[rows, indices]
    selected_error[~has_action] = np.nan
    native_selected = indices == requested_windows.size

    requested = np.where(
        native_selected,
        native_context,
        requested_windows[np.minimum(indices, requested_windows.size - 1)],
    ).astype(np.float64)
    grid_effective = effective_contexts[
        rows, np.minimum(indices, requested_windows.size - 1)
    ]
    selected_effective = np.where(native_selected, native_context, grid_effective)
    # Old numeric caches do not store effective_context. Their served rows have
    # at least the requested history, so requested W is the correct effective W.
    selected_effective = np.where(
        np.isfinite(selected_effective), selected_effective, requested
    )
    requested[~has_action] = np.nan
    selected_effective[~has_action] = np.nan
    native_selected &= has_action
    fraction = np.divide(
        selected_effective,
        native_context,
        out=np.full(selected_effective.shape, np.nan, dtype=np.float64),
        where=np.isfinite(native_context) & (native_context > 0),
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    return {
        "oracle_error": selected_error,
        "oracle_requested_context": requested,
        "oracle_effective_context": selected_effective,
        "native_selected": native_selected,
        "oracle_fraction": fraction,
        "has_action": has_action,
    }


def _discover_cells(
    datasets_root: Path,
    models: Optional[Iterable[str]],
    datasets: Optional[Iterable[str]],
) -> List[Tuple[str, str, str, Path]]:
    allowed_models = set(models or [])
    allowed_datasets = set(datasets or [])
    active = {
        (display, str(term))
        for _ge, term, display, _univariate in datasets_config.datasets_to_run()
    }
    found: List[Tuple[str, str, str, Path]] = []
    for dataset_dir in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
        if allowed_datasets and dataset_dir.name not in allowed_datasets:
            continue
        for model_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            if allowed_models and model_dir.name not in allowed_models:
                continue
            for term_dir in sorted(path for path in model_dir.glob("t*") if path.is_dir()):
                term = term_dir.name[1:]
                if (dataset_dir.name, term) not in active:
                    continue
                if (term_dir / "wfull_native" / "per_sample_metrics.npz").is_file():
                    found.append((model_dir.name, dataset_dir.name, term, term_dir))
    return found


def load_cell(
    model: str,
    dataset: str,
    term: str,
    term_dir: Path,
    metric: str,
) -> pd.DataFrame:
    native_path = term_dir / "wfull_native" / "per_sample_metrics.npz"
    numeric: List[Tuple[int, Path]] = []
    for window_dir in term_dir.glob("w*"):
        match = WINDOW_RE.fullmatch(window_dir.name)
        path = window_dir / "per_sample_metrics.npz"
        if match and path.is_file():
            numeric.append((int(match.group(1)), path))
    numeric.sort()
    if not numeric:
        return pd.DataFrame()
    all_paths = [native_path, *(path for _, path in numeric)]
    n = _cell_size(all_paths)
    native, native_counts, native_context, native_source = _metric_vector(
        native_path, n, metric
    )
    windows = np.asarray([window for window, _ in numeric], dtype=np.int64)
    errors = np.full((n, len(numeric)), np.nan, dtype=np.float64)
    effective = np.full_like(errors, np.nan)
    sources = {native_source}
    for column, (_window, path) in enumerate(numeric):
        values, _counts, eff, source = _metric_vector(path, n, metric)
        errors[:, column] = values
        effective[:, column] = eff
        sources.add(source)

    if not np.isfinite(native_context).any():
        # Compatibility fallback for caches created before full-native effective
        # lengths were persisted. Use the largest served numeric window per row.
        feasible_context = np.where(np.isfinite(errors), windows[None, :], np.nan)
        native_context = np.nanmax(feasible_context, axis=1)

    chosen = select_oracles(errors, windows, effective, native, native_context)
    keep = chosen["has_action"] & np.isfinite(native) & (native_counts > 0)
    source = "mase_gluonts" if "mase_gluonts" in sources else metric
    return pd.DataFrame({
        "model": model,
        "dataset_display": dataset,
        "term": term,
        "instance": np.arange(n, dtype=np.int64),
        "metric_source": source,
        "native_error": native,
        "native_context": native_context,
        "oracle_error": chosen["oracle_error"],
        "oracle_requested_context": chosen["oracle_requested_context"],
        "oracle_effective_context": chosen["oracle_effective_context"],
        "oracle_fraction": chosen["oracle_fraction"],
        "native_selected": chosen["native_selected"],
    }).loc[keep].reset_index(drop=True)


def _probabilities(values: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    counts, _ = np.histogram(values, bins=FRACTION_EDGES)
    total = counts.sum()
    return counts.astype(np.float64) / total if total else np.zeros(len(counts))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance in [0, 1], using base-2 logarithms."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        positive = left > 0
        return float(np.sum(left[positive] * np.log2(left[positive] / right[positive])))

    return float(math.sqrt(0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)))


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3 or np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")
    rank_left = pd.Series(left).rank(method="average").to_numpy()
    rank_right = pd.Series(right).rank(method="average").to_numpy()
    return float(np.corrcoef(rank_left, rank_right)[0, 1])


def summarize_cells(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    keys = ["model", "dataset_display", "term"]
    for key, group in frame.groupby(keys, sort=True):
        fractions = group["oracle_fraction"].to_numpy(dtype=float)
        contexts = group["oracle_effective_context"].to_numpy(dtype=float)
        probs = _probabilities(fractions)
        positive = probs > 0
        entropy = float(-np.sum(probs[positive] * np.log2(probs[positive])))
        normalized_entropy = entropy / math.log2(len(probs)) if len(probs) > 1 else 0.0
        modes = pd.Series(contexts).value_counts()
        rows.append({
            "model": key[0], "dataset_display": key[1], "term": key[2],
            "n_instances": int(len(group)),
            "oracle_context_mode": float(modes.index[0]),
            "mode_mass": float(modes.iloc[0] / len(group)),
            "native_selected_rate": float(group["native_selected"].mean()),
            "fraction_mean": float(np.mean(fractions)),
            "fraction_median": float(np.median(fractions)),
            "fraction_q10": float(np.quantile(fractions, 0.10)),
            "fraction_q25": float(np.quantile(fractions, 0.25)),
            "fraction_q75": float(np.quantile(fractions, 0.75)),
            "fraction_q90": float(np.quantile(fractions, 0.90)),
            "normalized_entropy": normalized_entropy,
            "n_distinct_contexts": int(modes.size),
        })
    return pd.DataFrame(rows)


def cross_model_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    keys = ["dataset_display", "term"]
    for (dataset, term), cell in frame.groupby(keys, sort=True):
        models = sorted(cell["model"].unique())
        for left_model, right_model in itertools.combinations(models, 2):
            columns = ["instance", "oracle_fraction", "oracle_effective_context"]
            left = cell[cell["model"] == left_model][columns]
            right = cell[cell["model"] == right_model][columns]
            merged = left.merge(right, on="instance", suffixes=("_left", "_right"))
            if merged.empty:
                continue
            lf = merged["oracle_fraction_left"].to_numpy(dtype=float)
            rf = merged["oracle_fraction_right"].to_numpy(dtype=float)
            lc = merged["oracle_effective_context_left"].to_numpy(dtype=float)
            rc = merged["oracle_effective_context_right"].to_numpy(dtype=float)
            ratio = np.divide(
                np.maximum(lc, rc), np.minimum(lc, rc),
                out=np.full(lc.shape, np.inf), where=np.minimum(lc, rc) > 0,
            )
            rows.append({
                "dataset_display": dataset,
                "term": term,
                "model_left": left_model,
                "model_right": right_model,
                "n_shared_instances": int(len(merged)),
                "spearman_fraction": _rank_correlation(lf, rf),
                "pearson_log2_fraction": float(np.corrcoef(
                    np.log2(np.maximum(lf, 1 / 32768)),
                    np.log2(np.maximum(rf, 1 / 32768)),
                )[0, 1]) if len(merged) >= 3 else float("nan"),
                "exact_effective_context_rate": float(np.mean(lc == rc)),
                "within_factor_2_rate": float(np.mean(ratio <= 2.0)),
                "mean_abs_fraction_difference": float(np.mean(np.abs(lf - rf))),
                "distribution_js_distance": jensen_shannon(
                    _probabilities(lf), _probabilities(rf)),
            })
    return pd.DataFrame(rows)


def summarize_model_pairs(agreement: pd.DataFrame) -> pd.DataFrame:
    if agreement.empty:
        return agreement.copy()
    rows: List[dict] = []
    for key, group in agreement.groupby(["model_left", "model_right"], sort=True):
        weights = group["n_shared_instances"].to_numpy(dtype=float)
        row = {"model_left": key[0], "model_right": key[1],
               "n_cells": int(len(group)), "n_shared_instances": int(weights.sum())}
        for column in (
            "spearman_fraction", "pearson_log2_fraction",
            "exact_effective_context_rate", "within_factor_2_rate",
            "mean_abs_fraction_difference", "distribution_js_distance",
        ):
            values = group[column].to_numpy(dtype=float)
            valid = np.isfinite(values) & (weights > 0)
            row[f"weighted_mean_{column}"] = (
                float(np.average(values[valid], weights=weights[valid]))
                if valid.any() else float("nan")
            )
            row[f"median_{column}"] = (
                float(np.median(values[valid])) if valid.any() else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def intra_dataset_stability(
    frame: pd.DataFrame, repeats: int, seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for key, group in frame.groupby(["model", "dataset_display", "term"], sort=True):
        values = group["oracle_fraction"].to_numpy(dtype=float)
        distances: List[float] = []
        if len(values) >= 4:
            half = len(values) // 2
            for _ in range(repeats):
                order = rng.permutation(len(values))
                distances.append(jensen_shannon(
                    _probabilities(values[order[:half]]),
                    _probabilities(values[order[half:2 * half]]),
                ))
        arr = np.asarray(distances, dtype=float)
        rows.append({
            "model": key[0], "dataset_display": key[1], "term": key[2],
            "n_instances": int(len(values)), "split_repeats": int(len(arr)),
            "split_half_js_mean": float(np.nanmean(arr)) if arr.size else float("nan"),
            "split_half_js_median": float(np.nanmedian(arr)) if arr.size else float("nan"),
            "split_half_js_q95": float(np.nanquantile(arr, 0.95)) if arr.size else float("nan"),
        })
    return pd.DataFrame(rows)


def _plot_pair_heatmap(pair_summary: pd.DataFrame, output: Path) -> None:
    if pair_summary.empty:
        return
    models = sorted(set(pair_summary["model_left"]) | set(pair_summary["model_right"]))
    index = {model: idx for idx, model in enumerate(models)}
    matrix = np.full((len(models), len(models)), np.nan)
    np.fill_diagonal(matrix, 1.0)
    for row in pair_summary.itertuples(index=False):
        i, j = index[row.model_left], index[row.model_right]
        value = row.weighted_mean_within_factor_2_rate
        matrix[i, j] = matrix[j, i] = value
    size = max(6.5, 0.58 * len(models))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(models)), models, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=8)
    ax.set_title("Oracle contexts within a factor of two (instance-weighted)")
    fig.colorbar(image, ax=ax, label="Agreement rate")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_model_fractions(frame: pd.DataFrame, output: Path) -> None:
    models = sorted(frame["model"].unique())
    values = [frame.loc[frame["model"] == model, "oracle_fraction"].to_numpy() for model in models]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(models)), 5.5))
    ax.boxplot(values, tick_labels=models, showfliers=False)
    ax.set_yscale("log", base=2)
    ax.set_ylim(max(1 / 32768, min(np.nanmin(v) for v in values) * 0.8), 1.1)
    ax.set_ylabel("Oracle effective context / native context")
    ax.set_title("Per-instance oracle context fractions")
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_stability(cell_summary: pd.DataFrame, stability: pd.DataFrame, output: Path) -> None:
    merged = cell_summary.merge(
        stability, on=["model", "dataset_display", "term", "n_instances"], how="inner"
    )
    if merged.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for model, group in merged.groupby("model"):
        ax.scatter(group["normalized_entropy"], group["split_half_js_mean"],
                   s=18, alpha=0.65, label=model)
    ax.set_xlabel("Oracle-distribution entropy (0=concentrated, 1=diffuse)")
    ax.set_ylabel("Random half-split JS distance (lower=stable)")
    ax.set_title("Within-dataset oracle heterogeneity and stability")
    if merged["model"].nunique() <= 12:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    datasets_root = _resolve_datasets_root(args.run_dir)
    cells = _discover_cells(datasets_root, args.models, args.datasets)
    if not cells:
        raise SystemExit("No complete cells with wfull_native per-sample metrics found.")
    frames: List[pd.DataFrame] = []
    for model, dataset, term, term_dir in cells:
        cell = load_cell(model, dataset, term, term_dir, args.metric)
        if not cell.empty:
            frames.append(cell)
    if not frames:
        raise SystemExit("Cells were found, but none had aligned numeric-window metrics.")
    frame = pd.concat(frames, ignore_index=True)
    output = Path(args.output_dir or (Path(args.run_dir) / "oracle_distribution_analysis"))
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    cell_summary = summarize_cells(frame)
    agreement = cross_model_agreement(frame)
    pair_summary = summarize_model_pairs(agreement)
    stability = intra_dataset_stability(frame, args.split_repeats, args.seed)

    frame.to_csv(output / "instance_oracles.csv.gz", index=False, compression="gzip")
    cell_summary.to_csv(output / "cell_summary.csv", index=False)
    agreement.to_csv(output / "cross_model_cell_agreement.csv", index=False)
    pair_summary.to_csv(output / "model_pair_summary.csv", index=False)
    stability.to_csv(output / "intra_dataset_stability.csv", index=False)
    _plot_pair_heatmap(pair_summary, figures / "cross_model_factor2_agreement.png")
    _plot_model_fractions(frame, figures / "oracle_fraction_by_model.png")
    _plot_stability(cell_summary, stability, figures / "intra_dataset_stability.png")

    print(
        f"Oracle analysis: {frame['model'].nunique()} models, "
        f"{frame[['dataset_display', 'term']].drop_duplicates().shape[0]} cells, "
        f"{len(frame)} model-instances -> {output}"
    )
    if not pair_summary.empty:
        columns = ["model_left", "model_right", "n_cells",
                   "weighted_mean_spearman_fraction",
                   "weighted_mean_within_factor_2_rate",
                   "weighted_mean_distribution_js_distance"]
        print(pair_summary[columns].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="logs/experiments/window_ablation_gifteval/general",
        help="Ablation general/ directory or its datasets/ child.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model display names; default: every cached model.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset display names; default: all active cached datasets.")
    parser.add_argument(
        "--metric", choices=["mase_gluonts_real", "mase_gluonts"],
        default="mase_gluonts_real",
    )
    parser.add_argument("--split-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.split_repeats < 1:
        parser.error("--split-repeats must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
