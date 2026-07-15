"""
Aggregation of raw schema rows into analysis-ready tables (spec §12 order:
seeds within sample -> samples -> datasets).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from context_interpretability.schema import load_results

BLOCK_KEYS = ["model", "dataset", "method", "perturbation_type",
              "context_length", "block_index", "lookback_start",
              "lookback_end", "layer"]


def collapse_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Average repeated perturbation seeds WITHIN each sample first — seeds are
    not independent observations (spec §12)."""
    if df.empty:
        return df
    keys = [k for k in BLOCK_KEYS if k in df.columns] + ["sample_id",
                                                         "severity"]
    num = [c for c in df.columns
           if c not in keys + ["seed"] and pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(keys, dropna=False)[num].mean().reset_index()
    return grouped


def block_effects(df: pd.DataFrame, value: str = "loss_delta"
                  ) -> pd.DataFrame:
    """Sample-level paired effects per (block, context, method): one row per
    group with the per-sample values as a list column (for the stats layer)."""
    d = collapse_seeds(df)
    if d.empty:
        return d
    keys = [k for k in BLOCK_KEYS if k in d.columns]
    rows = []
    for key, g in d.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        rec["values"] = g[value].to_numpy()
        rec["n_samples"] = len(g)
        rows.append(rec)
    return pd.DataFrame(rows)


def heatmap_matrix(df: pd.DataFrame, value: str = "loss_delta",
                   agg: str = "mean") -> pd.DataFrame:
    """(lookback_start x context_length) pivot of the chosen effect."""
    d = collapse_seeds(df)
    if d.empty:
        return pd.DataFrame()
    return d.pivot_table(index="lookback_start", columns="context_length",
                         values=value, aggfunc=agg)


def layer_block_matrix(df: pd.DataFrame, value: str = "recovery_score",
                       context_length: Optional[int] = None) -> pd.DataFrame:
    """(layer x lookback_start) pivot for the activation-patching heatmap."""
    d = collapse_seeds(df)
    if context_length is not None:
        d = d[d["context_length"] == context_length]
    if d.empty:
        return pd.DataFrame()
    piv = d.pivot_table(index="layer", columns="lookback_start", values=value,
                        aggfunc="mean")
    return piv.reindex(sorted(piv.index, key=_layer_sort_key))


def _layer_sort_key(name: str):
    import re
    m = re.search(r"(\d+)$", str(name))
    return (0, int(m.group(1))) if m else (1, str(name))


def lens_matrices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Forecast-lens heatmaps A/B/C: layer x context_length pivots."""
    d = df[df["method"] == "forecast_lens"]
    if d.empty:
        return {}
    out = {}
    for key, col in [("error", "intervened_loss"),
                     ("dist_to_full", "prediction_distance"),
                     ("dist_to_same_norm", "prediction_distance_norm")]:
        piv = d.pivot_table(index="layer", columns="context_length",
                            values=col, aggfunc="mean")
        out[key] = piv.reindex(sorted(piv.index, key=_layer_sort_key))
    return out


def sufficient_context_from_curve(windows: List[int], curve: np.ndarray,
                                  tolerance: float = 0.05) -> int:
    best = np.nanmin(curve)
    for w, e in zip(windows, curve):
        if e <= (1.0 + tolerance) * best:
            return int(w)
    return int(windows[-1])


def instance_sufficient_context(loss_matrix: np.ndarray, windows: List[int],
                                tolerance: float = 0.05) -> np.ndarray:
    """Per-instance sufficient context from a (N, K) loss matrix (figure 10)."""
    out = np.empty(loss_matrix.shape[0], dtype=np.int64)
    for i in range(loss_matrix.shape[0]):
        out[i] = sufficient_context_from_curve(windows, loss_matrix[i],
                                               tolerance)
    return out


def load_run(run_dir: str) -> pd.DataFrame:
    """All schema rows of one run directory."""
    return load_results(run_dir)


def method_block_profile(df: pd.DataFrame, method: str,
                         value: str, context_length: int) -> pd.Series:
    """Mean effect vs lookback_start for one method at one context length —
    the common currency of the cross-method comparison (spec §8.6)."""
    d = collapse_seeds(df[(df["method"] == method)
                          & (df["context_length"] == context_length)])
    if d.empty:
        return pd.Series(dtype=float)
    return d.groupby("lookback_start")[value].mean()
