"""
Common machine-readable results schema (spec §9).

Every experiment — including the pre-existing attention masking (exp0) — writes
one row per (sample, intervention) into the SAME tabular schema so the final
cross-method analysis can compare them directly. Rows are buffered and flushed
to a per-cell ``results.csv``; ``load_results`` globs a run tree back into one
DataFrame.

Column conventions
------------------
* ``context_length`` is the EFFECTIVE (patch-aligned, cap-respecting) length;
  the requested value is kept in ``requested_context_length``.
* ``block_index`` is lookback-relative: 0 = most recent block, B-1 = most
  distant. Interventions that are not block-scoped (e.g. forecast lens) use -1.
* ``lookback_start`` / ``lookback_end`` are timestep distances from the
  forecast origin (start < end; block b of length P has start = b*P).
* ``layer`` is the adapter layer name, or "" for input-level methods.
* Unused numeric fields are NaN, never 0 (0 is a meaningful effect size).
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Exact spec §9 column set, plus bookkeeping columns appended after `seed`.
SCHEMA_COLUMNS: List[str] = [
    "model",
    "dataset",
    "sample_id",
    "context_length",
    "block_index",
    "lookback_start",
    "lookback_end",
    "method",
    "perturbation_type",
    "layer",
    "clean_loss",
    "intervened_loss",
    "loss_delta",
    "prediction_distance",
    "recovery_score",
    "attribution_score",
    "seed",
    # -- bookkeeping (not in the minimal spec table, always emitted) ----------
    "requested_context_length",
    "horizon",
    "metric",                        # loss metric name the *_loss columns use
    "prediction_distance_norm",      # normalized prediction distance (§3.5)
    "severity",                      # perturbation severity (noise scale, ...)
    "loss_recovery",                 # forecast-loss recovery (§5.6 companion)
]

_STR_COLUMNS = {"model", "dataset", "sample_id", "method", "perturbation_type",
                "layer", "metric"}
_INT_COLUMNS = {"context_length", "block_index", "lookback_start",
                "lookback_end", "seed", "requested_context_length", "horizon"}


def make_row(**kwargs) -> Dict[str, object]:
    """Build one schema row; unknown keys are rejected, missing ones defaulted."""
    unknown = set(kwargs) - set(SCHEMA_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown schema columns: {sorted(unknown)}")
    row: Dict[str, object] = {}
    for col in SCHEMA_COLUMNS:
        if col in kwargs and kwargs[col] is not None:
            v = kwargs[col]
            if col in _INT_COLUMNS:
                v = int(v)
            elif col in _STR_COLUMNS:
                v = str(v)
            else:
                v = float(v)
            row[col] = v
        else:
            row[col] = "" if col in _STR_COLUMNS else (
                -1 if col in _INT_COLUMNS else math.nan)
    return row


class ResultsWriter:
    """Buffered writer for one experiment cell (atomic on flush).

    Writes ``<cell_dir>/results.csv`` plus a ``done.json`` marker on
    :meth:`finalize`, mirroring the repo's per-cell resume convention: a cell
    is complete iff its done-marker exists.
    """

    def __init__(self, cell_dir: str):
        self.cell_dir = cell_dir
        self.path = os.path.join(cell_dir, "results.csv")
        self._rows: List[Dict[str, object]] = []
        os.makedirs(cell_dir, exist_ok=True)

    def add(self, **kwargs) -> None:
        self._rows.append(make_row(**kwargs))

    def add_rows(self, rows: Iterable[Dict[str, object]]) -> None:
        for r in rows:
            self.add(**r)

    def __len__(self) -> int:
        return len(self._rows)

    def flush(self) -> None:
        if not self._rows:
            return
        df = pd.DataFrame(self._rows, columns=SCHEMA_COLUMNS)
        tmp = self.path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.path)

    def finalize(self, extra_meta: Optional[dict] = None) -> None:
        self.flush()
        marker = {"n_rows": len(self._rows)}
        if extra_meta:
            marker.update(extra_meta)
        tmp = os.path.join(self.cell_dir, "done.json.tmp")
        with open(tmp, "w") as f:
            json.dump(marker, f, indent=2, default=_json_default)
        os.replace(tmp, os.path.join(self.cell_dir, "done.json"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def cell_done(cell_dir: str) -> bool:
    return os.path.exists(os.path.join(cell_dir, "done.json"))


def load_results(root: str, method: Optional[str] = None) -> pd.DataFrame:
    """Concatenate every ``results.csv`` under ``root`` into one DataFrame."""
    frames = []
    for dirpath, _dirs, files in os.walk(root):
        if "results.csv" in files:
            try:
                frames.append(pd.read_csv(os.path.join(dirpath, "results.csv")))
            except Exception as exc:  # noqa: BLE001 — a corrupt cell must not sink the analysis
                print(f"[schema] WARNING: unreadable {dirpath}/results.csv: {exc}")
    if not frames:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    if method is not None:
        df = df[df["method"] == method].reset_index(drop=True)
    return df
