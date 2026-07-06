"""Standalone: reproduce the GiftEval *leaderboard* aggregate from cached cells.

Why this exists
---------------
Our per-instance MASE (`mase_gluonts` / `mase_gluonts_real`) already matches the
leaderboard to <1%. What does NOT match is the headline number, because the
GiftEval leaderboard does not report ``geomean(MASE)``. It reports a
**seasonal-naive-normalised geometric mean**:

    score(model) = exp( mean_over_cells( log( MASE_model / MASE_seasonalnaive ) ) )

i.e. for every ``(dataset, term)`` cell it divides the model's MASE by the
Seasonal-Naive baseline's MASE on that same cell, THEN geomeans across cells.
That is why Seasonal Naive sits at exactly 1.000 on the board.

This script reads the ablation cache (the `metrics.json` files stage 3 writes)
and computes, per model:

  * raw geomean of the chosen MASE metric (what our summary currently reports),
  * the leaderboard-style naive-normalised geomean (what the board reports),
  * arithmetic mean, median, and cell count,
  * a coverage report (missing naive baselines, missing cells).

NO TSFM inference happens — it's a pure read over cached numbers.

Cache layout it walks (default `--metric mase_gluonts`):
    <run-dir>/datasets/<dataset>/<model>/t<term>/w<window>/metrics.json
    <run-dir>/datasets/<dataset>/_naive_seasonal/t<term>/metrics.json   (denominator)

Window selection per cell (`--window`):
    full  -> the largest w<N> present   (default; matches leaderboard full-context)
    best  -> the w<N> with the smallest chosen metric (oracle)
    <int> -> a specific window size

Caveats it prints (so the comparison is honest):
  * Legacy Seasonal-Naive caches (written before the gluonts-season fix) tiled the
    naive forecast with the PROJECT seasonality map (D->7, W->52, S->86400) instead
    of gluonts' (D->1, W->1, S->3600), so for D/W/S datasets their `mase_gluonts`
    denominator is not the leaderboard's baseline. Such cells are detected via the
    `_gluonts_naive_faithful` sentinel and flagged. Re-run stage 3 (or --force 3)
    to refresh them. (H/T/M/Q/etc. agree regardless, so they're always exact.)
  * The geomean is only comparable to the board if the cell set matches the
    board's. Missing cells are listed.

Usage (on the SERVER):
    python -m experiments.check_leaderboard_mase \
        --run-dir logs/experiments/window_ablation_gifteval/general
    python -m experiments.check_leaderboard_mase --metric mase_gluonts_real --window full
    python -m experiments.check_leaderboard_mase --models Chronos2-Small Moirai2-Small
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Frequencies where the project seasonality map disagrees with gluonts', so a
# LEGACY (non-faithful) Seasonal-Naive denominator would be the wrong baseline.
# Derived from the two maps in gifteval_mase.py (gluonts: D->1,W->1,S->3600) vs
# test_window_ablation_gifteval_v5._get_seasonality (project: D->7,W->52,S->86400).
# Only matters when the naive cache predates the gluonts-season fix (detected via
# the `_gluonts_naive_faithful` sentinel).
_SEASON_MISMATCH_SUFFIXES = ("-D", "-W", "-S")


def _term_dirs(model_dir: str) -> List[Tuple[str, str]]:
    """(term, path) for every t<term> subdir of a model dir."""
    out = []
    if not os.path.isdir(model_dir):
        return out
    for name in os.listdir(model_dir):
        if name.startswith("t") and os.path.isdir(os.path.join(model_dir, name)):
            out.append((name[1:], os.path.join(model_dir, name)))
    return out


def _windows(term_dir: str) -> List[Tuple[int, str]]:
    """(window_size, metrics_path) for every w<N> under a t<term> dir that has a
    metrics.json."""
    out = []
    for name in os.listdir(term_dir):
        m = re.fullmatch(r"w(\d+)", name)
        if not m:
            continue
        mp = os.path.join(term_dir, name, "metrics.json")
        if os.path.isfile(mp):
            out.append((int(m.group(1)), mp))
    return sorted(out)


def _load(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _pick_window(windows: List[Tuple[int, str]], how: str, metric: str
                 ) -> Optional[Tuple[int, dict]]:
    """Return (window_size, metrics_dict) for the chosen selection strategy, or
    None if no window has a finite value for `metric`."""
    loaded = []
    for w, mp in windows:
        d = _load(mp)
        if d is None:
            continue
        v = d.get(metric)
        if v is None or not math.isfinite(v):
            continue
        loaded.append((w, d, v))
    if not loaded:
        return None
    if how == "full":
        w, d, _ = max(loaded, key=lambda t: t[0])
    elif how == "best":
        w, d, _ = min(loaded, key=lambda t: t[2])
    else:  # explicit int
        want = int(how)
        match = [t for t in loaded if t[0] == want]
        if not match:
            return None
        w, d, _ = match[0]
    return w, d


def _geomean(vals: List[float]) -> float:
    vals = [v for v in vals if math.isfinite(v) and v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def collect(run_dir: str, metric: str, window: str,
            models_filter: Optional[List[str]]) -> None:
    datasets_root = os.path.join(run_dir, "datasets")
    if not os.path.isdir(datasets_root):
        raise SystemExit(
            f"No 'datasets/' under {run_dir!r}. Point --run-dir at the folder that "
            f"contains datasets/ (e.g. .../window_ablation_gifteval/general).")

    # per model: list of (dataset, term, model_mase, naive_mase_or_None, mismatch)
    rows: Dict[str, List[tuple]] = defaultdict(list)
    naive_missing: Dict[str, List[str]] = defaultdict(list)

    for dataset in sorted(os.listdir(datasets_root)):
        ddir = os.path.join(datasets_root, dataset)
        if not os.path.isdir(ddir):
            continue

        # --- naive baseline per term (the denominator) --------------------
        naive_dir = os.path.join(ddir, "_naive_seasonal")
        naive_by_term: Dict[str, Optional[float]] = {}
        naive_faithful: Dict[str, bool] = {}
        for term, tdir in _term_dirs(naive_dir):
            nd = _load(os.path.join(tdir, "metrics.json"))
            v = nd.get(metric) if nd else None
            naive_by_term[term] = v if (v is not None and math.isfinite(v)) else None
            naive_faithful[term] = bool(nd.get("_gluonts_naive_faithful")) if nd else False

        # A legacy (non-faithful) naive cache only mis-normalises D/W/S datasets,
        # where the project and gluonts seasons differ. Faithful caches are exact.
        legacy_suffix = dataset.endswith(_SEASON_MISMATCH_SUFFIXES)

        # --- each model -----------------------------------------------------
        for model in sorted(os.listdir(ddir)):
            if model == "_naive_seasonal":
                continue
            mdir = os.path.join(ddir, model)
            if not os.path.isdir(mdir):
                continue
            if models_filter and model not in models_filter:
                continue
            for term, tdir in _term_dirs(mdir):
                picked = _pick_window(_windows(tdir), window, metric)
                if picked is None:
                    continue
                _w, md = picked
                mv = md.get(metric)
                if mv is None or not math.isfinite(mv):
                    continue
                nv = naive_by_term.get(term)
                if nv is None:
                    naive_missing[model].append(f"{dataset}/t{term}")
                # Flag only when the denominator is actually wrong: legacy cache
                # AND a frequency where the two season maps disagree.
                mismatch = legacy_suffix and not naive_faithful.get(term, False)
                rows[model].append((dataset, term, float(mv), nv, mismatch))

    if not rows:
        raise SystemExit(f"No model cells found under {datasets_root} for metric={metric!r}.")

    _report(rows, naive_missing, metric, window)


def _report(rows, naive_missing, metric, window) -> None:
    print("=" * 92)
    print(f"GiftEval leaderboard-style aggregate  |  metric={metric}  window={window}")
    print("=" * 92)
    print(f"{'model':<22} {'cells':>5} {'raw_gmean':>10} {'norm_gmean':>11} "
          f"{'mean':>9} {'median':>9} {'no_naive':>8} {'season?':>8}")
    print("-" * 92)

    board = []
    for model in sorted(rows):
        cells = rows[model]
        raw = [mv for _, _, mv, _, _ in cells]
        ratios = [mv / nv for _, _, mv, nv, _ in cells if nv]
        n_mismatch = sum(1 for _, _, _, nv, mm in cells if nv and mm)
        srt = sorted(raw)
        median = srt[len(srt) // 2] if srt else float("nan")
        raw_g = _geomean(raw)
        norm_g = _geomean(ratios)
        board.append((model, norm_g))
        print(f"{model:<22} {len(cells):>5} {raw_g:>10.4f} {norm_g:>11.4f} "
              f"{(sum(raw)/len(raw)):>9.4f} {median:>9.4f} "
              f"{len(naive_missing.get(model, [])):>8} {n_mismatch:>8}")

    print("-" * 92)
    print("norm_gmean = leaderboard value: exp(mean(log( MASE_model / MASE_seasonalnaive )))")
    print("raw_gmean  = exp(mean(log(MASE_model)))  -- NOT normalised (what the summary reports)")
    print()

    # coverage / caveat detail
    any_missing = any(naive_missing.values())
    any_mismatch = any(mm for cells in rows.values() for *_, mm in cells)
    if any_missing:
        print("!! Cells with NO naive baseline (excluded from norm_gmean):")
        for model, miss in naive_missing.items():
            if miss:
                shown = ", ".join(miss[:8]) + (" ..." if len(miss) > 8 else "")
                print(f"   {model}: {len(miss)} cells -> {shown}")
        print()
    if any_mismatch:
        print("!! 'season?' counts D/W/S cells whose Seasonal-Naive cache is LEGACY")
        print("   (no _gluonts_naive_faithful sentinel): its denominator tiled with the")
        print("   PROJECT season map, not gluonts' (D->1,W->1,S->3600). Re-run stage 3")
        print("   (or --force 3) to refresh those cells for an exact leaderboard match.")
        print()
    print("Reminder: norm_gmean only matches the board if the cell set matches the")
    print("board's ~97 (dataset x term) configs. Check 'cells' per model above.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="logs/experiments/window_ablation_gifteval/general",
                   help="Folder containing datasets/ (the shared 'general' ablation tree).")
    p.add_argument("--metric", default="mase_gluonts",
                   choices=["mase_gluonts", "mase_gluonts_real", "mase"],
                   help="Metric column to aggregate (default: mase_gluonts).")
    p.add_argument("--window", default="full",
                   help="Per-cell window: 'full' (max, default), 'best' (argmin), or an int.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Restrict to these model display names (default: all found).")
    return p.parse_args()


def main():
    args = parse_args()
    collect(args.run_dir, args.metric, args.window, args.models)


if __name__ == "__main__":
    main()
