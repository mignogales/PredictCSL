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

The denominator always comes from the shipped official GIFT-Eval
``leaderboard_reference/seasonal_naive_all_results.csv``.

Window selection per cell (`--window`):
    full  -> the largest w<N> present   (default; matches leaderboard full-context)
    best  -> the w<N> with the smallest chosen metric (oracle)
    <int> -> a specific window size

Caveat it prints (so the comparison is honest):
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

from experiments.gifteval_reference import published_naive_by_display


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


def _cell_value(md: dict, metric: str, prefer_real: bool) -> Optional[float]:
    """The MASE value for a cell. With ``prefer_real`` we use the ACTUAL gluonts
    machinery value (``mase_gluonts_real``) when it's finite, falling back to the
    requested ``metric`` (the numpy port) otherwise — so a run mixing fresh cells
    (real populated) and cached cells (port only) still aggregates cleanly."""
    if prefer_real:
        rv = md.get("mase_gluonts_real")
        if rv is not None and math.isfinite(rv):
            return float(rv)
    v = md.get(metric)
    return float(v) if (v is not None and math.isfinite(v)) else None


def _pick_window(windows: List[Tuple[int, str]], how: str, metric: str,
                 prefer_real: bool) -> Optional[Tuple[int, dict, float]]:
    """Return (window_size, metrics_dict, value) for the chosen selection strategy,
    or None if no window has a finite value."""
    loaded = []
    for w, mp in windows:
        d = _load(mp)
        if d is None:
            continue
        v = _cell_value(d, metric, prefer_real)
        if v is None:
            continue
        loaded.append((w, d, v))
    if not loaded:
        return None
    if how == "full":
        return max(loaded, key=lambda t: t[0])
    elif how == "best":
        return min(loaded, key=lambda t: t[2])
    else:  # explicit int
        want = int(how)
        match = [t for t in loaded if t[0] == want]
        return match[0] if match else None


def _geomean(vals: List[float]) -> float:
    vals = [v for v in vals if math.isfinite(v) and v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def collect(run_dir: str, metric: str, window: str,
            models_filter: Optional[List[str]], prefer_real: bool,
            common_cells: bool, degen_mase: float, breakdown: int) -> None:
    datasets_root = os.path.join(run_dir, "datasets")
    if not os.path.isdir(datasets_root):
        raise SystemExit(
            f"No 'datasets/' under {run_dir!r}. Point --run-dir at the folder that "
            f"contains datasets/ (e.g. .../window_ablation_gifteval/general).")

    # per model: list of Cell(dataset, term, model_mase,
    #                         published_naive_mase_or_None, degenerate)
    rows: Dict[str, List[tuple]] = defaultdict(list)
    naive_missing: Dict[str, List[str]] = defaultdict(list)
    published_naive = published_naive_by_display()

    for dataset in sorted(os.listdir(datasets_root)):
        ddir = os.path.join(datasets_root, dataset)
        if not os.path.isdir(ddir):
            continue

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
                picked = _pick_window(_windows(tdir), window, metric, prefer_real)
                if picked is None:
                    continue
                _w, _md, mv = picked
                nv = published_naive.get(
                    f"{dataset}/t{term}", {}).get("mase_gluonts_real")
                if nv is None:
                    naive_missing[model].append(f"{dataset}/t{term}")
                # Degenerate: model or naive MASE is astronomically large, i.e. the
                # per-instance seasonal error hit its 1e-9 floor (near-constant /
                # intermittent series). These cancel in the ratio but wreck raw
                # geomean/mean — flag so they can be excluded from the diagnosis.
                degen = mv > degen_mase or (nv is not None and nv > degen_mase)
                rows[model].append((dataset, term, mv, nv, degen))

    if not rows:
        raise SystemExit(f"No model cells found under {datasets_root} for metric={metric!r}.")

    # Optionally restrict to the (dataset, term) cells present for EVERY model, so
    # the per-model geomeans are computed over an identical set and are actually
    # comparable (raw/norm then collapses to one shared constant).
    if common_cells and len(rows) > 1:
        cell_sets = [set((c[0], c[1]) for c in cells) for cells in rows.values()]
        shared = set.intersection(*cell_sets)
        for model in rows:
            rows[model] = [c for c in rows[model] if (c[0], c[1]) in shared]
        # Rebuild naive_missing from the intersected rows, else the no_naive column
        # keeps its stale pre-intersection count (cells already dropped here).
        naive_missing.clear()
        for model, cells in rows.items():
            for ds, tm, _mv, nv, _dg in cells:
                if nv is None:
                    naive_missing[model].append(f"{ds}/t{tm}")
        print(f"[--common-cells] intersecting to {len(shared)} cells shared by all "
              f"{len(rows)} models.\n")

    _report(rows, naive_missing, metric, window, prefer_real, degen_mase, breakdown)


def _report(rows, naive_missing, metric, window, prefer_real, degen_mase,
            breakdown) -> None:
    print("=" * 100)
    src = "mase_gluonts_real (gluonts machinery), port fallback" if prefer_real else metric
    print(f"GiftEval leaderboard-style aggregate  |  value={src}  window={window}")
    print("=" * 100)
    # norm_g   : the leaderboard value (geomean of model/naive over ALL cells)
    # norm_g*  : same but EXCLUDING degenerate (seasonal-error-floor) cells
    # G_naive  : geomean of the naive denominator over the normalised cells
    #            (= raw_g/norm_g; a shared constant iff cells match across models)
    print(f"{'model':<20} {'cells':>5} {'raw_g':>8} {'norm_g':>8} {'norm_g*':>8} "
          f"{'G_naive':>8} {'median':>8} {'degen':>6} {'no_nv':>6}")
    print("-" * 100)

    for model in sorted(rows):
        cells = rows[model]
        raw = [mv for _, _, mv, _, _ in cells]
        ratios = [mv / nv for _, _, mv, nv, _ in cells if nv]
        ratios_clean = [mv / nv for _, _, mv, nv, dg in cells if nv and not dg]
        naive_vals = [nv for _, _, _, nv, _ in cells if nv]
        n_degen = sum(1 for *_, dg in cells if dg)
        srt = sorted(raw)
        median = srt[len(srt) // 2] if srt else float("nan")
        raw_g = _geomean(raw)
        norm_g = _geomean(ratios)
        norm_g_clean = _geomean(ratios_clean)
        g_naive = _geomean(naive_vals)
        print(f"{model:<20} {len(cells):>5} {raw_g:>8.4f} {norm_g:>8.4f} "
              f"{norm_g_clean:>8.4f} {g_naive:>8.4f} {median:>8.4f} "
              f"{n_degen:>6} {len(naive_missing.get(model, [])):>6}")

    print("-" * 100)
    print("norm_g  = LEADERBOARD value: exp(mean(log( MASE_model / MASE_seasonalnaive )))")
    print("norm_g* = same, excluding degenerate cells (model or naive MASE > "
          f"{degen_mase:g}; seasonal-error floor)")
    print("G_naive = geomean of the seasonal-naive denominator; raw_g / norm_g")
    print()

    # --- Per-dataset breakdown: which cells drive the geomean --------------
    # Sorted by ratio: the smallest ratios (models crushing seasonal naive) pull
    # the geomean DOWN; compare these against the public per-dataset leaderboard
    # numbers to find where your run diverges (missing hard cells, different
    # forecasts, or degenerate series).
    if breakdown > 0:
        for model in sorted(rows):
            scored = sorted(
                ((c[0], c[1], c[2], c[3], c[2] / c[3], c[4])
                 for c in rows[model] if c[3]),
                key=lambda t: t[4])
            if not scored:
                continue
            print(f"--- {model}: {len(scored)} normalised cells, "
                  f"{breakdown} lowest & highest ratios (model/naive) ---")
            print(f"    {'dataset/term':<28} {'model':>12} {'naive':>12} {'ratio':>8}  flag")
            def _line(t):
                ds, tm, mv, nv, rt, dg = t
                flag = "DEGEN" if dg else ""
                print(f"    {ds + '/t' + tm:<28} {mv:>12.3g} {nv:>12.3g} {rt:>8.3f}  {flag}")
            if len(scored) <= 2 * breakdown:
                for t in scored:                       # few cells: show all once
                    _line(t)
            else:
                for t in scored[:breakdown]:
                    _line(t)
                print(f"    {'...':<28}")
                for t in scored[-breakdown:]:
                    _line(t)
            print()

    # --- coverage / caveat detail -----------------------------------------
    any_missing = any(naive_missing.values())
    if any_missing:
        print("!! Cells with NO naive baseline (excluded from norm_g):")
        for model, miss in naive_missing.items():
            if miss:
                shown = ", ".join(miss[:8]) + (" ..." if len(miss) > 8 else "")
                print(f"   {model}: {len(miss)} cells -> {shown}")
        print()
    print("Reminder: norm_g matches the board ONLY over the board's exact 97 configs.")
    print("If G_naive differs across models, their cell sets differ — use --common-cells")
    print("for an apples-to-apples cross-model comparison.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="logs/experiments/window_ablation_gifteval/general",
                   help="Folder containing datasets/ (the shared 'general' ablation tree).")
    p.add_argument("--metric", default="mase_gluonts",
                   choices=["mase_gluonts", "mase_gluonts_real"],
                   help="GluonTS MASE column to aggregate (default: "
                        "mase_gluonts). The legacy project MASE is intentionally "
                        "excluded because it is incompatible with the published "
                        "GIFT-Eval denominator.")
    p.add_argument("--window", default="full",
                   help="Per-cell window: 'full' (max, default), 'best' (argmin), or an int.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Restrict to these model display names (default: all found).")
    p.add_argument("--prefer-real", action="store_true",
                   help="Use the ACTUAL gluonts-machinery value (mase_gluonts_real) "
                        "per cell when finite, falling back to --metric. Rules out "
                        "port-vs-machinery as the source of any leaderboard gap.")
    p.add_argument("--common-cells", action="store_true",
                   help="Restrict to the (dataset, term) cells present for EVERY "
                        "selected model, so cross-model geomeans are comparable.")
    p.add_argument("--degen-mase", type=float, default=100.0,
                   help="A cell whose model or naive MASE exceeds this is treated as "
                        "degenerate (seasonal-error floor); reported and excluded "
                        "from norm_g* (default: 100).")
    p.add_argument("--breakdown", type=int, default=0, metavar="N",
                   help="Print, per model, the N lowest- and highest-ratio cells so "
                        "you can see which datasets drive the geomean (default: 0=off).")
    return p.parse_args()


def main():
    args = parse_args()
    collect(args.run_dir, args.metric, args.window, args.models,
            args.prefer_real, args.common_cells, args.degen_mase, args.breakdown)


if __name__ == "__main__":
    main()
