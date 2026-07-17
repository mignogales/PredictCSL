"""Compare TimesFM sanity aggregation against the main ablation cache.

Use this after per-cell forecast checks show tiny differences. It reads:

  * sanity comparison.csv from logs/experiments/sanity_leaderboard/<tag>/, and
  * main v5 metrics.json cells from logs/experiments/window_ablation_gifteval/...

No model inference happens. The report focuses on aggregation causes: cell-set
coverage, chosen main window, seasonal-naive denominators, per-cell ratios, and
geomeans.

Examples
--------
    python -m experiments.compare_timesfm_aggregation \
        --sanity-dir logs/experiments/sanity_leaderboard/timesfm-2.5-200m-pytorch \
        --main-run-dir logs/experiments/window_ablation_gifteval/general \
        --main-model TimesFM2.5-200M

    python -m experiments.compare_timesfm_aggregation \
        --sanity-dir logs/experiments/sanity_leaderboard/TimesFM2.5-200M_pipeline_full \
        --main-window 8192 --top 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _float(x) -> Optional[float]:
    return float(x) if _finite(x) else None


def _geomean(vals: List[float]) -> float:
    vals = [float(v) for v in vals if _finite(v) and float(v) > 0.0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _mean(vals: List[float]) -> float:
    vals = [float(v) for v in vals if _finite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _load_sanity(sanity_dir: str) -> Dict[str, dict]:
    path = os.path.join(sanity_dir, "comparison.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"No comparison.csv in sanity dir: {sanity_dir}")
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = row["dataset"]
            rows[key] = {
                "mase": _float(row.get("mase_ours")),
                "mase_official": _float(row.get("mase_official")),
                "sn": _float(row.get("sn_official")),
                "norm": _float(row.get("norm_ours")),
                "norm_official": _float(row.get("norm_official")),
            }
    return rows


def _term_dirs(model_dir: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(model_dir):
        return []
    out = []
    for name in os.listdir(model_dir):
        if name.startswith("t") and os.path.isdir(os.path.join(model_dir, name)):
            out.append((name[1:], os.path.join(model_dir, name)))
    return out


def _windows(term_dir: str) -> List[Tuple[int, str]]:
    out = []
    if not os.path.isdir(term_dir):
        return out
    for name in os.listdir(term_dir):
        m = re.fullmatch(r"w(\d+)", name)
        if not m:
            continue
        path = os.path.join(term_dir, name, "metrics.json")
        if os.path.isfile(path):
            out.append((int(m.group(1)), path))
    return sorted(out)


def _pick_main_metric(term_dir: str, how: str, metric: str,
                      prefer_real: bool) -> Optional[Tuple[int, float, dict]]:
    loaded = []
    for w, path in _windows(term_dir):
        d = _load_json(path)
        if not d:
            continue
        v = None
        if prefer_real:
            v = _float(d.get("mase_gluonts_real"))
        if v is None:
            v = _float(d.get(metric))
        if v is not None:
            loaded.append((w, v, d))
    if not loaded:
        return None
    if how == "full":
        return max(loaded, key=lambda t: t[0])
    if how == "best":
        return min(loaded, key=lambda t: t[1])
    want = int(how)
    matches = [t for t in loaded if t[0] == want]
    return matches[0] if matches else None


def _load_main(run_dir: str, model: str, metric: str, window: str,
               prefer_real: bool) -> Dict[str, dict]:
    root = os.path.join(run_dir, "datasets")
    if not os.path.isdir(root):
        raise SystemExit(f"No datasets/ under main run dir: {run_dir}")
    rows = {}
    for dataset in sorted(os.listdir(root)):
        ddir = os.path.join(root, dataset)
        if not os.path.isdir(ddir):
            continue

        naive_by_term = {}
        naive_dir = os.path.join(ddir, "_naive_seasonal")
        for term, tdir in _term_dirs(naive_dir):
            nd = _load_json(os.path.join(tdir, "metrics.json")) or {}
            naive_by_term[term] = {
                "mase_gluonts": _float(nd.get("mase_gluonts")),
                "mase_gluonts_real": _float(nd.get("mase_gluonts_real")),
                "faithful": bool(nd.get("_gluonts_naive_faithful")),
                "ver": nd.get("_gluonts_naive_ver"),
            }

        model_dir = os.path.join(ddir, model)
        for term, tdir in _term_dirs(model_dir):
            picked = _pick_main_metric(tdir, window, metric, prefer_real)
            if picked is None:
                continue
            w, mase, md = picked
            ninfo = naive_by_term.get(term, {})
            # Prefer the actual naive gluonts machinery value if present; fall
            # back to the port so older caches can still be diagnosed.
            sn = ninfo.get("mase_gluonts_real") or ninfo.get("mase_gluonts")
            key = f"{dataset}/{term}"
            rows[key] = {
                "mase": mase,
                "sn": sn,
                "norm": (mase / sn if sn and sn > 0 else None),
                "window": w,
                "raw_metric": md,
                "naive": ninfo,
            }
    return rows


def _summarize(label: str, rows: Dict[str, dict]) -> dict:
    mase = [r["mase"] for r in rows.values() if r.get("mase") is not None]
    norm = [r["norm"] for r in rows.values() if r.get("norm") is not None]
    sn = [r["sn"] for r in rows.values() if r.get("sn") is not None]
    return {
        "label": label,
        "cells": len(rows),
        "mase_geomean": _geomean(mase),
        "norm_geomean": _geomean(norm),
        "sn_geomean": _geomean(sn),
        "norm_mean": _mean(norm),
    }


def _print_summary(title: str, s: dict) -> None:
    print(f"{title:<18} cells={s['cells']:>3}  "
          f"raw_g={s['mase_geomean']:.6f}  "
          f"norm_g={s['norm_geomean']:.6f}  "
          f"sn_g={s['sn_geomean']:.6f}  norm_mean={s['norm_mean']:.6f}")


def _row_diff(sanity_rows: Dict[str, dict], main_rows: Dict[str, dict],
              key: str) -> dict:
    sr = sanity_rows[key]
    mr = main_rows[key]
    sm, mm = sr.get("mase"), mr.get("mase")
    ss, ms = sr.get("sn"), mr.get("sn")
    sn, mn = sr.get("norm"), mr.get("norm")
    return {
        "dataset": key,
        "sanity_mase": sm,
        "main_mase": mm,
        "mase_abs_diff": abs(mm - sm) if sm is not None and mm is not None else None,
        "mase_rel_pct": (100.0 * (mm - sm) / sm
                         if sm not in (None, 0.0) and mm is not None else None),
        "sanity_sn": ss,
        "main_sn": ms,
        "sn_abs_diff": abs(ms - ss) if ss is not None and ms is not None else None,
        "sanity_norm": sn,
        "main_norm": mn,
        "norm_abs_diff": abs(mn - sn) if sn is not None and mn is not None else None,
        "main_window": mr.get("window"),
        "naive_faithful": mr.get("naive", {}).get("faithful"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sanity-dir", required=True,
                    help="logs/experiments/sanity_leaderboard/<tag> containing comparison.csv")
    ap.add_argument("--main-run-dir",
                    default="logs/experiments/window_ablation_gifteval/general",
                    help="Main ablation run dir containing datasets/")
    ap.add_argument("--main-model", default="TimesFM2.5-200M")
    ap.add_argument("--main-metric", default="mase_gluonts_real",
                    choices=["mase_gluonts", "mase_gluonts_real", "mase"])
    ap.add_argument("--main-window", default="full",
                    help="'full' = largest cached window, 'best', or an integer")
    ap.add_argument("--no-prefer-real", action="store_true",
                    help="Do not fall back to mase_gluonts_real when available.")
    ap.add_argument("--top", type=int, default=15,
                    help="Show top-N cells by normalised absolute difference.")
    ap.add_argument("--out", default=None,
                    help="Optional JSON report path.")
    args = ap.parse_args()

    sanity_rows = _load_sanity(args.sanity_dir)
    main_rows = _load_main(
        args.main_run_dir, args.main_model, args.main_metric, args.main_window,
        prefer_real=not args.no_prefer_real)

    sanity_keys = set(sanity_rows)
    main_keys = set(main_rows)
    overlap = sorted(sanity_keys & main_keys)
    sanity_only = sorted(sanity_keys - main_keys)
    main_only = sorted(main_keys - sanity_keys)

    print("=" * 100)
    print("Aggregation Comparison")
    print("=" * 100)
    _print_summary("sanity all", _summarize("sanity_all", sanity_rows))
    _print_summary("main all", _summarize("main_all", main_rows))
    _print_summary("sanity overlap", _summarize("sanity_overlap", {k: sanity_rows[k] for k in overlap}))
    _print_summary("main overlap", _summarize("main_overlap", {k: main_rows[k] for k in overlap}))
    print(f"\noverlap={len(overlap)}  sanity_only={len(sanity_only)}  main_only={len(main_only)}")
    if sanity_only:
        print("sanity_only:", ", ".join(sanity_only[:20]) + (" ..." if len(sanity_only) > 20 else ""))
    if main_only:
        print("main_only:", ", ".join(main_only[:20]) + (" ..." if len(main_only) > 20 else ""))

    diffs = [_row_diff(sanity_rows, main_rows, k) for k in overlap]
    diffs_sorted = sorted(
        diffs,
        key=lambda r: (
            -1.0 if r["norm_abs_diff"] is None else -float(r["norm_abs_diff"]),
            r["dataset"],
        ),
    )
    print(f"\nTop {args.top} normalised cell differences:")
    print(f"{'dataset':<34} {'san_norm':>10} {'main_norm':>10} {'abs':>10} "
          f"{'san_mase':>10} {'main_mase':>10} {'san_sn':>10} {'main_sn':>10} {'w':>7} {'faith':>5}")
    for r in diffs_sorted[: args.top]:
        def fmt(v):
            return f"{v:.5g}" if v is not None and _finite(v) else "nan"
        print(f"{r['dataset']:<34} {fmt(r['sanity_norm']):>10} "
              f"{fmt(r['main_norm']):>10} {fmt(r['norm_abs_diff']):>10} "
              f"{fmt(r['sanity_mase']):>10} {fmt(r['main_mase']):>10} "
              f"{fmt(r['sanity_sn']):>10} {fmt(r['main_sn']):>10} "
              f"{str(r['main_window']):>7} {str(r['naive_faithful']):>5}")

    report = {
        "summary": {
            "sanity_all": _summarize("sanity_all", sanity_rows),
            "main_all": _summarize("main_all", main_rows),
            "sanity_overlap": _summarize("sanity_overlap", {k: sanity_rows[k] for k in overlap}),
            "main_overlap": _summarize("main_overlap", {k: main_rows[k] for k in overlap}),
            "overlap": len(overlap),
            "sanity_only": sanity_only,
            "main_only": main_only,
        },
        "diffs": diffs_sorted,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
