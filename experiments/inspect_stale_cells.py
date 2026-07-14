"""
For the cells ``analyze_leftover_cells`` reports as ``no_gluonts_curve``, look at
the ACTUAL on-disk per-window cache and explain why the cheap backfill can't fill
them — so you know whether ``--force 3`` (backfill) suffices or the cell needs true
re-inference (delete + rerun).

Background: stage 4 needs a non-NaN ``real_curve_gluonts[_real]`` in each
``compare_*.npz``. That curve is assembled from every window-cell's
``mase_gluonts`` column, which stage 3 backfills from the cached per-instance MAE
(``per_sample_metrics.npz``) — NO TSFM re-inference. ``_backfill_mase_gluonts``
returns ``None`` (leaving the cell NaN) when that per-sample cache is missing OR
its instance count doesn't match the dataset. If a cell's windows are all in that
state, its whole gluonts curve stays NaN and the bar drops it.

This tool, per model, finds the offending cells and for each window reports:
  metrics.json present? mase_gluonts value / _mase_gluonts_ver / real-standin?
  per_sample_metrics.npz present? has 'mae'? shape vs the servable-instance count.
Then a per-cell VERDICT:
  * backfillable   -> per_sample present & matches: a ``--force 3`` run WILL fix it
                      (if you already ran one and it's still NaN, check the ver /
                      that stage 3 actually re-ran — not skipped by the done-marker)
  * needs-reinfer  -> per_sample missing/mismatched: delete the cell dir(s) so v5
                      recomputes fresh (emitted as rm -rf lines)

Run on the SERVER. Examples:
  python -m experiments.inspect_stale_cells --model Chronos2-Small
  python -m experiments.inspect_stale_cells --model Chronos2-Small --emit-rm > refresh.sh
  # sanity-compare against a model that works:
  python -m experiments.inspect_stale_cells --model Chronos2-Small --ref-model Chronos2-Base
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments import datasets_config

try:
    from colorama import Fore, init as _cinit
    _cinit()
except Exception:
    class _F:
        def __getattr__(self, _):
            return ""
    Fore = _F()                                     # type: ignore

DEFAULT_RUN_DIR = "logs/experiments/window_ablation_gifteval/general"
CURVE_KEY = {"mase_gluonts": "real_curve_gluonts",
             "mase_gluonts_real": "real_curve_gluonts_real"}


def _npz_filename(disp: str, term: str, model: str) -> str:
    return f"compare_{disp}_t{term}_{model}.npz"


def _cell_dir(run_dir: str, disp: str, model: str, term: str, w: int) -> str:
    return os.path.join(run_dir, "datasets", disp, model, f"t{term}", f"w{w}")


def _run_cells() -> List[Tuple[str, str]]:
    seen: Dict[Tuple[str, str], None] = {}
    for d in datasets_config.CATALOG:
        if d.run:
            seen.setdefault((d.display, d.term), None)
    return list(seen)


def _curve_all_nan(npz_path: str, metric: str) -> Optional[bool]:
    """True if the compare npz lacks a usable curve for `metric` (and its port
    fallback). None if the npz is missing/unreadable."""
    if not os.path.isfile(npz_path):
        return None
    try:
        data = np.load(npz_path)
    except Exception:
        return None
    keys = [CURVE_KEY[metric]]
    if metric == "mase_gluonts_real":
        keys.append("real_curve_gluonts")           # port stand-in fallback
    for k in keys:
        if k in data.files and bool(np.any(~np.isnan(data[k]))):
            return False
    return True


def _inspect_cell(run_dir: str, disp: str, model: str, term: str) -> dict:
    """Walk a cell's window dirs; summarise backfill-ability."""
    wdirs = sorted(glob.glob(_cell_dir(run_dir, disp, model, term, 0).replace("w0", "w*")))
    windows = []
    for wd in wdirs:
        w = os.path.basename(wd)                     # "w128"
        mj = os.path.join(wd, "metrics.json")
        ps = os.path.join(wd, "per_sample_metrics.npz")
        rec = {"w": w, "metrics": os.path.isfile(mj), "per_sample": os.path.isfile(ps),
               "mase_gluonts": None, "ver": None, "standin": None, "ps_n": None}
        if rec["metrics"]:
            try:
                with open(mj) as f:
                    m = json.load(f)
                rec["mase_gluonts"] = m.get("mase_gluonts")
                rec["ver"] = m.get("_mase_gluonts_ver")
                rec["standin"] = m.get("_mase_gluonts_real_standin")
            except Exception:
                pass
        if rec["per_sample"]:
            try:
                with np.load(ps) as d:
                    rec["ps_n"] = int(d["mae"].shape[0]) if "mae" in d else -1
            except Exception:
                rec["ps_n"] = -2
        windows.append(rec)
    return {"dir": os.path.dirname(_cell_dir(run_dir, disp, model, term, 0)),
            "windows": windows}


def _verdict(info: dict) -> str:
    ws = info["windows"]
    if not ws:
        return "no-cells"                            # cell was never computed at all
    any_ps = any(w["per_sample"] and (w["ps_n"] or 0) > 0 for w in ws)
    any_mg = any(w["mase_gluonts"] not in (None,) and not (
        isinstance(w["mase_gluonts"], float) and np.isnan(w["mase_gluonts"]))
        for w in ws)
    if any_mg:
        return "has-mase_gluonts (backfilled; npz just needs a stage-3 rewrite)"
    if any_ps:
        return "backfillable (per_sample present -> --force 3 fills it)"
    return "needs-reinfer (no usable per_sample -> delete cell dir + rerun)"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--model", required=True, help="model_short to inspect")
    ap.add_argument("--ref-model", default=None,
                    help="a working model to compare the SAME cells against")
    ap.add_argument("--mase-metric", default="mase_gluonts_real",
                    choices=list(CURVE_KEY))
    ap.add_argument("--emit-rm", action="store_true",
                    help="print rm -rf lines for needs-reinfer cells")
    ap.add_argument("--verbose", action="store_true",
                    help="dump per-window detail for each offending cell")
    args = ap.parse_args()

    compare_dir = os.path.join(args.run_dir, "models", args.model,
                               "compare_real_vs_predicted")
    if not os.path.isdir(compare_dir):
        raise SystemExit(f"No compare dir for {args.model}: {compare_dir}")

    offenders: List[Tuple[str, str]] = []
    for disp, term in _run_cells():
        npz = os.path.join(compare_dir, _npz_filename(disp, term, args.model))
        if _curve_all_nan(npz, args.mase_metric):
            offenders.append((disp, term))

    print(Fore.CYAN + f"{args.model}: {len(offenders)} no_gluonts_curve cells "
          f"(metric={args.mase_metric})" + Fore.RESET)
    rm_lines: List[str] = []
    verdict_counts: Dict[str, int] = {}
    for disp, term in offenders:
        info = _inspect_cell(args.run_dir, disp, args.model, term)
        v = _verdict(info)
        verdict_counts[v.split(" ")[0]] = verdict_counts.get(v.split(" ")[0], 0) + 1
        line = f"  {disp:<20} t={term:<7} {v}"
        if args.ref_model:
            rinfo = _inspect_cell(args.run_dir, disp, args.ref_model, term)
            line += f"   | {args.ref_model}: {_verdict(rinfo)}"
        print(line)
        if args.verbose:
            for w in info["windows"]:
                print(f"       {w['w']:<7} metrics={int(w['metrics'])} "
                      f"per_sample={int(w['per_sample'])} ps_n={w['ps_n']} "
                      f"mase_gluonts={w['mase_gluonts']} ver={w['ver']} "
                      f"standin={w['standin']}")
        if v.startswith("needs-reinfer") or v.startswith("no-cells"):
            rm_lines.append(f"rm -rf '{info['dir']}'")

    print(Fore.CYAN + "\nVerdict summary: "
          + "  ".join(f"{k}={n}" for k, n in sorted(verdict_counts.items()))
          + Fore.RESET)
    if args.emit_rm and rm_lines:
        print(Fore.YELLOW + "\n# needs-reinfer cells — delete then re-run stage 3 "
              "(--force 3):" + Fore.RESET)
        for ln in rm_lines:
            print(ln)


if __name__ == "__main__":
    main()
