"""
Diagnose why a model's bar-aggregate says ``n=<80>`` instead of the full 97.

The GiftEval catalog contributes **97 (dataset/freq/term) configs** — NOT "97
datasets". ``datasets_config.CATALOG`` is the single source of truth (each row is
one ``(ge_name, term)`` cell; every row currently has ``run=True``). Stage 3
(``test_window_ablation_gifteval_v5``) evaluates each config and writes a
``compare_*.npz`` per (dataset_display, term, model). Stage 4
(``compare_window_strategies_gifteval``) then loads those and, per model, plots a
bar whose title reads ``n=<rows> dataset-terms`` — where ``rows`` is how many of
the 97 cells survived every filter.

This script replays the SAME dispositions stage 4 applies and reports, per model,
exactly which catalog cells are left behind and WHY. Buckets, in the order stage 4
would hit them:

  disabled_in_config  run=False in datasets_config          (won't be evaluated)
  no_stage3_cell      not in the model's compare_summary.csv (ablation never made
                      it — usually gift_eval rejected that (name, term), or the
                      stage-3 run for this model is incomplete)
  missing_npz         summary lists it but the .npz is gone
  npz_error           .npz present but unreadable
  no_gluonts_curve    .npz has no usable curve for the chosen MASE metric AND no
                      port fallback -> stage 4 skips it (re-run stage 3 backfill)
  no_valid_mase       curve present but all-NaN
  survived            counted in the bar's base n

Then, because appending the Period / v3 / v4 strategies makes the bar
``dropna`` to rows that ALSO carry that strategy's MASE, we report how each
present variant would further shrink n (this is the usual 97 -> 80 culprit).

Run on the SERVER (outputs live there), e.g.:

    python -m experiments.analyze_leftover_cells \
        --run-dir logs/experiments/window_ablation_gifteval/general

    python -m experiments.analyze_leftover_cells --model TimesFM2_5-200M --list
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from experiments import datasets_config

try:
    from colorama import Fore, init as _cinit
    _cinit()
except Exception:                       # colorama optional
    class _F:                           # noqa: D401 - shim
        def __getattr__(self, _):
            return ""
    Fore = _F()                         # type: ignore


DEFAULT_RUN_DIR = "logs/experiments/window_ablation_gifteval/general"

# Which npz curve backs each MASE metric (mirrors the stage-4 loader).
CURVE_KEY = {
    "mase_gluonts": "real_curve_gluonts",
    "mase_gluonts_real": "real_curve_gluonts_real",
}

# Variant trees whose presence would further dropna the bar (mirrors
# discover_pred_variants: sibling run dirs named "<base>_v3", "<base>_v4", ...).
KNOWN_VARIANT_SUFFIXES = ["_v3", "_v4", "_v2"]

BUCKET_ORDER = [
    "survived",
    "disabled_in_config",
    "no_stage3_cell",
    "missing_npz",
    "npz_error",
    "no_gluonts_curve",
    "no_valid_mase",
]


def _npz_filename(dataset_display: str, term: str, model_short: str) -> str:
    return f"compare_{dataset_display}_t{term}_{model_short}.npz"


def _usable(data: "np.lib.npyio.NpzFile", key: str) -> bool:
    return key in data.files and bool(np.any(~np.isnan(data[key])))


def _expected_cells() -> List[Tuple[str, str, bool]]:
    """The 97-cell catalog as unique (display, term, run) in table order."""
    seen: Dict[Tuple[str, str], bool] = {}
    for d in datasets_config.CATALOG:
        seen.setdefault((d.display, d.term), d.run)
    return [(disp, term, run) for (disp, term), run in seen.items()]


def _discover_variant_trees(run_dir: str) -> List[Tuple[str, str]]:
    """(label, models_root) for sibling variant trees that exist on disk."""
    run_dir = os.path.normpath(run_dir)
    parent = os.path.dirname(run_dir)
    base = os.path.basename(run_dir)
    out: List[Tuple[str, str]] = []
    for suf in KNOWN_VARIANT_SUFFIXES:
        cand = os.path.join(parent, base + suf)
        mroot = os.path.join(cand, "models")
        if os.path.isdir(mroot):
            out.append((suf.lstrip("_"), mroot))
    return out


def _classify_model(
    models_root: str, model_short: str, mase_metric: str
) -> Dict[Tuple[str, str], str]:
    """Bucket every catalog cell for one model. Key: (display, term)."""
    curve_key = CURVE_KEY[mase_metric]
    compare_dir = os.path.join(models_root, model_short, "compare_real_vs_predicted")
    summary_path = os.path.join(compare_dir, "compare_summary.csv")

    summary_cells: set = set()
    if os.path.isfile(summary_path):
        summ = pd.read_csv(summary_path)
        for _, row in summ.iterrows():
            summary_cells.add((str(row["dataset_display"]), str(row["term"])))

    out: Dict[Tuple[str, str], str] = {}
    for disp, term, run in _expected_cells():
        key = (disp, term)
        if not run:
            out[key] = "disabled_in_config"
            continue
        if key not in summary_cells:
            out[key] = "no_stage3_cell"
            continue
        npz_path = os.path.join(compare_dir, _npz_filename(disp, term, model_short))
        if not os.path.isfile(npz_path):
            out[key] = "missing_npz"
            continue
        try:
            data = np.load(npz_path)
        except Exception:
            out[key] = "npz_error"
            continue
        # curve usable for the requested metric, or the port fallback for _real
        if _usable(data, curve_key):
            curve = data[curve_key]
        elif mase_metric == "mase_gluonts_real" and _usable(data, "real_curve_gluonts"):
            curve = data["real_curve_gluonts"]          # stand-in (loud in stage 4)
        else:
            out[key] = "no_gluonts_curve"
            continue
        if not bool(np.any(~np.isnan(curve))):
            out[key] = "no_valid_mase"
            continue
        out[key] = "survived"
    return out


def _variant_present_cells(
    variant_models_root: str, model_short: str
) -> set:
    """(display, term) cells for which this variant tree has an npz (its pred
    curve), i.e. rows that would keep a non-NaN ``<variant>_mase`` in the bar."""
    compare_dir = os.path.join(
        variant_models_root, model_short, "compare_real_vs_predicted")
    if not os.path.isdir(compare_dir):
        return set()
    cells = set()
    prefix = "compare_"
    suffix = f"_{model_short}.npz"
    for fn in os.listdir(compare_dir):
        if fn.startswith(prefix) and fn.endswith(suffix):
            mid = fn[len(prefix):-len(suffix)]          # "<display>_t<term>"
            if "_t" in mid:
                disp, term = mid.rsplit("_t", 1)
                cells.add((disp, term))
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR,
                    help=f"stage-3/4 run dir (default: {DEFAULT_RUN_DIR})")
    ap.add_argument("--mase-metric", default="mase_gluonts_real",
                    choices=["mase_gluonts", "mase_gluonts_real"],
                    help="which curve backs the bar (default: mase_gluonts_real)")
    ap.add_argument("--model", default=None,
                    help="restrict to one model_short (dir under models/)")
    ap.add_argument("--list", action="store_true",
                    help="list every non-survived cell with its reason")
    args = ap.parse_args()

    models_root = os.path.join(args.run_dir, "models")
    if not os.path.isdir(models_root):
        raise SystemExit(f"No models/ dir under {args.run_dir!r}")

    expected = _expected_cells()
    n_expected = len(expected)
    n_run = sum(1 for _, _, run in expected if run)
    print(Fore.CYAN + f"GiftEval catalog: {n_expected} (display,term) cells "
          f"({n_run} with run=True) — this is the ceiling for the bar's n.")
    print(f"Run dir: {args.run_dir}   metric: {args.mase_metric}" + Fore.RESET)

    model_dirs = sorted(
        d for d in os.listdir(models_root)
        if os.path.isdir(os.path.join(models_root, d))
        and (args.model is None or d == args.model)
    )
    if not model_dirs:
        raise SystemExit("No matching model dirs found.")

    variant_trees = _discover_variant_trees(args.run_dir)
    if variant_trees:
        print(Fore.CYAN + "Variant trees present (each further dropna's the bar to "
              "rows it also covers): "
              + ", ".join(lbl for lbl, _ in variant_trees) + Fore.RESET)

    # ---- per-model table -----------------------------------------------------
    header = f"{'model':<24} {'survived':>8}  " + "  ".join(
        f"{b.replace('_',' ')[:12]:>12}" for b in BUCKET_ORDER[1:])
    print("\n" + header)
    print("-" * len(header))

    all_leftover: List[Tuple[str, str, str, str]] = []   # model, disp, term, reason
    for model_short in model_dirs:
        buckets = _classify_model(models_root, model_short, args.mase_metric)
        cnt = Counter(buckets.values())
        n_surv = cnt.get("survived", 0)
        row = f"{model_short:<24} {n_surv:>8}  " + "  ".join(
            f"{cnt.get(b, 0):>12}" for b in BUCKET_ORDER[1:])
        # flag models below the full ceiling
        if n_surv < n_run:
            row = Fore.YELLOW + row + Fore.RESET
        print(row)

        for (disp, term), reason in buckets.items():
            if reason != "survived":
                all_leftover.append((model_short, disp, term, reason))

        # ---- variant shrink (the usual 97 -> 80 driver) ----------------------
        if variant_trees and n_surv:
            surv_cells = {k for k, v in buckets.items() if v == "survived"}
            notes = []
            for lbl, vroot in variant_trees:
                vcells = _variant_present_cells(vroot, model_short)
                keep = len(surv_cells & vcells)
                if keep < n_surv:
                    notes.append(f"+{lbl}: bar n -> {keep} "
                                 f"(drops {n_surv - keep})")
            if notes:
                print(Fore.MAGENTA + "    " + " | ".join(notes) + Fore.RESET)

    # ---- reason rollup -------------------------------------------------------
    print(Fore.CYAN + "\nLeft-behind cells by reason (all selected models):" + Fore.RESET)
    reason_counts = Counter(r for _, _, _, r in all_leftover)
    for reason in BUCKET_ORDER[1:]:
        if reason_counts.get(reason):
            print(f"  {reason:<20} {reason_counts[reason]}")

    # cells dropped for EVERY model -> systemic (bad config guess / gift_eval
    # rejection), vs dropped for some models only -> incomplete stage 3 there.
    by_cell: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    for model_short, disp, term, reason in all_leftover:
        by_cell[(disp, term)].append((model_short, reason))
    systemic = {c: v for c, v in by_cell.items() if len(v) == len(model_dirs)}
    if systemic:
        print(Fore.CYAN + f"\nCells missing for ALL {len(model_dirs)} models "
              "(systemic — config/gift_eval, not a per-model gap):" + Fore.RESET)
        for (disp, term), v in sorted(systemic.items()):
            reasons = ", ".join(sorted(set(r for _, r in v)))
            print(f"  {disp:<22} t={term:<7} {reasons}")

    if args.list:
        print(Fore.CYAN + "\nEvery left-behind cell:" + Fore.RESET)
        for model_short, disp, term, reason in sorted(all_leftover):
            print(f"  {model_short:<24} {disp:<22} t={term:<7} {reason}")


if __name__ == "__main__":
    main()
