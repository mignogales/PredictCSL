#!/usr/bin/env python
"""
Combined cross-predictor overview — every requested predictor on one figure.

The recomputation master uses ``general_v3`` (cheap curve regression) as the
base run. Sibling discovery folds in Mamba, soft-label classification, and the
risk-aware PatchTST tree against the same real GiftEval curves. Outputs go to ``general_all``;
none of the source trees is modified.

It is pure post-processing: no TSFM inference, no predictor training. It just
re-reads the cached strategy records the ablation stage already produced.

Exactly TWO metrics are emitted, each as its own suffixed set: ``_gluonts_real``
(gluonts' own evaluate_forecasts machinery — the leaderboard-faithful metric,
which also drives the wall-clock overview) and ``_gluonts`` (the numpy port).
The legacy project ``mase`` is not consumed here at all.

Usage (run on the SERVER, where logs/ lives)::

    python -m experiments.rollup_all_predictors
    python -m experiments.rollup_all_predictors --models Moirai2-Small TimesFM2.5-200M
    # dataset-wise + instance-wise predictors:
    python -m experiments.rollup_all_predictors --plot-strategies \
      pred_cheap pred_mamba pred_cheap_instance pred_mamba_instance
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from colorama import Fore

from experiments.compare_window_strategies_gifteval import (
    CACHE_ROOT,
    DEFAULT_PATCH_SIZES,
    discover_pred_variants,
    load_strategy_records,
    run_has_gluonts_curve,
    run_has_gluonts_real_curve,
    write_run_rollup,
    write_run_time_rollup,
)

# Standalone defaults retain the historical logs root. The master passes its
# self-contained recomputation paths explicitly.
ABLATION_ROOT = "logs/experiments/window_ablation_gifteval"
BASE_RUN_DIR  = os.path.join(ABLATION_ROOT, "general_v3")
OUT_DIR       = os.path.join(ABLATION_ROOT, "general_all")

INSTANCE_REPORTS = {
    "pred_cheap_instance": "strategy_comparison_v3_instance",
    "pred_mamba_instance": "strategy_comparison_v4_instance",
}


def fold_instance_strategies(df: pd.DataFrame, run_dir: str) -> pd.DataFrame:
    """Attach Phase-6 per-instance MASE/FLOPs columns to Stage-4 records.

    Instance reports are scored on ``mase_gluonts_real`` and live beside each
    model's v3/v4 folders, one row per dataset/term. Missing or partial reports
    remain NaN so resumability never drops otherwise valid rollup rows.
    """
    out = df.copy()
    root = os.path.dirname(os.path.normpath(run_dir))
    for key, subdir in INSTANCE_REPORTS.items():
        out[f"{key}_mase"] = np.nan
        out[f"{key}_flops"] = np.nan
        for model in out["model_short"].dropna().astype(str).unique():
            path = os.path.join(root, model, subdir, "comparison.csv")
            if not os.path.isfile(path):
                continue
            report = pd.read_csv(path)
            required = {"dataset_display", "term", "pred_mase", "pred_flops"}
            if not required.issubset(report.columns):
                print(Fore.YELLOW
                      + f"Incomplete instance report, skipping: {path}"
                      + Fore.RESET)
                continue
            lookup = {
                (str(row.dataset_display), str(row.term)):
                    (float(row.pred_mase), float(row.pred_flops))
                for row in report.itertuples(index=False)
            }
            model_mask = out["model_short"].astype(str) == model
            for idx in out.index[model_mask]:
                row_key = (
                    str(out.at[idx, "dataset_display"]),
                    str(out.at[idx, "term"]),
                )
                values = lookup.get(row_key)
                if values is not None:
                    out.at[idx, f"{key}_mase"] = values[0]
                    out.at[idx, f"{key}_flops"] = values[1]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=str, default=BASE_RUN_DIR,
                   help=f"Base ablation tree to read (default {BASE_RUN_DIR}).")
    p.add_argument("--output-dir", type=str, default=OUT_DIR,
                   help=f"Where to write the combined overview (default {OUT_DIR}). "
                        "A pure output sink — never read back as a run dir.")
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="Restrict to these model_short names (default: all present).")
    p.add_argument("--plot-strategies", type=str, nargs="+", default=None,
                   help="Restrict the overview figures to these strategies "
                        "(e.g. pred_cheap pred_mamba pred_cheap_instance "
                        "pred_mamba_instance). Default: all present.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = os.path.normpath(args.run_dir)
    out_dir = os.path.normpath(args.output_dir)

    if os.path.normpath(out_dir) == run_dir:
        raise SystemExit(
            f"--output-dir ({out_dir}) must differ from --run-dir ({run_dir}); "
            "general_all is an output sink, not a tree to read from.")

    variants = discover_pred_variants(run_dir)
    print(Fore.CYAN + f"Base run dir : {run_dir}  (\"pred\" = its own predictor)"
          + Fore.RESET)
    if variants:
        print(Fore.GREEN + "Folded-in variants: "
              + ", ".join(f"{key} <- {os.path.basename(tree)}"
                          for key, _lbl, _clr, tree in variants) + Fore.RESET)
    else:
        print(Fore.YELLOW + "No sibling predictor variants found — overview "
              "will hold only the base predictor."
              + Fore.RESET)

    def _load(metric: str):
        # The run is self-contained: its metrics live under
        # ``<run_dir>/datasets``.  Using the parent here makes every
        # ``_load_metrics`` lookup miss and causes all cells to be reported as
        # having no valid MASE, even though the comparison NPZs are present.
        cache_root = run_dir
        try:
            df = load_strategy_records(
                run_dir, cache_root, dict(DEFAULT_PATCH_SIZES),
                mase_metric=metric, models=args.models)
        except RuntimeError as exc:
            if str(exc) != "No valid records found — check the run directory.":
                raise
            print(Fore.YELLOW
                  + f"No valid {metric} records found — skipping that rollup."
                  + Fore.RESET)
            return None
        if args.models:
            wanted = set(args.models)
            available = sorted(df["model_short"].unique())
            df = df[df["model_short"].isin(wanted)].reset_index(drop=True)
            if df.empty:
                raise SystemExit(f"No records for {sorted(wanted)} in {run_dir}. "
                                 f"Available: {available}")
        return df

    # Emit each available metric as an unambiguous suffixed set:
    # `mase_gluonts_real` (gluonts machinery — the leaderboard-faithful one) and
    # `mase_gluonts` (numpy port).  The legacy project `mase` is not consumed.
    if not run_has_gluonts_real_curve(run_dir) and not run_has_gluonts_curve(run_dir):
        raise SystemExit(
            f"No gluonts curves (real_curve_gluonts[_real]) found in {run_dir} — "
            "re-run stage 3 (cheap backfill, no TSFM re-inference) first.")

    df_gr = _load("mase_gluonts_real")
    if df_gr is not None:
        df_gr = fold_instance_strategies(df_gr, run_dir)
        write_run_rollup(df_gr, out_dir, plot_strategies=args.plot_strategies,
                         suffix="_gluonts_real", metric_label="MASE (gluonts-real)")
        if not run_has_gluonts_real_curve(run_dir):
            print(Fore.YELLOW + "Note: no real_curve_gluonts_real anywhere in the run — "
                  "the _gluonts_real set above was scored on the port stand-in curves. "
                  "Re-run stage 3 (--force 3 for cached cells) to populate the machinery "
                  "values." + Fore.RESET)

    df_g = _load("mase_gluonts")
    if df_g is not None:
        write_run_rollup(df_g, out_dir, plot_strategies=args.plot_strategies,
                         suffix="_gluonts", metric_label="MASE (gluonts)")

    if df_gr is None and df_g is None:
        print(Fore.YELLOW
              + f"No valid gluonts-real or gluonts records found in {run_dir}"
              + (f" for models {sorted(args.models)}" if args.models else "")
              + "; no combined overview was written."
              + Fore.RESET)
        return

    # Timing columns are metric-independent. Prefer the leaderboard-faithful
    # frame, but retain the overview when only the port metric is usable.
    write_run_time_rollup(df_gr if df_gr is not None else df_g, out_dir,
                          plot_strategies=args.plot_strategies)

    print(Fore.GREEN + f"\nCombined overview written to: {out_dir}" + Fore.RESET)


if __name__ == "__main__":
    main()
