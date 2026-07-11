#!/usr/bin/env python
"""
Combined cross-predictor overview — every predictor variant on ONE figure.

The per-variant orchestrators each emit their own ``model_strategy_overview.png``
rooted at their own ablation tree, so none of them shows the v1 *big* predictor
and the v3/v4 *small* predictors together:

  * ``run_all.py``    rollup is rooted at ``general/``    -> "pred" = v1 big,
    and discovers ``general_v3`` / ``general_v4`` as sibling variants. This is
    already the all-in-one figure, but it WRITES INTO ``general/``.
  * ``run_all_v3.py`` rollup is rooted at ``general_v3/`` -> "pred" = v3 cheap,
    the v1 big predictor is never folded in.
  * ``run_all_v4.py`` rollup is rooted at ``general_v4/`` -> "pred" = v4 Mamba.

This script produces the combined overview WITHOUT touching any of those trees:
it READS from the base ``general/`` tree (so "pred" is the v1 big predictor and
``discover_pred_variants`` folds in ``general_v3`` as ``pred_cheap`` and
``general_v4`` as ``pred_mamba`` against the same real curves), and WRITES every
artefact into a dedicated ``general_all/`` sink. Nothing is written back into
``general/``, ``general_v3/`` or ``general_v4/`` — ``general_all`` is an output
directory only.

It is pure post-processing: no TSFM inference, no predictor training. It just
re-reads the cached strategy records the ablation stage already produced.

When the ablation wrote the leaderboard gluonts curves it also emits the
``_gluonts`` / ``_gluonts_real`` twins of the overview (re-scoring the same
strategies on those metrics), so ``model_strategy_overview_gluonts_real.png`` is
produced beside the default-``mase`` one.

Usage (run on the SERVER, where logs/ lives)::

    python -m experiments.rollup_all_predictors
    python -m experiments.rollup_all_predictors --models Moirai2-Small TimesFM2.5-200M
    python -m experiments.rollup_all_predictors --plot-strategies pred pred_cheap best
    # oracle + both cheap predictors, on the gluonts-real leaderboard MASE:
    python -m experiments.rollup_all_predictors --plot-strategies best pred_cheap pred_mamba
"""

from __future__ import annotations

import argparse
import os

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

# Mirror run_all.py's tree layout. The base run dir MUST keep the bare "general"
# basename: discover_pred_variants recovers the stem by peeling a known suffix,
# so only "general" finds the _v3/_v4 siblings (a "general_all" run-dir would
# look for "general_all_v3" and find nothing). Hence general_all is OUTPUT only.
ABLATION_ROOT = "logs/experiments/window_ablation_gifteval"
BASE_RUN_DIR  = os.path.join(ABLATION_ROOT, "general")
OUT_DIR       = os.path.join(ABLATION_ROOT, "general_all")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=str, default=BASE_RUN_DIR,
                   help=f"Base ablation tree to read (default {BASE_RUN_DIR}). "
                        "Must be the bare 'general' tree for variant discovery.")
    p.add_argument("--output-dir", type=str, default=OUT_DIR,
                   help=f"Where to write the combined overview (default {OUT_DIR}). "
                        "A pure output sink — never read back as a run dir.")
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="Restrict to these model_short names (default: all present).")
    p.add_argument("--plot-strategies", type=str, nargs="+", default=None,
                   help="Restrict the overview figures to these strategies "
                        "(e.g. pred pred_cheap pred_mamba best). Default: all present.")
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
        print(Fore.YELLOW + "No sibling predictor variants found (general_v3 / "
              "general_v4 absent) — overview will hold only the base predictor."
              + Fore.RESET)

    df = load_strategy_records(run_dir, CACHE_ROOT, dict(DEFAULT_PATCH_SIZES))
    if args.models:
        wanted = set(args.models)
        available = sorted(df["model_short"].unique())
        df = df[df["model_short"].isin(wanted)].reset_index(drop=True)
        if df.empty:
            raise SystemExit(
                f"No records for {sorted(wanted)} in {run_dir}. Available: {available}")

    write_run_rollup(df, out_dir, plot_strategies=args.plot_strategies)
    write_run_time_rollup(df, out_dir, plot_strategies=args.plot_strategies)

    # Parallel leaderboard-MASE twins. When the ablation wrote the gluonts curves
    # (real_curve_gluonts / real_curve_gluonts_real), re-score every strategy —
    # including the folded-in v3/v4 predictors — on those metrics so the combined
    # overview is available on `mase_gluonts` and `mase_gluonts_real` too, without
    # any TSFM re-inference. Mirrors compare_window_strategies_gifteval.main().
    if run_has_gluonts_curve(run_dir):
        df_g = load_strategy_records(run_dir, CACHE_ROOT, dict(DEFAULT_PATCH_SIZES),
                                     mase_metric="mase_gluonts")
        if args.models:
            df_g = df_g[df_g["model_short"].isin(set(args.models))].reset_index(drop=True)
        write_run_rollup(df_g, out_dir, plot_strategies=args.plot_strategies,
                         suffix="_gluonts", metric_label="MASE (gluonts)")

    if run_has_gluonts_real_curve(run_dir):
        df_gr = load_strategy_records(run_dir, CACHE_ROOT, dict(DEFAULT_PATCH_SIZES),
                                      mase_metric="mase_gluonts_real")
        if args.models:
            df_gr = df_gr[df_gr["model_short"].isin(set(args.models))].reset_index(drop=True)
        write_run_rollup(df_gr, out_dir, plot_strategies=args.plot_strategies,
                         suffix="_gluonts_real", metric_label="MASE (gluonts-real)")

    print(Fore.GREEN + f"\nCombined overview written to: {out_dir}" + Fore.RESET)


if __name__ == "__main__":
    main()
