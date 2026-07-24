"""
PredictCSL pipeline orchestrator -- v2 (adds cadence-similarity period methods).

Same four stages as ``run_all.py`` (build -> predictor -> ablation -> compare),
but with a fifth stage inserted before the comparison that evaluates an
alternative, off-grid context-length strategy:

    Stage 5 (period) -- experiments.period_window_eval
        For every test instance, translate the official GiftEval cadences to
        sample counts, choose the cadence whose adjacent non-overlapping chunks
        are most similar, then evaluate max(2P, horizon) and max(3P, horizon).
        Because these lengths are generally NOT on the ablation grid, their
        metrics are computed here rather than read from cached cells.

    Stage 4 (compare) is then re-run so compare_window_strategies_gifteval.py
        surfaces the period strategy as a fourth column alongside
        full / best / predictor.

Everything else is reused from ``run_all.py``: stages 1-3 short-circuit on their
existing done-markers, so a v2 run on top of a completed v1 run only does the new
period evaluation + the (cheap, CPU-only) comparison re-run.  The period stage
itself resumes per (model, dataset, term), so re-running is safe.

Usage
-----
    python -m experiments.run_all_v2                      # all models, reuse v1 caches
    python -m experiments.run_all_v2 --models Chronos2-Small
    python -m experiments.run_all_v2 --skip-stages 1 2 3  # only period + compare
    python -m experiments.run_all_v2 --quantize none      # exact per-series widths
    python -m experiments.run_all_v2 --force-period       # recompute period sidecars
"""

from __future__ import annotations

import argparse
import sys
import time

from colorama import Fore
from tqdm.auto import tqdm

# Reuse v1's orchestration machinery verbatim (catalog, roots, the _run bar
# helper, the stage-1/2/3 launchers, and their done-checks). v2 only adds the
# period stage and re-runs the comparison.
import experiments.run_all as ra


PERIOD_STAGES = ("1", "2", "3")   # reused-from-v1 stages that may be skipped/cached


def stage_5_period(display: str, quantize: str, n_buckets: int, extra) -> float:
    """Run the per-series period-window evaluator for one model.

    Writes period_<dataset>_t<term>_<display>.json sidecars under
    ABLATION_GENERAL/models/<display>/compare_real_vs_predicted/, which stage 4
    then folds into the comparison. The script resumes per (dataset, term), so
    re-runs only fill gaps unless --force-period is passed (threaded via extra).
    """
    cmd = [sys.executable, "-m", "experiments.period_window_eval",
           "--models", display,
           "--run-dir", ra.ABLATION_GENERAL,
           "--quantize", quantize,
           "--n-buckets", str(n_buckets),
           *extra]
    return ra._run(cmd, stage="5/period", display=display)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of display names from run_all.MODELS_TO_RUN to run.")
    p.add_argument("--skip-stages", nargs="+", default=[], choices=list(PERIOD_STAGES),
                   help="Reused v1 stages to skip (1=build, 2=predictor, 3=ablation). "
                        "The period + compare stages always run.")
    p.add_argument("--force", nargs="*", default=None, choices=list(PERIOD_STAGES),
                   help="Force re-run of cached v1 stages (e.g. --force 3), or "
                        "--force with no arg to force all of 1/2/3.")
    p.add_argument("--force-period", action="store_true",
                   help="Recompute period sidecars even when they already exist.")
    p.add_argument("--quantize", choices=["log", "none"], default="none",
                   help="Window quantization for the period stage (see period_window_eval).")
    p.add_argument("--n-buckets", type=int, default=48,
                   help="Log-spaced window buckets when --quantize log.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Stream subprocess output instead of the quiet tqdm bars.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ra._QUIET = not args.verbose

    selected = [m for m in ra.MODELS_TO_RUN
                if args.models is None or m[2] in args.models]
    if not selected:
        raise SystemExit(
            f"No models selected. MODELS_TO_RUN={[m[2] for m in ra.MODELS_TO_RUN]} "
            f"--models={args.models}")

    if args.force is None:
        forced = set()
    elif args.force == []:
        forced = set(PERIOD_STAGES)
    else:
        forced = set(args.force)

    active_v1 = [s for s in PERIOD_STAGES if s not in set(args.skip_stages)]
    period_extra = ["--force"] if args.force_period else []

    print(Fore.CYAN + f"[v2] Models: {[m[2] for m in selected]}" + Fore.RESET)
    print(Fore.CYAN + f"[v2] Reused stages: {active_v1}  + period + compare" + Fore.RESET)
    if forced:
        print(Fore.YELLOW + f"[v2] Forced v1 re-runs: {sorted(forced)}" + Fore.RESET)

    stage_fns = {
        "1": lambda mid, fam, disp: ra.stage_1_build(mid, disp, fam, []),
        "2": lambda mid, fam, disp: ra.stage_2_predictor(disp, fam, []),
        "3": lambda mid, fam, disp: ra.stage_3_ablation(
            disp, fam, ["--short-context-mode", "skip"]),
    }

    t_start = time.perf_counter()
    for model_id, family, display in selected:
        # ---- Reused v1 stages 1-3 (skip when cached) -----------------------
        for sid in active_v1:
            name = ra.STAGES[sid][0]
            label = f"{display} · {sid}/{name}"
            done, summary = ra.DONE_CHECKS[sid](family, display)
            if done and sid not in forced:
                tqdm.write(Fore.WHITE + f"{label} … · cached ({summary}) — skipping" + Fore.RESET)
                continue
            if done and sid in forced:
                tqdm.write(Fore.YELLOW + f"  ! {label} cached but --force — re-running" + Fore.RESET)
            stage_fns[sid](model_id, family, display)

        # ---- New: per-series period-window evaluation ----------------------
        stage_5_period(display, args.quantize, args.n_buckets, period_extra)

        # ---- Re-run the comparison so 'period' becomes a 4th strategy ------
        # Always re-run (CPU-only, cheap): the v1 summary_stats.json predates the
        # period sidecars, so a cached marker would otherwise hide the new column.
        ra.stage_4_compare(display, family, [])

    total = time.perf_counter() - t_start
    print(Fore.GREEN + f"\n[v2] All done in {total/60:.1f} min." + Fore.RESET)


if __name__ == "__main__":
    main()
