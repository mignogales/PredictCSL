"""
PredictCSL pipeline orchestrator -- v5 / run_all_5 (adds robust wall-clock timing).

Same four stages as ``run_all.py`` (build -> predictor -> ablation -> compare),
plus a fifth stage that produces *robust* forward-pass timings and a final
re-run of the comparison so the wall-clock ("clock-wise") strategy comparison is
backed by warmed-up, repeated measurements (mean +/- std) instead of the
single-shot ``elapsed_seconds`` recorded during the ablation:

    Stage 5 (timing) -- experiments.benchmark_window_timing_gifteval
        For every (model, dataset, term) it reads the strategy comparison's
        comparison.csv, collects the on-grid windows the strategies actually use
        (full / best / pred / predictor-variant ``*_window``), and times each one
        with ``--warmup`` discarded + ``--repeats`` timed forward passes, each
        GPU-synced. Writes a ``timing.json`` sidecar into the SAME per-cell dir as
        the ablation's metrics.json, so every variant's comparison folds it in.

    Stage 4 (compare) is then re-run with ``--use-robust-timing`` so the
        wall-clock figures (``model_strategy_overview_time.png``,
        ``time_savings*.csv``) use the robust mean and draw std error bars.

Everything else is reused from ``run_all.py``: stages 1-4 short-circuit on their
existing done-markers, so a run_all_5 on top of a completed run_all only does the
new timing pass + the (cheap, CPU-only) comparison re-run. The timing stage
resumes per cell (skips a cell whose timing.json already has enough repeats), so
re-running is safe.

Usage
-----
    python -m experiments.run_all_5                       # all models, reuse caches
    python -m experiments.run_all_5 --models Chronos2-Small
    python -m experiments.run_all_5 --skip-stages 1 2 3 4 # only timing + compare re-run
    python -m experiments.run_all_5 --repeats 10 --warmup 3
    python -m experiments.run_all_5 --force-timing        # re-time every cell
"""

from __future__ import annotations

import argparse
import sys
import time

from colorama import Fore
from tqdm.auto import tqdm

# Reuse v1's orchestration machinery verbatim (catalog, roots, the _run bar
# helper, the stage launchers, and their done-checks). run_all_5 only adds the
# timing stage and re-runs the comparison with robust timing.
import experiments.run_all as ra

TIMING_DEFAULT_REPEATS = 10
TIMING_DEFAULT_WARMUP  = 3


def stage_5_timing(display: str, repeats: int, warmup: int, extra) -> float:
    """Run the robust forward-pass timing benchmark for one model.

    Times the strategy-selected windows on the shared ``general/`` tree (so the
    timing is predictor-independent and reused by every variant's comparison),
    writing per-cell timing.json sidecars. Resumes per cell unless ``--force`` is
    threaded via ``extra``.
    """
    cmd = [sys.executable, "-m", "experiments.benchmark_window_timing_gifteval",
           "--run-dir", ra.ABLATION_GENERAL,
           "--models", display,
           "--repeats", str(repeats),
           "--warmup", str(warmup),
           *extra]
    return ra._run(cmd, stage="5/timing", display=display)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of display names from run_all.MODELS_TO_RUN to run.")
    p.add_argument("--skip-stages", nargs="+", default=[], choices=list(ra.STAGES),
                   help="Reused v1 stages to skip (1=build, 2=predictor, 3=ablation, "
                        "4=compare). The timing stage + compare re-run always run.")
    p.add_argument("--only-stages", nargs="+", default=None, choices=list(ra.STAGES),
                   help="If set, run only these reused v1 stages (mutually exclusive "
                        "with --skip-stages). Timing + compare re-run still run.")
    p.add_argument("--force", nargs="*", default=None, choices=list(ra.STAGES),
                   help="Force re-run of cached v1 stages (e.g. --force 3), or "
                        "--force with no arg to force all active.")
    p.add_argument("--repeats", type=int, default=TIMING_DEFAULT_REPEATS,
                   help=f"Timed forward passes per cell (default {TIMING_DEFAULT_REPEATS}).")
    p.add_argument("--warmup", type=int, default=TIMING_DEFAULT_WARMUP,
                   help=f"Warm-up forward passes per cell (default {TIMING_DEFAULT_WARMUP}).")
    p.add_argument("--force-timing", action="store_true",
                   help="Re-time every cell even when a complete timing.json exists.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Stream subprocess output instead of the quiet tqdm bars.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ra._QUIET = not args.verbose

    if args.only_stages and args.skip_stages:
        raise SystemExit("Use either --skip-stages or --only-stages, not both.")
    active = (set(args.only_stages) if args.only_stages
              else set(ra.STAGES) - set(args.skip_stages))

    selected = [m for m in ra.MODELS_TO_RUN
                if args.models is None or m[2] in args.models]
    if not selected:
        raise SystemExit(
            f"No models selected. MODELS_TO_RUN={[m[2] for m in ra.MODELS_TO_RUN]} "
            f"--models={args.models}")

    if args.force is None:
        forced: set = set()
    elif args.force == []:
        forced = set(ra.STAGES)
    else:
        forced = set(args.force)

    # Stage 3 needs the short-context mode; stage 4 folds in robust timing (on by
    # default in compare, but passed explicitly for clarity).
    extras = {"1": [], "2": [], "3": ["--short-context-mode", "skip"],
              "4": ["--use-robust-timing"]}
    timing_extra = ["--force"] if args.force_timing else []

    print(Fore.CYAN + f"[5] Models: {[m[2] for m in selected]}" + Fore.RESET)
    print(Fore.CYAN + f"[5] Reused stages: {sorted(active)}  + timing + compare" + Fore.RESET)
    print(Fore.CYAN + f"[5] Robust timing: warmup={args.warmup}, repeats={args.repeats}"
          + Fore.RESET)
    if forced:
        print(Fore.YELLOW + f"[5] Forced v1 re-runs: {sorted(forced)}" + Fore.RESET)

    def _maybe_run(sid: str, model_id: str, family: str, display: str) -> None:
        name = ra.STAGES[sid][0]
        label = f"{display} · {sid}/{name}"
        done, summary = ra.DONE_CHECKS[sid](family, display)
        if done and sid not in forced:
            tqdm.write(Fore.WHITE + f"{label} … · cached ({summary}) — skipping" + Fore.RESET)
            return
        if done and sid in forced:
            tqdm.write(Fore.YELLOW + f"  ! {label} cached ({summary}) but --force — re-running" + Fore.RESET)
        if sid == "1":
            ra.stage_1_build(model_id, display, family, extras["1"])
        elif sid == "2":
            ra.stage_2_predictor(display, family, extras["2"])
        elif sid == "3":
            ra.stage_3_ablation(display, family, extras["3"])
        else:
            ra.stage_4_compare(display, family, extras["4"])

    ordered_stages = [s for s in ("1", "2", "3", "4") if s in active]

    t_start = time.perf_counter()
    for model_id, family, display in selected:
        # ---- Reused v1 stages 1-4 (skip when cached). Stage 4 must run before
        # timing: the timing stage reads comparison.csv to know which windows to
        # measure. ------------------------------------------------------------
        for sid in ordered_stages:
            _maybe_run(sid, model_id, family, display)

        # ---- New: robust forward-pass timing of the strategy-selected windows --
        stage_5_timing(display, args.repeats, args.warmup, timing_extra)

        # ---- Re-run the comparison so the wall-clock figures use robust mean+std.
        # Always re-run (CPU-only, cheap): the first stage-4 pass predates the
        # timing.json sidecars, so a cached marker would otherwise hide them.
        ra.stage_4_compare(display, family, ["--use-robust-timing"])

    # Cross-model roll-up (overview figures + grand-total CSVs) with robust timing.
    ra.stage_4_rollup(
        ["--use-robust-timing"],
        models=[display for _model_id, _family, display in selected],
    )

    total = time.perf_counter() - t_start
    print(Fore.GREEN + f"\n[5] All done in {total/60:.1f} min." + Fore.RESET)


if __name__ == "__main__":
    main()
