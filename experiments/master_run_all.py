"""
PredictCSL master orchestrator -- fuses every ``run_all*`` variant into one run,
WITHOUT recomputing any shared stage.

Each ``run_all*`` orchestrator re-walks the four base stages (build -> predictor
-> ablation -> compare); many of those stages are *shared* across variants and
must run exactly once:

  * Stage 1 (synthetic dataset labeling) is common to ALL variants.
  * The shared stage-2/3/4 on the base ``general/`` tree is common to v1, v2 and
    run_all_5.
  * Only v3 / v4 have a genuinely distinct predictor + ablation (their own
    ``*_v3`` / ``*_v4`` roots), and even there the expensive GiftEval cells are
    SYMLINKED from ``general/`` — so the TSFM inference itself still happens once.

This script runs Stage 1 once up front, then invokes each variant as a subprocess
(needed so import-time env vars like ``PREDICTCSL_PREDICTOR_ARCH=mamba`` stay
isolated per variant) with an explicit ``--skip-stages`` set covering everything
an earlier entry already produced. Done-markers remain the safety net.

Order (per selected model, shared caches):
    0. run_all  --only-stages 1            (dataset labeling, once)
    1. run_all      --skip-stages 1        (v1 predictor + ablation + compare)
    2. run_all_v2   --skip-stages 1 2 3    (+ period strategy, reuse v1 2/3)
    3. run_all_v3   --skip-stages 1        (cheap predictor; symlinked cells)
    4. run_all_v4   --skip-stages 1        (Mamba predictor; symlinked cells)
    5. run_all_5    --skip-stages 1 2 3 4  (robust timing + compare re-run)
    6. rollup_all_predictors               (combined cross-predictor overview)

-------------------------------------------------------------------------------
MAINTENANCE RULE: when you add a new ``run_all_*`` orchestrator, add it to the
``VARIANTS`` registry below with the stages it should SKIP (everything already
produced by an earlier entry). That keeps master_run_all the single
fuse-everything entry point and guarantees no shared stage is recomputed.
-------------------------------------------------------------------------------

Usage
-----
    python -m experiments.master_run_all                       # everything, all models
    python -m experiments.master_run_all --models Chronos2-Small
    python -m experiments.master_run_all --only-variants v1 v5
    python -m experiments.master_run_all --skip-variants v3 v4
    python -m experiments.master_run_all --repeats 10 --warmup 3   # forwarded to v5
    python -m experiments.master_run_all --test               # smoke-test base pipeline only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List

from colorama import Fore


@dataclass
class Variant:
    name: str                       # short id used by --only/--skip-variants
    module: str                     # python -m target
    skip_stages: List[str]          # base stages already produced upstream
    extra: List[str] = field(default_factory=list)   # variant-specific flags
    takes_repeats: bool = False     # forward --repeats/--warmup (run_all_5 only)
    label: str = ""                 # human description for the banner


# Authoritative variant registry (see MAINTENANCE RULE in the module docstring).
# Order matters: each entry's skip set assumes everything above it has run.
VARIANTS: List[Variant] = [
    Variant("v1", "experiments.run_all",    ["1"],
            label="base predictor (build/predictor/ablation/compare)"),
    Variant("v2", "experiments.run_all_v2", ["1", "2", "3"],
            label="+ 2×period strategy"),
    Variant("v3", "experiments.run_all_v3", ["1"],
            label="cheap constrained PatchTST predictor"),
    Variant("v4", "experiments.run_all_v4", ["1"],
            label="Mamba predictor"),
    Variant("v5", "experiments.run_all_5",  ["1", "2", "3", "4"], takes_repeats=True,
            label="robust wall-clock timing + compare re-run"),
]


def _banner(text: str) -> None:
    bar = "═" * 78
    print(Fore.CYAN + f"\n{bar}\n  {text}\n{bar}" + Fore.RESET)


def _run(cmd: List[str], title: str) -> None:
    """Stream a subprocess to the terminal (so each variant's own bars show),
    timing it and raising on failure."""
    _banner(title)
    print(Fore.MAGENTA + f"  $ {' '.join(cmd)}" + Fore.RESET)
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, check=False).returncode
    dt = time.perf_counter() - t0
    if rc != 0:
        raise SystemExit(
            Fore.RED + f"  ✗ {title} failed (exit {rc}) after {dt/60:.1f} min." + Fore.RESET)
    print(Fore.GREEN + f"  ✓ {title} done in {dt/60:.1f} min." + Fore.RESET)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    names = [v.name for v in VARIANTS]
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of display names from run_all.MODELS_TO_RUN to run.")
    p.add_argument("--only-variants", nargs="+", default=None, choices=names,
                   help=f"Run only these variants (default: all of {names}).")
    p.add_argument("--skip-variants", nargs="+", default=[], choices=names,
                   help="Variants to skip.")
    p.add_argument("--repeats", type=int, default=None,
                   help="Robust-timing repeats, forwarded to run_all_5.")
    p.add_argument("--warmup", type=int, default=None,
                   help="Robust-timing warm-up passes, forwarded to run_all_5.")
    p.add_argument("--no-rollup", action="store_true",
                   help="Skip the final cross-predictor rollup_all_predictors pass.")
    p.add_argument("--test", action="store_true",
                   help="Smoke-test the BASE pipeline only (run_all --test) and exit; "
                        "the variant fusion runs on real trees and has no --test.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Pass -v to each variant (stream its subprocess output).")
    return p.parse_args()


def _common_flags(args) -> List[str]:
    flags: List[str] = []
    if args.models:
        flags += ["--models", *args.models]
    if args.verbose:
        flags += ["-v"]
    return flags


def main() -> None:
    args = parse_args()

    # ---- Smoke test: just exercise the base pipeline end-to-end and exit. ----
    if args.test:
        cmd = [sys.executable, "-m", "experiments.run_all", "--test"]
        if args.models:
            cmd += ["--models", *args.models]
        _run(cmd, "SMOKE TEST — run_all --test (base pipeline)")
        return

    wanted = set(args.only_variants) if args.only_variants else {v.name for v in VARIANTS}
    wanted -= set(args.skip_variants)
    variants = [v for v in VARIANTS if v.name in wanted]
    if not variants:
        raise SystemExit(f"No variants selected. Known: {[v.name for v in VARIANTS]}")

    common = _common_flags(args)
    print(Fore.CYAN + f"Master run — variants: {[v.name for v in variants]}" + Fore.RESET)
    print(Fore.CYAN + f"Models: {args.models or 'ALL (MODELS_TO_RUN)'}" + Fore.RESET)

    t_start = time.perf_counter()

    # ---- Stage 1 once, up front (every variant then skips it). ---------------
    _run([sys.executable, "-m", "experiments.run_all", "--only-stages", "1", *common],
         "Stage 1 — dataset labeling (once, shared by all variants)")

    # ---- Each variant, with its shared stages skipped. -----------------------
    for v in variants:
        cmd = [sys.executable, "-m", v.module]
        if v.skip_stages:
            cmd += ["--skip-stages", *v.skip_stages]
        cmd += v.extra + common
        if v.takes_repeats:
            if args.repeats is not None:
                cmd += ["--repeats", str(args.repeats)]
            if args.warmup is not None:
                cmd += ["--warmup", str(args.warmup)]
        _run(cmd, f"{v.name} — {v.label}  (skip stages {v.skip_stages or 'none'})")

    # ---- Final combined cross-predictor overview (pure post-processing). -----
    if not args.no_rollup:
        rollup = [sys.executable, "-m", "experiments.rollup_all_predictors"]
        if args.models:
            rollup += ["--models", *args.models]
        _run(rollup, "rollup_all_predictors — combined cross-predictor overview")

    total = time.perf_counter() - t_start
    print(Fore.GREEN + f"\nMaster run complete in {total/60:.1f} min." + Fore.RESET)


if __name__ == "__main__":
    main()
