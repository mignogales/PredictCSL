"""
PredictCSL pipeline orchestrator -- v3 (cheap, severely-constrained predictor).

Same four stages as ``run_all.py`` (build -> predictor -> ablation -> compare),
but stage 2 trains a *constrained* predictor: the hyperparameter search is pinned
to the low-FLOP corner of the space (large patches -> few tokens, narrow
d_model=128, shallow 2-4 layers) and shortened to 30 trials. The point is a
predictor whose own per-series inference cost is negligible against the labeled
TSFM (worst case ~0.12 GMAC, ~2% of a Moirai2 forward pass), so the
``pred_window`` compute savings in stage 4 are honest even without bookkeeping
the predictor's cost.

The constraint + trial count are passed to stage 2 through environment variables
(PREDICTCSL_CHEAP_PREDICTOR=1, PREDICTCSL_N_TRIALS=30), resolved at import by
predict_context_length.py so they also reach its spawned per-GPU trial workers.

Cache strategy — repeat as little as possible
----------------------------------------------
The only thing that genuinely changes in v3 is the predictor. So:

  * Stage 1 (dataset, the expensive labeling) is predictor-independent and is
    REUSED from the shared ``context_length_dataset/`` tree — its done-marker
    short-circuits instantly.

  * Stage 2 (predictor) writes to a SEPARATE root,
    ``context_length_predictor_v3/``, so the v1/v2 predictors are never touched
    and v3's own trials resume across re-runs.

  * Stage 3 (ablation) keeps a SEPARATE run dir, ``general_v3/``, but its
    ``datasets/`` cell cache (the expensive per-(dataset,model,term,window)
    GiftEval TSFM inference) is SYMLINKED to the shared ``general/datasets/``.
    The window grid comes from the dataset meta, not the HP search, so the cells
    are identical -> no TSFM re-inference; only the cheap predictor overlay is
    recomputed against the constrained predictor.

  * Stage 4 (compare) writes a separate ``strategy_comparison_v3/`` per model.

Everything reused from ``run_all.py`` (catalog, the _run bar helper, the stage
launchers, the done-checks) is imported verbatim; v3 only redirects roots and
sets the constraint env vars.

Usage
-----
    python -m experiments.run_all_v3                      # all models, reuse caches
    python -m experiments.run_all_v3 --models Moirai2-Small
    python -m experiments.run_all_v3 --skip-stages 1      # dataset already built
    python -m experiments.run_all_v3 --force 2            # re-run constrained search
    python -m experiments.run_all_v3 --n-trials 40        # override the 30-trial default
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

from colorama import Fore
from tqdm.auto import tqdm

# Reuse v1's orchestration machinery verbatim (catalog, roots, the _run bar
# helper, the stage launchers, and their done-checks). v3 only redirects the
# predictor + ablation roots and sets the constraint env vars before stage 2.
import experiments.run_all as ra

# v3-specific roots (siblings of the v1 ones, so nothing collides).
PREDICTOR_ROOT_V3   = "logs/experiments/context_length_predictor_v3"
ABLATION_GENERAL_V3 = os.path.join(ra.ABLATION_ROOT, "general_v3")
STRATEGY_SUBDIR_V3  = "strategy_comparison_v3"

N_TRIALS_V3_DEFAULT = 30


def _link_shared_cells(general_v3: str, shared_general: str) -> None:
    """Symlink general_v3/datasets -> shared general/datasets so stage 3 reuses
    the expensive per-cell GiftEval TSFM inference instead of recomputing it.

    No-op (with a clear message) if the shared cells don't exist yet, or if a
    real ``datasets/`` dir is already present under general_v3 (we never clobber
    actual data). The window grid is dataset-derived, not HP-derived, so the
    cached cells are valid for the constrained predictor too.
    """
    shared_cells = os.path.join(shared_general, "datasets")
    v3_cells     = os.path.join(general_v3, "datasets")
    os.makedirs(general_v3, exist_ok=True)

    if os.path.islink(v3_cells):
        return  # already linked from a previous run
    if os.path.isdir(v3_cells):
        print(Fore.YELLOW
              + f"[v3] {v3_cells} is a real directory (not a link) — leaving it "
                "as-is; stage 3 will use whatever cells live there."
              + Fore.RESET)
        return
    # The reduced master no longer runs the full/base predictor first. Create
    # the canonical predictor-independent cell store on demand; the first
    # variant fills it and every later curve/classification variant reuses it.
    os.makedirs(shared_cells, exist_ok=True)

    # Relative link so the tree stays portable if logs/ is moved/synced.
    rel = os.path.relpath(shared_cells, start=general_v3)
    os.symlink(rel, v3_cells)
    print(Fore.GREEN
          + f"[v3] Linked shared GiftEval cells: {v3_cells} -> {rel}" + Fore.RESET)


def _apply_output_root(output_root: str) -> str:
    """Point every reused pipeline root at one self-contained master tree."""
    root = os.path.normpath(output_root)
    os.environ["PREDICTCSL_MASTER_ROOT"] = root
    os.environ["PREDICTCSL_DATASET_ROOT"] = os.path.join(
        root, "context_length_dataset")
    os.environ["PREDICTCSL_ABLATION_ROOT"] = os.path.join(
        root, "window_ablation_gifteval")
    os.environ["PREDICTCSL_RUN_LOG_ROOT"] = os.path.join(
        root, "run_all_logs")
    ra.DATASET_ROOT = os.environ["PREDICTCSL_DATASET_ROOT"]
    ra.ABLATION_ROOT = os.environ["PREDICTCSL_ABLATION_ROOT"]
    ra.RUN_LOG_ROOT = os.environ["PREDICTCSL_RUN_LOG_ROOT"]
    return root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of display names from run_all.MODELS_TO_RUN to run.")
    p.add_argument(
        "--output-root", default=None,
        help=("Use one self-contained master output tree and configure its "
              "dataset, predictor, ablation, and run-log roots together. "
              "Use this when resuming a master_run_all pipeline directly."),
    )
    p.add_argument("--skip-stages", nargs="+", default=[], choices=list(ra.STAGES),
                   help="Stage numbers to skip (1=build, 2=predictor, 3=ablation, 4=compare).")
    p.add_argument("--only-stages", nargs="+", default=None, choices=list(ra.STAGES),
                   help="If set, run only these stages (mutually exclusive with --skip-stages).")
    p.add_argument("--force", nargs="*", default=None, choices=list(ra.STAGES),
                   help="Force re-run of cached stages (e.g. --force 2), or --force "
                        "with no arg to force every active stage.")
    p.add_argument("--n-trials", type=int, default=N_TRIALS_V3_DEFAULT,
                   help=f"Constrained-search trial count (default {N_TRIALS_V3_DEFAULT}).")
    p.add_argument("--training-objective",
                   choices=["curve", "classification", "risk"],
                   default="curve",
                   help=("Predict the z-scored curve (original), classify the "
                         "best window, or train calibrated risk-aware relative "
                         "error with asymmetric harm versus full context."))
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Stream subprocess output instead of the quiet tqdm bars.")
    p.add_argument("--verbose-ablation", action="store_true",
                   help="Stream stage 3 output live; keep other stages quiet.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ra._QUIET = not args.verbose
    ra._VERBOSE_ABLATION = args.verbose_ablation

    if args.only_stages and args.skip_stages:
        raise SystemExit("Use either --skip-stages or --only-stages, not both.")

    # ---- Constraint env vars (read at import by predict_context_length.py) ---
    # subprocess.Popen inherits os.environ, so setting these here propagates to
    # the stage-2 child and, in turn, to its spawned per-GPU trial workers.
    objective_suffix = {
        "curve": "",
        "classification": "_classification",
        "risk": "_risk",
    }[args.training_objective]
    test_mode = os.environ.get("PREDICTCSL_TEST") == "1"
    master_root = (_apply_output_root(args.output_root) if args.output_root
                   else os.environ.get("PREDICTCSL_MASTER_ROOT"))
    if master_root:
        predictor_base = os.path.join(master_root, "context_length_predictor_v3")
    else:
        predictor_base = PREDICTOR_ROOT_V3
    predictor_root = predictor_base + objective_suffix
    ablation_general = os.path.join(
        ra.ABLATION_ROOT, "general_v3" + objective_suffix)
    strategy_subdir = STRATEGY_SUBDIR_V3 + objective_suffix
    n_trials = 2 if test_mode else args.n_trials

    os.environ["PREDICTCSL_CHEAP_PREDICTOR"] = "1"
    os.environ["PREDICTCSL_N_TRIALS"]        = str(n_trials)
    os.environ["PREDICTCSL_TRAINING_OBJECTIVE"] = args.training_objective
    # Stage 2 + stage 3 both resolve the predictor root from this env var.
    os.environ["PREDICTCSL_PREDICTOR_ROOT"]  = predictor_root

    # ---- Redirect roots on the reused run_all machinery ----------------------
    # Stage 1 stays on the shared DATASET_ROOT (reused). Stage 2 -> v3 predictor
    # root; stage 3 -> general_v3 (with linked cells); stage 4 -> *_v3 subdir.
    ra.PREDICTOR_ROOT   = predictor_root
    ra.ABLATION_GENERAL = ablation_general
    ra.STRATEGY_SUBDIR  = strategy_subdir

    # Reuse the expensive per-cell GiftEval inference.
    _link_shared_cells(ablation_general, os.path.join(ra.ABLATION_ROOT, "general"))

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

    # Stage 3 needs the v3 predictor + general_v3 run dir explicitly (run_all's
    # stage_3 launcher reads ra.PREDICTOR_ROOT / ra.ABLATION_GENERAL, already
    # redirected above, so no extra args are required). Same for stage 4.
    extras = {"1": [], "2": [], "3": ["--short-context-mode", "skip"], "4": []}
    if test_mode:
        extras["1"] += ["--n-series", str(ra.TEST_N_SERIES)]
        extras["3"] += ["--no-plots",
                        "--test-datasets", str(ra.TEST_N_DATASETS),
                        "--test-datasets-seed", "42"]

    print(Fore.CYAN + f"[v3] Models: {[m[2] for m in selected]}" + Fore.RESET)
    print(Fore.CYAN + f"[v3] Active stages: {sorted(active)}" + Fore.RESET)
    print(Fore.CYAN + f"[v3] Constrained search: {n_trials} trials, "
          "patch∈{64,128}, d_model=128, layers∈{2,4}" + Fore.RESET)
    print(Fore.CYAN + f"[v3] Training objective: {args.training_objective}" + Fore.RESET)
    print(Fore.CYAN + f"[v3] Predictor root: {predictor_root}" + Fore.RESET)
    print(Fore.CYAN + f"[v3] Ablation run dir: {ablation_general}" + Fore.RESET)
    if forced:
        print(Fore.YELLOW + f"[v3] Forced re-runs: {sorted(forced)}" + Fore.RESET)

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
        for sid in ordered_stages:
            _maybe_run(sid, model_id, family, display)

    # Cross-model roll-up (overview figure + grand-total CSV) over general_v3.
    if "4" in active:
        ra.stage_4_rollup(
            extras["4"],
            models=[display for _model_id, _family, display in selected],
        )

    total = time.perf_counter() - t_start
    print(Fore.GREEN + f"\n[v3] All done in {total/60:.1f} min." + Fore.RESET)


if __name__ == "__main__":
    main()
