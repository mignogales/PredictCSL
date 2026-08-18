"""
PredictCSL pipeline orchestrator -- v4 (Mamba predictor, cheap to run).

Same four stages as ``run_all.py`` (build -> predictor -> ablation -> compare),
but stage 2 trains a **Mamba** (selective state-space) predictor instead of the
PatchTST-Transformer. Where v3 makes the predictor cheap by *constraining the HP
search* of the same O(N^2) Transformer, v4 attacks the cost at the architecture
level: the bidirectional Mamba encoder is O(N) in the patch-token count, so the
predictor's own per-series inference cost is much lower against the labeled
TSFM. Its unconstrained standalone search can explore smaller patches, while
the master passes ``--cheap`` for a like-sized comparison with v3. The
``pred_window`` compute savings reported in stage 4 include the predictor cost.

The arch switch is passed to stage 2 through an environment variable
(PREDICTCSL_PREDICTOR_ARCH=mamba), resolved at import by
predict_context_length.py so it also reaches its spawned per-GPU trial workers.
Pass ``--cheap`` to pin patch size, width, and depth to the same constrained
corner used by v3 (PREDICTCSL_CHEAP_PREDICTOR=1).

Requires the ``mamba-ssm`` package (and ``causal-conv1d``) on the GPU server:
    pip install mamba-ssm causal-conv1d
The import is lazy, so only the actual v4 *run* needs it; the patchtst path
(run_all.py / run_all_v3.py) keeps importing predict_context_length.py without
it.

Cache strategy — repeat as little as possible
----------------------------------------------
Identical reasoning to v3; only the predictor changes:

  * Stage 1 (dataset, the expensive labeling) is predictor-independent and is
    REUSED from the shared ``context_length_dataset/`` tree.

  * Stage 2 (predictor) writes to a SEPARATE root,
    ``context_length_predictor_v4/``, so the v1/v3 predictors are untouched and
    v4's own trials resume across re-runs.

  * Stage 3 (ablation) keeps a SEPARATE run dir, ``general_v4/``, but SYMLINKS
    its ``datasets/`` cell cache to the shared ``general/datasets/`` (the window
    grid is dataset-derived, not arch-derived, so the expensive per-cell
    GiftEval TSFM inference is identical -> no re-inference; only the cheap
    Mamba predictor overlay is recomputed).

  * Stage 4 (compare) writes a separate ``strategy_comparison_v4/`` per model.

Everything reused from ``run_all.py`` is imported verbatim; v4 only redirects
roots and sets the arch env var.

Usage
-----
    python -m experiments.run_all_v4                      # all models, reuse caches
    python -m experiments.run_all_v4 --models Moirai2-Small
    python -m experiments.run_all_v4 --skip-stages 1      # dataset already built
    python -m experiments.run_all_v4 --force 2            # re-run the Mamba search
    python -m experiments.run_all_v4 --cheap              # pin the cheap Mamba corner
    python -m experiments.run_all_v4 --n-trials 40        # override the trial count
    python -m experiments.run_all_v4 --cached-only-ablation # never run the TSFM in stage 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from colorama import Fore
from tqdm.auto import tqdm

# Reuse v1's orchestration machinery verbatim (catalog, roots, the _run bar
# helper, the stage launchers, and their done-checks). v4 only redirects the
# predictor + ablation roots and sets the arch env var before stage 2.
import experiments.run_all as ra
# Reuse v3's shared-cell symlink helper verbatim (module-level, no side effects
# at import — env vars are only set inside its main()).
import experiments.run_all_v3 as ra_v3

# v4-specific roots (siblings of the v1/v3 ones, so nothing collides).
PREDICTOR_ROOT_V4   = "logs/experiments/context_length_predictor_v4"
ABLATION_GENERAL_V4 = os.path.join(ra.ABLATION_ROOT, "general_v4")
STRATEGY_SUBDIR_V4  = "strategy_comparison_v4"

# Match v3's trial count. Passing --cheap additionally matches v3's patch sizes,
# width, and depth, keeping stage-2 wall-clock meaningfully comparable.
N_TRIALS_V4_DEFAULT = 30


def _preflight_mamba_env() -> None:
    """Fail before launching trials if Mamba's compiled extensions are mismatched."""
    probe = r"""
import sys
print("python", sys.executable)
try:
    import torch
    print("torch", torch.__version__, "cuda", getattr(torch.version, "cuda", None))
except Exception as exc:
    print("torch import failed:", repr(exc))
    raise
from mamba_ssm import Mamba
print("mamba-ssm import OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0:
        return

    details = "\n".join(
        line for line in (proc.stdout + proc.stderr).strip().splitlines()
        if line.strip()
    )
    raise SystemExit(
        Fore.RED
        + "[v4] Mamba preflight failed before launching predictor trials.\n"
        + details
        + "\n\nThis usually means torch was upgraded after installing the "
        + "torch2.4-tagged mamba-ssm / causal-conv1d wheels. Repair the active "
        + "server env with:\n"
        + "  bash envs/repair-main-mamba.sh predictcsl-main\n"
        + "Then rerun the same run_all_v4 command; failed trials are cached as "
        + "NaN and will retry automatically."
        + Fore.RESET
    )


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
    p.add_argument("--n-trials", type=int, default=N_TRIALS_V4_DEFAULT,
                   help=f"Mamba random-search trial count (default {N_TRIALS_V4_DEFAULT}, "
                        "matching v3; override to widen/shorten the search).")
    p.add_argument("--cheap", action="store_true",
                   help="Also pin the constrained Mamba corner "
                        "(PREDICTCSL_CHEAP_PREDICTOR=1): match v3's patch sizes, "
                        "d_model, and layer counts.")
    p.add_argument("--training-objective",
                   choices=["curve", "classification", "pairwise", "acceptable"],
                   default="curve",
                   help=("Predict the curve shape (original), classify the best "
                         "window with soft top-3 labels, learn cost-weighted "
                         "preferences between adjacent windows, or classify "
                         "every near-oracle window as acceptable."))
    p.add_argument(
        "--cached-only-ablation", action="store_true",
        help=("Pass --cached-only to stage 3, refusing to load the foundation "
              "model if a native forecast cell is missing or stale. Used by "
              "master_run_all when predictor work runs outside the model's "
              "dedicated environment."),
    )
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Stream subprocess output instead of the quiet tqdm bars.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ra._QUIET = not args.verbose

    if args.only_stages and args.skip_stages:
        raise SystemExit("Use either --skip-stages or --only-stages, not both.")

    # ---- Arch + constraint env vars (read at import by predict_context_length) -
    # subprocess.Popen inherits os.environ, so setting these here propagates to
    # the stage-2 child and, in turn, to its spawned per-GPU trial workers.
    objective_suffix = (
        "" if args.training_objective == "curve"
        else f"_{args.training_objective}")
    test_mode = os.environ.get("PREDICTCSL_TEST") == "1"
    master_root = (ra_v3._apply_output_root(args.output_root) if args.output_root
                   else os.environ.get("PREDICTCSL_MASTER_ROOT"))
    if master_root:
        predictor_base = os.path.join(master_root, "context_length_predictor_v4")
    else:
        predictor_base = PREDICTOR_ROOT_V4
    predictor_root = predictor_base + objective_suffix
    ablation_general = os.path.join(
        ra.ABLATION_ROOT, "general_v4" + objective_suffix)
    strategy_subdir = STRATEGY_SUBDIR_V4 + objective_suffix
    n_trials = 2 if test_mode else args.n_trials

    os.environ["PREDICTCSL_PREDICTOR_ARCH"] = "mamba"
    if args.cheap:
        os.environ["PREDICTCSL_CHEAP_PREDICTOR"] = "1"
    os.environ["PREDICTCSL_N_TRIALS"] = str(n_trials)
    os.environ["PREDICTCSL_TRAINING_OBJECTIVE"] = args.training_objective
    # Stage 2 + stage 3 both resolve the predictor root from this env var.
    os.environ["PREDICTCSL_PREDICTOR_ROOT"] = predictor_root

    # ---- Redirect roots on the reused run_all machinery ----------------------
    # Stage 1 stays on the shared DATASET_ROOT (reused). Stage 2 -> v4 predictor
    # root; stage 3 -> general_v4 (with linked cells); stage 4 -> *_v4 subdir.
    ra.PREDICTOR_ROOT   = predictor_root
    ra.ABLATION_GENERAL = ablation_general
    ra.STRATEGY_SUBDIR  = strategy_subdir

    # Reuse the expensive per-cell GiftEval inference (same helper v3 uses).
    ra_v3._link_shared_cells(ablation_general,
                             os.path.join(ra.ABLATION_ROOT, "general"))

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

    # Stage 3 reads ra.PREDICTOR_ROOT / ra.ABLATION_GENERAL (redirected above),
    # so no extra args are required beyond the short-context mode (skip, to match
    # v3 — short instances are excluded at each window rather than padded).
    extras = {"1": [], "2": [], "3": ["--short-context-mode", "skip"], "4": []}
    if args.cached_only_ablation:
        extras["3"].append("--cached-only")
    if test_mode:
        extras["1"] += ["--n-series", str(ra.TEST_N_SERIES)]
        extras["3"] += ["--no-plots",
                        "--test-datasets", str(ra.TEST_N_DATASETS),
                        "--test-datasets-seed", "42"]

    print(Fore.CYAN + f"[v4] Models: {[m[2] for m in selected]}" + Fore.RESET)
    print(Fore.CYAN + f"[v4] Active stages: {sorted(active)}" + Fore.RESET)
    print(Fore.CYAN + "[v4] Predictor arch: mamba (bidirectional, O(N))"
          + ("  + cheap corner" if args.cheap else "") + Fore.RESET)
    print(Fore.CYAN + f"[v4] Trials: {n_trials}" + Fore.RESET)
    print(Fore.CYAN + f"[v4] Training objective: {args.training_objective}" + Fore.RESET)
    print(Fore.CYAN + f"[v4] Predictor root: {predictor_root}" + Fore.RESET)
    print(Fore.CYAN + f"[v4] Ablation run dir: {ablation_general}" + Fore.RESET)
    print(Fore.YELLOW + "[v4] Requires mamba-ssm + causal-conv1d on the GPU server."
          + Fore.RESET)
    if forced:
        print(Fore.YELLOW + f"[v4] Forced re-runs: {sorted(forced)}" + Fore.RESET)
    if {"2", "3"} & active:
        _preflight_mamba_env()

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

    # Cross-model roll-up (overview figure + grand-total CSV) over general_v4.
    if "4" in active:
        ra.stage_4_rollup(
            extras["4"],
            models=[display for _model_id, _family, display in selected],
        )

    total = time.perf_counter() - t_start
    print(Fore.GREEN + f"\n[v4] All done in {total/60:.1f} min." + Fore.RESET)


if __name__ == "__main__":
    main()
