"""
PredictCSL end-to-end pipeline orchestrator.

For every entry in MODELS_TO_RUN, runs the four pipeline stages with
deterministic per-family output paths:

    Stage 1 — build_context_length_dataset.py --model-idx <i>
              -> logs/experiments/context_length_dataset/<family>/

    Stage 2 — predict_context_length.py --dataset-dir <ds_root>/<family>
              -> logs/experiments/context_length_predictor/<family>/

    Stage 3 — test_window_ablation_gifteval_v5.py
                  --models <display> --predictor-dir <pred_root>/<family>
                  --cache-root <win_root>/general
              -> logs/experiments/window_ablation_gifteval/general/
                 (shared gifteval ablation results for ALL models — raw metrics
                  under datasets/, real-vs-predicted curves under
                  models/<display>/, combined results.csv)

    Stage 4 — compare_window_strategies_gifteval.py
                  --run-dir <win_root>/general --models <display>
                  --output-dir <win_root>/<display>/strategy_comparison
              -> logs/experiments/window_ablation_gifteval/<display>/strategy_comparison/
                 (per-model comparison between window-selection methods:
                  full window vs oracle-best vs predictor)

Comment out entries in MODELS_TO_RUN to skip families. Re-running is safe:
each stage caches its own work (per-shard for stage 1, per-trial for stage 2,
per-(dataset, model, term, window) cell for stage 3, derived artefacts for
stage 4) and overwrites in place.

The orchestrator additionally short-circuits a stage when its done-marker is
already on disk, so a re-run only spawns subprocesses for the stages that
still need work. Markers checked:

    stage 1 — meta.json with shards_done == shards_total
    stage 2 — best_model.pt + best_config.json
    stage 3 — predictor_meta.json (last file written by v5)
    stage 4 — strategy_comparison/summary_stats.json

Pass --force to re-run cached stages anyway (--force 2 3 for specific ones,
or --force with no arg to force everything active).

Usage
-----
    python -m experiments.run_all                       # all stages, all listed models
    python -m experiments.run_all --skip-stages 1 2     # only stages 3 and 4
    python -m experiments.run_all --models Chronos2-Small Moirai2-Small
    python -m experiments.run_all --force 4             # re-run stage 4 even if cached
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Callable, List, Sequence, Tuple

from colorama import Fore
from tqdm.auto import tqdm

# Authoritative model catalog. --model-idx for stage 1 indexes into THIS list,
# so we resolve indices from it directly rather than from a hand-copied mirror.
from experiments.build_context_length_dataset import MODELS as BUILD_MODELS


# Master catalog: (model_id, family, display). Order is free — stage 1 resolves
# --model-idx by matching model_id against build_context_length_dataset.MODELS
# (see _model_idx), so reordering or renaming here can't mislabel a model.
# Every model_id must exist in that build catalog. Comment out a row to skip a
# family end-to-end.
MODELS_TO_RUN: List[Tuple[str, str, str]] = [
    ("autogluon/chronos-2-small",       "chronos2",    "Chronos2-Small"),
    ("autogluon/chronos-2-synth",       "chronos2",    "Chronos2-Synth"),
    ("google/timesfm-2.5-200m-pytorch", "timesfm",     "TimesFM2.5-200M"),
    ("ibm-research/patchtst-fm-r1",     "patchtst_fm", "PatchTST-FM-R1"),
    ("thuml/sundial-base-128m",         "sundial",     "Sundial-Base-128M"),
    ("Maple728/TimeMoE-200M",           "timemoe",     "TimeMoE-200M"),
    ("Salesforce/moirai-2.0-R-small",   "moirai",      "Moirai2-Small"),
]

DATASET_ROOT   = "logs/experiments/context_length_dataset"
PREDICTOR_ROOT = "logs/experiments/context_length_predictor"
ABLATION_ROOT  = "logs/experiments/window_ablation_gifteval"
# Shared, cross-model gifteval ablation output (stage 3). One folder for every
# model — v5 keys its artefacts by (dataset, model_short), so models coexist
# here without collision and share the dataset cache + naive baselines.
ABLATION_GENERAL = os.path.join(ABLATION_ROOT, "general")


def _model_idx(model_id: str, display: str) -> int:
    # Resolve --model-idx against the build script's OWN MODELS list (imported
    # as BUILD_MODELS), keyed by the unique checkpoint id. This is the only safe
    # key: --model-idx is positional into build's MODELS, so matching by id
    # guarantees the index always points at the same checkpoint the build script
    # will load — no hand-maintained mirror to drift out of sync. (Matching by
    # display would break for variants whose display differs from build's, e.g.
    # a synthetic-data variant reusing the same checkpoint.)
    for i, (mid, _, _) in enumerate(BUILD_MODELS):
        if mid == model_id:
            return i
    available = "\n".join(
        f"    [{i}] {mid}  ({disp})" for i, (mid, _, disp) in enumerate(BUILD_MODELS))
    raise ValueError(
        f"Model id {model_id!r} (display {display!r}) is not in "
        f"experiments/build_context_length_dataset.py MODELS, so stage 1 cannot "
        f"build it. Add it there first. Available checkpoints:\n{available}"
    )


# Overall progress bar (set in main). Orchestrator-level messages are routed
# through _emit so they print *above* the bar instead of clobbering it; the
# subprocess output of each stage still streams raw and will scroll the bar,
# which tqdm redraws on the next update.
_BAR: "tqdm | None" = None


def _emit(msg: str) -> None:
    if _BAR is not None:
        _BAR.write(msg)
    else:
        print(msg)


def _banner(text: str) -> None:
    bar = "═" * 78
    _emit(Fore.CYAN + f"\n{bar}\n  {text}\n{bar}" + Fore.RESET)


def _run(cmd: Sequence[str], stage: str) -> None:
    """Subprocess wrapper: stream output, raise on non-zero."""
    _emit(Fore.MAGENTA + f"  $ {' '.join(cmd)}" + Fore.RESET)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, check=False)
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(
            f"Stage {stage} failed (exit {res.returncode}) after {dt:.1f}s.")
    _emit(Fore.GREEN + f"  ✓ stage {stage} done in {dt:.1f}s" + Fore.RESET)


def stage_1_build(model_id: str, display: str, family: str,
                  extra: Sequence[str]) -> None:
    idx = _model_idx(model_id, display)
    _run(
        [sys.executable, "-m", "experiments.build_context_length_dataset",
         "--model-idx", str(idx), *extra],
        stage="1/build",
    )


def stage_2_predictor(display: str, family: str, extra: Sequence[str]) -> None:
    dataset_dir = os.path.join(DATASET_ROOT, display)
    _run(
        [sys.executable, "-m", "experiments.predict_context_length",
         "--dataset-dir", dataset_dir, *extra],
        stage="2/predictor",
    )


def stage_3_ablation(display: str, family: str, extra: Sequence[str]) -> None:
    predictor_dir = os.path.join(PREDICTOR_ROOT, display)
    _run(
        [sys.executable, "-m", "experiments.test_window_ablation_gifteval_v5",
         "--models", display,
         "--predictor-dir", predictor_dir,
         "--cache-root", ABLATION_GENERAL, *extra],
        stage="3/ablation",
    )


def stage_4_compare(display: str, family: str, extra: Sequence[str]) -> None:
    out_dir = os.path.join(ABLATION_ROOT, display, "strategy_comparison")
    _run(
        [sys.executable, "-m", "experiments.compare_window_strategies_gifteval",
         "--run-dir", ABLATION_GENERAL,
         "--models", display,
         "--output-dir", out_dir, *extra],
        stage="4/compare",
    )


STAGES = {
    "1": ("build",     stage_1_build),
    "2": ("predictor", stage_2_predictor),
    "3": ("ablation",  stage_3_ablation),
    "4": ("compare",   stage_4_compare),
}


# ==============================================================================
#  DONE CHECKS  (stage-level skip)
# ==============================================================================
#  Each returns (done, summary). `done` short-circuits the subprocess call;
#  `summary` is shown to the user so they know what was cached. These checks
#  are intentionally cheap: a few stat() calls and one JSON read per stage.

def _done_stage_1(family: str, display: str = "") -> Tuple[bool, str]:
    """Stage 1 is done when every labeling shard for this model is on disk.
    meta.json carries shards_done / shards_total — written each invocation."""
    meta_path = os.path.join(DATASET_ROOT, display, "meta.json")
    if not os.path.isfile(meta_path):
        return False, "no meta.json"
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, "meta.json unreadable"
    done = int(meta.get("shards_done", 0))
    total = int(meta.get("shards_total", 0))
    if total > 0 and done >= total:
        return True, f"{done}/{total} shards"
    return False, f"{done}/{total} shards"


def _done_stage_2(family: str, display: str = "") -> Tuple[bool, str]:
    """Stage 2 is done when the predictor's final artefacts are persisted.
    Trials are sweep-internal cache; what downstream stages need is
    best_model.pt + best_config.json."""
    fdir = os.path.join(PREDICTOR_ROOT, display)
    have_model = os.path.isfile(os.path.join(fdir, "best_model.pt"))
    have_cfg   = os.path.isfile(os.path.join(fdir, "best_config.json"))
    trial_dir  = os.path.join(fdir, "trials")
    n_trials = (sum(1 for n in os.listdir(trial_dir)
                    if n.startswith("trial_") and n.endswith(".json"))
                if os.path.isdir(trial_dir) else 0)
    if have_model and have_cfg:
        return True, f"best_model.pt + {n_trials} trials"
    return False, (
        "missing best_model.pt/best_config.json "
        f"({n_trials} trials cached so far)"
    )


def _done_stage_3(family: str, display: str = "") -> Tuple[bool, str]:
    """Stage 3 (for one model) is done when v5 wrote that model's
    compare_real_vs_predicted/compare_summary.csv into the shared general
    folder — the terminal per-model artefact of v5's main()."""
    compare_dir = os.path.join(
        ABLATION_GENERAL, "models", display, "compare_real_vs_predicted")
    marker = os.path.join(compare_dir, "compare_summary.csv")
    if not os.path.isfile(marker):
        return False, "no compare_summary.csv"
    n_npz = sum(1 for n in os.listdir(compare_dir) if n.endswith(".npz"))
    return True, f"{n_npz} comparison .npz files"


def _done_stage_4(family: str, display: str = "") -> Tuple[bool, str]:
    """Stage 4 is done when summary_stats.json exists in the model's
    strategy_comparison/ folder."""
    out = os.path.join(ABLATION_ROOT, display, "strategy_comparison",
                       "summary_stats.json")
    if os.path.isfile(out):
        return True, "summary_stats.json present"
    return False, "summary_stats.json missing"


DONE_CHECKS: dict[str, Callable[..., Tuple[bool, str]]] = {
    "1": _done_stage_1,
    "2": _done_stage_2,
    "3": _done_stage_3,
    "4": _done_stage_4,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of display names from MODELS_TO_RUN to actually run.",
    )
    p.add_argument(
        "--skip-stages", nargs="+", default=[], choices=list(STAGES),
        help="Stage numbers to skip (1=build, 2=predictor, 3=ablation, 4=compare).",
    )
    p.add_argument(
        "--only-stages", nargs="+", default=None, choices=list(STAGES),
        help="If set, run only these stages (mutually exclusive with --skip-stages).",
    )
    p.add_argument(
        "--force", nargs="*", default=None, choices=list(STAGES),
        help=("Re-run stages even when their done-marker is present. "
              "Pass stage numbers (e.g. --force 2 3) or use --force with no "
              "argument to force every active stage."),
    )
    # Pass-through extras per stage.
    for sid, (name, _) in STAGES.items():
        p.add_argument(
            f"--{name}-args", nargs=argparse.REMAINDER, default=[],
            help=f"Extra CLI args appended to stage {sid} ({name}).",
        )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.only_stages and args.skip_stages:
        raise SystemExit("Use either --skip-stages or --only-stages, not both.")
    active = (set(args.only_stages) if args.only_stages
              else set(STAGES) - set(args.skip_stages))

    selected = [m for m in MODELS_TO_RUN
                if args.models is None or m[2] in args.models]
    if not selected:
        raise SystemExit(
            f"No models selected. MODELS_TO_RUN={[m[2] for m in MODELS_TO_RUN]} "
            f"--models={args.models}")

    extras_by_stage = {
        "1": args.build_args,
        "2": args.predictor_args,
        "3": args.ablation_args,
        "4": args.compare_args,
    }

    # --force semantics: None = no override, [] = force all active, [...] = force listed
    if args.force is None:
        forced = set()
    elif args.force == []:
        forced = set(STAGES)
    else:
        forced = set(args.force)

    print(Fore.CYAN + f"Models: {[m[2] for m in selected]}" + Fore.RESET)
    print(Fore.CYAN + f"Active stages: {sorted(active)}" + Fore.RESET)
    if forced:
        print(Fore.YELLOW + f"Forced re-runs: {sorted(forced)}" + Fore.RESET)

    def _maybe_run(sid: str, model_id: str, family: str, display: str) -> None:
        name = STAGES[sid][0]
        done, summary = DONE_CHECKS[sid](family, display)
        if done and sid not in forced:
            _emit(Fore.WHITE
                  + f"  · stage {sid}/{name} cached ({summary}) — skipping"
                  + Fore.RESET)
            return
        if done and sid in forced:
            _emit(Fore.YELLOW
                  + f"  ! stage {sid}/{name} cached ({summary}) but --force given — re-running"
                  + Fore.RESET)
        else:
            _emit(Fore.CYAN
                  + f"  → stage {sid}/{name} — {summary}"
                  + Fore.RESET)
        extras = extras_by_stage[sid]
        if sid == "1":
            stage_1_build(model_id, display, family, extras)
        elif sid == "2":
            stage_2_predictor(display, family, extras)
        elif sid == "3":
            stage_3_ablation(display, family, extras)
        else:
            stage_4_compare(display, family, extras)

    # One overall bar across every (model × active stage) unit of work. Its
    # description names the model + stage currently in flight so a long run is
    # legible at a glance: "Chronos2-Small · 3/ablation".
    ordered_stages = [s for s in ("1", "2", "3", "4") if s in active]
    total_units = len(selected) * len(ordered_stages)

    global _BAR
    t_start = time.perf_counter()
    with tqdm(total=total_units, desc="pipeline", unit="stage",
              dynamic_ncols=True) as bar:
        _BAR = bar
        try:
            for model_id, family, display in selected:
                _banner(f"{display}  ({family})  —  {model_id}")
                for sid in ordered_stages:
                    name = STAGES[sid][0]
                    bar.set_description(f"{display} · {sid}/{name}")
                    _maybe_run(sid, model_id, family, display)
                    bar.update(1)
        finally:
            _BAR = None

    total = time.perf_counter() - t_start
    print(Fore.GREEN + f"\nAll done in {total/60:.1f} min." + Fore.RESET)


if __name__ == "__main__":
    main()
