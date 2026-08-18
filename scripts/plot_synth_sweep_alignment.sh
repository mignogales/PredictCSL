#!/usr/bin/env bash
# STD-focused alignment report; inference-free and safe to rerun.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_BIN="${CONDA_BIN:-$REPO_ROOT/../../miniconda3/bin/conda}"
MAIN_ENV="${MAIN_ENV:-predictcsl-main}"
SWEEP_ROOT="${PREDICTCSL_SWEEP_ROOT:-logs/experiments/synth_param_sweeps}"
STRICT="${STRICT:-0}"
PER_CELL="${PER_CELL:-1}"

MODELS=(
  Chronos2-Small Moirai2-Small TimesFM2.5-200M PatchTST-FM-R1
  Sundial-Base-128M Chronos2-Synth Chronos2-Base ChronosBolt-Base
  Toto-2.0-313m FlowState-R1 TiRex2
)

args=(
  python -m experiments.plot_synth_sweep_alignment
  --root "$SWEEP_ROOT"
  --models "${MODELS[@]}"
)
if [[ "$STRICT" == "1" ]]; then
  args+=(--strict)
fi
if [[ "$PER_CELL" == "1" ]]; then
  args+=(--per-cell)
fi
"$CONDA_BIN" run --no-capture-output -n "$MAIN_ENV" "${args[@]}"
