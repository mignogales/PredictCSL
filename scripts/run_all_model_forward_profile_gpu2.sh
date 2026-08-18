#!/usr/bin/env bash
# Profile neural-forward CUDA time for every active non-TiRex model on GPU 2.
# The three Chronos2 variants are already complete and are merged at the end.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID="${GPU_ID:-2}"
CONDA_ROOT="${CONDA_ROOT:-/home/mnogales/miniconda3/envs}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/experiments/master_recompute/context_selection_reviewer_followups/forward_profile}"
LOG_DIR="${LOG_DIR:-logs/experiments/master_recompute/context_selection_reviewer_followups/launch_logs}"
LOG_FILE="$LOG_DIR/all_model_forward_profile_gpu${GPU_ID}.log"
STATUS_FILE="$LOG_DIR/all_model_forward_profile_gpu${GPU_ID}_status.tsv"
LOCK_DIR="$LOG_DIR/.all_model_forward_profile_gpu${GPU_ID}.lock"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Forward-profile queue already active: $LOCK_DIR" >&2
  exit 2
fi
cleanup() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES="$GPU_ID"

run_model() {
  local env_name="$1"
  local model="$2"
  printf '%s START %s (%s)\n' "$(date --iso-8601=seconds)" "$model" "$env_name" | tee -a "$LOG_FILE"
  if PYTHONPATH=. "$CONDA_ROOT/$env_name/bin/python" \
      -m experiments.profile_all_models_forward \
      --models "$model" --output-dir "$OUTPUT_DIR" \
      >>"$LOG_FILE" 2>&1; then
    printf '%s\tcomplete\n' "$model" >>"$STATUS_FILE"
  else
    local code=$?
    printf '%s\tfailed:%s\n' "$model" "$code" >>"$STATUS_FILE"
  fi
}

run_model predictcsl-main ChronosBolt-Base
run_model predictcsl-main FlowState-R1
run_model predictcsl-main Moirai2-Small
run_model TSFM_PATCH PatchTST-FM-R1
run_model predictcsl-toto Toto-2.0-313m
run_model predictcsl-main TimesFM2.5-200M
run_model predictcsl-legacy Sundial-Base-128M

PYTHONPATH=. "$CONDA_ROOT/predictcsl-main/bin/python" \
  -m experiments.profile_all_models_forward \
  --combine-only --output-dir "$OUTPUT_DIR" >>"$LOG_FILE" 2>&1
printf '%s ALL COMPLETE\n' "$(date --iso-8601=seconds)" | tee -a "$LOG_FILE"
