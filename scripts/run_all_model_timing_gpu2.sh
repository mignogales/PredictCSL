#!/usr/bin/env bash
# Complete the robust 3-warmup / 10-repeat timing matrix on physical GPU 2.
# Each model runs in its pinned environment. Existing complete timing.json cells
# are skipped, making this queue safe to stop and resume.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID="${GPU_ID:-2}"
WAIT_PID="${WAIT_PID:-3527970}"
CONDA_ROOT="${CONDA_ROOT:-/home/mnogales/miniconda3/envs}"
MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
CACHE_ROOT="${CACHE_ROOT:-$MASTER_ROOT/window_ablation_gifteval/general}"
LOG_DIR="$MASTER_ROOT/launch_logs"
STATUS="$LOG_DIR/all_model_timing_gpu${GPU_ID}_status.tsv"
LOG_FILE="$LOG_DIR/all_model_timing_gpu${GPU_ID}.log"
LOCK_DIR="$LOG_DIR/.all_model_timing_gpu${GPU_ID}.lock"

mkdir -p "$LOG_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Timing queue already active: $LOCK_DIR" >&2
  exit 2
fi
cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

log() {
  printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

gpu_used_mib() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' '
}

if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    log "Waiting for PID $WAIT_PID to release physical GPU $GPU_ID."
    sleep 60
  done
fi
while (( $(gpu_used_mib) > 1024 )); do
  log "GPU $GPU_ID still has $(gpu_used_mib) MiB allocated; checking again in 60s."
  sleep 60
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"
COMMON=(
  -m experiments.benchmark_window_timing_gifteval
  --run-dir "$CACHE_ROOT"
  --device cuda
  --num-gpus 1
  --warmup 3
  --repeats 10
)

run_model() {
  local env_name="$1"
  local model="$2"
  log "START $model ($env_name)"
  if PYTHONPATH=. "$CONDA_ROOT/$env_name/bin/python" "${COMMON[@]}" \
      --models "$model" >>"$LOG_FILE" 2>&1; then
    printf '%s\tcomplete\n' "$model" >>"$STATUS"
    log "DONE $model"
  else
    local status=$?
    printf '%s\tfailed:%s\n' "$model" "$status" >>"$STATUS"
    log "FAIL $model (exit $status); continuing"
  fi
}

run_model predictcsl-main Chronos2-Small
run_model predictcsl-main Chronos2-Synth
run_model predictcsl-main Chronos2-Base
run_model predictcsl-main ChronosBolt-Base
run_model predictcsl-main TimesFM2.5-200M
run_model predictcsl-main FlowState-R1
run_model TSFM_PATCH PatchTST-FM-R1
run_model predictcsl-toto Toto-2.0-313m
run_model predictcsl-legacy Sundial-Base-128M
run_model predictcsl-main Moirai2-Small
run_model predictcsl-tirex TiRex2

# Per-model invocations overwrite the run-level CSV. Rebuild it once across all
# active models without importing or loading any model-specific dependencies.
PYTHONPATH=. "$CONDA_ROOT/predictcsl-main/bin/python" "${COMMON[@]}" \
  --summary-only >>"$LOG_FILE" 2>&1
log "ALL MODEL TIMING QUEUE COMPLETE"
