#!/usr/bin/env bash
# Run one fixed half of the corrected forecast-timing benchmark. This staged
# pass uses the already-complete ExtraTrees/full-score histogram; the final
# reviewer launcher later unions compact-tree selections and resumes missing
# cells without repeating schema-v2 sidecars.
set -Eeuo pipefail

[[ $# -eq 2 ]] || {
  echo "Usage: $0 PHYSICAL_GPU_ID SHARD_ID" >&2
  exit 2
}
GPU_ID="$1"
SHARD_ID="$2"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: $GPU_ID" >&2; exit 2; }
[[ "$SHARD_ID" == 0 || "$SHARD_ID" == 1 ]] || {
  echo "SHARD_ID must be 0 or 1" >&2
  exit 2
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
GRID_V1="${GRID_V1:-$MASTER_ROOT/window_ablation_gifteval/general}"
OUT_ROOT="${OUT_ROOT:-$MASTER_ROOT/context_selection_reviewer_followups}"
CONDA_ROOT="${CONDA_ROOT:-/home/mnogales/miniconda3/envs}"
TIMING_MAX_SERIES="${TIMING_MAX_SERIES:-64}"
GPU_FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-1024}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

if [[ "${WAIT_FOR_GPU:-0}" == 1 ]]; then
  while true; do
    used_mib="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr -d ' ')"
    if (( used_mib <= GPU_FREE_THRESHOLD_MIB )); then
      break
    fi
    sleep "$GPU_POLL_SECONDS"
  done
fi

MODEL_SPECS=(
  "Chronos2-Base|chronos2_base|predictcsl-main"
  "Chronos2-Small|chronos2_small|predictcsl-main"
  "Chronos2-Synth|chronos2_synth|predictcsl-main"
  "ChronosBolt-Base|chronos_bolt_base|predictcsl-main"
  "FlowState-R1|flowstate_r1|predictcsl-main"
  "Moirai2-Small|moirai2_small|predictcsl-main"
  "PatchTST-FM-R1|patchtst_fm_r1|TSFM_PATCH"
  "Sundial-Base-128M|sundial_base_128m|predictcsl-legacy"
  "TimesFM2.5-200M|timesfm2p5|predictcsl-main"
  "Toto-2.0-313m|toto_2_0_313m|predictcsl-toto"
)

LOG_DIR="$OUT_ROOT/launch_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/speed_staged_gpu${GPU_ID}_shard${SHARD_ID}_$(date +%Y%m%d_%H%M%S).out"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Starting staged timing: physical GPU $GPU_ID, dataset shard $SHARD_ID/2"
echo "Log: $LOG_FILE"

for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model slug env_name <<<"$spec"
  teacher_hist="$OUT_ROOT/fronts/$slug/full_score/selected_window_histograms.csv"
  [[ -s "$teacher_hist" ]] || {
    echo "Missing teacher histogram: $teacher_hist" >&2
    exit 2
  }
  echo "TIMING $model ($env_name), shard=$SHARD_ID on physical GPU $GPU_ID"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH=. \
    "$CONDA_ROOT/$env_name/bin/python" \
    -m experiments.benchmark_window_timing_gifteval \
    --run-dir "$GRID_V1" --models "$model" --device cuda --num-gpus 1 \
    --shard-id "$SHARD_ID" --num-shards 2 \
    --warmup 3 --repeats 10 --max-series "$TIMING_MAX_SERIES" \
    --selection-csvs "$teacher_hist" \
    --selection-methods full_native balanced efficiency max_efficiency
done

echo "Completed staged timing: physical GPU $GPU_ID, dataset shard $SHARD_ID/2"
