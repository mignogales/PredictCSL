#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID="${GPU_ID:-3}"
CONDA_ROOT="${CONDA_ROOT:-/home/mnogales/miniconda3/envs}"
ROOT="${OUTPUT_ROOT:-logs/experiments/master_recompute/window_ablation_gifteval_grid_v2/general}"
WINDOWS=(40 56 80 112 160 224 320 448 640 896 1280 1792 2304 2816 3584 5120 7168 10240 13824)

mkdir -p "$ROOT/launch_logs"
STATUS="$ROOT/launch_logs/run_status.tsv"
touch "$STATUS"

run_group() {
  local env_name="$1"
  local cap="$2"
  shift 2
  local models=("$@")
  local selected=()
  local window
  for window in "${WINDOWS[@]}"; do
    if (( window <= cap )); then
      selected+=("$window")
    fi
  done
  local tag="${models[0]//\//_}"
  if env PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "$CONDA_ROOT/$env_name/bin/python" \
    -m experiments.test_window_ablation_gifteval_v5 \
    --forecast-only \
    --window-grid "${selected[@]}" \
    --models "${models[@]}" \
    --cache-root "$ROOT" \
    --no-full-native-baseline \
    --device cuda \
    --num-gpus 1 \
    > "$ROOT/launch_logs/${tag}.log" 2>&1; then
    printf '%s\tcomplete\n' "${models[*]}" >> "$STATUS"
  else
    local rc=$?
    printf '%s\tfailed:%s\n' "${models[*]}" "$rc" >> "$STATUS"
    return "$rc"
  fi
}

run_group predictcsl-main 8192 Chronos2-Small Chronos2-Base Chronos2-Synth
run_group predictcsl-main 2048 ChronosBolt-Base
run_group predictcsl-main 8192 Moirai2-Small
run_group predictcsl-main 15360 TimesFM2.5-200M
run_group predictcsl-main 4096 FlowState-R1
run_group TSFM_PATCH 8192 PatchTST-FM-R1
run_group predictcsl-legacy 2880 Sundial-Base-128M
run_group predictcsl-toto 4096 Toto-2.0-313m
run_group predictcsl-tirex 8192 TiRex2
