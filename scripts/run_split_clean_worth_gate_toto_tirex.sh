#!/usr/bin/env bash
set -u

# Complete clean worth-gate runs for native environments without mamba-ssm.
# The selector is prepared in predictcsl-main, forecasting happens in the
# model-native env, and inference-free gate fitting returns to main.

CUDA_DEVICE="${CUDA_DEVICE:-0}"
CONDA_ROOT="${CONDA_ROOT:-/home/mnogales/miniconda3/envs}"
ROOT="${OUTPUT_ROOT:-logs/experiments/gifteval_clean_worth_gate_all_models}"
TEST_ROOT="${TEST_ROOT:-logs/experiments/gifteval_clean_worth_gate_test_caches}"
SELECTOR_ROOT="${SELECTOR_ROOT:-logs/experiments/master_recompute/context_length_predictor_v4}"
STATUS="$ROOT/split_run_status.tsv"

mkdir -p "$ROOT/run_logs"
: > "$STATUS"

run_stage() {
  local model="$1"
  local env_name="$2"
  local stage="$3"
  local log="$ROOT/run_logs/${model//\//_}_${stage}_split.log"
  local output="$ROOT/$model"
  local checkpoint="$SELECTOR_ROOT/$model/best_model.pt"
  local config="$SELECTOR_ROOT/$model/best_config.json"

  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
    "$CONDA_ROOT/$env_name/bin/python" \
    experiments/train_gifteval_clean_worth_gate.py \
    --model-short "$model" \
    --checkpoint "$checkpoint" \
    --legacy-config "$config" \
    --output-dir "$output" \
    --test-cache "$TEST_ROOT/$model/temporal_gate_dataset.npz" \
    --device cuda \
    --stage "$stage" \
    --origins-per-series 4 \
    --max-items-per-cell 256 \
    --batch-size 32 \
    --n-estimators 500 > "$log" 2>&1
}

run_model() {
  local model="$1"
  local native_env="$2"
  local output="$ROOT/$model"
  mkdir -p "$output"

  if ! run_stage "$model" predictcsl-main prepare; then
    printf '%s\tprepare_failed\n' "$model" >> "$STATUS"
    return
  fi
  local prepared_count
  prepared_count=$(find "$output/prepared_cells" -type f -name '*.npz' 2>/dev/null | wc -l)
  if [[ $prepared_count -lt 90 ]]; then
    printf '%s\tprepare_incomplete_%s\n' "$model" "$prepared_count" >> "$STATUS"
    return
  fi

  if ! run_stage "$model" "$native_env" forecast; then
    printf '%s\tforecast_failed\n' "$model" >> "$STATUS"
    return
  fi
  local oracle_count
  oracle_count=$(find "$output/cells" -type f -name '*.npz' 2>/dev/null | wc -l)
  if [[ $oracle_count -lt 90 ]]; then
    printf '%s\tforecast_incomplete_%s\n' "$model" "$oracle_count" >> "$STATUS"
    return
  fi

  if ! run_stage "$model" predictcsl-main train; then
    printf '%s\ttrain_failed\n' "$model" >> "$STATUS"
    return
  fi
  if [[ -f "$output/report.json" ]]; then
    printf '%s\tcomplete\n' "$model" >> "$STATUS"
  else
    printf '%s\treport_missing\n' "$model" >> "$STATUS"
  fi
}

run_model Toto-2.0-313m predictcsl-toto
run_model TiRex2 predictcsl-tirex
