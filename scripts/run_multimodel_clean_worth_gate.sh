#!/usr/bin/env bash
set -u

# Reproducible sequential CUDA run for the model-wise clean worth-gate table.
# Run from the PredictCSL repository root.

CUDA_DEVICE="${CUDA_DEVICE:-0}"
CONDA_ROOT="${CONDA_ROOT:-/home/mnogales/miniconda3/envs}"
ROOT="${OUTPUT_ROOT:-logs/experiments/gifteval_clean_worth_gate_all_models}"
TEST_ROOT="${TEST_ROOT:-logs/experiments/gifteval_clean_worth_gate_test_caches}"
ABLATION_ROOT="${ABLATION_ROOT:-logs/experiments/master_recompute/window_ablation_gifteval}"
SELECTOR_ROOT="${SELECTOR_ROOT:-logs/experiments/master_recompute/context_length_predictor_v4}"

mkdir -p "$ROOT/run_logs" "$TEST_ROOT"
: > "$ROOT/run_status.tsv"

run_model() {
  local model="$1"
  local model_env="$2"
  local safe_model="${model//\//_}"
  local checkpoint="$SELECTOR_ROOT/$model/best_model.pt"
  local config="$SELECTOR_ROOT/$model/best_config.json"
  local cache_dir="$TEST_ROOT/$model"
  local output_dir="$ROOT/$model"
  local cache_log="$ROOT/run_logs/${safe_model}_test_cache.log"
  local gate_log="$ROOT/run_logs/${safe_model}_gate.log"

  mkdir -p "$cache_dir" "$output_dir"
  if [[ ! -f "$cache_dir/temporal_gate_dataset.npz" ]]; then
    env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
      "$CONDA_ROOT/predictcsl-main/bin/python" \
      experiments/train_real_temporal_worth_gate.py \
      --model-short "$model" \
      --checkpoint "$checkpoint" \
      --legacy-config "$config" \
      --output-dir "$cache_dir" \
      --ablation-root "$ABLATION_ROOT" \
      --device cuda:0 \
      --dataset-only > "$cache_log" 2>&1
    local cache_rc=$?
    if [[ $cache_rc -ne 0 ]]; then
      printf '%s\tcache_failed\t%s\n' "$model" "$cache_rc" >> "$ROOT/run_status.tsv"
      return
    fi
  fi

  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
    "$CONDA_ROOT/$model_env/bin/python" \
    experiments/train_gifteval_clean_worth_gate.py \
    --model-short "$model" \
    --checkpoint "$checkpoint" \
    --legacy-config "$config" \
    --output-dir "$output_dir" \
    --test-cache "$cache_dir/temporal_gate_dataset.npz" \
    --device cuda \
    --stage all \
    --origins-per-series 4 \
    --max-items-per-cell 256 \
    --batch-size 32 \
    --n-estimators 500 > "$gate_log" 2>&1
  local gate_rc=$?
  if [[ $gate_rc -eq 0 && -f "$output_dir/report.json" ]]; then
    printf '%s\tcomplete\t0\n' "$model" >> "$ROOT/run_status.tsv"
  else
    printf '%s\tgate_failed\t%s\n' "$model" "$gate_rc" >> "$ROOT/run_status.tsv"
  fi
}

# Chronos2-Small and Chronos2-Base are already complete in the current study.
run_model Chronos2-Synth predictcsl-main
run_model ChronosBolt-Base predictcsl-main
run_model Moirai2-Small predictcsl-main
run_model TimesFM2.5-200M predictcsl-main
run_model FlowState-R1 predictcsl-main
# This server still uses the legacy alias. The master runner resolves
# predictcsl-patchtst -> TSFM_PATCH in the same way.
run_model PatchTST-FM-R1 TSFM_PATCH
run_model Sundial-Base-128M predictcsl-legacy
run_model Toto-2.0-313m predictcsl-toto
run_model TiRex2 predictcsl-tirex
