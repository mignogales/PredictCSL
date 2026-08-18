#!/usr/bin/env bash
set -u

# Sequential clean GIFT-Eval alpha calibration.  Run from the repository root.

CUDA_DEVICE="${CUDA_DEVICE:-1}"
CONDA_ROOT="${CONDA_ROOT:-/home/mnogales/miniconda3/envs}"
ROOT="${OUTPUT_ROOT:-logs/experiments/master_recompute/serieswise_predictor/clean_alpha_calibration_all_models}"
SELECTOR_ROOT="${SELECTOR_ROOT:-logs/experiments/master_recompute/context_length_predictor_v4}"
MAX_ITEMS="${MAX_ITEMS:-64}"

mkdir -p "$ROOT/run_logs"
STATUS="$ROOT/run_status.tsv"
touch "$STATUS"

run_stage() {
  local model="$1"
  local model_env="$2"
  local stage="$3"
  local env_name="predictcsl-main"
  if [[ "$stage" == "forecast" ]]; then
    env_name="$model_env"
  fi
  local safe_model="${model//\//_}"
  env PYTHONPATH=. CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
    "$CONDA_ROOT/$env_name/bin/python" \
    experiments/calibrate_alpha_gifteval_clean.py \
    --model-short "$model" \
    --checkpoint "$SELECTOR_ROOT/$model/best_model.pt" \
    --legacy-config "$SELECTOR_ROOT/$model/best_config.json" \
    --output-dir "$ROOT/$model" \
    --stage "$stage" \
    --device cuda \
    --batch-size 32 \
    --max-items-per-cell "$MAX_ITEMS" \
    > "$ROOT/run_logs/${safe_model}_${stage}.log" 2>&1
}

run_model() {
  local model="$1"
  local model_env="$2"
  if [[ -f "$ROOT/$model/validation_alpha_selection.json" ]]; then
    printf '%s\talready_complete\n' "$model" >> "$STATUS"
    return
  fi
  mkdir -p "$ROOT/$model"
  if ! run_stage "$model" "$model_env" prepare; then
    printf '%s\tprepare_failed\n' "$model" >> "$STATUS"
    return
  fi
  if ! run_stage "$model" "$model_env" forecast; then
    printf '%s\tforecast_failed\n' "$model" >> "$STATUS"
    return
  fi
  if ! run_stage "$model" "$model_env" aggregate; then
    printf '%s\taggregate_failed\n' "$model" >> "$STATUS"
    return
  fi
  printf '%s\tcomplete\n' "$model" >> "$STATUS"
}

run_model Chronos2-Small predictcsl-main
run_model Chronos2-Base predictcsl-main
run_model Chronos2-Synth predictcsl-main
run_model ChronosBolt-Base predictcsl-main
run_model Moirai2-Small predictcsl-main
run_model TimesFM2.5-200M predictcsl-main
run_model FlowState-R1 predictcsl-main
run_model PatchTST-FM-R1 TSFM_PATCH
run_model Sundial-Base-128M predictcsl-legacy
run_model Toto-2.0-313m predictcsl-toto
run_model TiRex2 predictcsl-tirex
