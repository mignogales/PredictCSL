#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/mnogales/miniconda3/envs/predictcsl-main/bin/python"
SYNTH_ROOT="logs/experiments/master_recompute/context_length_dataset"
TEACHER_ROOT="logs/experiments/master_recompute/calibrated_context_risk_extreme"
COMPACT_ROOT="logs/experiments/master_recompute/compact_context_risk_all_models"
CACHE_A="logs/experiments/master_recompute/window_ablation_gifteval/general"
CACHE_B="logs/experiments/master_recompute/window_ablation_gifteval_grid_v2/general"
MAX_JOBS="${MAX_JOBS:-4}"

MODELS=(
  "Chronos2-Base|chronos2_base"
  "Chronos2-Small|chronos2_small"
  "Chronos2-Synth|chronos2_synth"
  "ChronosBolt-Base|chronos_bolt_base"
  "FlowState-R1|flowstate_r1"
  "Moirai2-Small|moirai2_small"
  "PatchTST-FM-R1|patchtst_fm_r1"
  "Sundial-Base-128M|sundial_base_128m"
  "TimesFM2.5-200M|timesfm2p5"
  "Toto-2.0-313m|toto_2_0_313m"
)

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    wait -n
  done
}

run_tree() {
  local model="$1" slug="$2"
  local out="$COMPACT_ROOT/$slug/distill_tree_depth8"
  "$PYTHON_BIN" -m experiments.calibrated_context_risk \
    --stage evaluate \
    --model-short "$model" \
    --synthetic-dir "$SYNTH_ROOT/$model" \
    --cache-roots "$CACHE_A" "$CACHE_B" \
    --output-dir "$out" \
    --n-jobs 1 > "$out/dense_real_run.log" 2>&1
}

run_teacher() {
  local model="$1" slug="$2"
  local out="$COMPACT_ROOT/$slug/teacher_dense"
  "$PYTHON_BIN" -m experiments.evaluate_context_risk_profile_override \
    --policy "$TEACHER_ROOT/$slug/policy.joblib" \
    --profiles-json "$out/synthetic_calibration.json" \
    --output-dir "$out" \
    --model-short "$model" \
    --synthetic-dir "$SYNTH_ROOT/$model" \
    --cache-roots "$CACHE_A" "$CACHE_B" \
    --n-jobs 1 > "$out/dense_real_run.log" 2>&1
}

for spec in "${MODELS[@]}"; do
  IFS='|' read -r model slug <<< "$spec"
  wait_for_slot
  run_tree "$model" "$slug" &
  wait_for_slot
  run_teacher "$model" "$slug" &
done
wait

