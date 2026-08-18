#!/usr/bin/env bash
# Matched-resolution achieved-policy Pareto: ExtraTrees vs one depth-8 tree.
# Profile calibration and GiftEval scoring are CPU-only and use cached forecasts.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python}"
MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
SYNTH_ROOT="$MASTER_ROOT/context_length_dataset"
GRID_V1="$MASTER_ROOT/window_ablation_gifteval/general"
GRID_V2="$MASTER_ROOT/window_ablation_gifteval_grid_v2/general"
EXTRA_ROOT="$MASTER_ROOT/context_selection_reviewer_followups/fronts"
COMPACT_POLICY_ROOT="$MASTER_ROOT/compact_context_risk_all_models"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MASTER_ROOT/context_selection_reviewer_followups/selector_pareto}"
JOBS="${SELECTOR_PARETO_JOBS:-3}"
LOCK_DIR="$OUTPUT_ROOT/.run.lock"
LOG_DIR="$OUTPUT_ROOT/logs"

MODEL_SPECS=(
  "Chronos2-Base|chronos2_base"
  "Chronos2-Small|chronos2_small"
  "Chronos2-Synth|chronos2_synth"
  "ChronosBolt-Base|chronos_bolt_base"
  "FlowState-R1|flowstate_r1"
  "Moirai2-Small|moirai2_small"
  "PatchTST-FM-R1|patchtst_fm_r1"
  "Sundial-Base-128M|sundial_base_128m"
  "TiRex2|tirex2"
  "TimesFM2.5-200M|timesfm2p5"
  "Toto-2.0-313m|toto_2_0_313m"
)

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SELECTOR_PARETO_JOBS must be a positive integer, got: $JOBS" >&2
  exit 2
fi
mkdir -p "$LOG_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Selector Pareto suite already active: $LOCK_DIR" >&2
  exit 2
fi
cleanup() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
trap cleanup EXIT

run_one() {
  local model="$1" slug="$2"
  local policy="$COMPACT_POLICY_ROOT/$slug/distill_tree_depth8/policy.joblib"
  local profiles_root="$OUTPUT_ROOT/compact_profiles/$slug"
  local profiles="$profiles_root/full_score/synthetic_calibration.json"
  local evaluation="$OUTPUT_ROOT/compact_dense/$slug/full_score"
  local report="$evaluation/real_evaluation.json"
  local log="$LOG_DIR/$slug.log"
  local extra="$EXTRA_ROOT/$slug/full_score/real_evaluation.json"
  for required in "$policy" "$extra"; do
    if [[ ! -s "$required" ]]; then
      echo "Missing prerequisite: $required" >&2
      return 1
    fi
  done
  echo "START $model $(date --iso-8601=seconds)" >"$log"
  if [[ ! -s "$profiles" || "$profiles" -ot "$policy" ]]; then
    PYTHONPATH=. "$PYTHON_BIN" \
      -m experiments.prepare_context_risk_ablation_profiles \
      --policy "$policy" --model-short "$model" \
      --synthetic-dir "$SYNTH_ROOT/$model" \
      --output-root "$profiles_root" --dense-points 101 >>"$log" 2>&1
  fi
  if [[ ! -s "$report" || "$report" -ot "$profiles" ]]; then
    PYTHONPATH=. "$PYTHON_BIN" \
      -m experiments.evaluate_context_risk_profile_override \
      --policy "$policy" --profiles-json "$profiles" \
      --output-dir "$evaluation" --model-short "$model" \
      --synthetic-dir "$SYNTH_ROOT/$model" \
      --cache-roots "$GRID_V1" "$GRID_V2" --n-jobs 1 >>"$log" 2>&1
  fi
  echo "DONE $model $(date --iso-8601=seconds)" >>"$log"
}

echo "Launching matched ExtraTrees/depth-8 Pareto inputs with concurrency=$JOBS"
pids=()
active=0
failed=0
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model slug <<<"$spec"
  run_one "$model" "$slug" &
  pids+=("$!")
  ((active += 1))
  if (( active >= JOBS )); then
    if ! wait "${pids[0]}"; then
      failed=1
    fi
    pids=("${pids[@]:1}")
    ((active -= 1))
  fi
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed )); then
  echo "At least one selector Pareto input failed; inspect $LOG_DIR" >&2
  exit 1
fi

summary_args=()
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model slug <<<"$spec"
  summary_args+=(--model "$model=$slug")
done
PYTHONPATH=. "$PYTHON_BIN" -m experiments.summarize_selector_pareto \
  --extra-root "$EXTRA_ROOT" \
  --compact-root "$OUTPUT_ROOT/compact_dense" \
  --output-dir "$OUTPUT_ROOT/combined" \
  "${summary_args[@]}" >"$LOG_DIR/combined.log" 2>&1

echo "SELECTOR PARETO COMPLETE: $OUTPUT_ROOT"
