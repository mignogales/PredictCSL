#!/usr/bin/env bash
# Validation-only pilot of non-forest explainable selector alternatives.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python}"
MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MASTER_ROOT/explainable_selector_alternatives_pilot}"
JOBS="${ALTERNATIVE_JOBS:-3}"
mkdir -p "$OUTPUT_ROOT/logs"

# Representative cases: one successful rank-distillation case and the two
# clearest regressions.  No GiftEval evaluator is invoked by this launcher.
MODEL_SPECS=(
  "Chronos2-Base|chronos2_base"
  "FlowState-R1|flowstate_r1"
  "Moirai2-Small|moirai2_small"
)

run_one() {
  local model="$1" slug="$2"
  local output="$OUTPUT_ROOT/$slug"
  local summary="$output/explainable_alternatives_screen.json"
  local log="$OUTPUT_ROOT/logs/$slug.log"
  if [[ -s "$summary" ]]; then
    echo "SKIP $model (complete)" >>"$log"
    return
  fi
  echo "START $model $(date --iso-8601=seconds)" >"$log"
  PYTHONPATH=. "$PYTHON_BIN" -m experiments.explainable_selector_alternatives \
    --model-short "$model" \
    --synthetic-dir "$MASTER_ROOT/context_length_dataset/$model" \
    --teacher-policy "$MASTER_ROOT/calibrated_context_risk_extreme/$slug/policy.joblib" \
    --output-root "$output" \
    --tree-depths 8 10 --tree-min-samples-leaves 16 32 \
    --ordinal-bins 16 32 --random-tree-seeds 42 137 \
    --model-tree-depths 4 5 --model-tree-alphas 1 100 \
    --dense-points 101 >>"$log" 2>&1
  echo "DONE $model $(date --iso-8601=seconds)" >>"$log"
}

pids=()
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model slug <<<"$spec"
  run_one "$model" "$slug" &
  pids+=("$!")
  if ((${#pids[@]} >= JOBS)); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "EXPLAINABLE ALTERNATIVES PILOT COMPLETE: $OUTPUT_ROOT"
