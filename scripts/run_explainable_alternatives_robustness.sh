#!/usr/bin/env bash
# Independent-split robustness check for the winning model-tree family.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python}"
MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MASTER_ROOT/explainable_selector_alternatives_robustness}"
JOBS="${ROBUSTNESS_JOBS:-3}"
mkdir -p "$OUTPUT_ROOT/logs"

MODEL_SPECS=(
  "Chronos2-Base|chronos2_base"
  "FlowState-R1|flowstate_r1"
  "Moirai2-Small|moirai2_small"
)
SEEDS=(7 137)

run_one() {
  local model="$1" slug="$2" seed="$3"
  local output="$OUTPUT_ROOT/$slug/seed_$seed"
  local summary="$output/explainable_alternatives_screen.json"
  local log="$OUTPUT_ROOT/logs/${slug}_seed${seed}.log"
  if [[ -s "$summary" ]]; then
    echo "SKIP $model seed=$seed (complete)" >>"$log"
    return
  fi
  echo "START $model seed=$seed $(date --iso-8601=seconds)" >"$log"
  PYTHONPATH=. "$PYTHON_BIN" -m experiments.explainable_selector_alternatives \
    --model-short "$model" \
    --synthetic-dir "$MASTER_ROOT/context_length_dataset/$model" \
    --teacher-policy "$MASTER_ROOT/calibrated_context_risk_extreme/$slug/policy.joblib" \
    --output-root "$output" --seed "$seed" --families model_tree \
    --model-tree-depths 4 5 --model-tree-alphas 1 100 \
    --dense-points 101 >>"$log" 2>&1
  echo "DONE $model seed=$seed $(date --iso-8601=seconds)" >>"$log"
}

pids=()
active=0
for seed in "${SEEDS[@]}"; do
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r model slug <<<"$spec"
    run_one "$model" "$slug" "$seed" &
    pids+=("$!")
    ((active += 1))
    if ((active >= JOBS)); then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
      ((active -= 1))
    fi
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "EXPLAINABLE ROBUSTNESS COMPLETE: $OUTPUT_ROOT"
