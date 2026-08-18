#!/usr/bin/env bash
# Re-evaluate completed calibrated-risk policies after changing only the
# analytical TSFM MAC proxy. This consumes cached forecasts and trained trees;
# it never loads a TSFM or uses a GPU.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python}"
MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
GRID_V1="${GRID_V1:-$MASTER_ROOT/window_ablation_gifteval/general}"
GRID_V2="${GRID_V2:-$MASTER_ROOT/window_ablation_gifteval_grid_v2/general}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

OUTPUT_ROOTS=(
  calibrated_context_risk
  calibrated_context_risk_very_aggressive
  calibrated_context_risk_very_very_aggressive
  calibrated_context_risk_extreme
)

# These checkpoints were affected by this audit: Chronos2 Small was previously
# priced as Base, Base used Small's output-head width, FlowState used a
# Transformer proxy, Moirai omitted recursive quantile paths, and TiRex omitted
# its fixed native grid and sign-flip TTA. MODEL_SPECS may select a comma-separated
# subset, for example "Moirai2-Small|moirai2_small,TiRex2|tirex2".
MODELS=(
  "Chronos2-Small|chronos2_small"
  "Chronos2-Base|chronos2_base"
  "FlowState-R1|flowstate_r1"
  "Moirai2-Small|moirai2_small"
  "TiRex2|tirex2"
)
if [[ -n "${MODEL_SPECS:-}" ]]; then
  IFS=',' read -r -a MODELS <<<"$MODEL_SPECS"
fi

run_one() {
  local root="$1"
  local model="$2"
  local slug="$3"
  local output="$MASTER_ROOT/$root/$slug"
  if [[ ! -s "$output/policy.joblib" ]]; then
    printf 'SKIP %s/%s (policy.joblib missing)\n' "$root" "$model"
    return 0
  fi
  printf 'START %s/%s\n' "$root" "$model"
  if PYTHONPATH=. "$PYTHON_BIN" -m experiments.calibrated_context_risk \
      --stage evaluate \
      --model-short "$model" \
      --synthetic-dir "$MASTER_ROOT/context_length_dataset/$model" \
      --cache-roots "$GRID_V1" "$GRID_V2" \
      --output-dir "$output" \
      --n-jobs 10 >"$output/recompute_flops.log" 2>&1; then
    printf 'DONE %s/%s\n' "$root" "$model"
  else
    local status=$?
    printf 'FAIL %s/%s (exit %s; see %s/recompute_flops.log)\n' \
      "$root" "$model" "$status" "$output" >&2
    return "$status"
  fi
}

if (( MAX_PARALLEL < 1 )); then
  printf 'MAX_PARALLEL must be at least 1\n' >&2
  exit 2
fi

failed=0
pids=()
names=()
for root in "${OUTPUT_ROOTS[@]}"; do
  for spec in "${MODELS[@]}"; do
    IFS='|' read -r model slug <<<"$spec"
    run_one "$root" "$model" "$slug" &
    pids+=("$!")
    names+=("$root/$model")
    if (( ${#pids[@]} >= MAX_PARALLEL )); then
      for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
          printf 'Batch member failed: %s\n' "${names[$index]}" >&2
          failed=1
        fi
      done
      pids=()
      names=()
    fi
  done
done

for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf 'Batch member failed: %s\n' "${names[$index]}" >&2
    failed=1
  fi
done

exit "$failed"
