#!/usr/bin/env bash
# Multi-model, long-dependency follow-up for Exp1 + Exp4 + Exp7.
#
# The master launcher routes each model to its pinned conda environment and
# performs one final analysis pass for the cross-model figures. Outputs are
# isolated under logs/experiments/context_interpretability_long_lag.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MAIN_ENV="${MAIN_ENV:-predictcsl-main}"
CONFIG="${INTERPRETABILITY_CONFIG:-context_interpretability/configs/experiments_long_lag.yaml}"
DEVICE="${INTERPRETABILITY_DEVICE:-cuda:0}"
# Default to one reference plus five architecturally distinct comparators that
# all support Exp7. Use INTERPRETABILITY_MODELS to request the full run set.
MODEL_STRING="${INTERPRETABILITY_MODELS:-Chronos2-Small Moirai2-Small TimesFM2.5-200M PatchTST-FM-R1 Sundial-Base-128M Toto-2.0-313m}"
EXPERIMENT_STRING="${INTERPRETABILITY_EXPERIMENTS:-exp1 exp4 exp7}"
LOG_DIR="${INTERPRETABILITY_LOG_DIR:-logs/experiments/context_interpretability_long_lag/launch_logs}"

read -r -a MODELS <<< "$MODEL_STRING"
read -r -a EXPERIMENTS <<< "$EXPERIMENT_STRING"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/multimodel_long_lag_${STAMP}.log"

echo "== Multi-model long-lag interpretability ==" | tee -a "$LOG_FILE"
echo "models: ${MODELS[*]}" | tee -a "$LOG_FILE"
echo "experiments: ${EXPERIMENTS[*]}" | tee -a "$LOG_FILE"
echo "config: $CONFIG" | tee -a "$LOG_FILE"
echo "device: $DEVICE" | tee -a "$LOG_FILE"

conda run --no-capture-output -n "$MAIN_ENV" \
  python -m context_interpretability.master_run \
  --models "${MODELS[@]}" \
  --experiments "${EXPERIMENTS[@]}" \
  --source synthetic \
  --config "$CONFIG" \
  --device "$DEVICE" 2>&1 | tee -a "$LOG_FILE"

echo "Finished. Consolidated log: $LOG_FILE" | tee -a "$LOG_FILE"
