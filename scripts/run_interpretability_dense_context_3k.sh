#!/usr/bin/env bash
# Re-estimate the adaptive context reach on a denser grid capped at 3,000.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${INTERPRETABILITY_CONFIG:-context_interpretability/configs/experiments_long_lag_dense_3k.yaml}"
MODEL_STRING="${INTERPRETABILITY_MODELS:-Chronos2-Small Chronos2-Base Chronos2-Synth Moirai2-Small TimesFM2.5-200M PatchTST-FM-R1 Sundial-Base-128M Toto-2.0-313m FlowState-R1}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to one free physical GPU before launching." >&2
  exit 2
fi

INTERPRETABILITY_MODELS="$MODEL_STRING" \
INTERPRETABILITY_EXPERIMENTS="exp4" \
INTERPRETABILITY_CONFIG="$CONFIG" \
INTERPRETABILITY_LOG_DIR="logs/experiments/context_interpretability_long_lag_dense_3k/launch_logs" \
INTERPRETABILITY_DEVICE="cuda:0" \
  bash scripts/run_interpretability_multimodel.sh
