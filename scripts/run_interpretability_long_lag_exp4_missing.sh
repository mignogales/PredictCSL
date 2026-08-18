#!/usr/bin/env bash
# Complete the paper's missing long-lag Exp4 models without rerunning Toto.
#
# This script starts immediately when invoked. Set CUDA_VISIBLE_DEVICES to the
# desired physical GPU; inside the routed environments it is addressed as
# cuda:0. The underlying experiment runner resumes already completed cells.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_STRING="${INTERPRETABILITY_MODELS:-Chronos2-Small Moirai2-Small TimesFM2.5-200M PatchTST-FM-R1 Sundial-Base-128M}"
CONFIG="${INTERPRETABILITY_CONFIG:-context_interpretability/configs/experiments_long_lag_comparison.yaml}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to one free physical GPU before launching." >&2
  echo "Example: CUDA_VISIBLE_DEVICES=3 bash $0" >&2
  exit 2
fi

echo "Running long-lag Exp4 for: $MODEL_STRING"
echo "Visible physical GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG"

INTERPRETABILITY_MODELS="$MODEL_STRING" \
INTERPRETABILITY_EXPERIMENTS="exp4" \
INTERPRETABILITY_CONFIG="$CONFIG" \
INTERPRETABILITY_DEVICE="cuda:0" \
  bash scripts/run_interpretability_multimodel.sh
