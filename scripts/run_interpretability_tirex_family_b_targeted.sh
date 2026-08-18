#!/usr/bin/env bash
# Targeted TiRex Family-B completion for the paper plot.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${INTERPRETABILITY_CONFIG:-context_interpretability/configs/experiments_long_lag_tirex_family_b_targeted.yaml}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to one free physical GPU." >&2
  exit 2
fi

INTERPRETABILITY_MODELS="TiRex2" \
INTERPRETABILITY_EXPERIMENTS="exp4" \
INTERPRETABILITY_CONFIG="$CONFIG" \
INTERPRETABILITY_DEVICE="cuda:0" \
  bash scripts/run_interpretability_multimodel.sh

