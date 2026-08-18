#!/usr/bin/env bash
# Targeted planted-lag perturbation audit. Run after the active long-lag job;
# it resumes independently and must finish before the final speed benchmark.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_STRING="${INTERPRETABILITY_MODELS:-Chronos2-Base Chronos2-Small Chronos2-Synth PatchTST-FM-R1 Sundial-Base-128M}"
CONFIG="${INTERPRETABILITY_CONFIG:-context_interpretability/configs/experiments_long_lag_forced_d2048.yaml}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to one free physical GPU." >&2
  exit 2
fi

INTERPRETABILITY_MODELS="$MODEL_STRING" \
INTERPRETABILITY_EXPERIMENTS="exp4" \
INTERPRETABILITY_CONFIG="$CONFIG" \
INTERPRETABILITY_DEVICE="cuda:0" \
  bash scripts/run_interpretability_multimodel.sh
