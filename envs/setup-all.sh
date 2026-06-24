#!/usr/bin/env bash
# =============================================================================
# Build all three PredictCSL envs in the correct order.
#   1. predictcsl-main   (workhorse)
#   2. predictcsl-legacy (clone of main -> MUST run after main)
#   3. predictcsl-mamba  (from-scratch, run_all_v4 Mamba predictor)
#
# Run from the repo root:  bash envs/setup-all.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$HERE/setup-main.sh"
bash "$HERE/setup-legacy.sh"
bash "$HERE/setup-mamba.sh"

echo
echo "All three envs built: predictcsl-main, predictcsl-legacy, predictcsl-mamba."
