#!/usr/bin/env bash
# =============================================================================
# Build all PredictCSL envs in the correct order.
#   1. predictcsl-main   (workhorse; includes the Mamba predictor)
#   2. predictcsl-legacy (clone of main -> MUST run after main)
#   3. predictcsl-toto   (Toto-2.0-313m; Python 3.12, independent of the others)
#
# Run from the repo root:  bash envs/setup-all.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$HERE/setup-main.sh"
bash "$HERE/setup-legacy.sh"
bash "$HERE/setup-toto.sh"

echo
echo "Built: predictcsl-main, predictcsl-legacy, predictcsl-toto."
