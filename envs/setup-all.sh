#!/usr/bin/env bash
# =============================================================================
# Build all PredictCSL envs in the correct order.
#   1. predictcsl-main   (workhorse; includes the Mamba predictor)
#   2. predictcsl-patchtst (clone of main -> torch 2.8, published PatchTST path)
#   3. predictcsl-legacy (clone of main -> MUST run after main)
#   4. predictcsl-toto   (Toto-2.0-313m; Python 3.12, independent of the others)
#   5. predictcsl-tirex  (TiRex2; torch>=2.8 / numpy 2.x)
#
# Run from the repo root:  bash envs/setup-all.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$HERE/setup-main.sh"
bash "$HERE/setup-patchtst.sh"
bash "$HERE/setup-legacy.sh"
bash "$HERE/setup-toto.sh"
bash "$HERE/setup-tirex.sh"

echo
echo "Built: predictcsl-main, predictcsl-patchtst, predictcsl-legacy, predictcsl-toto, predictcsl-tirex."
