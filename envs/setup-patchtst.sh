#!/usr/bin/env bash
# =============================================================================
# predictcsl-patchtst — PatchTST-FM-R1 leaderboard-compatible inference.
#
# The published GIFT-Eval row used granite-tsfm commit e4d488689 with its
# forward(inputs=...) API and torch 2.8 SDPA behavior. The modern main env uses
# torch 2.4 for its binary Mamba wheels; on fully masked padded query rows that
# runtime produces NaNs, so it cannot reproduce the official wrapper naturally.
# Official CPython-3.11 / torch-2.8 wheels provide the Mamba predictor here too.
#
# Prereq: predictcsl-main must already exist (run envs/setup-main.sh first).
# Run:    bash envs/setup-patchtst.sh
# =============================================================================
set -euo pipefail

ENV=predictcsl-patchtst

conda create --name "$ENV" --clone predictcsl-main -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

pip install --upgrade "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade "transformers==4.56.0"
pip install --no-deps --force-reinstall \
    "git+https://github.com/ibm-granite/granite-tsfm.git@e4d48868969281f2f4cbc520bd8354c9f9ea3d48"

# Replace main's torch-2.4 binary extensions with the matching official
# CPython-3.11 / CUDA-12 / torch-2.8 / C++11-ABI wheels.
pip install --no-deps --force-reinstall \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl" \
    "https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

echo
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
python -c "import inspect, tsfm_public; from tsfm_public import PatchTSTFMForPrediction as M; s=inspect.signature(M.forward); assert 'inputs' in s.parameters and 'past_values' not in s.parameters; assert 'e4d488689' in tsfm_public.__version__; print('granite-tsfm', tsfm_public.__version__, s)"
python -c "import causal_conv1d, mamba_ssm; from mamba_ssm import Mamba; assert causal_conv1d.__version__.startswith('1.5.4'); assert mamba_ssm.__version__.startswith('2.2.5'); print('causal-conv1d', causal_conv1d.__version__, 'mamba-ssm', mamba_ssm.__version__)"
echo "predictcsl-patchtst ready (published PatchTST path + both Mamba predictor variants)."
