#!/usr/bin/env bash
# =============================================================================
# predictcsl-main — the workhorse env.  (Mirror of the working server env TSFM_moirai,
# captured from `pip freeze` on 2026-06-24. Python 3.11.14, torch 2.4.1+cu121.)
#
# Runs: stage-1 labeling for the modern families except PatchTST-FM, Toto,
#       Sundial, TimeMoE, and TiRex2; BOTH stage-2 predictors — PatchTST and the Mamba variant
#       (run_all_v4), since mamba-ssm lives here; stages 3-5 for the compatible
#       families, period eval, embedding saturation.
#
# Run:  bash envs/setup-main.sh
#
# KEY DESIGN POINT: the TSFM packages are installed from GIT at pinned commits,
# NOT from PyPI. PyPI granite-tsfm pins torch>=2.10 (and PyPI gluonts/chronos pull
# newer, conflicting deps); the pinned git commits below resolve cleanly against
# torch 2.4.1. Don't "simplify" these to `pip install granite-tsfm` etc.
# =============================================================================
set -euo pipefail

ENV=predictcsl-main
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINTS="$SCRIPT_DIR/predictcsl-main-constraints.txt"

conda create -n "$ENV" python=3.11 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

# 1) torch FIRST, cu121 build (<= the server's CUDA 12.2 driver ceiling).
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 2) Core scientific stack. numpy<2 (project scipy built against numpy 1.x).
pip install -c "$CONSTRAINTS" \
    numpy==1.26.4 scipy==1.11.4 pandas==2.1.4 matplotlib==3.9.4 \
    tqdm==4.67.3 colorama==0.4.6 python-dotenv==1.0.0 einops==0.7.0

# 3) HF / transformers stack (modern 4.56; legacy 4.40.1 lives in predictcsl-legacy).
pip install -c "$CONSTRAINTS" \
    transformers==4.56.0 tokenizers==0.22.2 huggingface_hub==0.36.2 \
    accelerate==1.13.0 safetensors==0.7.0 datasets==2.17.1

# 4) TSFM packages from PyPI (these two are fine on PyPI).
#    TiRex2 is intentionally NOT installed here: tirex-2 requires torch>=2.8 and
#    numpy 2.x, which conflicts with this env's torch 2.4 / numpy 1.26 stack.
#    See envs/setup-tirex.sh.
pip install -c "$CONSTRAINTS" gluonts==0.14.4 uni2ts==2.0.0

# 5) TSFM packages from GIT at the exact commits running on the server.
pip install -c "$CONSTRAINTS" "git+https://github.com/amazon-science/chronos-forecasting.git@f951d9aefa06f5389b2ed6b0e51fd5a1a4cf194b"
pip install -c "$CONSTRAINTS" "git+https://github.com/ibm-granite/granite-tsfm.git@e4d48868969281f2f4cbc520bd8354c9f9ea3d48"
pip install -c "$CONSTRAINTS" "git+https://github.com/google-research/timesfm.git@8a755c9c755fd5b1fe2f0c8af3b86d7a5b846160"
pip install -c "$CONSTRAINTS" "git+https://github.com/SalesforceAIResearch/gift-eval.git@d8184bb51079bb5021332f8e5d7486c378a52202"

# 6) Mamba predictor (run_all_v4): prebuilt cp311 / torch2.4 / cu122 wheels.
#    --no-deps so they can't drag in / clobber the pinned torch.
pip install --no-deps \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
pip install --no-deps \
    "https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

echo
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
python -c "from mamba_ssm import Mamba; print('mamba-ssm import OK')"
echo "predictcsl-main ready (PatchTST-FM/Toto/TiRex2 use dedicated envs — see envs/README.md)."
