#!/usr/bin/env bash
# =============================================================================
# predictcsl-mamba — minimal env for run_all_v4's bidirectional Mamba predictor.
#
# Used ONLY for: stage-2 Mamba predictor (PREDICTCSL_PREDICTOR_ARCH=mamba) and
# the cheap GiftEval stages 3-4. The heavy TSFM loaders are lazy and v4 symlinks
# the `general/` cells, so NO TSFM stack is needed here.
#
# ORDER IS LOAD-BEARING. Build from scratch — do NOT clone a TSFM env (granite-tsfm
# pins torch>=2.10 and the resolver will re-pull a too-new torch, breaking cu121).
#
# Hard constraints (learned the hard way — see memory project_server_cuda_driver):
#   * Server driver caps at CUDA 12.2  -> target cu121 (torch>=2.6 dropped cu121 wheels)
#   * torch==2.5.1+cu121 ships triton 3.1.0
#   * mamba-ssm==2.2.2  (2.3.x calls triton.set_allocator, absent in triton 3.1)
#   * causal-conv1d==1.6.2
#   * conda nvcc on the server is 11.5 (too old) -> install cuda-toolkit 12.1 in-env
# =============================================================================
set -euo pipefail

ENV=predictcsl-mamba

# 1) Fresh minimal env (no TSFM packages).
conda create -n "$ENV" python=3.11 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

# 2) Plain runtime deps (numpy<2: project scipy is built against numpy 1.x).
pip install "numpy<2" scipy pandas matplotlib colorama tqdm python-dotenv \
            einops ninja packaging

# 3) torch built for cu121 (<= the 12.2 driver ceiling). MUST come before mamba.
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121

# 4) CUDA toolkit 12.1 INTO the env so nvcc matches torch (system nvcc is 11.5).
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"

# 5) mamba-ssm __init__ eagerly imports an LM head -> needs hub + transformers.
#    Pin transformers<5 so it can't bump torch off cu121.
pip install huggingface_hub "transformers<5"

# 6) Mamba kernels. Install with --no-build-isolation AND --no-deps so they don't
#    drag in (and clobber) the pinned cu121 torch. Force a from-source build
#    (--no-binary / --no-cache-dir) to avoid stale ABI-mismatched cached wheels
#    (undefined symbol: _ZN3c10... / ncclCommResume).
pip install causal-conv1d==1.6.2 \
    --no-build-isolation --no-deps --no-binary :all: --no-cache-dir
pip install mamba-ssm==2.2.2 \
    --no-build-isolation --no-deps --no-binary :all: --no-cache-dir

# 7) gift_eval for stages 3-4 (install from source — confirm URL/ref on server).
#    Stages 1-2 of run_all_v4 don't import it, so validate stage 2 FIRST:
#       PREDICTCSL_PREDICTOR_ARCH=mamba python -m experiments.run_all_v4 --only-stages 2
#    then add gift_eval:
# pip install "git+https://github.com/SalesforceAIResearch/gift-eval.git"

echo
echo "Sanity checks:"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch fell back to CPU -> wrong CUDA build (see README, driver caps at 12.2)"
from mamba_ssm import Mamba
print("mamba-ssm import OK")
PY
echo "predictcsl-mamba ready."
