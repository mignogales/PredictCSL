#!/usr/bin/env bash
# Repair an existing predictcsl-main env whose torch was upgraded after the
# torch2.4-tagged Mamba wheels were installed.
set -euo pipefail

ENV="${1:-predictcsl-main}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "Before repair:"
python - <<'PY'
import sys
print("python", sys.executable)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
except Exception as exc:
    print("torch import failed:", repr(exc))
PY

pip uninstall -y mamba-ssm causal-conv1d
pip install --force-reinstall \
    torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install --force-reinstall --no-deps \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
pip install --force-reinstall --no-deps \
    "https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

echo
echo "After repair:"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda?", torch.cuda.is_available())
from mamba_ssm import Mamba
print("mamba-ssm import OK")
PY
