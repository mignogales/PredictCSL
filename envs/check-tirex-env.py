"""Fail-fast validation for the dedicated TiRex2 environment."""

from importlib.metadata import version
from pathlib import Path
import sys

# Executing this file by path makes envs/ (rather than the repository root)
# sys.path[0]. Add the root so the shared TiRex compatibility checks import
# reliably even when setup-tirex.sh is launched from another directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

import numpy as np
import pyarrow as pa
import torch
from datasets.formatting.formatting import NumpyArrowExtractor
from gift_eval.data import Dataset  # noqa: F401
from tirex2 import TimeseriesType, load_model  # noqa: F401

from experiments.tirex_compat import (
    TirexCheckpointAccessError,
    require_tirex_checkpoint_access,
)


def require_version(distribution: str, expected: str) -> None:
    actual = version(distribution)
    print(distribution, actual)
    # CUDA wheels may append a PEP 440 local suffix (for example +cu126).
    if actual != expected and not actual.startswith(expected + "+"):
        raise SystemExit(
            f"predictcsl-tirex expects {distribution}=={expected}, found {actual}"
        )


print("torch", torch.__version__, "cuda", torch.version.cuda,
      "cuda?", torch.cuda.is_available())
require_version("torch", "2.8.0")
require_version("numpy", "2.1.3")
require_version("datasets", "2.21.0")
require_version("tirex-2", "0.1.1")

# Import-only checks missed the datasets 2.17.x / NumPy 2 failure: it occurs
# lazily when GIFT-Eval reads an Arrow target. Exercise that exact formatter
# boundary without requiring the benchmark data to be present on this machine.
table = pa.table({"target": [[1.0, 2.0], [3.0, 4.0]]})
targets = NumpyArrowExtractor().extract_column(table)
if targets.shape != (2, 2) or not np.array_equal(
        targets, np.asarray([[1.0, 2.0], [3.0, 4.0]])):
    raise SystemExit(
        f"datasets NumPy formatter returned unexpected target data: {targets!r}"
    )

print("tirex2 import OK")
print("gift_eval import OK")
print("datasets NumPy-2 formatter OK")
try:
    require_tirex_checkpoint_access()
except TirexCheckpointAccessError as exc:
    raise SystemExit(str(exc)) from exc
print("TiRex GIFT-Eval checkpoint access OK")
