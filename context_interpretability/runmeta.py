"""
Reproducibility metadata for every run (spec §10.2).

Collected once at run start, finalized (runtime, peak GPU memory, per-experiment
sample counts, capability skips) at exit, and written as ``run_meta.json`` in
the run directory. Any subsampling an experiment performs MUST be recorded here
via :meth:`RunMeta.note` — never silently.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Optional


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True).strip()
    except Exception:  # noqa: BLE001 — not a git checkout on the server is fine
        return "unknown"


def _package_versions() -> Dict[str, str]:
    out: Dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("torch", "numpy", "pandas", "scipy", "transformers",
                 "gluonts", "tsfm_public"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "?"))
        except Exception:  # noqa: BLE001
            out[name] = "not installed"
    return out


class RunMeta:
    """Accumulates run metadata; write with :meth:`save` (safe to call often)."""

    def __init__(self, run_dir: str, config: dict, model: str, device: str,
                 seed: int):
        self.run_dir = run_dir
        self._t0 = time.time()
        self.data: Dict[str, object] = {
            "git_commit": _git_commit(),
            "model": model,
            "device": device,
            "seed": seed,
            "platform": platform.platform(),
            "package_versions": _package_versions(),
            "config": config,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "precision": None,            # filled by the adapter after load
            "checkpoint": None,
            "effective_context_lengths": None,
            "n_samples": {},              # experiment -> evaluated sample count
            "subsampling": {},            # experiment -> note (spec §11)
            "skipped_capabilities": [],   # explicit unsupported-method log
            "runtime_seconds": None,
            "peak_gpu_memory_bytes": None,
        }

    def note(self, key: str, value) -> None:
        self.data[key] = value

    def note_samples(self, experiment: str, n: int) -> None:
        self.data["n_samples"][experiment] = int(n)  # type: ignore[index]

    def note_subsampling(self, experiment: str, note: str) -> None:
        self.data["subsampling"][experiment] = note  # type: ignore[index]

    def skip(self, experiment: str, reason: str) -> None:
        """Log an explicit capability skip (spec §10.1) — also printed."""
        entry = {"experiment": experiment, "reason": reason}
        self.data["skipped_capabilities"].append(entry)  # type: ignore[union-attr]
        print(f"[SKIP] {self.data['model']} / {experiment}: {reason}")

    def finalize(self) -> None:
        self.data["runtime_seconds"] = round(time.time() - self._t0, 1)
        try:
            import torch
            if torch.cuda.is_available():
                self.data["peak_gpu_memory_bytes"] = int(
                    torch.cuda.max_memory_allocated())
        except Exception:  # noqa: BLE001
            pass
        self.save()

    def save(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        tmp = os.path.join(self.run_dir, "run_meta.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        os.replace(tmp, os.path.join(self.run_dir, "run_meta.json"))
