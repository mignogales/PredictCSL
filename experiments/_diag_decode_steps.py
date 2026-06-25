"""
Diagnostic: how many backbone forward passes does each TSFM make for ONE
predict() call at a long horizon?

We hook every repeated transformer block and count how many times it fires
during a single `predict_*` call. The count tells us empirically what the decode
structure is:

  * count == 1  -> the whole horizon is produced one-shot (no per-output-patch
                   axis exists; saturation is purely context-length).
  * count  > 1  -> the horizon is rolled out autoregressively; each firing after
                   the prefill is one output patch/step, and the token-axis
                   length at that firing reveals the patch size.

This settles the "does Chronos2 really do one forward for H=1024?" question with
ground truth instead of guessing per-library internals, and tells us exactly
which models get the `e(L, k)` per-output-patch saturation capture.

Run on the SERVER (legacy/toto families need their own env):
    python -m experiments._diag_decode_steps
    python -m experiments._diag_decode_steps --models TimesFM2.5-200M Sundial-Base-128M
    python -m experiments._diag_decode_steps --horizon 1024 --window 512
    conda run -n predictcsl-legacy python -m experiments._diag_decode_steps \
        --models Sundial-Base-128M TimeMoE-200M
"""

from __future__ import annotations

import argparse
import collections
import re
import traceback

import numpy as np
import torch

from experiments import build_context_length_dataset as bcl
from experiments import models_config
from experiments.embedding_saturation import (
    DEFAULT_HOOK_PATTERN,
    _predict_dispatch,
    _resolve_backbone,
    _resolve_backbone_from,
)


class FiringCounter:
    """Count forward-hook firings per matched block; trace token-axis lengths."""

    def __init__(self, model: torch.nn.Module, pattern: str):
        rx = re.compile(pattern)
        self.counts: "collections.Counter[str]" = collections.Counter()
        self.token_lens: dict[str, list[int]] = collections.defaultdict(list)
        self.matched: list[str] = []
        self._handles = []
        for name, mod in model.named_modules():
            if rx.search(name):
                self.matched.append(name)
                self._handles.append(mod.register_forward_hook(self._make(name)))

    def _make(self, name: str):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t) or t.dim() < 2:
                return
            T = int(t.shape[-2]) if t.dim() >= 3 else 1   # token axis (B, ..., T, d)
            self.counts[name] += 1
            self.token_lens[name].append(T)
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _make_context(batch: int, window: int) -> torch.Tensor:
    """A benign (B, W, 1) batch — values don't affect firing counts."""
    t = np.arange(window, dtype=np.float32)
    base = np.sin(2 * np.pi * t / 64.0)[None, :] + 0.05 * np.random.RandomState(0).randn(batch, window)
    return torch.from_numpy(base.astype(np.float32)).unsqueeze(-1)


def _runner_and_backbone(family, base, model_id, window, horizon, batch, device):
    if family == "moirai":
        runner = bcl._build_moirai(base, horizon, window, device)
        return runner, _resolve_backbone_from(runner)
    if family == "timesfm":
        runner = bcl.load_timesfm(model_id, window, horizon, batch)
        return runner, _resolve_backbone_from(runner)
    return base, _resolve_backbone(family, base, model_id, device)


def probe(model_id, family, display, window, horizon, batch, device, pattern):
    base = bcl.setup_model(family, model_id, device)
    if isinstance(base, torch.nn.Module):
        base.eval()
    runner, backbone = _runner_and_backbone(
        family, base, model_id, window, horizon, batch, device)
    cap = FiringCounter(backbone, pattern)
    try:
        with torch.no_grad():
            _predict_dispatch(family, runner, _make_context(batch, window), horizon, device)
    finally:
        cap.remove()
        if family in ("moirai", "timesfm"):
            del runner
        del base
        if bcl._is_cuda(device):
            torch.cuda.empty_cache()

    if not cap.counts:
        print(f"  {display:<22} no block fired (matched {len(cap.matched)} modules) "
              f"— refine --hook-pattern")
        return
    n_passes = max(cap.counts.values())                 # forwards through the stack
    deepest = cap.matched[-1]                            # last-indexed block
    trace = cap.token_lens[deepest][:12]
    kind = "one-shot" if n_passes == 1 else f"autoregressive ({n_passes} passes)"
    print(f"  {display:<22} blocks={len(cap.matched):<3} passes={n_passes:<4} {kind}")
    print(f"      token-axis per firing of {deepest!r}: {trace}"
          + (" …" if len(cap.token_lens[deepest]) > 12 else ""))


def _models(args):
    catalog = {disp: (mid, fam) for (mid, fam, disp) in models_config.catalog()}
    names = args.models or [disp for (_m, _f, disp) in models_config.models_to_run()]
    for n in names:
        if n not in catalog:
            raise SystemExit(f"Unknown model {n!r}. Known: {sorted(catalog)}")
        mid, fam = catalog[n]
        yield mid, fam, n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--horizon", type=int, default=1024)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--hook-pattern", default=DEFAULT_HOOK_PATTERN)
    args = p.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  window={args.window}  horizon={args.horizon}\n")
    for mid, fam, disp in _models(args):
        try:
            probe(mid, fam, disp, args.window, args.horizon, args.batch,
                  device, args.hook_pattern)
        except Exception as e:                            # noqa: BLE001
            print(f"  {disp:<22} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
