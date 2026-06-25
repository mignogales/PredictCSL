"""
Embedding-saturation experiment — the *representational* twin of the error-vs-
context ablation.

Idea
----
"Useful context length" is normally defined behaviorally: the input window that
minimizes a TSFM's forecast error (that is what build_context_length_dataset.py
measures and predict_context_length.py learns). This script asks the same
question mechanistically:

    As we feed a model more tail context, when does its internal summary of the
    series stop changing?

For one series we take the nested tail windows in WINDOW_GRID. At each window L
we run the model exactly as the real pipeline does (same normalization / NaN
padding) and capture the *context embedding* e(L): the final-block hidden state,
read at the most-recent-aligned position (last token) and mean-pooled. Every
window forecasts the SAME future, so the embedding's job is held fixed and only
how much past it sees varies. As L grows e(L) should converge — that convergence
is saturation.

Two curves per series (cosine distance, raw + dataset-centered):
  * to-asymptote  s(L)  = cos_dist(e(L), e(L_max))      -> decreases to 0
  * marginal      d(L)  = cos_dist(e(L), e(L_prev))     -> "did this extra chunk
                                                            of history move me?"

The saturation length L*_embed = smallest grid window where the marginal update
falls below a threshold (plateau). Non-monotonicity in d(L) — the marginal
update rising again after a plateau — is the embedding-space fingerprint of
"stale context perturbs the representation", i.e. more-context-hurts.

Per-output-step capture (autoregressive models)
-----------------------------------------------
Many TSFMs do NOT produce a long horizon in one forward — they roll it out
autoregressively in output patches, re-running the backbone once per step
(measured with experiments._diag_decode_steps: ChronosBolt 16×64, Moirai2
16×64, TimeMoE 64-pt steps, Sundial ~512-pt steps; ChronosBolt even re-encodes
the *growing* context each step). We therefore log the final-block embedding at
EVERY generation step, giving e(L, k): the representation as the model emits
output-patch k. Embeddings are stored as (N, K_windows, S_steps, d). One-shot
models (Chronos2, Toto, FlowState, and — when it forecasts in a single step —
PatchTST-FM) have S==1 and collapse to the original single-embedding case.

This adds a second saturation axis read "as we generate". The S axis is aligned
to the final output patch (step -1 = most-recent generation step), so it stays
comparable across context windows even for models (TimesFM) whose prefill firing
count grows with context. The canonical context-axis curve uses that final step
(reproducing the original single-embedding behavior); `*_steps` arrays carry the
per-step context curves; and `gen_marginal[k, s]` asks whether emitting step s
moved the representation — so long horizons can reveal later output patches
saturating at a different context length than earlier ones.

Headline test (done in post-processing, not here): does L*_embed line up with
L*_err = argmin of the error curve from the stage-1 / stage-3 ablation? If yes,
useful context is detectable from internal states with NO forecast scoring, no
horizon, no ground truth.

Sources
-------
* gifteval  — reuses the v5 GiftEvalCache + DATASETS, so cells line up 1:1 with
  the error ablation for the L*_embed vs L*_err scatter. This is the default
  source (see --sources).
* synthetic — reuses build_context_length_dataset.generate_dataset (the same
  non-stationary pool). We know each series' regime count, so saturation can be
  checked against regime structure. Still wired up but OFF by default.

NOTE (for later — synthetic saturation): the embedding saturation of the
synthetic pool *on its own* isn't very interesting — it's just "does the model
settle". It becomes interesting only once we **control the synthetic generator**
and draw parallels between saturation behavior and known data characteristics
(regime count / shift magnitude / per-segment trend-AR-seasonality / spike
density / wave type). Then L*_embed becomes a probe of *which* data property
drives representational saturation. generate_dataset already returns n_segments
per series; the next step is to expose the rest of the per-series generative
metadata and correlate it against L*_embed. Deferred for now; gifteval first.

Hooking
-------
We do NOT hand-maintain per-architecture module paths. Instead we hook every
sub-module whose qualified name ends in a repeated-block index
(``.layers.N`` / ``.blocks.N`` / ``.block.N`` / ``.h.N`` — the standard HF
ModuleList naming for T5, PatchTST, Llama/TimeMoE, GPT2-style, Moirai, … — plus
``.stacked_xf.N`` for TimesFM 2.5, whose blocks use a non-standard name) and
keep the output of the LAST one that fires (= the final transformer block).
Forward hooks fire in execution order, so this is robust across the whole model
zoo. For encoder-decoder models (chronos / T5) we prefer the last *encoder*
block (the context summary) over decoder blocks. Use ``--dump-modules`` to print
the matched modules for a model on the server, and ``--hook-pattern`` to refine.

Outputs (logs/experiments/embedding_saturation/)
------------------------------------------------
  synthetic/<model>/saturation.npz   per-series curves + L*_embed + metadata
  gifteval/<dataset>/<model>/t<term>/saturation.npz
  .../summary.json                   scalars + config (also the done-marker)
  .../plots/*.png                    saturation curves + L*_embed histogram
  (raw embeddings only with --save-embeddings; they are large)

Run on the SERVER, e.g.:
  python -m experiments.embedding_saturation --test          # tiny smoke run
  python -m experiments.embedding_saturation                  # full run
  python -m experiments.embedding_saturation --sources synthetic --models TiRex
  python -m experiments.embedding_saturation --dump-modules --models TiRex
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from colorama import Fore
except Exception:                                            # pragma: no cover
    class _NoColor:
        def __getattr__(self, _):  # noqa: D401
            return ""
    Fore = _NoColor()                                        # type: ignore

try:
    from dotenv import load_dotenv
except Exception:                                            # pragma: no cover
    def load_dotenv(*_a, **_k):                              # type: ignore
        return False

try:
    from tqdm.auto import tqdm
except Exception:                                            # pragma: no cover
    def tqdm(it=None, *_a, **_k):                            # type: ignore
        return it if it is not None else iter(())

from experiments import build_context_length_dataset as bcl
from experiments import models_config
# Cross-env routing: single source of truth lives in master_run_all so the
# experiment dispatches Sundial/TimeMoE -> predictcsl-legacy and Toto ->
# predictcsl-toto exactly like the rest of the pipeline.
from experiments.master_run_all import FAMILY_ENV, _env_label, _py

load_dotenv()


class _Tee:
    """Mirror a stream to a log file so dump-modules / runs leave a readable log."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        self._fh.flush()

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _install_log(path: str) -> None:
    """Tee stdout+stderr into `path` (append). Kept open for the process life."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fh = open(path, "a", buffering=1)                        # noqa: SIM115
    import datetime as _dt
    fh.write(f"\n===== {_dt.datetime.now().isoformat()}  "
             f"argv={' '.join(sys.argv[1:])} =====\n")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    print(Fore.CYAN + f"Logging to {path}" + Fore.RESET)

# ==============================================================================
#  CONFIG
# ==============================================================================
OUT_ROOT = "logs/experiments/embedding_saturation"

# Repeated-block naming across transformer stacks: T5/PatchTST `block.N`,
# Llama/TimeMoE/Moirai `layers.N`, GPT2 `h.N`, and TimesFM 2.5's `stacked_xf.N`.
# TimesFM's blocks are NOT named layers/blocks, so without `stacked_xf` here its
# trunk is invisible to the hook and find_backbone gives up (the observed
# "Could not locate backbone for per-width runner"). `_block_level` keeps only
# the top-level blocks, so adding a stack name to this alternation is safe.
DEFAULT_HOOK_PATTERN = r"(?:^|\.)(?:layers?|blocks?|h|stacked_xf)\.\d+$"

# Marginal-update plateau thresholds (cosine distance between consecutive
# windows) -> L*_embed. First grid window at/under the threshold.
MARGINAL_THRESHOLDS = (0.05, 0.02, 0.01)
# To-asymptote thresholds (cosine distance to the longest-context embedding).
ASYMPTOTE_THRESHOLDS = (0.10, 0.05, 0.02)

# Synthetic pool: this is an analysis experiment, not labeling, so a few
# thousand series is plenty (and keeps embedding arrays small).
DEFAULT_SYNTH_SERIES = 2000
DEFAULT_SYNTH_SEED = 1234

DEFAULT_BATCH_SIZE = 64
# A small, fast, variable-length model makes the best smoke-test default: it
# exercises the whole path quickly across every dataset.
TEST_MODEL = "Chronos2-Small"


# ==============================================================================
#  HIDDEN-STATE CAPTURE
# ==============================================================================
def _reduce_hidden(h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce a block output to ``(last_token (B, d), mean_pooled (B, d))``.

    General shape is (B, ..., T, d). Most stacks emit (B, T, d), but some (e.g.
    Moirai) insert an extra grouping axis — (B, group, T, d) — so we keep batch
    (dim 0) and features (last dim) and collapse everything in between:
      last-token  = last index along the token axis, then mean any grouping axes;
      mean-pooled = mean over all axes except batch and feature.
    """
    if h.dim() == 2:                                         # already pooled (B, d)
        return h, h
    last = h[..., -1, :]                                     # drop token axis
    while last.dim() > 2:
        last = last.mean(dim=1)
    mean = h
    while mean.dim() > 2:
        mean = mean.mean(dim=1)
    return last, mean


class HiddenCapture:
    """Log block hidden states across a forward/generate, per generation step.

    Hooks every repeated block (the regex matches both a block and its repeated
    sub-layers). We log every firing whose output is a usable tensor, in
    execution order, then :meth:`read_steps` returns the FINAL block's per-step
    embeddings ``(B, S, d)`` (last-token + mean-pooled). "Final block" = the last
    *valid* firing in execution order, preferring the deepest *encoder* block
    (the context summary for enc-dec models; ChronosBolt re-runs it each step
    over the growing context). Picking the last *valid* firing — rather than a
    fixed top-level block — degrades gracefully when a parent block returns a
    non-tensor tuple (Chronos2) and falls back to its deepest tensor-returning
    sub-layer, matching the original capture; where the parent does return a
    tensor it fires last and wins (post-residual block output, not an MLP
    intermediate). For a single-forward (one-shot) model S==1. :meth:`read`
    keeps the legacy single-embedding contract (final step). After the first
    forward the chosen block name is cached, so subsequent batches log only that
    one module and the transient log stays bounded.
    """

    def __init__(self, model: torch.nn.Module, pattern: str):
        self._rx = re.compile(pattern)
        self._handles = []
        self.matched: List[str] = []
        for name, mod in model.named_modules():
            if self._rx.search(name):
                self.matched.append(name)
                self._handles.append(
                    mod.register_forward_hook(self._make_hook(name)))
        # execution-order log: list[(name, last (B, d) cpu, mean (B, d) cpu)]
        self._log: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        self._final_name: Optional[str] = None   # cached after first read

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            # Once the final block is known, only log it (bounds the log size).
            if self._final_name is not None and name != self._final_name:
                return
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t) or t.dim() < 2:
                return
            last, mean = _reduce_hidden(t.detach().to(torch.float32))
            self._log.append((name, last.cpu(), mean.cpu()))
        return hook

    def reset(self) -> None:
        self._log = []                            # keep _final_name across batches

    def _pick_name(self) -> Optional[str]:
        """Final block whose per-step firings we read: deepest encoder if any
        encoder block fired (enc-dec context summary), else the last block fired
        overall (decoder-only / single trunk). Honors the cached name."""
        if self._final_name is not None:
            return self._final_name if any(
                n == self._final_name for (n, _, _) in self._log) else None
        if not self._log:
            return None
        enc = [n for (n, _, _) in self._log if "encoder" in n.lower()]
        return enc[-1] if enc else self._log[-1][0]

    def read_steps(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """``(last (B, S, d), mean (B, S, d))`` over the S generation steps."""
        name = self._pick_name()
        if name is None:
            return None
        seq = [(l, m) for (n, l, m) in self._log if n == name]
        if not seq:
            return None
        last = torch.stack([l for (l, _) in seq], dim=1)     # (B, S, d)
        mean = torch.stack([m for (_, m) in seq], dim=1)
        self._final_name = name                              # cache for next batch
        return last, mean

    def read(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        steps = self.read_steps()
        if steps is None:
            return None
        last, mean = steps
        return last[:, -1, :], mean[:, -1, :]                # legacy: final step

    def remove(self) -> None:
        for hd in self._handles:
            hd.remove()
        self._handles.clear()


def _predict_dispatch(family, runner, xb, horizon, device):
    """Drive the real forward pass (used only for its side effect: the hook)."""
    fn = {
        "chronos2": bcl.predict_chronos2,
        "chronos_bolt": bcl.predict_chronos_bolt,
        "moirai": bcl.predict_moirai,
        "timesfm": bcl.predict_timesfm,
        "patchtst_fm": bcl.predict_patchtst_fm,
        "sundial": bcl.predict_sundial,
        "timemoe": bcl.predict_timemoe,
        "toto": bcl.predict_toto,
        "flowstate": bcl.predict_flowstate,
        "tirex": bcl.predict_tirex,
    }.get(family)
    if fn is None:
        raise ValueError(f"Unknown model family: {family}")
    return fn(runner, xb, horizon, device)


def _resolve_backbone(family, base, model_id, device):
    """Return the nn.Module whose sub-tree carries the transformer blocks.

    Pipelines (chronos) wrap the nn.Module; moirai/timesfm build per-width. We
    try a few common attribute paths and fall back to the object itself if it is
    already an nn.Module.
    """
    if family in ("moirai", "timesfm"):
        return None                                          # resolved per width
    for attr_path in ("", "model", "inner_model", "model.model", "module"):
        obj = base
        ok = True
        for a in filter(None, attr_path.split(".")):
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok and isinstance(obj, torch.nn.Module):
            # Prefer the deepest candidate that actually contains block modules.
            if any(re.search(DEFAULT_HOOK_PATTERN, n) for n, _ in obj.named_modules()):
                return obj
    if isinstance(base, torch.nn.Module):
        return base
    raise RuntimeError(
        f"Could not locate an nn.Module backbone for family={family}. "
        f"Run with --dump-modules to inspect.")


def extract_uniform(
    family, base, model_id, x_all, width, horizon, batch_size, device, pattern,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Embeddings for a uniform-width batch.

    Returns ``(last (n, S, d), mean (n, S, d))`` where S is the number of
    generation steps the model used for this horizon (1 for one-shot models)."""
    n = x_all.shape[0]
    if family == "moirai":
        runner = bcl._build_moirai(base, horizon, width, device)
        backbone = _resolve_backbone_from(runner)
    elif family == "timesfm":
        runner = bcl.load_timesfm(model_id, width, horizon, batch_size)
        backbone = _resolve_backbone_from(runner)
    else:
        runner = base
        backbone = _resolve_backbone(family, base, model_id, device)

    cap = HiddenCapture(backbone, pattern)
    last_chunks: List[torch.Tensor] = []
    mean_chunks: List[torch.Tensor] = []
    try:
        with torch.no_grad():
            for start in range(0, n, batch_size):
                xb = x_all[start:start + batch_size]
                cap.reset()
                _predict_dispatch(family, runner, xb, horizon, device)
                got = cap.read_steps()
                if got is None:
                    raise RuntimeError(
                        f"No hidden state captured for family={family} "
                        f"(pattern={pattern!r}, matched {len(cap.matched)} modules). "
                        f"Use --dump-modules / --hook-pattern.")
                last, mean = got
                last_chunks.append(last.cpu())
                mean_chunks.append(mean.cpu())
    finally:
        cap.remove()
        if family in ("moirai", "timesfm"):
            del runner
            if bcl._is_cuda(device):
                torch.cuda.empty_cache()

    return (torch.cat(last_chunks).numpy(),
            torch.cat(mean_chunks).numpy(),
            cap.matched)


def find_backbone(obj, pattern=DEFAULT_HOOK_PATTERN, max_depth=5):
    """Recursively locate the nn.Module whose sub-tree carries the blocks.

    Wrappers (timesfm/moirai/pipelines) bury the torch module under arbitrary
    attribute names, so instead of guessing names we walk ``__dict__`` + ``dir``
    and return the nn.Module with the MOST pattern-matching sub-modules (the
    trunk), shallowest on ties. Also returns the largest nn.Module seen as a
    fallback for diagnostics when nothing matches.

    Returns ``(match, fallback)`` — either may be None.
    """
    rx = re.compile(pattern)
    seen = set()
    best = None        # (match_count, -depth, module)
    fallback = None    # (total_modules, module)

    def visit(o, depth):
        nonlocal best, fallback
        if depth > max_depth or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, torch.nn.Module):
            names = [n for n, _ in o.named_modules()]
            cnt = sum(1 for n in names if rx.search(n))
            if fallback is None or len(names) > fallback[0]:
                fallback = (len(names), o)
            if cnt > 0:
                key = (cnt, -depth)
                if best is None or key > best[:2]:
                    best = (cnt, -depth, o)
            return                      # named_modules already covered the subtree
        children = []
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict):
            children.extend(d.values())
        for name in dir(o):
            if name.startswith("__"):
                continue
            try:
                children.append(getattr(o, name))
            except Exception:           # noqa: BLE001  (properties may raise)
                continue
        for v in children:
            if isinstance(v, torch.nn.Module) or (
                    hasattr(v, "__dict__")
                    and not isinstance(v, (str, bytes, int, float, bool,
                                           np.ndarray, torch.Tensor, type))):
                visit(v, depth + 1)

    visit(obj, 0)
    return (best[2] if best else None, fallback[1] if fallback else None)


def _resolve_backbone_from(runner):
    """Backbone for per-width runners (moirai forecast obj / timesfm wrapper)."""
    if isinstance(runner, torch.nn.Module) and any(
            re.search(DEFAULT_HOOK_PATTERN, n) for n, _ in runner.named_modules()):
        return runner
    match, _ = find_backbone(runner)
    if match is not None:
        return match
    raise RuntimeError("Could not locate backbone for per-width runner.")


# ==============================================================================
#  SATURATION METRICS
# ==============================================================================
def _cos_dist_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine distance between (n, d) arrays."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.sum(an * bn, axis=1)


def saturation_curves(emb: np.ndarray) -> Dict[str, np.ndarray]:
    """emb: (N, K, d) over the K windows (ascending). Returns curves over K.

    to_asymp[:, k] = cos_dist(e_k, e_{K-1})    (0 at k=K-1)
    marginal[:, k] = cos_dist(e_k, e_{k-1})    (0 at k=0 by convention)
    """
    N, K, _ = emb.shape
    to_asymp = np.zeros((N, K), dtype=np.float32)
    marginal = np.zeros((N, K), dtype=np.float32)
    e_last = emb[:, -1, :]
    for k in range(K):
        to_asymp[:, k] = _cos_dist_rows(emb[:, k, :], e_last)
        if k > 0:
            marginal[:, k] = _cos_dist_rows(emb[:, k, :], emb[:, k - 1, :])
    return {"to_asymp": to_asymp, "marginal": marginal}


def _first_crossing(curve: np.ndarray, windows: np.ndarray, thr: float) -> np.ndarray:
    """Smallest window whose curve value <= thr; else the max window."""
    below = curve <= thr
    out = np.full(curve.shape[0], windows[-1], dtype=np.int64)
    has = below.any(axis=1)
    first = np.argmax(below, axis=1)                          # first True index
    out[has] = windows[first[has]]
    return out


def summarize_saturation(
    curves: Dict[str, np.ndarray], windows: np.ndarray,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    # marginal[:, 0] is 0 by convention (no previous window) — mask it so the
    # plateau detector never trivially fires at the smallest window.
    marg = curves["marginal"].copy()
    marg[:, 0] = np.inf
    for thr in MARGINAL_THRESHOLDS:
        out[f"Lstar_marginal_{thr}"] = _first_crossing(marg, windows, thr)
    for thr in ASYMPTOTE_THRESHOLDS:
        out[f"Lstar_asymp_{thr}"] = _first_crossing(
            curves["to_asymp"], windows, thr)
    # Drift: largest marginal update occurring AFTER the first plateau (rise of
    # the representation once stale context is added). 0 = cleanly saturating.
    K = marg.shape[1]
    plateau_idx = np.argmax(marg <= MARGINAL_THRESHOLDS[0], axis=1)
    drift = np.zeros(marg.shape[0], dtype=np.float32)
    for i in range(marg.shape[0]):
        p = plateau_idx[i]
        if p + 1 < K:
            drift[i] = float(marg[i, p + 1:].max())
    out["drift_score"] = drift
    return out


def saturation_curves_steps(emb: np.ndarray) -> Dict[str, np.ndarray]:
    """Context-axis saturation curves per generation step.

    emb: (N, K, S, d). Returns (N, K, S): for each output step s, the same
    to_asymp/marginal curves over the K context windows. Step 0 is the
    context-only encoding; later steps fold in self-generated history.
    """
    N, K, S, _ = emb.shape
    to_asymp = np.zeros((N, K, S), dtype=np.float32)
    marginal = np.zeros((N, K, S), dtype=np.float32)
    for s in range(S):
        c = saturation_curves(emb[:, :, s, :])
        to_asymp[:, :, s] = c["to_asymp"]
        marginal[:, :, s] = c["marginal"]
    return {"to_asymp": to_asymp, "marginal": marginal}


def generation_curves(emb: np.ndarray) -> Dict[str, np.ndarray]:
    """Step-axis saturation: how the representation moves AS we generate.

    emb: (N, K, S, d). Returns (N, K, S):
      gen_to_asymp[:, k, s] = cos_dist(e(L_k, s), e(L_k, S-1))  (0 at the last step)
      gen_marginal[:, k, s] = cos_dist(e(L_k, s), e(L_k, s-1))  (0 at s=0)
    For one-shot models (S==1) both are all-zero. Non-trivial gen_marginal at
    late steps is "emitting this output-patch perturbed the representation".
    """
    N, K, S, _ = emb.shape
    gen_to_asymp = np.zeros((N, K, S), dtype=np.float32)
    gen_marginal = np.zeros((N, K, S), dtype=np.float32)
    for k in range(K):
        e = emb[:, k, :, :]                                   # (N, S, d)
        e_last = e[:, -1, :]
        for s in range(S):
            gen_to_asymp[:, k, s] = _cos_dist_rows(e[:, s, :], e_last)
            if s > 0:
                gen_marginal[:, k, s] = _cos_dist_rows(e[:, s, :], e[:, s - 1, :])
    return {"gen_to_asymp": gen_to_asymp, "gen_marginal": gen_marginal}


# ==============================================================================
#  EMBEDDING COLLECTION OVER THE GRID
# ==============================================================================
def collect_grid_embeddings(
    family, base, model_id, contexts: np.ndarray, real_lengths: np.ndarray,
    horizon: int, windows: List[int], batch_size: int, device: str, pattern: str,
    progress_desc: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """(N, K, S, d) embeddings (last-token) for every series at every window.

    Mirrors forecast_window: each series is fed its last min(L, real_len)
    genuine samples, bucketed down to the grid, so the embedding flattens once a
    short series runs out of context. The S axis holds per-generation-step
    embeddings (S==1 for one-shot models), aligned to the final output patch
    (step -1 = most-recent), so e[:, k, -1] is the final-block embedding after
    the last generated patch at context L_k. When a model's firing count varies
    with context (TimesFM), S is the running minimum across the cell and earlier
    steps are trimmed from the front. We keep last-token embeddings (most-recent
    aligned, robust to NaN-padded models); mean-pooled stored separately.

    ``progress_desc`` (if given) labels a per-window tqdm bar so a long cell
    shows how far the embedding sweep has progressed.
    """
    N = contexts.shape[0]
    grid = np.asarray(sorted(set(bcl.WINDOW_GRID)))
    emb_last: Optional[np.ndarray] = None
    emb_mean: Optional[np.ndarray] = None
    matched: List[str] = []

    d0: Optional[int] = None                                  # first-seen d_model
    s0: Optional[int] = None                                  # running MIN n_steps
    # Most models use a fixed number of generation steps per horizon, but some
    # (TimesFM) interleave context re-encoding with generation, so the firing
    # count grows with context length and varies across windows AND across the
    # bucket widths inside one window. We therefore align the step axis to the
    # final output patch (the most-recent step, == the legacy single embedding):
    # keep the running minimum S and trim already-stored steps from the FRONT so
    # every cell ends up with the same S = min steps, indexed backward from the
    # last generation step. Constant-S models keep all their steps (no trim).
    bar = tqdm(total=len(windows), desc=progress_desc, unit="win",
               leave=False, disable=progress_desc is None)
    try:
      # Largest windows first: the slow (long-context) cells run early so tqdm
      # OVER-estimates the remaining time rather than under-estimating it. k is
      # kept as the true window index, so the embedding slots are unaffected.
      for k in reversed(range(len(windows))):
        L = windows[k]
        bar.set_postfix_str(f"L={int(L)}", refresh=False)
        eff = np.minimum(int(L), np.asarray(real_lengths))
        eff_buck = np.minimum(
            eff, grid[np.clip(np.searchsorted(grid, eff, side="right") - 1, 0, None)])
        for width in np.unique(eff_buck):
            idx = np.flatnonzero(eff_buck == width)
            x_grp = torch.from_numpy(
                np.ascontiguousarray(contexts[idx, -int(width):])).unsqueeze(-1)
            last, mean, matched = extract_uniform(
                family, base, model_id, x_grp, int(width), horizon,
                batch_size, device, pattern)
            # --- shape diagnostics: shape mismatches here are the failure mode
            # the user has hit, so we make them loud and actionable rather than
            # letting a bare numpy broadcast error bubble up context-free.
            if last.ndim != 3 or last.shape[0] != idx.shape[0]:
                raise RuntimeError(
                    f"embedding row count mismatch: family={family} "
                    f"window L={int(L)} bucket-width={int(width)} expected "
                    f"{idx.shape[0]} rows got last.shape={tuple(last.shape)}, "
                    f"mean.shape={tuple(mean.shape)}, x_grp.shape={tuple(x_grp.shape)}. "
                    f"(N={N}, n_windows={len(windows)})")
            S_w = last.shape[1]
            if d0 is None:
                d0, s0 = last.shape[2], S_w
            if last.shape[2] != d0:
                raise RuntimeError(
                    f"embedding d_model changed across windows: family={family} "
                    f"window L={int(L)} bucket-width={int(width)} got "
                    f"d={last.shape[2]} but first window gave d={d0}. The hooked "
                    f"module ({len(matched)} matched) returns a different feature "
                    f"width per context size — refine --hook-pattern / inspect "
                    f"with --dump-modules.")
            if emb_last is None:
                emb_last = np.zeros((N, len(windows), s0, d0), dtype=np.float32)
                emb_mean = np.zeros((N, len(windows), s0, d0), dtype=np.float32)
            elif S_w < s0:                       # shrink stored steps to new min (tail)
                emb_last = np.ascontiguousarray(emb_last[:, :, s0 - S_w:, :])
                emb_mean = np.ascontiguousarray(emb_mean[:, :, s0 - S_w:, :])
                s0 = S_w
            emb_last[idx, k, :, :] = last[:, S_w - s0:, :]     # this extract's last s0
            emb_mean[idx, k, :, :] = mean[:, S_w - s0:, :]
        bar.update(1)
    finally:
      bar.close()
    return emb_last, emb_mean, matched


# ==============================================================================
#  PLOTS
# ==============================================================================
def _plots(out_dir, windows, curves, summary, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, c in (("marginal", curves["marginal"]), ("to_asymp", curves["to_asymp"])):
        a = ax[0] if name == "marginal" else ax[1]
        med = np.median(c, axis=0)
        q1, q3 = np.percentile(c, [25, 75], axis=0)
        a.plot(windows, med, marker="o", label="median")
        a.fill_between(windows, q1, q3, alpha=0.2, label="IQR")
        a.set_xscale("log"); a.set_xlabel("context length"); a.set_title(name)
        a.set_ylabel("cosine distance"); a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "saturation_curves.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ls = summary[f"Lstar_marginal_{MARGINAL_THRESHOLDS[0]}"]
    ax.hist(np.log2(ls.astype(np.float64)), bins=len(windows))
    ax.set_xlabel("log2  L*_embed (marginal<=%.2f)" % MARGINAL_THRESHOLDS[0])
    ax.set_ylabel("count"); ax.set_title(f"{title}\nsaturation length")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "Lstar_hist.png"), dpi=120)
    plt.close(fig)


def _gen_plot(out_dir, gen_curves, title):
    """Step-axis drift at the longest context: did each output-patch move the rep?"""
    g = gen_curves["gen_marginal"]                            # (N, K, S)
    if g.shape[2] < 2:                                        # one-shot: nothing to show
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    gm = g[:, -1, :]                                          # longest window, (N, S)
    steps = np.arange(gm.shape[1])
    med = np.median(gm, axis=0)
    q1, q3 = np.percentile(gm, [25, 75], axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, med, marker="o", label="median")
    ax.fill_between(steps, q1, q3, alpha=0.2, label="IQR")
    ax.set_xlabel("generation step (output patch)")
    ax.set_ylabel("cosine dist to previous step")
    ax.set_title(f"{title}\nrepresentation drift as we generate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plots", "generation_drift.png"), dpi=120)
    plt.close(fig)


def _write_cell(out_dir, windows, emb_last, emb_mean, meta, save_embeddings, title):
    os.makedirs(out_dir, exist_ok=True)
    # emb_last/emb_mean: (N, K, S, d), with the S axis aligned to the final
    # output patch (step -1 = most-recent generation step). The canonical
    # context-axis saturation uses that final step, which exactly reproduces the
    # original single-embedding behavior (for one-shot models S==1). The full
    # per-step picture lives in the *_steps / gen_* arrays below.
    S = emb_last.shape[2]
    ef = emb_last[:, :, -1, :]                                # (N, K, d), final step
    curves = saturation_curves(ef)
    # Centered (anisotropy-robust) saturation alongside raw: subtract per-window
    # mean embedding across the cell before the cosine.
    centered = ef - ef.mean(axis=0, keepdims=True)
    curves_centered = saturation_curves(centered)
    summary_arrays = summarize_saturation(curves, np.asarray(windows))
    # Per-step context curves + step-axis "as we generate" curves.
    step_curves = saturation_curves_steps(emb_last)          # (N, K, S)
    gen_curves = generation_curves(emb_last)                 # (N, K, S)

    np.savez_compressed(
        os.path.join(out_dir, "saturation.npz"),
        windows=np.asarray(windows), n_steps=np.int64(S),
        to_asymp=curves["to_asymp"], marginal=curves["marginal"],
        to_asymp_centered=curves_centered["to_asymp"],
        marginal_centered=curves_centered["marginal"],
        to_asymp_steps=step_curves["to_asymp"],
        marginal_steps=step_curves["marginal"],
        gen_to_asymp=gen_curves["gen_to_asymp"],
        gen_marginal=gen_curves["gen_marginal"],
        **summary_arrays,
        **{k: np.asarray(v) for k, v in meta.items() if np.ndim(v) >= 1},
    )
    if save_embeddings:
        np.savez_compressed(
            os.path.join(out_dir, "embeddings.npz"),
            emb_last=emb_last, emb_mean=emb_mean, windows=np.asarray(windows))

    summary = {
        "n_series": int(emb_last.shape[0]),
        "n_windows": int(emb_last.shape[1]),
        "n_steps": int(S),
        "d_model": int(emb_last.shape[3]),
        "windows": list(map(int, windows)),
        **{k: float(np.median(v)) for k, v in summary_arrays.items()},
        **{k: v for k, v in meta.items() if np.ndim(v) == 0},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _plots(out_dir, windows, curves, summary_arrays, title)
    _gen_plot(out_dir, gen_curves, title)
    return summary


# ==============================================================================
#  SOURCES
# ==============================================================================
def run_synthetic(args, family, model_id, display, base, device, windows):
    out_dir = os.path.join(args.out_root, "synthetic", display)
    if _done(out_dir) and not args.force:
        print(Fore.YELLOW + f"  [skip] synthetic/{display} (cached)" + Fore.RESET)
        return
    contexts, _targets, n_segments, real_lengths = bcl.generate_dataset(
        args.synth_series, args.synth_seed)
    horizon = bcl.HORIZON_GRID[-1]
    emb_last, emb_mean, matched = collect_grid_embeddings(
        family, base, model_id, contexts, real_lengths, horizon, windows,
        args.batch_size, device, args.hook_pattern,
        progress_desc=f"    synthetic {display}")
    meta = {"n_segments": n_segments, "real_lengths": real_lengths,
            "horizon": horizon, "source": "synthetic", "model": display,
            "hooked_modules": len(matched)}
    summary = _write_cell(out_dir, windows, emb_last, emb_mean, meta,
                          args.save_embeddings, f"synthetic — {display}")
    print(Fore.GREEN + f"  [done] synthetic/{display}  "
          f"L*med={summary[f'Lstar_marginal_{MARGINAL_THRESHOLDS[0]}']}" + Fore.RESET)


def run_gifteval(args, family, model_id, display, base, device, windows,
                 shard_id, num_shards):
    from experiments.test_window_ablation_gifteval_v5 import (
        DATASETS, GiftEvalCache)
    from gift_eval.data import Dataset as GiftEvalDataset

    wanted = _selected_datasets(args, DATASETS)
    # Cells this shard owns — enumerate them so we can print true progress
    # (i/total) instead of a silent stream that gives no sense of how far in we are.
    my_cells = [(d_idx, row) for d_idx, row in enumerate(wanted)
                if num_shards <= 1 or d_idx % num_shards == shard_id]
    total = len(my_cells)
    shard_tag = f" shard {shard_id}/{num_shards}" if num_shards > 1 else ""
    print(Fore.CYAN + f"  gifteval{shard_tag}: {total} cells for {display}"
          + Fore.RESET)
    n_done = n_skip = n_fail = 0
    for pos, (d_idx, (ge_name, term, dataset_display, to_univariate)) in enumerate(
            my_cells, start=1):
        prog = f"[{pos}/{total}]"
        out_dir = os.path.join(
            args.out_root, "gifteval", dataset_display, display, f"t{term}")
        if _done(out_dir) and not args.force:
            n_skip += 1
            print(Fore.YELLOW + f"  {prog} [skip] {dataset_display}/t{term} (cached)"
                  + Fore.RESET)
            continue
        try:
            ge = GiftEvalDataset(name=ge_name, term=term, to_univariate=to_univariate)
            cache = GiftEvalCache(ge, dataset_display)
        except Exception as e:                               # noqa: BLE001
            n_fail += 1
            print(Fore.RED + f"  {prog} [warn] load failed {dataset_display}/t{term}: {e}"
                  + Fore.RESET)
            traceback.print_exc()
            continue
        # Each cell is isolated: a shape mismatch in one dataset must not abort
        # the remaining cells for this model. Log the failing cell's full shape
        # context so the mismatch is diagnosable from the log alone.
        try:
            print(Fore.CYAN + f"  {prog} {dataset_display}/t{term} …" + Fore.RESET)
            contexts = np.stack([
                np.pad(c, (max(0, windows[-1] - len(c)), 0))[-windows[-1]:]
                if len(c) < windows[-1] else c[-windows[-1]:]
                for c in cache.contexts]).astype(np.float32)
            real_lengths = np.minimum(
                cache.context_lengths, windows[-1]).astype(np.int64)
            horizon = cache.horizon
            grid = [w for w in windows if w <= cache.max_context] or [windows[0]]
            emb_last, emb_mean, matched = collect_grid_embeddings(
                family, base, model_id, contexts, real_lengths, horizon, grid,
                args.batch_size, device, args.hook_pattern,
                progress_desc=f"    {prog} {display} {dataset_display}/t{term}")
            meta = {"real_lengths": real_lengths, "horizon": horizon,
                    "source": "gifteval", "dataset": dataset_display, "term": term,
                    "model": display, "hooked_modules": len(matched)}
            _write_cell(out_dir, grid, emb_last, emb_mean, meta,
                        args.save_embeddings, f"{dataset_display} t{term} — {display}")
            n_done += 1
            print(Fore.GREEN + f"  {prog} [done] gifteval/{dataset_display}/{display}"
                  f"/t{term}" + Fore.RESET)
        except Exception as e:                               # noqa: BLE001
            n_fail += 1
            print(Fore.RED + f"  {prog} [FAIL] {dataset_display}/t{term} ({display}): "
                  f"{type(e).__name__}: {e}" + Fore.RESET)
            print(Fore.RED + f"    n_series={len(cache.contexts)} "
                  f"max_context={cache.max_context} horizon={cache.horizon} "
                  f"grid={[w for w in windows if w <= cache.max_context] or [windows[0]]}"
                  + Fore.RESET)
            traceback.print_exc()
            continue
    print(Fore.CYAN + f"  gifteval{shard_tag} {display}: "
          f"{n_done} done, {n_skip} skipped, {n_fail} failed (of {total})"
          + Fore.RESET)


def _selected_datasets(args, DATASETS):
    # Use EVERY (dataset, term) cell — same 91 cells as the stage-3 error
    # ablation, so the L*_embed vs L*_err scatter lines up 1:1. `to_univariate`
    # is NOT an inclusion flag: it tells GiftEvalDataset to flatten a natively
    # multivariate dataset into one univariate series per variate. Each such
    # series then enters the saturation tensor as its own row (instance) and is
    # aggregated over axis 0 exactly like any other instance.
    sel = list(DATASETS)
    if args.datasets:
        sel = [r for r in sel if r[2] in args.datasets or r[0] in args.datasets]
    if args.test_datasets:
        rng = np.random.RandomState(0)
        idx = sorted(rng.choice(len(sel), min(args.test_datasets, len(sel)),
                                replace=False))
        sel = [sel[i] for i in idx]
    return sel


def _done(out_dir) -> bool:
    return os.path.exists(os.path.join(out_dir, "summary.json"))


# ==============================================================================
#  WORKER / COORDINATOR
# ==============================================================================
def _models_to_run(args) -> List[Tuple[str, str, str]]:
    catalog = {disp: (mid, fam) for (mid, fam, disp) in models_config.catalog()}
    if args.test and not args.models:
        names = [TEST_MODEL]
    elif args.models:
        names = args.models
    else:
        names = [disp for (_m, _f, disp) in models_config.models_to_run()]
    out = []
    for n in names:
        if n not in catalog:
            raise SystemExit(f"Unknown model {n!r}. Known: {sorted(catalog)}")
        mid, fam = catalog[n]
        out.append((mid, fam, n))
    return out


def run_worker(args, device, shard_id, num_shards):
    for (model_id, family, display) in _models_to_run(args):
        print(Fore.CYAN + f"\n=== {display} ({family}) on {device} ===" + Fore.RESET)
        try:
            base = bcl.setup_model(family, model_id, device)
            if base is not None and isinstance(base, torch.nn.Module):
                base.eval()
        except Exception as e:                               # noqa: BLE001
            print(Fore.RED + f"  load failed: {e}" + Fore.RESET)
            traceback.print_exc()
            continue
        if args.dump_modules:
            _dump_modules(family, base, model_id, device, args.hook_pattern)
            continue
        try:
            if "synthetic" in args.sources:
                run_synthetic(args, family, model_id, display, base,
                              device, args.windows)
            if "gifteval" in args.sources:
                run_gifteval(args, family, model_id, display, base, device,
                             args.windows, shard_id, num_shards)
        except Exception as e:                               # noqa: BLE001
            print(Fore.RED + f"  run failed for {display}: {e}" + Fore.RESET)
            traceback.print_exc()
        finally:
            del base
            if bcl._is_cuda(device):
                torch.cuda.empty_cache()


def _dump_modules(family, base, model_id, device, pattern):
    backbone = None
    if family in ("moirai", "timesfm"):
        # These build a fresh runner per context width, so inspect one concrete
        # per-width backbone (the same object the extractor hooks at run time).
        try:
            if family == "moirai":
                runner = bcl._build_moirai(base, 64, 512, device)
            else:
                runner = bcl.load_timesfm(model_id, 512, 64, 8)
        except Exception as e:                               # noqa: BLE001
            print(Fore.RED + f"  per-width runner build failed: {e}" + Fore.RESET)
            return
        print("  (per-width runner backbone — built at window=512)")
        match, fb = find_backbone(runner, pattern)
        if match is None:
            print(Fore.RED + "  no pattern match. Largest nn.Module found — "
                  "sample of ITS module names (to refine --hook-pattern):"
                  + Fore.RESET)
            if fb is not None:
                names = [n for n, _ in fb.named_modules() if n]
                for n in names[:40]:
                    print(f"    {n}")
                print(f"    … ({len(names)} total submodules)")
            else:
                print("    (no nn.Module reachable from the runner at all)")
            return
        backbone = match
    else:
        try:
            backbone = _resolve_backbone(family, base, model_id, device)
        except Exception as e:                               # noqa: BLE001
            print(Fore.RED + f"  backbone resolve failed: {e}" + Fore.RESET)
            backbone = base if isinstance(base, torch.nn.Module) else None
    if backbone is None:
        print("  (no nn.Module backbone found)")
        return
    rx = re.compile(pattern)
    matched = [n for n, _ in backbone.named_modules() if rx.search(n)]
    print(f"  pattern {pattern!r} matched {len(matched)} modules:")
    shown = matched if len(matched) <= 12 else matched[:6] + ["…"] + matched[-6:]
    for n in shown:
        print(f"    {n}")


def _resolve_env_groups(args) -> "Dict[Optional[str], List[str]]":
    """Group selected models by the conda env they must load in (main first)."""
    triples = _models_to_run(args)                       # [(model_id, family, display)]
    groups: Dict[Optional[str], List[str]] = {None: []}  # None == main env
    for (_mid, family, display) in triples:
        groups.setdefault(FAMILY_ENV.get(family), []).append(display)
    if not groups[None]:
        del groups[None]
    return groups


def _reconstruct_flags(args, displays: List[str]) -> List[str]:
    """Serialize this run's config into flags for a per-env/per-shard child.

    --test is intentionally NOT re-passed (windows/synth sizes are already
    concrete and PREDICTCSL_TEST is inherited via the environment); --gpus /
    --shard-id / --num-shards / --log-file are set per dispatch.
    """
    f = ["--sources", *args.sources, "--models", *displays,
         "--windows", *map(str, args.windows),
         "--batch-size", str(args.batch_size),
         "--synth-series", str(args.synth_series),
         "--synth-seed", str(args.synth_seed),
         "--hook-pattern", args.hook_pattern,
         "--out-root", args.out_root]
    if args.datasets:
        f += ["--datasets", *args.datasets]
    if args.test_datasets is not None:
        f += ["--test-datasets", str(args.test_datasets)]
    if args.save_embeddings:
        f += ["--save-embeddings"]
    if args.force:
        f += ["--force"]
    if args.dump_modules:
        f += ["--dump-modules"]
    return f


def _group_cmd(args, env, displays, shard_id, num_shards) -> List[str]:
    label = _env_label(env)
    stem = (f"dump_modules_{label}" if args.dump_modules
            else f"run_{label}_shard{shard_id}")
    flags = _reconstruct_flags(args, displays) + [
        "--gpus", "1", "--shard-id", str(shard_id), "--num-shards", str(num_shards),
        "--log-file", os.path.join(args.out_root, stem + ".log")]
    return _py(env, "experiments.embedding_saturation", *flags)


def coordinate(args):
    """Dispatch per conda-env group; within a group, shard datasets across GPUs.

    Main-env models reuse the current interpreter; legacy/toto models are
    launched via ``conda run -n <env>`` (FAMILY_ENV from master_run_all).
    """
    n_gpus = args.gpus or (torch.cuda.device_count()
                           if torch.cuda.is_available() else 0)
    groups = _resolve_env_groups(args)
    fails = 0
    for env, displays in groups.items():
        print(Fore.CYAN + f"\n### env {_env_label(env)}: {displays}" + Fore.RESET)
        if args.dump_modules or n_gpus <= 1:
            cmd = _group_cmd(args, env, displays, shard_id=0, num_shards=1)
            print(Fore.MAGENTA + f"  $ {' '.join(cmd)}" + Fore.RESET)
            fails += subprocess.run(cmd, env=os.environ).returncode != 0
        else:
            procs = []
            for i in range(n_gpus):
                cenv = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
                cmd = _group_cmd(args, env, displays, shard_id=i, num_shards=n_gpus)
                procs.append(subprocess.Popen(cmd, env=cenv))
            fails += sum(p.wait() != 0 for p in procs)
    if fails:
        raise SystemExit(Fore.RED + f"{fails} group/worker(s) failed." + Fore.RESET)


# ==============================================================================
#  CLI
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", nargs="+", default=["gifteval"],
                   choices=["synthetic", "gifteval"],
                   help="Default: gifteval only. Add 'synthetic' to also run the "
                        "synthetic pool (see the NOTE in the module docstring).")
    p.add_argument("--models", nargs="+", default=None,
                   help="Display names (default: full run set; TEST_MODEL with --test).")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Restrict gifteval to these dataset display/ge names.")
    p.add_argument("--windows", type=int, nargs="+", default=None,
                   help="Override the context grid (default WINDOW_GRID).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--synth-series", type=int, default=DEFAULT_SYNTH_SERIES)
    p.add_argument("--synth-seed", type=int, default=DEFAULT_SYNTH_SEED)
    p.add_argument("--hook-pattern", default=DEFAULT_HOOK_PATTERN)
    p.add_argument("--save-embeddings", action="store_true",
                   help="Also dump raw embeddings.npz (large).")
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--force", action="store_true", help="Recompute cached cells.")
    p.add_argument("--gpus", type=int, default=0, help="0 = all visible GPUs.")
    p.add_argument("--dump-modules", action="store_true",
                   help="Print hook-matched modules per model and exit.")
    p.add_argument("--log-file", default=None,
                   help="Tee stdout+stderr here (default: <out-root>/<run>.log).")
    p.add_argument("--test", action="store_true",
                   help="Smoke run: TEST_MODEL, tiny grid, few series + datasets.")
    p.add_argument("--test-datasets", type=int, default=None,
                   help="Sample this many gifteval datasets (seeded).")
    p.add_argument("--shard-id", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    if args.test:
        os.environ["PREDICTCSL_TEST"] = "1"                  # collapses WINDOW_GRID
        if args.windows is None:
            args.windows = [32, 128, 512, 2048, 8192]
        args.synth_series = min(args.synth_series, 64)
        if args.test_datasets is None:
            args.test_datasets = 2
    if args.windows is None:
        args.windows = list(bcl.WINDOW_GRID)
    return args


def main():
    args = parse_args()
    if args.log_file is None:
        stem = "dump_modules" if args.dump_modules else "run"
        if args.shard_id is not None:
            stem += f"_shard{args.shard_id}"
        args.log_file = os.path.join(args.out_root, f"{stem}.log")
    _install_log(args.log_file)
    if args.shard_id is not None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        run_worker(args, device, args.shard_id, args.num_shards)
    else:
        coordinate(args)


if __name__ == "__main__":
    main()
