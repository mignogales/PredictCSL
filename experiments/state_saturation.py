"""
State-saturation experiment — the *recurrent* twin of the attention-masking /
context-restriction study, for the two models the masking ablation had to
exclude.

Why this exists
---------------
``context_attention_mask.py`` restricts a transformer's *attention span* to probe
"useful context length" mechanistically. Its docstring calls out that FlowState
(SSM) and TiRex (xLSTM) are **out of scope** there: they have no attention matrix
to mask. Both are *recurrent, linear-cost* models — they carry a fixed-size
internal **state** that is updated token-by-token as the series is scanned. The
mechanistic question for them is therefore not "what can it attend to" but:

    As the recurrent state absorbs more history, when does it stop changing?

That is **state saturation**. If the state saturates after L tokens, feeding more
history than L cannot move the model's summary of the series — the recurrent
analog of the attention curve flattening. This is why FLOWSTATE_MAX_CONTEXT /
TIREX_MAX_CONTEXT exist in the labeling code ("curve flattens past it"); here we
measure that flattening in state space directly.

Two axes (both stored in one ``saturation.npz``)
------------------------------------------------
1. **Within-sequence trajectory** (ONE forward over the full window). For every
   position t along the scan we read the recurrent state h_t and ask how much the
   last chunk of history moved it:
        traj_marginal[t] = cos_dist(h_t, h_{t-1})     (-> 0 once saturated)
        traj_to_asymp[t] = cos_dist(h_t, h_T)         (distance to final state)
   ``L*_state_traj`` (in **timesteps**) = first t whose marginal update falls
   below a threshold. This is the genuine recurrent probe and is *cheap* — the
   whole trajectory comes from the single largest-window pass.

2. **Across-window final state** (K forward passes over WINDOW_GRID, the
   ``embedding_saturation.py`` pattern). We compare the FINAL recurrent state
   e(L) across nested tail windows L:
        win_marginal[k] = cos_dist(e(L_k), e(L_{k-1}))
        win_to_asymp[k] = cos_dist(e(L_k), e(L_max))
   ``L*_state_win`` = first grid window that plateaus. This one lines up 1:1 with
   the stage-3 error ablation cells, so ``L*_state_win`` vs ``L*_err`` (argmin of
   the error curve) is a direct scatter — is useful context readable from the
   recurrent state with no forecast scoring?

Both sweeps share the SAME forward passes: the across-window sweep already runs
the model at every L; the largest-L pass additionally yields the full per-position
trajectory, so axis (1) is free.

Where the "state" is read (best-effort true state, robust fallback)
-------------------------------------------------------------------
The user asked for the *true* recurrent state where reachable. The scan inside
FlowState / TiRex is usually fused (chunked-parallel), so per-position genuine
state tensors are typically NOT exposed to a forward hook. We therefore read
state at two fidelities, per firing:

  * **per-position readout** (always available) — the recurrent BLOCK's output
    sequence ``(B, T, d)``. Position t's output is a deterministic function of the
    recurrent state after absorbing tokens 0..t, so it traces the state's
    evolution. This drives the **trajectory** axis.
  * **true final state** (best-effort) — if a recurrent submodule returns its
    carry state (e.g. xLSTM ``(c, n, m)`` memory, or an SSM ``h``) alongside the
    output, we capture that compact tensor and use it as e(L) for the
    **across-window** axis. When no such tensor is exposed we fall back to the
    block output's last position. ``state_source`` in summary.json records which
    was used, and ``--dump-modules`` prints the captured shapes so the heuristic
    can be verified on the server.

Restricted to the recurrent families (``flowstate``, ``tirex``). Everything else
is an attention model and belongs in ``context_attention_mask.py`` /
``embedding_saturation.py``.

Outputs (logs/experiments/state_saturation/)
--------------------------------------------
  synthetic/<model>/saturation.npz
  gifteval/<dataset>/<model>/t<term>/saturation.npz
  .../summary.json         scalars + config (also the done-marker)
  .../plots/*.png          trajectory + across-window curves, L* histograms

Run on the SERVER, e.g.:
  python -m experiments.state_saturation --test                       # smoke run
  python -m experiments.state_saturation                              # full run
  python -m experiments.state_saturation --sources synthetic --models TiRex
  python -m experiments.state_saturation --dump-modules --models FlowState-R1
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
        def __getattr__(self, _):
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
# Reuse the saturation math + logging/error helpers verbatim so the two
# saturation experiments stay definition-identical (same cosine, same plateau
# crossing). These are the generic pieces of embedding_saturation.
from experiments.embedding_saturation import (
    _dump_error_report,
    _first_crossing,
    _install_log,
    find_backbone,
    saturation_curves,
)
# Cross-env routing (single source of truth) — flowstate/tirex are main-env, but
# routing through here keeps the coordinator identical to the rest of the pipeline.
from experiments.master_run_all import FAMILY_ENV, _env_label, _py

load_dotenv()


# ==============================================================================
#  CONFIG
# ==============================================================================
OUT_ROOT = "logs/experiments/state_saturation"

# Only the recurrent (state-carrying) families. Attention models -> use
# context_attention_mask.py / embedding_saturation.py instead.
RECURRENT_FAMILIES = ("flowstate", "tirex")
FAMILY_MAX_CONTEXT = {
    "flowstate": bcl.FLOWSTATE_MAX_CONTEXT,
    "tirex": bcl.TIREX_MAX_CONTEXT,
}

# Repeated-block naming for the recurrent stacks: FlowState SSM `layers.N` /
# `blocks.N` / `mixer` and TiRex xLSTM `blocks.N`. The block output is the
# per-position state readout (B, T, d).
DEFAULT_BLOCK_PATTERN = r"(?:^|\.)(?:layers?|blocks?|mixers?)\.\d+$"
# Submodules whose lowercased name hints at the recurrent carry (probed for a
# TRUE state tensor in their output, best-effort).
STATE_NAME_HINTS = ("mlstm", "slstm", "xlstm", "ssm", "s4", "mamba", "scan",
                    "recurrent", "state", "cell")

# Marginal-update plateau thresholds (cosine distance) -> L*_state.
MARGINAL_THRESHOLDS = (0.05, 0.02, 0.01)
ASYMPTOTE_THRESHOLDS = (0.10, 0.05, 0.02)

DEFAULT_SYNTH_SERIES = 2000
DEFAULT_SYNTH_SEED = 1234
DEFAULT_BATCH_SIZE = 64
TEST_MODEL = "TiRex"                                          # small + fast recurrent


# ==============================================================================
#  STATE CAPTURE
# ==============================================================================
def _seq_from_output(out) -> Optional[torch.Tensor]:
    """Reduce a block output to a per-position sequence ``(B, T, d)`` (or None).

    Accepts a tensor or a (tuple/list) whose first tensor element is the hidden
    sequence. Extra middle axes (grouping) are mean-collapsed so only batch,
    time, feature survive."""
    t = out[0] if isinstance(out, (tuple, list)) and out else out
    if not torch.is_tensor(t) or t.dim() < 2:
        return None
    t = t.detach().to(torch.float32)
    if t.dim() == 2:                                         # (B, d) — single step
        return t.unsqueeze(1)                               # (B, 1, d)
    # General (B, ..., T, d): collapse everything between batch and (T, d).
    while t.dim() > 3:
        t = t.mean(dim=1)
    return t                                                 # (B, T, d)


def _iter_tensors(obj, depth=0):
    """Yield every tensor nested in a (tuple/list/dict) output, shallowly."""
    if depth > 3:
        return
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for v in obj:
            yield from _iter_tensors(v, depth + 1)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v, depth + 1)
    else:                                                   # namedtuple / dataclass-ish
        for attr in ("state", "hidden", "carry", "memory", "c", "n", "m", "h"):
            v = getattr(obj, attr, None)
            if v is not None:
                yield from _iter_tensors(v, depth + 1)


def _state_from_output(out, batch: int, seq_len: int) -> Optional[torch.Tensor]:
    """Best-effort TRUE recurrent state ``(B, d_state)`` from a cell output.

    Scans the (possibly nested) output for a compact carry tensor: leading dim ==
    batch and NOT the per-position sequence (its non-batch element count differs
    from a ``T``-length axis). Picks the largest such tensor (the memory matrix
    dominates), flattening its non-batch dims. Returns None if nothing qualifies
    — the caller then falls back to the block-output last position."""
    best = None
    best_sz = -1
    for t in _iter_tensors(out):
        if not torch.is_tensor(t) or t.dim() < 2 or t.shape[0] != batch:
            continue
        # Skip the per-position sequence itself (has a time axis of length seq_len).
        if any(t.shape[ax] == seq_len for ax in range(1, t.dim())) and t.dim() >= 3:
            continue
        sz = int(t.numel() // batch)
        if sz > best_sz:
            best, best_sz = t, sz
    if best is None:
        return None
    return best.detach().to(torch.float32).reshape(batch, -1)


class StateCapture:
    """Capture the final recurrent block's per-position readout + best-effort
    true final state over a forward/generate.

    Hooks (a) every repeated block matching ``block_pattern`` — its output is the
    per-position sequence ``(B, T, d)`` — keeping the LAST valid firing as the
    final block; and (b) every submodule whose name hints at a recurrent carry,
    probing its output for a compact true-state tensor. ``read()`` returns
    ``(seq (B, T, d), state (B, d_state) or None, source)`` where ``source`` is
    "true_state" when a carry tensor was captured for the final position, else
    "block_output"."""

    def __init__(self, backbone: torch.nn.Module, block_pattern: str):
        self._brx = re.compile(block_pattern)
        self._handles = []
        self.matched_blocks: List[str] = []
        self.matched_state: List[str] = []
        self._batch: Optional[int] = None
        # execution-order logs
        self._seq_log: List[Tuple[str, torch.Tensor]] = []      # (name, (B,T,d) cpu)
        self._state_log: List[Tuple[str, torch.Tensor]] = []    # (name, (B,dS) cpu)
        for name, mod in backbone.named_modules():
            if not name:
                continue
            if self._brx.search(name):
                self.matched_blocks.append(name)
                self._handles.append(
                    mod.register_forward_hook(self._block_hook(name)))
            elif any(h in name.lower() for h in STATE_NAME_HINTS):
                self.matched_state.append(name)
                self._handles.append(
                    mod.register_forward_hook(self._state_hook(name)))

    def _block_hook(self, name):
        def hook(_m, _i, out):
            seq = _seq_from_output(out)
            if seq is None:
                return
            if self._batch is not None and seq.shape[0] != self._batch:
                return                                        # folded-axis firing
            self._seq_log.append((name, seq.cpu()))
        return hook

    def _state_hook(self, name):
        def hook(_m, _i, out):
            if self._batch is None:
                return
            # infer this firing's seq length (if any block already fired) to avoid
            # mistaking the hidden sequence for the carry.
            seq_len = self._seq_log[-1][1].shape[1] if self._seq_log else -1
            st = _state_from_output(out, self._batch, seq_len)
            if st is not None:
                self._state_log.append((name, st.cpu()))
        return hook

    def reset(self, batch: int) -> None:
        self._batch = batch
        self._seq_log = []
        self._state_log = []

    def read(self) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], str]]:
        if not self._seq_log:
            return None
        # final block = last valid firing at the true batch (prefill, most-recent).
        seq = self._seq_log[-1][1]                             # (B, T, d)
        if self._state_log:
            return seq, self._state_log[-1][1], "true_state"
        return seq, seq[:, -1, :], "block_output"

    def remove(self) -> None:
        for hd in self._handles:
            hd.remove()
        self._handles.clear()


def _predict_dispatch(family, runner, xb, horizon, device):
    """Drive the real forward pass (side effect: fires the capture hooks)."""
    fn = {"flowstate": bcl.predict_flowstate, "tirex": bcl.predict_tirex}.get(family)
    if fn is None:
        raise ValueError(
            f"state_saturation only supports {RECURRENT_FAMILIES}; got {family!r}.")
    return fn(runner, xb, horizon, device)


def _resolve_recurrent_backbone(family, base) -> torch.nn.Module:
    """nn.Module carrying the recurrent blocks.

    FlowState's ``FlowStateForPrediction`` is itself an nn.Module; TiRex's
    ``load_model`` returns a forecaster wrapper burying the torch module under an
    attribute, so we walk to the nn.Module with the most block matches."""
    if isinstance(base, torch.nn.Module) and any(
            re.search(DEFAULT_BLOCK_PATTERN, n) for n, _ in base.named_modules()):
        return base
    match, fallback = find_backbone(base, DEFAULT_BLOCK_PATTERN)
    if match is not None:
        return match
    if isinstance(base, torch.nn.Module):
        return base
    if fallback is not None:
        return fallback
    raise RuntimeError(
        f"Could not locate an nn.Module backbone for family={family}. "
        f"Run with --dump-modules to inspect.")


# ==============================================================================
#  METRICS
# ==============================================================================
def _traj_curves(seq: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-position saturation of a state trajectory ``(n, T, d)``.

    Reuses ``saturation_curves`` (which reads a (N, K, d) window axis) with the
    time axis in the K slot, so the cosine / crossing definitions are identical to
    the across-window and embedding experiments."""
    c = saturation_curves(seq)                                # keys marginal/to_asymp
    return {"marginal": c["marginal"], "to_asymp": c["to_asymp"]}


def _lstar_positions(marginal: np.ndarray, thr: float) -> np.ndarray:
    """First position index whose marginal update <= thr (else last position).

    ``marginal[:, 0]`` is 0 by convention (no previous state) so it is masked to
    +inf before the crossing, mirroring summarize_saturation."""
    m = marginal.copy()
    m[:, 0] = np.inf
    pos = np.arange(m.shape[1])
    return _first_crossing(m, pos, thr)                       # position index


# ==============================================================================
#  COLLECTION OVER THE GRID (both axes in one sweep)
# ==============================================================================
def collect_state(
    family, base, model_id, contexts: np.ndarray, real_lengths: np.ndarray,
    horizon: int, windows: List[int], batch_size: int, device: str,
    block_pattern: str, progress_desc: Optional[str] = None,
) -> Dict[str, object]:
    """Run the WINDOW_GRID sweep once, returning both saturation axes.

    across-window : ``e_final`` (N, K, d_state) — final recurrent state per window.
    trajectory    : from the LARGEST window pass only — per-series marginal/
                    to_asymp over scan positions (N, T_max), start-aligned and
                    NaN-padded at the tail for short series, plus per-series
                    ``traj_stride`` (timesteps per state position = width / T) to
                    convert a position L* into timesteps.
    """
    N = contexts.shape[0]
    grid = np.asarray(sorted(set(bcl.WINDOW_GRID)))
    backbone = _resolve_recurrent_backbone(family, base)
    cap = StateCapture(backbone, block_pattern)

    e_final: Optional[np.ndarray] = None                      # (N, K, d_state)
    d_state: Optional[int] = None
    source_seen = set()
    K = len(windows)
    L_max = int(windows[-1])                                  # trajectory window

    # trajectory accumulators (filled only on the L_max pass)
    traj_marg: Optional[np.ndarray] = None                    # (N, T_max)
    traj_asym: Optional[np.ndarray] = None
    traj_stride = np.full(N, np.nan, dtype=np.float32)
    traj_len = np.zeros(N, dtype=np.int64)

    bar = tqdm(total=K, desc=progress_desc, unit="win", leave=False,
               disable=progress_desc is None)
    try:
        with torch.no_grad():
            for k in range(K):
                L = int(windows[k])
                bar.set_postfix_str(f"L={L}", refresh=False)
                eff = np.minimum(L, np.asarray(real_lengths))
                eff_buck = np.minimum(
                    eff, grid[np.clip(
                        np.searchsorted(grid, eff, side="right") - 1, 0, None)])
                for width in np.unique(eff_buck):
                    idx = np.flatnonzero(eff_buck == width)
                    x_grp = torch.from_numpy(np.ascontiguousarray(
                        contexts[idx, -int(width):])).unsqueeze(-1)
                    for start in range(0, len(idx), batch_size):
                        bidx = idx[start:start + batch_size]
                        xb = x_grp[start:start + batch_size]
                        cap.reset(xb.shape[0])
                        _predict_dispatch(family, base, xb, horizon, device)
                        got = cap.read()
                        if got is None:
                            raise RuntimeError(
                                f"No state captured for family={family} "
                                f"(block_pattern={block_pattern!r}, matched "
                                f"{len(cap.matched_blocks)} blocks). Use "
                                f"--dump-modules / --block-pattern.")
                        seq, state, src = got                 # (b,T,d), (b,dS), str
                        source_seen.add(src)
                        sv = state.numpy()                    # (b, d_state)
                        if d_state is None:
                            d_state = sv.shape[1]
                            e_final = np.zeros((N, K, d_state), dtype=np.float32)
                        if sv.shape[1] != d_state:            # width-varying state -> pad/trim
                            sv = _fit_width(sv, d_state)
                        e_final[bidx, k, :] = sv
                        if L == L_max:                        # trajectory pass
                            sq = seq.numpy()                  # (b, T, d)
                            T = sq.shape[1]
                            c = _traj_curves(sq)
                            if traj_marg is None:
                                traj_marg = np.full((N, T), np.nan, np.float32)
                                traj_asym = np.full((N, T), np.nan, np.float32)
                            elif T > traj_marg.shape[1]:      # grow the time axis
                                pad = T - traj_marg.shape[1]
                                traj_marg = np.pad(
                                    traj_marg, ((0, 0), (0, pad)),
                                    constant_values=np.nan)
                                traj_asym = np.pad(
                                    traj_asym, ((0, 0), (0, pad)),
                                    constant_values=np.nan)
                            traj_marg[bidx, :T] = c["marginal"]
                            traj_asym[bidx, :T] = c["to_asymp"]
                            traj_len[bidx] = T
                            traj_stride[bidx] = float(width) / max(1, T)
                bar.update(1)
    finally:
        cap.remove()
        bar.close()

    if e_final is None or traj_marg is None:
        raise RuntimeError(f"state collection produced no data (family={family}).")

    return {
        "e_final": e_final,                                   # (N, K, d_state)
        "traj_marginal": traj_marg,                           # (N, T_max)
        "traj_to_asymp": traj_asym,
        "traj_len": traj_len,                                 # per-series #positions
        "traj_stride": traj_stride,                           # timesteps / position
        "state_source": "+".join(sorted(source_seen)),
        "matched_blocks": cap.matched_blocks,
        "matched_state": cap.matched_state,
    }


def _fit_width(a: np.ndarray, d: int) -> np.ndarray:
    """Pad/trim a (b, *) state to feature width d so all windows stack."""
    if a.shape[1] == d:
        return a
    if a.shape[1] > d:
        return a[:, :d]
    return np.pad(a, ((0, 0), (0, d - a.shape[1])))


# ==============================================================================
#  SUMMARIZE + WRITE
# ==============================================================================
def _summ_traj(res, thr_list) -> Dict[str, np.ndarray]:
    """Per-series trajectory L* in TIMESTEPS for each marginal threshold."""
    out: Dict[str, np.ndarray] = {}
    marg = res["traj_marginal"]
    stride = res["traj_stride"]
    # NaN tail (short series) must not trigger a spurious plateau: treat NaN as
    # +inf so a series never "crosses" on padding.
    m = np.where(np.isnan(marg), np.inf, marg)
    for thr in thr_list:
        pos = _lstar_positions(m, thr)                        # position index
        out[f"Lstar_traj_{thr}"] = ((pos + 1) * stride).astype(np.float32)
    return out


def _write_cell(out_dir, windows, res, meta, title):
    os.makedirs(out_dir, exist_ok=True)
    win = np.asarray(windows)
    win_curves = saturation_curves(res["e_final"])            # over K windows
    traj_summ = _summ_traj(res, MARGINAL_THRESHOLDS)

    # across-window L* (plateau over the grid), same crossing as the trajectory.
    wm = win_curves["marginal"].copy()
    wm[:, 0] = np.inf
    win_summ = {f"Lstar_win_{thr}": _first_crossing(wm, win, thr)
                for thr in MARGINAL_THRESHOLDS}
    win_summ.update({f"Lstar_win_asymp_{thr}": _first_crossing(
        win_curves["to_asymp"], win, thr) for thr in ASYMPTOTE_THRESHOLDS})

    np.savez_compressed(
        os.path.join(out_dir, "saturation.npz"),
        windows=win,
        win_marginal=win_curves["marginal"], win_to_asymp=win_curves["to_asymp"],
        traj_marginal=res["traj_marginal"], traj_to_asymp=res["traj_to_asymp"],
        traj_len=res["traj_len"], traj_stride=res["traj_stride"],
        **traj_summ, **win_summ,
        **{k: np.asarray(v) for k, v in meta.items() if np.ndim(v) >= 1},
    )

    summary = {
        "n_series": int(res["e_final"].shape[0]),
        "n_windows": int(res["e_final"].shape[1]),
        "d_state": int(res["e_final"].shape[2]),
        "T_max": int(res["traj_marginal"].shape[1]),
        "state_source": res["state_source"],
        "n_blocks_hooked": len(res["matched_blocks"]),
        "n_state_modules_hooked": len(res["matched_state"]),
        "windows": list(map(int, windows)),
        **{k: float(np.nanmedian(v)) for k, v in traj_summ.items()},
        **{k: float(np.median(v)) for k, v in win_summ.items()},
        **{k: v for k, v in meta.items() if np.ndim(v) == 0},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _plots(out_dir, windows, win_curves, res, traj_summ, title)
    return summary


# ==============================================================================
#  PLOTS
# ==============================================================================
def _plots(out_dir, windows, win_curves, res, traj_summ, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    pdir = os.path.join(out_dir, "plots")
    os.makedirs(pdir, exist_ok=True)

    # --- across-window curves ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, name in ((ax[0], "marginal"), (ax[1], "to_asymp")):
        c = win_curves[name]
        med = np.median(c, axis=0)
        q1, q3 = np.percentile(c, [25, 75], axis=0)
        a.plot(windows, med, marker="o", label="median")
        a.fill_between(windows, q1, q3, alpha=0.2, label="IQR")
        a.set_xscale("log"); a.set_xlabel("context length"); a.set_title(f"window {name}")
        a.set_ylabel("cosine distance"); a.legend()
    fig.suptitle(f"{title} — across-window final state")
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "window_saturation.png"), dpi=120)
    plt.close(fig)

    # --- within-sequence trajectory (nanmedian over series) ---
    tm, ta = res["traj_marginal"], res["traj_to_asymp"]
    pos = np.arange(tm.shape[1])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, c, name in ((ax[0], tm, "marginal"), (ax[1], ta, "to_asymp")):
        med = np.nanmedian(c, axis=0)
        q1 = np.nanpercentile(c, 25, axis=0)
        q3 = np.nanpercentile(c, 75, axis=0)
        a.plot(pos, med, label="median")
        a.fill_between(pos, q1, q3, alpha=0.2, label="IQR")
        a.set_xlabel("state position (patch token)"); a.set_title(f"trajectory {name}")
        a.set_ylabel("cosine distance"); a.legend()
    fig.suptitle(f"{title} — within-sequence state trajectory")
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "trajectory_saturation.png"), dpi=120)
    plt.close(fig)

    # --- L*_state_traj histogram (timesteps) ---
    ls = traj_summ[f"Lstar_traj_{MARGINAL_THRESHOLDS[0]}"]
    ls = ls[np.isfinite(ls) & (ls > 0)]
    if ls.size:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.log2(ls.astype(np.float64)), bins=min(30, len(windows) * 2))
        ax.set_xlabel(f"log2  L*_state_traj (marginal<={MARGINAL_THRESHOLDS[0]}, timesteps)")
        ax.set_ylabel("count"); ax.set_title(f"{title}\nstate-saturation length")
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, "Lstar_traj_hist.png"), dpi=120)
        plt.close(fig)


# ==============================================================================
#  SOURCES
# ==============================================================================
def _windows_for(family, windows, cap_ctx: bool, hard_max=None) -> List[int]:
    """Grid windows this recurrent model runs, capped at its trained context.

    FlowState/TiRex fidelity past FLOWSTATE_MAX_CONTEXT / TIREX_MAX_CONTEXT is
    unverified (the labeling code caps there), so by default we cap too;
    ``--probe-beyond`` lifts the cap to study saturation past the trained window.
    ``hard_max`` (dataset max context) further trims so we never feed a window
    longer than the data has."""
    ws = list(windows)
    if cap_ctx:
        cap = FAMILY_MAX_CONTEXT.get(family)
        if cap:
            ws = [w for w in ws if w <= cap]
    if hard_max is not None:
        ws = [w for w in ws if w <= hard_max] or [min(ws)]
    return ws or [windows[0]]


def run_synthetic(args, family, model_id, display, base, device, windows):
    out_dir = os.path.join(args.out_root, "synthetic", display)
    if _done(out_dir) and not args.force:
        print(Fore.YELLOW + f"  [skip] synthetic/{display} (cached)" + Fore.RESET)
        return
    contexts, _targets, n_segments, real_lengths = bcl.generate_dataset(
        args.synth_series, args.synth_seed)
    horizon = bcl.HORIZON_GRID[-1]
    ws = _windows_for(family, windows, not args.probe_beyond)
    real_lengths = np.minimum(real_lengths, ws[-1]).astype(np.int64)
    contexts = contexts[:, -ws[-1]:].astype(np.float32)
    res = collect_state(family, base, model_id, contexts, real_lengths, horizon,
                        ws, args.batch_size, device, args.block_pattern,
                        progress_desc=f"    synthetic {display}")
    meta = {"n_segments": n_segments, "real_lengths": real_lengths,
            "horizon": horizon, "source": "synthetic", "model": display}
    summary = _write_cell(out_dir, ws, res, meta, f"synthetic — {display}")
    print(Fore.GREEN + f"  [done] synthetic/{display}  src={summary['state_source']} "
          f"L*traj={summary[f'Lstar_traj_{MARGINAL_THRESHOLDS[0]}']:.0f}" + Fore.RESET)


def run_gifteval(args, family, model_id, display, base, device, windows,
                 shard_id, num_shards):
    from experiments.test_window_ablation_gifteval_v5 import (
        DATASETS, GiftEvalCache)
    from gift_eval.data import Dataset as GiftEvalDataset

    wanted = _selected_datasets(args, DATASETS)
    my_cells = [(d_idx, row) for d_idx, row in enumerate(wanted)
                if num_shards <= 1 or d_idx % num_shards == shard_id]
    total = len(my_cells)
    tag = f" shard {shard_id}/{num_shards}" if num_shards > 1 else ""
    print(Fore.CYAN + f"  gifteval{tag}: {total} cells for {display}" + Fore.RESET)
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
            _dump_error_report(
                args.out_root, f"{display}__{dataset_display}__t{term}__load",
                {"phase": "dataset load", "model": display, "family": family,
                 "dataset": dataset_display, "term": term,
                 "error": f"{type(e).__name__}: {e}"})
            continue
        try:
            print(Fore.CYAN + f"  {prog} {dataset_display}/t{term} …" + Fore.RESET)
            ws = _windows_for(family, windows, not args.probe_beyond,
                              hard_max=cache.max_context)
            wmax = ws[-1]
            contexts = np.stack([
                np.pad(c, (max(0, wmax - len(c)), 0))[-wmax:]
                if len(c) < wmax else c[-wmax:]
                for c in cache.contexts]).astype(np.float32)
            real_lengths = np.minimum(cache.context_lengths, wmax).astype(np.int64)
            res = collect_state(
                family, base, model_id, contexts, real_lengths, cache.horizon,
                ws, args.batch_size, device, args.block_pattern,
                progress_desc=f"    {prog} {display} {dataset_display}/t{term}")
            meta = {"real_lengths": real_lengths, "horizon": cache.horizon,
                    "source": "gifteval", "dataset": dataset_display, "term": term,
                    "model": display}
            _write_cell(out_dir, ws, res, meta,
                        f"{dataset_display} t{term} — {display}")
            n_done += 1
            print(Fore.GREEN + f"  {prog} [done] gifteval/{dataset_display}/{display}"
                  f"/t{term}" + Fore.RESET)
        except Exception as e:                               # noqa: BLE001
            n_fail += 1
            print(Fore.RED + f"  {prog} [FAIL] {dataset_display}/t{term} ({display}): "
                  f"{type(e).__name__}: {e}" + Fore.RESET)
            traceback.print_exc()
            _dump_error_report(
                args.out_root, f"{display}__{dataset_display}__t{term}",
                {"phase": "state extraction", "model": display, "family": family,
                 "dataset": dataset_display, "term": term,
                 "error": f"{type(e).__name__}: {e}",
                 "n_series": len(cache.contexts), "max_context": cache.max_context,
                 "horizon": cache.horizon, "block_pattern": args.block_pattern,
                 "batch_size": args.batch_size, "device": device})
            continue
    print(Fore.CYAN + f"  gifteval{tag} {display}: "
          f"{n_done} done, {n_skip} skipped, {n_fail} failed (of {total})"
          + Fore.RESET)


def _selected_datasets(args, DATASETS):
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
    else:                                                    # default: recurrent run set
        names = [disp for (_m, fam, disp) in models_config.models_to_run()
                 if fam in RECURRENT_FAMILIES]
    out = []
    for n in names:
        if n not in catalog:
            raise SystemExit(f"Unknown model {n!r}. Known: {sorted(catalog)}")
        mid, fam = catalog[n]
        if fam not in RECURRENT_FAMILIES:
            raise SystemExit(
                f"{n!r} is family {fam!r}; state_saturation only handles the "
                f"recurrent families {RECURRENT_FAMILIES}. Use "
                f"embedding_saturation / context_attention_mask for {fam!r}.")
        out.append((mid, fam, n))
    return out


def _dump_modules(family, base, pattern):
    backbone = _resolve_recurrent_backbone(family, base)
    rx = re.compile(pattern)
    blocks = [n for n, _ in backbone.named_modules() if n and rx.search(n)]
    states = [n for n, _ in backbone.named_modules()
              if n and any(h in n.lower() for h in STATE_NAME_HINTS)]
    print(f"  block pattern {pattern!r} matched {len(blocks)} modules:")
    for n in (blocks if len(blocks) <= 12 else blocks[:6] + ["…"] + blocks[-6:]):
        print(f"    {n}")
    print(f"  state-hint modules ({len(states)} — probed for a true carry tensor):")
    for n in (states if len(states) <= 12 else states[:6] + ["…"] + states[-6:]):
        print(f"    {n}")


def run_worker(args, device, shard_id, num_shards):
    for (model_id, family, display) in _models_to_run(args):
        print(Fore.CYAN + f"\n=== {display} ({family}) on {device} ===" + Fore.RESET)
        try:
            base = bcl.setup_model(family, model_id, device)
            if isinstance(base, torch.nn.Module):
                base.eval()
        except Exception as e:                               # noqa: BLE001
            print(Fore.RED + f"  load failed: {e}" + Fore.RESET)
            traceback.print_exc()
            _dump_error_report(
                args.out_root, f"{display}__model_load",
                {"phase": "model load", "model": display, "family": family,
                 "model_id": model_id, "device": device,
                 "error": f"{type(e).__name__}: {e}"})
            continue
        if args.dump_modules:
            _dump_modules(family, base, args.block_pattern)
            del base
            if bcl._is_cuda(device):
                torch.cuda.empty_cache()
            continue
        try:
            if "synthetic" in args.sources:
                run_synthetic(args, family, model_id, display, base, device,
                              args.windows)
            if "gifteval" in args.sources:
                run_gifteval(args, family, model_id, display, base, device,
                             args.windows, shard_id, num_shards)
        except Exception as e:                               # noqa: BLE001
            print(Fore.RED + f"  run failed for {display}: {e}" + Fore.RESET)
            traceback.print_exc()
            _dump_error_report(
                args.out_root, f"{display}__run",
                {"phase": "run", "model": display, "family": family,
                 "device": device, "error": f"{type(e).__name__}: {e}"})
        finally:
            del base
            if bcl._is_cuda(device):
                torch.cuda.empty_cache()


def _resolve_env_groups(args) -> "Dict[Optional[str], List[str]]":
    groups: Dict[Optional[str], List[str]] = {None: []}
    for (_mid, family, display) in _models_to_run(args):
        groups.setdefault(FAMILY_ENV.get(family), []).append(display)
    if not groups[None]:
        del groups[None]
    return groups


def _reconstruct_flags(args, displays):
    f = ["--sources", *args.sources, "--models", *displays,
         "--windows", *map(str, args.windows),
         "--batch-size", str(args.batch_size),
         "--synth-series", str(args.synth_series),
         "--synth-seed", str(args.synth_seed),
         "--block-pattern", args.block_pattern,
         "--out-root", args.out_root]
    if args.datasets:
        f += ["--datasets", *args.datasets]
    if args.test_datasets is not None:
        f += ["--test-datasets", str(args.test_datasets)]
    if args.probe_beyond:
        f += ["--probe-beyond"]
    if args.force:
        f += ["--force"]
    if args.dump_modules:
        f += ["--dump-modules"]
    return f


def _group_cmd(args, env, displays, shard_id, num_shards):
    label = _env_label(env)
    stem = (f"dump_modules_{label}" if args.dump_modules
            else f"run_{label}_shard{shard_id}")
    flags = _reconstruct_flags(args, displays) + [
        "--gpus", "1", "--shard-id", str(shard_id), "--num-shards", str(num_shards),
        "--log-file", os.path.join(args.out_root, stem + ".log")]
    return _py(env, "experiments.state_saturation", *flags)


def coordinate(args):
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
                   help="Default: gifteval only. Add 'synthetic' for the pool.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Display names (default: the recurrent run set — "
                        "FlowState + TiRex; TEST_MODEL with --test).")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Restrict gifteval to these dataset display/ge names.")
    p.add_argument("--windows", type=int, nargs="+", default=None,
                   help="Override the context grid (default WINDOW_GRID).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--synth-series", type=int, default=DEFAULT_SYNTH_SERIES)
    p.add_argument("--synth-seed", type=int, default=DEFAULT_SYNTH_SEED)
    p.add_argument("--block-pattern", default=DEFAULT_BLOCK_PATTERN,
                   help="Regex for the recurrent BLOCK modules (per-position "
                        "state readout). Inspect with --dump-modules.")
    p.add_argument("--probe-beyond", action="store_true",
                   help="Lift the FLOWSTATE_MAX_CONTEXT / TIREX_MAX_CONTEXT cap to "
                        "study saturation past the trained context.")
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--force", action="store_true", help="Recompute cached cells.")
    p.add_argument("--gpus", type=int, default=0, help="0 = all visible GPUs.")
    p.add_argument("--dump-modules", action="store_true",
                   help="Print hook-matched block + state modules and exit.")
    p.add_argument("--log-file", default=None)
    p.add_argument("--test", action="store_true",
                   help="Smoke run: TEST_MODEL, tiny grid, few series + datasets.")
    p.add_argument("--test-datasets", type=int, default=None)
    p.add_argument("--shard-id", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    if args.test:
        os.environ["PREDICTCSL_TEST"] = "1"                  # collapses WINDOW_GRID
        if args.windows is None:
            args.windows = [32, 128, 512, 2048]
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
