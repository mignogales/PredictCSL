"""
Context restriction via *attention masking* (companion to the slicing ablation).

The context-length saturation study (``build_context_length_dataset`` ->
``forecast_window``) shortens context by *slicing*: it feeds a TSFM only the last
``L`` genuine timesteps. That changes three things at once relative to the full
run — the attention span, the instance-normalization statistics (mean/std over
``L``), and the positional indices of the surviving tokens.

This module implements the *masking* counterpart, to isolate the **attention
span** alone. The idea (user's framing): feed the SAME full-window input on every
grid point, but add an attention mask so that only the last ``L`` timesteps are
reachable as attention *keys*; everything older is ``-inf``'d out. Because the old
tokens stay physically present in the sequence:

  * normalization stats are still computed over the full window, and
  * positional / RoPE indices of the visible tail are its absolute full-window
    positions (NOT renumbered to ``0..L`` the way slicing / left-padding would).

So any gap between the masking curve and the slicing curve is attributable to the
normalization + position change, not the attention span. If the two coincide, the
saturation effect is purely a matter of what the model attends to.

Only transformer families are supported (there must be an attention matrix to
mask): PatchTST-FM (encoder), Sundial / TimeMoE (decoder-only, RoPE). FlowState
(SSM) and TiRex (xLSTM) have no attention and are out of scope. TimesFM 2.5 goes
through a compiled decode path (``compiled_decode``) that a runtime hook can't
reliably reach, so it's excluded here too — use Sundial/TimeMoE as the decoder
representative.

Mechanism, by family:
  * **patchtst_fm** — the model already builds an additive attention mask
    (``make_attn_mask``: ``-inf`` on padded key patches) and threads it to every
    block's ``scaled_dot_product_attention``. We wrap that one function so it also
    sets the oldest key-patch columns to ``-inf``. The input tensor is untouched,
    so RevIN still normalizes over the full genuine window.
  * **sundial / timemoe** — both pass a 4-D additive ``(B, 1, q, kv)`` mask into
    SDPA. We register a ``forward_pre_hook`` on every attention submodule that sets
    the first ``hide_count`` key columns to ``-inf``. Tokens stay in the sequence
    so RoPE positions remain full-window absolute. ``hide_count`` (in *tokens*) is
    derived from the visible timestep budget and the prefill token count, then held
    fixed through autoregressive decode (the oldest columns are always the ones to
    hide, so a fixed leading count is correct as the KV cache grows).

NOTE (server verification): the PatchTST-FM path is written against the exact
granite-tsfm source (``models/patchtst_fm/{basic,modeling_patchtst_fm}.py``). The
decoder hook is written against the confirmed Sundial/TimeMoE attention signature
(4-D additive mask into SDPA); the attention *class names* and the ``d_patch``
config attribute should be sanity-checked against the installed packages on the
server (mirroring the existing "verify on the server" notes for toto/tirex).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional

import torch


# ==============================================================================
#  Encoder: PatchTST-FM
# ==============================================================================

def _patchtstfm_patch_len(model) -> int:
    """Timesteps per patch token (``d_patch``). Tried in order of likelihood."""
    cfg = model.config
    for attr in ("d_patch", "patch_length", "patch_len"):
        v = getattr(cfg, attr, None)
        if v:
            return int(v)
    # Fall back to context_length / n_patch if the split attrs aren't present.
    n_patch = getattr(cfg, "n_patch", None)
    ctx = getattr(cfg, "context_length", None)
    if n_patch and ctx:
        return int(ctx) // int(n_patch)
    raise AttributeError(
        "Could not infer PatchTST-FM patch length (d_patch) from config; "
        "check the attribute name on the server.")


@contextmanager
def _patchtstfm_mask(model, last_timesteps: int):
    """Wrap ``make_attn_mask`` so the oldest key patches are also ``-inf``'d.

    ``make_attn_mask(query_pad, key_pad)`` returns an additive float mask of shape
    ``(B, n_query, n_key)`` (``-inf`` on padded key positions). We keep only the
    last ``visible`` patch columns; the leading ``n_key - visible`` columns (which
    already includes any NaN-pad patches for short series) get ``-inf``.
    """
    import tsfm_public.models.patchtst_fm.basic as B
    import tsfm_public.models.patchtst_fm.modeling_patchtst_fm as M

    patch_len = _patchtstfm_patch_len(model)
    visible = max(1, int(last_timesteps) // patch_len)   # visible patch tokens
    orig = B.make_attn_mask
    # ``modeling`` may hold its own bound reference (``from .basic import ...``);
    # patch whichever name(s) exist so ``decode`` resolves the wrapped version.
    patched_modeling = getattr(M, "make_attn_mask", None) is orig

    def wrapped(query_pad, key_pad):
        m = orig(query_pad, key_pad)                     # (B, nq, nk) additive
        n_key = m.shape[-1]
        cutoff = max(0, n_key - visible)
        if cutoff > 0:
            m[..., :cutoff] = float("-inf")
        return m

    B.make_attn_mask = wrapped
    if patched_modeling:
        M.make_attn_mask = wrapped
    try:
        yield
    finally:
        B.make_attn_mask = orig
        if patched_modeling:
            M.make_attn_mask = orig


# ==============================================================================
#  Decoder-only (RoPE): Sundial, TimeMoE
# ==============================================================================

def _is_attention_module(module) -> bool:
    """Heuristic: a self-attention submodule of a decoder block.

    Matches by class name ending in ``Attention`` (``TimeMoeAttention``,
    ``SundialAttention``/``SundialSelfAttention``). Also requires q/k projections
    so we don't accidentally grab a wrapper container."""
    name = type(module).__name__
    if not name.endswith("Attention"):
        return False
    return any(hasattr(module, p) for p in ("q_proj", "k_proj", "qkv", "Wqkv"))


def _install_decoder_mask(model, last_timesteps: int, full_len: int) -> List:
    """Hook every attention module to ``-inf`` the oldest key columns.

    ``hide_count`` is computed once, at the prefill call (first time the hook sees
    a mask), from the prefill token count ``P``: with ``P`` tokens spanning
    ``full_len`` timesteps, the visible tail of ``last_timesteps`` is
    ``round(last_timesteps * P / full_len)`` tokens, so we hide the leading
    ``P - visible`` columns — and keep hiding exactly that many as the KV cache
    grows during decode (the hidden columns are always the oldest)."""
    state = {"hide_count": None}

    def pre_hook(module, args, kwargs):
        am = kwargs.get("attention_mask", None)
        from_kwargs = "attention_mask" in kwargs
        if am is None and len(args) >= 2:
            am = args[1]
        if not torch.is_tensor(am) or not am.is_floating_point():
            # No explicit additive mask (e.g. a pure ``is_causal`` SDPA path) —
            # nothing to edit. If this fires for a model you expect to mask, the
            # decoder path needs a per-module tweak (verify on the server).
            return None

        kv = am.shape[-1]
        if state["hide_count"] is None:
            visible = max(1, round(int(last_timesteps) * kv / int(full_len)))
            state["hide_count"] = max(0, kv - visible)
        hide = state["hide_count"]
        if hide <= 0:
            return None

        am = am.clone()
        am[..., :hide] = torch.finfo(am.dtype).min
        if from_kwargs:
            kwargs = {**kwargs, "attention_mask": am}
            return args, kwargs
        args = tuple(am if i == 1 else a for i, a in enumerate(args))
        return args, kwargs

    handles = []
    for module in model.modules():
        if _is_attention_module(module):
            handles.append(
                module.register_forward_pre_hook(pre_hook, with_kwargs=True))
    if not handles:
        raise RuntimeError(
            "No attention submodules found to hook — check "
            "_is_attention_module against the installed model on the server.")
    return handles


# ==============================================================================
#  Public entry point
# ==============================================================================

SUPPORTED_FAMILIES = {"patchtst_fm", "sundial", "timemoe"}


@contextmanager
def context_attention_mask(
    family: str,
    model,
    last_timesteps: int,
    full_len: int,
):
    """Restrict a transformer TSFM to attend only to the last ``last_timesteps``.

    Feed the model its full ``full_len``-timestep window as usual, but wrap the
    forecast call in this context manager so attention keys older than the last
    ``last_timesteps`` are ``-inf``'d out. A no-op when ``last_timesteps >=
    full_len`` (nothing to hide) so the top of the grid matches the full run.
    """
    if int(last_timesteps) >= int(full_len):
        yield
        return

    if family == "patchtst_fm":
        with _patchtstfm_mask(model, last_timesteps):
            yield
        return

    if family in ("sundial", "timemoe"):
        handles = _install_decoder_mask(model, last_timesteps, full_len)
        try:
            yield
        finally:
            for h in handles:
                h.remove()
        return

    raise NotImplementedError(
        f"Attention masking not implemented for family {family!r}. "
        f"Supported: {sorted(SUPPORTED_FAMILIES)}.")
