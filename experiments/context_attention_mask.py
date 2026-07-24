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
(SSM) and TiRex (xLSTM) have no attention and are out of scope. TimesFM 2.5 is
supported ONLY on its eager path: ``compiled_decode`` is (despite the name) an
eager closure — only ``model.forward`` gets ``torch.compile``d, and only when
loaded with ``torch_compile=True``. Load with ``torch_compile=False`` (the
interpretability adapter does) and the ``make_attn_mask`` wrap below is
reachable; under a compiled forward the wrap may be baked out and masking would
silently no-op — the ``_warn_if_noop`` guard cannot detect that case, so never
use the timesfm family here with a compiled model.

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


def _edit_attention_mask(am, state, last_timesteps: int, full_len: int):
    """Return ``am`` with the oldest key columns ``-inf``'d for the VISIBLE query
    rows only, or None if it isn't an editable additive mask.

    ``hide_count`` is computed once, at the prefill call (first mask we see), from
    the prefill token count ``P``: with ``P`` tokens spanning ``full_len``
    timesteps, the visible tail of ``last_timesteps`` is
    ``round(last_timesteps * P / full_len)`` tokens, so the leading ``P - visible``
    columns are "old" — and stay old as the KV cache grows during decode.

    RECTANGULAR, not a full column block: we only drop the old key columns for
    query rows whose absolute position is in the visible region (>= hide). Old
    query rows keep their normal causal mask. This is essential for a causal
    decoder — a full column block makes the oldest query rows fully ``-inf``,
    which NaNs their softmax; the NaN then flows into their K vectors, and since
    ``NaN + (-inf) = NaN`` the ``-inf`` can no longer suppress them, so the NaN
    floods every row (the empty-slice / garbage curves seen on Sundial). The
    forecast-relevant rows (the last position + any decode/future rows) still see
    only the last ``last_timesteps``, which is all we need."""
    if not torch.is_tensor(am) or not am.is_floating_point():
        state["saw_none"] += 1
        return None
    q = am.shape[-2]
    kv = am.shape[-1]
    if state["hide_count"] is None:
        visible = max(1, round(int(last_timesteps) * kv / int(full_len)))
        state["hide_count"] = max(0, kv - visible)
    hide = state["hide_count"]
    if hide <= 0:
        return None
    # The q rows are the LAST q positions of a length-kv sequence, so row r has
    # absolute position (kv - q) + r; mask old columns only where abs pos >= hide.
    start_row = max(0, hide - (kv - q))
    am = am.clone()
    am[..., start_row:, :hide] = torch.finfo(am.dtype).min
    state["applied"] += 1
    return am


def _install_decoder_mask(model, last_timesteps: int, full_len: int):
    """Wrap every attention module's ``forward`` to ``-inf`` its oldest key columns.

    We monkeypatch ``module.forward`` (rather than register a
    ``forward_pre_hook(..., with_kwargs=True)``) so this works on torch < 2.0 too
    — the legacy Sundial/TimeMoE env (transformers==4.40.1) may ship an older
    torch that lacks ``with_kwargs``. ``attention_mask`` is taken from a kwarg or
    the 2nd positional arg (``SundialAttention.forward(hidden_states,
    attention_mask, ...)``). ``state["applied"]`` counts real edits so the caller
    can fail loud if a model never hands us an additive mask."""
    state = {"hide_count": None, "applied": 0, "saw_none": 0}

    def make_wrapper(orig):
        def wrapper(*args, **kwargs):
            if "attention_mask" in kwargs:
                edited = _edit_attention_mask(
                    kwargs["attention_mask"], state, last_timesteps, full_len)
                if edited is not None:
                    kwargs = {**kwargs, "attention_mask": edited}
            elif len(args) >= 2:
                edited = _edit_attention_mask(
                    args[1], state, last_timesteps, full_len)
                if edited is not None:
                    args = tuple(edited if i == 1 else a for i, a in enumerate(args))
            return orig(*args, **kwargs)
        return wrapper

    patched = []
    for module in model.modules():
        if _is_attention_module(module):
            orig = module.forward
            module.forward = make_wrapper(orig)
            patched.append((module, orig))
    if not patched:
        raise RuntimeError(
            "No attention submodules found to hook — check "
            "_is_attention_module against the installed model on the server.")
    return patched, state


_DECODER_NOOP_WARNED = [False]


# ==============================================================================
#  Encoder with future-token readout: Chronos-2
# ==============================================================================
#  Chronos-2 is encoder-only but reads the forecast ONLY from appended *future*
#  patch tokens (``hidden_states[:, -num_output_patches:]``), which are the
#  trailing tokens; the *context* patches lead the sequence. So — unlike
#  PatchTST-FM, whose head reads every context token — hiding the oldest context
#  patches as attention keys genuinely restricts the forecast to the last
#  ``last_timesteps`` (old-context hidden states are never read). We mask only
#  ``TimeSelfAttention`` (temporal); ``GroupSelfAttention`` is cross-series and
#  trivial for our one-series-per-instance batches.

def _chronos2_inner_module(handle):
    """Locate the nn.Module inside the Chronos2Pipeline handle."""
    for attr in ("model", "inner_model", "module"):
        m = getattr(handle, attr, None)
        if isinstance(m, torch.nn.Module):
            return m
    if isinstance(handle, torch.nn.Module):
        return handle
    raise AttributeError(
        "Could not locate the Chronos-2 torch module on the pipeline handle; "
        "check the attribute name on the server.")


def _chronos2_patch_step(inner) -> int:
    """Timesteps advanced per context patch (input_patch_stride, or _size)."""
    cfg = getattr(inner, "config", None)
    cc = getattr(cfg, "chronos_config", None)
    def _get(obj, name):
        if obj is None:
            return None
        return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    for obj in (cc, cfg):
        for name in ("input_patch_stride", "input_patch_size", "patch_size"):
            v = _get(obj, name)
            if v:
                return int(v)
    raise AttributeError(
        "Could not infer Chronos-2 input patch stride/size from config; "
        "check the attribute names on the server.")


def _is_time_self_attention(module) -> bool:
    return type(module).__name__ == "TimeSelfAttention"


@contextmanager
def _chronos2_mask(handle, last_timesteps: int, full_len: int):
    inner = _chronos2_inner_module(handle)
    step = _chronos2_patch_step(inner)
    # hide_count is in CONTEXT-patch units. full_len must be <= Chronos-2's
    # context_length (see the caller's cap) so the context isn't internally
    # right-truncated and this patch count is exact; we only ever touch the
    # leading (oldest-context) columns, never the trailing future tokens.
    n_ctx = max(1, round(int(full_len) / step))
    vis = max(1, round(int(last_timesteps) / step))
    hide = max(0, n_ctx - vis)
    state = {"applied": 0, "saw_none": 0}

    def make_wrapper(orig):
        def wrapper(*args, **kwargs):
            if hide > 0:
                from_kwargs = "attention_mask" in kwargs
                am = kwargs.get("attention_mask", None)
                if am is None and len(args) >= 2:
                    am = args[1]
                if torch.is_tensor(am) and am.is_floating_point():
                    am = am.clone()
                    am[..., :hide] = torch.finfo(am.dtype).min
                    if from_kwargs:
                        kwargs = {**kwargs, "attention_mask": am}
                    else:
                        args = tuple(am if i == 1 else a for i, a in enumerate(args))
                    state["applied"] += 1
                else:
                    state["saw_none"] += 1
            return orig(*args, **kwargs)
        return wrapper

    patched = []
    for module in inner.modules():
        if _is_time_self_attention(module):
            orig = module.forward
            module.forward = make_wrapper(orig)
            patched.append((module, orig))
    if not patched:
        raise RuntimeError(
            "No TimeSelfAttention modules found on Chronos-2 — verify the class "
            "name against the installed chronos package on the server.")
    try:
        yield
    finally:
        for module, orig in patched:
            try:
                del module.forward
            except AttributeError:
                module.forward = orig
        if hide > 0 and state["applied"] == 0 and not _DECODER_NOOP_WARNED[0]:
            _DECODER_NOOP_WARNED[0] = True
            print("[context_attention_mask] WARNING: Chronos-2 TimeSelfAttention "
                  "never received an additive attention_mask to edit — the masked "
                  "forecast equals the full-context one. Verify the mask plumbing "
                  "on the server.")


def _warn_if_noop(family: str, state: dict) -> None:
    """Fail loud (once) if we intended to mask but never edited/created a mask."""
    if (state.get("applied", 0) == 0 and state.get("saw_none", 0) > 0
            and not _DECODER_NOOP_WARNED[0]):
        _DECODER_NOOP_WARNED[0] = True
        print(f"[context_attention_mask] WARNING: family {family!r} never received "
              "an editable attention mask — the masked forecast equals the "
              "full-context one. The hook needs a per-model tweak for this "
              "model/version (verify on the server).")


def _config_int(inner, names):
    """First present int among ``names`` on ``inner.config`` or its
    ``chronos_config`` (attr or dict)."""
    cfg = getattr(inner, "config", None)
    for obj in (getattr(cfg, "chronos_config", None), cfg):
        if obj is None:
            continue
        for n in names:
            v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
            if v:
                return int(v)
    return None


# ==============================================================================
#  T5 encoder-decoder: Chronos-Bolt
# ==============================================================================
#  Standard HF T5 (``T5Attention``). The forecast is a single decoder query token
#  cross-attending to the encoder; the same additive encoder mask governs encoder
#  self-attention AND decoder cross-attention. Hiding the oldest context-patch key
#  columns in every ``T5Attention`` whose key dim is the context length therefore
#  (a) keeps visible encoder reps from mixing in old context (self-attn) and (b)
#  stops the decoder query from reading old context (cross-attn). Decoder
#  self-attention has key dim 1 rather than the context-token layout and is
#  therefore skipped.

def _chronosbolt_patch_geometry(handle, last_timesteps: int, full_len: int):
    """Return the exact Chronos-Bolt context-token geometry.

    Chronos-Bolt's key sequence is ``[context patches..., REG?]``.  The REG
    token is not a timestep-bearing patch and must never consume one of the
    requested visible-patch slots.  Derive the patch count from the model
    config and mirror upstream's ``Patch.forward`` padding/unfold operation.

    Returns ``(n_context_patches, n_visible_patches, n_reg_tokens)``.
    """
    inner = _chronos2_inner_module(handle)
    cfg = getattr(inner, "config", None)
    cc = getattr(cfg, "chronos_config", None)

    def _get(name, default=None):
        if isinstance(cc, dict):
            return cc.get(name, default)
        return getattr(cc, name, default) if cc is not None else default

    patch_size = int(_get("input_patch_size", 0) or 0)
    patch_stride = int(_get("input_patch_stride", 0) or 0)
    if patch_size <= 0 or patch_stride <= 0:
        raise AttributeError(
            "Could not infer Chronos-Bolt input_patch_size/input_patch_stride "
            "from model.config.chronos_config.")
    if patch_stride != patch_size:
        raise ValueError(
            "Chronos-Bolt attention masking currently requires non-overlapping "
            "patches (input_patch_stride == input_patch_size); overlapping "
            "patches need an explicit suffix-to-patch overlap policy.")

    def _n_patches(length: int) -> int:
        length = max(1, int(length))
        padded = length + (-length % patch_size)
        return max(1, 1 + (padded - patch_size) // patch_stride)

    n_context = _n_patches(full_len)
    # We retain every tail patch that intersects the requested visible suffix.
    # Official Chronos-Bolt models use non-overlapping patches
    # (input_patch_stride == input_patch_size), for which this is exactly
    # ceil(last_timesteps / patch_size).
    n_visible = min(n_context, _n_patches(last_timesteps))
    n_reg = 1 if bool(_get("use_reg_token", False)) else 0
    return n_context, n_visible, n_reg


def _chronosbolt_mask(handle, last_timesteps: int, full_len: int):
    inner = _chronos2_inner_module(handle)
    n_context, n_visible, n_reg = _chronosbolt_patch_geometry(
        handle, last_timesteps, full_len)
    expected_kv = n_context + n_reg
    hide = max(0, n_context - n_visible)
    state = {
        "applied": 0,
        "saw_none": 0,
        "n_context_patches": n_context,
        "n_visible_patches": n_visible,
        "n_reg_tokens": n_reg,
        "expected_kv": expected_kv,
        "hide_count": hide,
    }

    def make_wrapper(orig):
        def wrapper(*args, **kwargs):
            # T5Attention.forward(self, hidden_states, mask=None, key_value_states, ...)
            from_kwargs = "mask" in kwargs
            m = kwargs.get("mask", None)
            if m is None and len(args) >= 2:
                m = args[1]
            # Encoder self-attention and decoder cross-attention both have
            # ``n_context_patches + n_reg_tokens`` keys. Decoder self-attention
            # has a different key length (normally 1) and must remain untouched.
            if (torch.is_tensor(m) and m.is_floating_point()
                    and m.shape[-1] == expected_kv and hide > 0):
                m = m.clone()
                # Context patches lead and REG, when present, is last. Hiding
                # only the leading context columns preserves REG explicitly.
                m[..., :hide] = torch.finfo(m.dtype).min
                if from_kwargs:
                    kwargs = {**kwargs, "mask": m}
                else:
                    args = tuple(m if i == 1 else a for i, a in enumerate(args))
                state["applied"] += 1
            elif (torch.is_tensor(m) and m.is_floating_point()
                  and m.shape[-1] > 1 and m.shape[-1] != expected_kv):
                # If upstream changes its token layout, fail loud through the
                # existing no-op warning instead of silently masking nothing.
                state["saw_none"] += 1
            return orig(*args, **kwargs)
        return wrapper

    patched = []
    for module in inner.modules():
        if type(module).__name__ == "T5Attention":
            orig = module.forward
            module.forward = make_wrapper(orig)
            patched.append((module, orig))
    if not patched:
        raise RuntimeError("No T5Attention modules found on Chronos-Bolt — verify "
                           "the class name on the server.")
    return patched, state


# ==============================================================================
#  TimesFM 2.5 (decoder-only, patch 32) — EAGER path only (torch_compile=False)
# ==============================================================================
#  timesfm.torch.transformer.MultiHeadAttention builds its own boolean SDPA mask
#  via the module-level ``make_attn_mask`` (True = attend): causal AND
#  ``kv_index >= num_all_masked_kv`` (leading pad patches dropped). We wrap that
#  function — same technique as PatchTST-FM — and additionally drop the oldest
#  ``hide`` CONTEXT-patch key columns for the visible query rows (rectangular:
#  old query rows keep their causal mask so no row is fully masked -> no NaN
#  softmax). ``hide`` is sized in patch (32) units of the fed window; decode
#  appends generated patches to the RIGHT, so the leading columns stay the
#  oldest ones as kv grows. NOTE (server): verify the function lives at
#  ``timesfm.torch.transformer.make_attn_mask`` and that no caller holds a
#  from-import rebinding (we patch any module in timesfm.torch that re-exports
#  it, mirroring the PatchTST-FM dual-binding handling).

TIMESFM_PATCH = 32


@contextmanager
def _timesfm_mask(model, last_timesteps: int, full_len: int):
    import timesfm.torch.transformer as T

    n_ctx = max(1, -(-int(full_len) // TIMESFM_PATCH))       # ceil div
    vis = max(1, -(-int(last_timesteps) // TIMESFM_PATCH))
    hide = max(0, n_ctx - vis)
    state = {"applied": 0, "saw_none": 0}
    orig = T.make_attn_mask

    # collect sibling modules that re-bound make_attn_mask via from-import
    import sys
    rebound = [m for name, m in list(sys.modules.items())
               if name.startswith("timesfm") and m is not None
               and getattr(m, "make_attn_mask", None) is orig and m is not T]

    def wrapped(*args, **kwargs):
        m = orig(*args, **kwargs)
        if hide > 0 and torch.is_tensor(m) and m.dtype == torch.bool:
            m = _rect_hide_bool(m.clone(), min(hide, m.shape[-1] - 1))
            state["applied"] += 1
        elif hide > 0:
            state["saw_none"] += 1
        return m

    T.make_attn_mask = wrapped
    for m in rebound:
        m.make_attn_mask = wrapped
    try:
        yield
    finally:
        T.make_attn_mask = orig
        for m in rebound:
            m.make_attn_mask = orig
        _warn_if_noop("timesfm", state)


# ==============================================================================
#  Moirai2 (decoder-only, single patch, uni2ts) + Toto (decoder-only, DataDog)
#  -- EXPERIMENTAL: both use custom attention whose internals we can only
#  partially see and cannot test locally. Validate on the server (the top-grid
#  sanity check: at L == full window the masked curve must equal the sliced one).
# ==============================================================================

def _rect_hide_bool(m, hide: int):
    """Set the oldest ``hide`` key columns to False (not-attend) for the visible
    query rows of a boolean (True=attend) SDPA mask. Rectangular so old query rows
    keep their causal mask (no fully-masked row)."""
    q, kv = m.shape[-2], m.shape[-1]
    start = max(0, hide - (kv - q))
    m[..., start:, :hide] = False
    return m


def _moirai2_patch_size(handle):
    v = _config_int(handle, ["patch_size", "patch_len"])
    if v:
        return v
    for a in ("patch_size", "patch_len"):
        x = getattr(handle, a, None)
        if x:
            return int(x)
    hp = getattr(handle, "hparams", None)
    if hp is not None:
        x = hp.get("patch_size") if isinstance(hp, dict) else getattr(hp, "patch_size", None)
        if x:
            return int(x)
    return None


def _moirai2_mask(handle, last_timesteps: int, full_len: int):
    """Edit uni2ts ``GroupedQueryAttention``'s boolean causal mask (True=attend) to
    drop the oldest context patches.

    uni2ts calls ``self.self_attn(x, x, x, attn_mask=mask, query_time_id=..., ...)``
    — RoPE lives in the time_id projection (relative), so keeping full-window
    positions is fine (this is why Sundial aligned). Two things the earlier version
    got wrong: the mask kwarg is ``attn_mask`` (not ``attention_mask``), and the
    hide must be sized in CONTEXT patches (patch=16) so the appended prediction
    tokens — which trail the context in the same sequence — are never touched."""
    patch = _moirai2_patch_size(handle)
    state = {"applied": 0, "saw_none": 0, "hide": None}

    def make_wrapper(orig):
        def wrapper(*args, **kwargs):
            key = ("attn_mask" if "attn_mask" in kwargs
                   else ("attention_mask" if "attention_mask" in kwargs else None))
            m = kwargs.get(key) if key else None
            if not (torch.is_tensor(m) and m.dtype == torch.bool):
                state["saw_none"] += 1
                return orig(*args, **kwargs)
            kv = m.shape[-1]
            if state["hide"] is None:
                if patch:
                    n_ctx = max(1, round(int(full_len) / patch))
                    vis = max(1, round(int(last_timesteps) / patch))
                    hide = n_ctx - vis
                else:  # no patch size found -> proportional over the whole seq
                    vis = max(1, round(int(last_timesteps) * kv / int(full_len)))
                    hide = kv - vis
                state["hide"] = max(0, min(kv - 1, hide))
            hide = state["hide"]
            if hide > 0:
                m = _rect_hide_bool(m.clone(), hide)
                kwargs = {**kwargs, key: m}
                state["applied"] += 1
            return orig(*args, **kwargs)
        return wrapper

    patched = []
    for module in handle.modules():
        if type(module).__name__.endswith("Attention"):
            orig = module.forward
            module.forward = make_wrapper(orig)
            patched.append((module, orig))
    if not patched:
        raise RuntimeError("No *Attention modules found on Moirai2 — verify class "
                           "names on the server.")
    return patched, state


def _toto_mask(handle, last_timesteps: int, full_len: int):
    """Wrap Toto's ``TimeWiseMultiheadAttention`` (causal, time axis) to build/edit
    a boolean SDPA mask restricting attention to the last ``last_timesteps``.
    Toto uses ``is_causal`` SDPA (no mask at prefill), so we CONSTRUCT a causal
    boolean mask with the oldest context columns dropped. NOTE: assumes Toto 2.0's
    single-pass (CPM) decode; verify patch size + seq layout on the server."""
    patch = _config_int(handle, ["patch_size", "input_patch_size", "patch_len"])
    state = {"applied": 0, "saw_none": 0}

    def _hide(seq_len):
        if patch:
            n_ctx = max(1, round(int(full_len) / patch))
            vis = max(1, round(int(last_timesteps) / patch))
            return max(0, min(seq_len - 1, n_ctx - vis))
        vis = max(1, round(int(last_timesteps) * seq_len / int(full_len)))
        return max(0, seq_len - vis)

    def make_wrapper(orig):
        def wrapper(*args, **kwargs):
            # forward(self, layer_idx, inputs[b,var,seq,emb], attention_mask, kv_cache)
            inputs = args[1] if len(args) >= 2 else kwargs.get("inputs")
            m = kwargs.get("attention_mask", None)
            if m is None and len(args) >= 3:
                m = args[2]
            if not torch.is_tensor(inputs):
                state["saw_none"] += 1
                return orig(*args, **kwargs)
            seq_len = inputs.shape[-2]
            hide = _hide(seq_len)
            if hide <= 0:
                return orig(*args, **kwargs)
            if torch.is_tensor(m) and m.dtype == torch.bool:
                m = _rect_hide_bool(m.clone(), hide)
            elif torch.is_tensor(m) and m.is_floating_point():
                q, kv = m.shape[-2], m.shape[-1]
                start = max(0, hide - (kv - q))
                m = m.clone()
                m[..., start:, :hide] = torch.finfo(m.dtype).min
            else:
                # No mask (is_causal path): build a causal boolean mask (True=attend)
                # then drop the oldest columns for the visible rows.
                causal = torch.ones((seq_len, seq_len), dtype=torch.bool,
                                    device=inputs.device).tril()
                causal[hide:, :hide] = False
                m = causal
            kwargs = {**kwargs, "attention_mask": m}
            # drop a positional mask if present so our kwarg wins
            if len(args) >= 3:
                args = args[:2] + (None,) + args[3:]
            state["applied"] += 1
            return orig(*args, **kwargs)
        return wrapper

    patched = []
    for module in handle.modules():
        if type(module).__name__ == "TimeWiseMultiheadAttention":
            orig = module.forward
            module.forward = make_wrapper(orig)
            patched.append((module, orig))
    if not patched:
        raise RuntimeError("No TimeWiseMultiheadAttention on Toto — verify class "
                           "name against the installed toto2 on the server.")
    return patched, state


def _restore_forwards(patched):
    for module, orig in patched:
        try:
            del module.forward
        except AttributeError:
            module.forward = orig


# ==============================================================================
#  Public entry point
# ==============================================================================

SUPPORTED_FAMILIES = {"patchtst_fm", "sundial", "timemoe", "chronos2",
                      "chronos_bolt", "moirai", "toto", "timesfm"}


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

    if family == "chronos2":
        with _chronos2_mask(model, last_timesteps, full_len):
            yield
        return

    if family == "timesfm":
        # EAGER models only (torch_compile=False) — see the module docstring.
        with _timesfm_mask(model, last_timesteps, full_len):
            yield
        return

    if family in ("chronos_bolt", "moirai", "toto"):
        installer = {"chronos_bolt": _chronosbolt_mask,
                     "moirai": _moirai2_mask, "toto": _toto_mask}[family]
        patched, state = installer(model, last_timesteps, full_len)
        try:
            yield
        finally:
            _restore_forwards(patched)
            _warn_if_noop(family, state)
        return

    if family in ("sundial", "timemoe"):
        patched, state = _install_decoder_mask(model, last_timesteps, full_len)
        try:
            yield
        finally:
            for module, orig in patched:
                try:
                    del module.forward           # restore the class method
                except AttributeError:
                    module.forward = orig
        # Fail loud (once) if we intended to mask but never edited a real mask —
        # otherwise the "masked" forecast is silently the full-context one.
        if state["applied"] == 0 and state["saw_none"] > 0 \
                and not _DECODER_NOOP_WARNED[0]:
            _DECODER_NOOP_WARNED[0] = True
            print(
                f"[context_attention_mask] WARNING: family {family!r} never "
                "received an additive attention_mask to edit (pure is_causal SDPA "
                "path?) — the masked forecast equals the full-context one. The "
                "decoder hook needs a per-module tweak for this model/transformers "
                "version before the masking curve is meaningful.")
        return

    raise NotImplementedError(
        f"Attention masking not implemented for family {family!r}. "
        f"Supported: {sorted(SUPPORTED_FAMILIES)}.")
