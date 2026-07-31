"""Shared PatchTST-FM inference safeguards for GIFT-Eval.

Granite-TSFM 0.3.6 builds an additive attention mask that masks both padded
keys and padded queries.  A padded query therefore receives an all-``-inf``
attention row.  CUDA scaled-dot-product attention can turn that row into NaNs;
after the residual connection, later layers can propagate those NaNs into the
real forecast tokens.  This is especially visible in context-window ablations,
where every context shorter than PatchTST-FM's native 8192 steps is left padded.

The official model still needs its pad mask: real query tokens must not attend
to padded keys.  We therefore change only fully masked *query* rows, giving each
one a single finite padding key.  Real query rows and their outputs are bitwise
unchanged by the mask transformation.
"""

from __future__ import annotations

from contextlib import contextmanager


def stabilize_fully_masked_attention_rows(mask):
    """Make all-``-inf`` additive-attention rows numerically well-defined.

    ``make_attn_mask`` returns a fresh tensor, so mutating it avoids another
    native-context-sized allocation.  The implementation also avoids a Python
    truth test on a CUDA tensor (and therefore an unnecessary synchronization).
    """
    if mask is None or getattr(mask, "dtype", None) is None:
        return mask

    import torch

    if not mask.dtype.is_floating_point or mask.ndim < 2 or mask.shape[-1] == 0:
        return mask
    empty_rows = torch.isneginf(mask).all(dim=-1)
    mask[..., 0].masked_fill_(empty_rows, 0.0)
    return mask


@contextmanager
def patchtst_padding_safe_attention():
    """Temporarily stabilize PatchTST-FM's internally constructed pad mask.

    Patch both module bindings because ``modeling_patchtst_fm`` imports
    ``make_attn_mask`` directly from ``basic``.  Wrapping the current binding
    (rather than assuming both names still point to the same function) composes
    correctly with the interpretability attention-mask context manager.
    """
    try:
        import tsfm_public.models.patchtst_fm.basic as basic
        import tsfm_public.models.patchtst_fm.modeling_patchtst_fm as modeling
    except (ImportError, AttributeError):
        # Unit tests use small API-compatible fake models and do not install the
        # heavyweight TSFM dependency.
        yield
        return

    original_basic = getattr(basic, "make_attn_mask", None)
    original_modeling = getattr(modeling, "make_attn_mask", None)
    if original_basic is None or original_modeling is None:
        yield
        return

    def wrap(original):
        def wrapped(*args, **kwargs):
            return stabilize_fully_masked_attention_rows(
                original(*args, **kwargs))

        return wrapped

    basic.make_attn_mask = wrap(original_basic)
    if original_modeling is original_basic:
        modeling.make_attn_mask = basic.make_attn_mask
    else:
        modeling.make_attn_mask = wrap(original_modeling)
    try:
        yield
    finally:
        basic.make_attn_mask = original_basic
        modeling.make_attn_mask = original_modeling


def require_finite_patchtst_forecast(forecast):
    """Reject a poisoned PatchTST forecast before it can enter cell caches."""
    import torch

    bad = ~torch.isfinite(forecast)
    if bool(bad.any()):
        raise RuntimeError(
            "PatchTST-FM returned non-finite forecasts after padding-safe "
            f"attention ({int(bad.sum().item())}/{forecast.numel()} values)."
        )
    return forecast
