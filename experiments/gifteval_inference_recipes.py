"""Versioned inference recipes shared by the GiftEval experiment stages.

The model forecast is the numerator of every error metric.  Keep a compact
recipe id beside expensive stage-1 labels and stage-3 forecasts so changing a
checkpoint, output head, precision, or missing-value policy cannot silently
reuse incompatible caches.
"""

from __future__ import annotations

from typing import Optional

from experiments.timesfm_gifteval import TIMESFM_GIFTEVAL_RECIPE


INFERENCE_RECIPES = {
    "timesfm": TIMESFM_GIFTEVAL_RECIPE,
    # Shared by Chronos2-Small, Chronos2-Base, and Chronos2-Synth. All three
    # must forecast each univariate series independently; cross-learning would
    # make a result depend on whichever unrelated series share its batch.
    "chronos2": "chronos2_univariate_no_cross_learning_v1",
    "chronos_bolt": "official_chronos_bolt_float32_quantile_v1",
    # Intentional ablation mode: checkpoint-native 8192 total points (context
    # plus forecast), not the leaderboard notebook's conservative 4000 context.
    "moirai": "moirai2_8192total_raw_quantile_v1",
    # v2 stabilizes fully masked left-padding query rows. Real query rows retain
    # the official key-padding mask; the change prevents CUDA SDPA NaNs only.
    "patchtst_fm": "official_patchtst_fm_list_q05_padding_safe_v2",
    "sundial": "official_sundial_100samples_lastvalue_v1",
    "tirex": "official_tirex2_gifteval_zs_v1",
}

# These official wrappers consume the original missing observations.  Each
# model then applies its own policy (Chronos/Moirai/TiRex internally,
# mean-imputation for PatchTST-FM, last-value imputation for Sundial).
RAW_CONTEXT_FAMILIES = frozenset({
    "timesfm", "chronos_bolt", "patchtst_fm", "sundial", "moirai", "tirex",
})


def inference_recipe(model_family: str) -> Optional[str]:
    """Return the cache recipe for a parity-sensitive family, if any."""
    return INFERENCE_RECIPES.get(model_family)


def preserves_missing(model_family: str) -> bool:
    """Whether batching must retain raw NaNs for the model-specific wrapper."""
    return model_family in RAW_CONTEXT_FAMILIES
