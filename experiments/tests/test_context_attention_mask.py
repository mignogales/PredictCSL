from types import SimpleNamespace
import unittest

import torch
from torch import nn

from experiments.context_attention_mask import (
    _chronosbolt_mask,
    _chronosbolt_patch_geometry,
    _patchtstfm_retained_patches,
    _restore_forwards,
)


class T5Attention(nn.Module):
    """Minimal exact-name stand-in for transformers.models.t5.T5Attention."""

    def __init__(self):
        super().__init__()
        self.last_mask = None

    def forward(self, hidden_states, mask=None, key_value_states=None):
        self.last_mask = mask
        return hidden_states


class _BoltInner(nn.Module):
    def __init__(self, use_reg_token=True):
        super().__init__()
        self.config = SimpleNamespace(chronos_config={
            "input_patch_size": 16,
            "input_patch_stride": 16,
            "use_reg_token": use_reg_token,
        })
        self.encoder_attention = T5Attention()
        self.cross_attention = T5Attention()
        self.decoder_self_attention = T5Attention()


class _Pipeline:
    def __init__(self, inner):
        self.model = inner


class TestChronosBoltAttentionMask(unittest.TestCase):
    def test_geometry_counts_reg_separately(self):
        handle = _Pipeline(_BoltInner(use_reg_token=True))
        self.assertEqual(
            _chronosbolt_patch_geometry(handle, 128, 2048),
            (128, 8, 1),
        )

    def test_masks_exact_context_patches_and_preserves_reg(self):
        inner = _BoltInner(use_reg_token=True)
        handle = _Pipeline(inner)
        patched, state = _chronosbolt_mask(handle, 128, 2048)
        try:
            context_mask = torch.zeros(2, 1, 129, 129)
            cross_mask = torch.zeros(2, 1, 1, 129)
            decoder_mask = torch.zeros(2, 1, 1, 1)
            hidden = torch.zeros(2, 129, 4)

            inner.encoder_attention(hidden, mask=context_mask)
            inner.cross_attention(
                hidden[:, :1], mask=cross_mask, key_value_states=hidden)
            inner.decoder_self_attention(hidden[:, :1], mask=decoder_mask)
        finally:
            _restore_forwards(patched)

        min_value = torch.finfo(context_mask.dtype).min
        # W=2048 -> 128 context patches. L=128 -> retain 8 patches, so hide 120.
        self.assertEqual(state["hide_count"], 120)
        self.assertEqual(state["n_visible_patches"], 8)
        self.assertEqual(state["applied"], 2)
        self.assertTrue(
            torch.all(inner.encoder_attention.last_mask[..., :120] == min_value))
        self.assertTrue(
            torch.all(inner.cross_attention.last_mask[..., :120] == min_value))
        # Eight tail context patches (120:128) and REG (128) remain visible.
        self.assertTrue(
            torch.all(inner.encoder_attention.last_mask[..., 120:] == 0))
        self.assertTrue(
            torch.all(inner.cross_attention.last_mask[..., 120:] == 0))
        # Decoder self-attention does not have the context-key layout.
        self.assertTrue(torch.equal(
            inner.decoder_self_attention.last_mask, decoder_mask))

    def test_without_reg_uses_context_only_key_length(self):
        inner = _BoltInner(use_reg_token=False)
        handle = _Pipeline(inner)
        patched, state = _chronosbolt_mask(handle, 48, 64)
        try:
            mask = torch.zeros(1, 1, 4, 4)
            inner.encoder_attention(torch.zeros(1, 4, 2), mask=mask)
        finally:
            _restore_forwards(patched)

        self.assertEqual(state["expected_kv"], 4)
        self.assertEqual(state["hide_count"], 1)
        self.assertEqual(state["applied"], 1)
        self.assertEqual(
            inner.encoder_attention.last_mask[0, 0, 0, 0].item(),
            torch.finfo(mask.dtype).min,
        )
        self.assertTrue(
            torch.all(inner.encoder_attention.last_mask[..., 1:] == 0))


class TestPatchTSTAttentionMask(unittest.TestCase):
    def test_forecast_patches_do_not_consume_context_budget(self):
        # Patch size 32 and horizon 64 means two forecast patches are always
        # retained *in addition to* the requested historical patches.
        self.assertEqual(_patchtstfm_retained_patches(32, 64, 32), (1, 2, 3))
        self.assertEqual(_patchtstfm_retained_patches(64, 64, 32), (2, 2, 4))
        self.assertEqual(_patchtstfm_retained_patches(128, 64, 32), (4, 2, 6))

    def test_partial_patches_round_up(self):
        self.assertEqual(_patchtstfm_retained_patches(33, 65, 32), (2, 3, 5))


if __name__ == "__main__":
    unittest.main()
