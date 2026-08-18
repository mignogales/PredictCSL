from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.context_normalization_override import (
    normalization_reference_override,
    normalization_stat_override,
)


class _BoltNorm(torch.nn.Module):
    eps = 1e-5

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x, loc_scale=None):
        self.seen = loc_scale
        loc, scale = loc_scale
        return (x - loc) / scale, (loc, scale)


class _PatchNorm:
    std_min = 1e-5

    def _get_statistics(self, x, mask=None):
        valid = (~mask.bool()).float()
        count = valid.sum(-1, keepdim=True)
        self.mean = (x * valid).sum(-1, keepdim=True) / count
        self.std = (((x - self.mean) * valid).square().sum(
            -1, keepdim=True) / count).sqrt()

    def transform(self, x):
        return (x - self.mean) / self.std

    def fit_transform(self, x, mask=None):
        self._get_statistics(x, mask)
        return self.transform(x)


class TestDirectNormalizationControls(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[0.0, 2.0, 10.0, 14.0]])

    def test_chronos_bolt_mean_only(self):
        norm = _BoltNorm()
        handle = SimpleNamespace(model=SimpleNamespace(instance_norm=norm))
        with normalization_stat_override(
                "chronos_bolt", handle, 2, tail_mean=True):
            norm(self.x)
        loc, scale = norm.seen
        self.assertAlmostEqual(loc.item(), 12.0)
        self.assertAlmostEqual(scale.item(), self.x.std(unbiased=False).item())

    def test_chronos_bolt_scale_only(self):
        norm = _BoltNorm()
        handle = SimpleNamespace(model=SimpleNamespace(instance_norm=norm))
        with normalization_stat_override(
                "chronos_bolt", handle, 2, tail_scale=True):
            norm(self.x)
        loc, scale = norm.seen
        self.assertAlmostEqual(loc.item(), self.x.mean().item())
        self.assertAlmostEqual(scale.item(), 2.0)

    def test_patchtst_uses_last_valid_history_not_future_slots(self):
        norm = _PatchNorm()
        handle = SimpleNamespace(backbone=SimpleNamespace(norm_fn=norm))
        # Leading pad and trailing forecast slots are masked; the last two
        # valid history values are 10 and 14.
        x = torch.tensor([[99.0, 0.0, 2.0, 10.0, 14.0, 88.0, 77.0]])
        mask = torch.tensor([[True, False, False, False, False, True, True]])
        with normalization_stat_override(
                "patchtst_fm", handle, 2,
                tail_mean=True, tail_scale=True):
            norm.fit_transform(x, mask)
        self.assertAlmostEqual(norm.mean.item(), 12.0)
        self.assertAlmostEqual(norm.std.item(), 2.0)

    def test_chronos_slice_can_use_full_history_stats(self):
        norm = _BoltNorm()
        handle = SimpleNamespace(model=SimpleNamespace(instance_norm=norm))
        sliced = self.x[:, -2:]
        with normalization_reference_override(
                "chronos_bolt", handle, self.x.numpy()):
            norm(sliced)
        loc, scale = norm.seen
        self.assertAlmostEqual(loc.item(), self.x.mean().item())
        self.assertAlmostEqual(scale.item(), self.x.std(unbiased=False).item())

    def test_patchtst_reference_matches_official_missing_imputation(self):
        norm = _PatchNorm()
        handle = SimpleNamespace(backbone=SimpleNamespace(norm_fn=norm))
        references = [np.array([0.0, np.nan, 4.0], dtype=np.float32)]
        with normalization_reference_override(
                "patchtst_fm", handle, references):
            norm.fit_transform(torch.tensor([[10.0, 14.0]]))
        # Official preprocessing turns [0, nan, 4] into [0, 2, 4].
        self.assertAlmostEqual(norm.mean.item(), 2.0)
        self.assertAlmostEqual(
            norm.std.item(), np.std([0.0, 2.0, 4.0]), places=6)

    def test_reference_override_repeats_for_recursive_quantile_expansion(self):
        norm = _BoltNorm()
        handle = SimpleNamespace(model=SimpleNamespace(instance_norm=norm))
        refs = np.array([[0.0, 2.0], [10.0, 14.0]], dtype=np.float32)
        with normalization_reference_override("chronos_bolt", handle, refs):
            norm(torch.tensor([[8.0], [12.0]]))
            np.testing.assert_allclose(norm.seen[0].flatten(), [1.0, 12.0])
            norm(torch.tensor([[8.0], [9.0], [12.0], [13.0]]))
            np.testing.assert_allclose(
                norm.seen[0].flatten(), [1.0, 1.0, 12.0, 12.0])


if __name__ == "__main__":
    unittest.main()
