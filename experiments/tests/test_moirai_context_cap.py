import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments import test_window_ablation_gifteval_v5 as ablation


class FullNativeContextCapTest(unittest.TestCase):
    def test_chronos2_full_native_cap_matches_registered_limit(self):
        self.assertEqual(
            ablation._full_native_context_cap("chronos2", 480, 104_640),
            8_192,
        )

    def test_chronos2_full_native_cap_does_not_expand_short_context(self):
        self.assertEqual(
            ablation._full_native_context_cap("chronos2", 480, 4_096),
            4_096,
        )

    def test_full_native_cap_reserves_the_forecast_horizon(self):
        self.assertEqual(
            ablation._full_native_context_cap("moirai", 480, 104_640),
            7_712,
        )

    def test_full_native_cap_does_not_expand_short_context(self):
        self.assertEqual(
            ablation._full_native_context_cap("moirai", 480, 4_096),
            4_096,
        )

    def test_cap_accounts_for_patch_padding(self):
        self.assertEqual(ablation._moirai_max_context(1), 8_176)

    def test_other_uncapped_family_still_uses_available_context(self):
        self.assertEqual(
            ablation._full_native_context_cap(
                "context_parroting", 480, 104_640
            ),
            104_640,
        )

    def test_grid_cap_is_reusable_only_when_every_context_reaches_it(self):
        cache = SimpleNamespace(
            max_context=128,
            context_lengths=np.array([128, 160, 256]),
        )
        self.assertEqual(
            ablation._full_native_reusable_grid_cap(
                cache, "context_parroting", 16, [64, 128]),
            128,
        )
        cache.context_lengths[0] = 127
        self.assertIsNone(ablation._full_native_reusable_grid_cap(
            cache, "context_parroting", 16, [64, 128]))
        cache.context_lengths[0] = 128
        self.assertIsNone(ablation._full_native_reusable_grid_cap(
            cache, "context_parroting", 16, [32, 64]))

    def test_equivalent_grid_cache_is_materialized_as_full_native(self):
        cache = SimpleNamespace(n_total=3)
        metrics = {
            "mae": 1.25,
            "mse": 2.0,
            "rmse": 2.0 ** 0.5,
            "mase_gluonts_real": 0.8,
            "_mase_gluonts_real_standin": False,
            "_mase_gluonts_ver": ablation.MASE_GLUONTS_VER,
            "_metric_suite_ver": ablation.METRIC_SUITE_VER,
            "elapsed_seconds": 3.5,
            "_dynamic_batch_sizes": [8],
        }
        per_sample = {
            "mae": np.array([1.0, 1.25, 1.5]),
            "served_index": np.arange(3, dtype=np.int32),
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                ablation, "CACHE_ROOT", tmp):
            ablation._save_result(
                "Example", "ContextParroting", "short", 128, metrics)
            ablation._save_per_sample_metrics(
                "Example", "ContextParroting", "short", 128, per_sample)

            self.assertTrue(ablation._reuse_grid_cell_as_full_native(
                cache, "Example", "short", "context_parroting",
                "ContextParroting", 128, "skip"))

            native_dir = ablation._cache_dir(
                "Example", "ContextParroting", "short",
                ablation.FULL_NATIVE_WINDOW)
            with open(os.path.join(native_dir, "metrics.json")) as f:
                native = json.load(f)
            self.assertTrue(native["_full_native_baseline"])
            self.assertEqual(native["_full_native_reused_from_grid"], 128)
            self.assertEqual(native["elapsed_seconds"], 3.5)
            with np.load(os.path.join(
                    native_dir, "per_sample_metrics.npz")) as data:
                np.testing.assert_array_equal(
                    data["effective_context"], np.full(3, 128))

    def test_existing_full_native_cache_can_materialize_grid_cap(self):
        cache = SimpleNamespace(n_total=2)
        metrics = {
            "mae": 0.75,
            "mse": 1.0,
            "rmse": 1.0,
            "mase_gluonts_real": 0.6,
            "_mase_gluonts_real_standin": False,
            "_mase_gluonts_ver": ablation.MASE_GLUONTS_VER,
            "_metric_suite_ver": ablation.METRIC_SUITE_VER,
            "_full_native_baseline": True,
            "_context_cap": 128,
            "_min_effective_context": 128,
            "_mean_effective_context": 128.0,
            "_max_effective_context": 128,
            "_n_width_groups": 1,
            "elapsed_seconds": 2.5,
        }
        per_sample = {
            "mae": np.array([0.5, 1.0]),
            "served_index": np.arange(2, dtype=np.int32),
            "effective_context": np.full(2, 128, dtype=np.int32),
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                ablation, "CACHE_ROOT", tmp):
            ablation._save_result(
                "Example", "ContextParroting", "short",
                ablation.FULL_NATIVE_WINDOW, metrics)
            ablation._save_per_sample_metrics(
                "Example", "ContextParroting", "short",
                ablation.FULL_NATIVE_WINDOW, per_sample)

            self.assertTrue(ablation._reuse_full_native_cell_as_grid(
                cache, "Example", "short", "context_parroting",
                "ContextParroting", 128))

            grid_dir = ablation._cache_dir(
                "Example", "ContextParroting", "short", 128)
            with open(os.path.join(grid_dir, "metrics.json")) as f:
                grid = json.load(f)
            self.assertTrue(grid["_grid_reused_from_full_native"])
            self.assertNotIn("_full_native_baseline", grid)
            self.assertNotIn("_context_cap", grid)
            self.assertEqual(grid["elapsed_seconds"], 2.5)
            with np.load(os.path.join(
                    grid_dir, "per_sample_metrics.npz")) as data:
                self.assertNotIn("effective_context", data.files)


if __name__ == "__main__":
    unittest.main()
