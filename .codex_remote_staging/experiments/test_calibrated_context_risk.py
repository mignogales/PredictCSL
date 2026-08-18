import unittest

import numpy as np

from experiments.calibrated_context_risk import (
    _calibrate_profiles,
    _context_flops,
    _requested_window_action,
    extract_series_features,
    make_pair_features,
    policy_metrics,
    select_shortest_safe,
)


class CalibratedContextRiskTests(unittest.TestCase):
    def test_context_flops_uses_model_cost_and_caches_duplicates(self):
        cache = {}
        result = _context_flops(
            "TimesFM2.5-200M", np.asarray([32, 320, 320]), 96, cache)
        self.assertGreater(float(result[1]), float(result[0]))
        self.assertEqual(float(result[1]), float(result[2]))
        self.assertEqual(len(cache), 2)

    def test_series_features_are_finite_and_affine_invariant(self):
        t = np.arange(256, dtype=np.float32)
        signal = np.sin(2 * np.pi * t / 24) + 0.01 * t
        contexts = np.stack([signal, 7.5 * signal + 31.0])
        features = extract_series_features(
            contexts, np.asarray([256, 256]), 256,
            scales=(32, 128), lags=(1, 24))
        self.assertEqual(features.shape[0], 2)
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[0], features[1], atol=2e-4, rtol=2e-4)

    def test_pair_features_reflect_effective_context(self):
        base = np.zeros((2, 3), dtype=np.float32)
        result = make_pair_features(
            base,
            np.asarray([80, 256]),
            np.asarray([128, 128]),
            np.asarray([24, 24]),
            256,
        )
        self.assertEqual(result.shape, (2, 13))
        # effective/native fraction is the fourth appended feature.
        self.assertAlmostEqual(float(result[0, 6]), 1.0)
        self.assertAlmostEqual(float(result[1, 6]), 0.5)

    def test_shortest_safe_respects_availability_and_abstains(self):
        score = np.asarray([
            [-0.2, -0.3, 0.1],
            [-0.2, -0.3, -0.4],
            [0.2, 0.3, 0.4],
        ])
        available = np.asarray([
            [False, True, True],
            [True, True, True],
            [True, True, True],
        ])
        action = select_shortest_safe(
            score, 0.0, np.asarray([32, 64, 128]),
            np.asarray([256, 48, 256]), available)
        np.testing.assert_array_equal(action, np.asarray([1, 0, -1]))

    def test_policy_metrics_uses_native_for_abstentions(self):
        errors = np.asarray([[0.8, 1.2], [1.1, 0.9]], dtype=np.float64)
        counts = np.ones_like(errors)
        metrics = policy_metrics(
            errors, counts, np.asarray([32, 64]),
            np.asarray([1.0, 1.0]), np.ones(2), np.asarray([128, 128]),
            np.asarray([0, -1]))
        self.assertAlmostEqual(metrics["mean_ratio"], 0.9)
        self.assertAlmostEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["harm5_rate"], 0.0)
        self.assertAlmostEqual(metrics["context_saved_pct"], 37.5)

    def test_requested_window_baseline_caps_and_uses_nearest_cache(self):
        available = np.ones((3, 3), dtype=bool)
        action = _requested_window_action(
            96, np.asarray([32, 64, 128]),
            np.asarray([256, 80, 96]), available)
        # 96 is equally far from 64 and 128, so the shorter cached action wins;
        # rows whose capped request reaches native abstain to native/full.
        np.testing.assert_array_equal(action, np.asarray([1, -1, -1]))

    def test_calibration_freezes_one_score_and_profiles_are_nested(self):
        n = 40
        predicted_mean = np.tile(np.asarray([-0.2, -0.08, -0.01]), (n, 1))
        predicted_std = np.zeros_like(predicted_mean)
        predicted_harm = np.tile(np.asarray([1.0, 0.0, 0.0]), (n, 1))
        errors = np.ones((n, 4), dtype=np.float64)
        errors[:, 0] = np.where(np.arange(n) % 2 == 0, 0.9, 1.1)
        errors[:, 1] = np.where(np.arange(n) % 5 == 0, 1.06, 0.99)
        errors[:, 2] = 0.995
        profiles, _ = _calibrate_profiles(
            predicted_mean, predicted_std, predicted_harm,
            errors, np.asarray([32, 64, 128, 256]),
            np.full(n, 256), uncertainty_weights=(0.0, 1.0),
            harm_weights=(0.0, 0.5), n_quantiles=9)
        score_definitions = {
            (row["config"]["uncertainty_weight"], row["config"]["harm_weight"])
            for row in profiles.values()
        }
        self.assertEqual(len(score_definitions), 1)
        savings = [
            profiles[name]["validation"]["context_saved_pct"]
            for name in (
                "conservative", "balanced", "aggressive", "very_aggressive")
        ]
        self.assertLessEqual(savings[0], savings[1])
        self.assertLessEqual(savings[1], savings[2])


if __name__ == "__main__":
    unittest.main()
