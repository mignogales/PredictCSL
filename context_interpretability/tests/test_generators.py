import unittest

import numpy as np

from context_interpretability.data.synthetic_generators import (
    ControlSpec, gate_detectable, generate_control, verify_distant_information)
from context_interpretability.data.validation_generators import (
    _KERNEL_BANK, load_harmonic_pools, load_kernelsynth_pool)


class TestSyntheticControls(unittest.TestCase):
    def test_shapes_and_determinism(self):
        spec = ControlSpec("A", "linear", 4, noise=0.1)
        c1, t1, g1 = generate_control(spec, 8, 128, 16, seed=3)
        c2, t2, g2 = generate_control(spec, 8, 128, 16, seed=3)
        self.assertEqual(c1.shape, (8, 128))
        self.assertEqual(t1.shape, (8, 16))
        self.assertIsNone(g1)
        np.testing.assert_array_equal(c1, c2)
        self.assertTrue(np.isfinite(c1).all())

    def test_stability_at_max_strength(self):
        spec = ControlSpec("B", "linear", 4, distant_lag=32, strength=1.0,
                           noise=0.25)
        c, _t, _g = generate_control(spec, 4, 2048, 8, seed=0)
        self.assertTrue(np.isfinite(c).all())
        self.assertLess(np.abs(c).max(), 100.0)         # no explosion

    def test_distant_lag_is_genuinely_predictive(self):
        spec = ControlSpec("B", "linear", 4, distant_lag=32, strength=1.0,
                           noise=0.1)
        c, _t, _g = generate_control(spec, 32, 1024, 8, seed=1)
        rep = verify_distant_information(c, spec, seed=1)
        self.assertTrue(rep["distant_predictive"],
                        f"oracle gain {rep['relative_gain']:.4f} not positive")

    def test_zero_strength_has_no_distant_information(self):
        spec = ControlSpec("B", "linear", 4, distant_lag=32, strength=0.0,
                           noise=0.1)
        c, _t, _g = generate_control(spec, 32, 1024, 8, seed=2)
        rep = verify_distant_information(c, spec, seed=2)
        self.assertLess(abs(rep["relative_gain"]), 0.05)

    def test_family_c_gates_and_scale(self):
        spec = ControlSpec("C", "linear", 4, distant_lag=32, strength=1.0,
                           noise=0.1)
        c, _t, g = generate_control(spec, 40, 1024, 8, seed=4)
        self.assertIsNotNone(g)
        frac = g.mean()
        self.assertGreater(frac, 0.25)
        self.assertLess(frac, 0.75)
        # unit-variance rescaling -> amplitude alone cannot separate groups
        stds = c.std(axis=1)
        self.assertLess(abs(stds[g].mean() - stds[~g].mean()), 0.05)
        # ...but the gate is inferable from the context itself
        self.assertGreater(gate_detectable(c, g, spec, seed=4), 0.6)


class TestMaskingValidationGenerators(unittest.TestCase):
    def test_harmonic_pools_are_deterministic_and_scale_matched(self):
        pools = load_harmonic_pools(
            4, series_length=128, horizon=16, seed=7,
            periods=[16, 32, 64], scales=[0.5, 1.0, 2.0])
        again = load_harmonic_pools(
            4, series_length=128, horizon=16, seed=7,
            periods=[16, 32, 64], scales=[0.5, 1.0, 2.0])
        self.assertEqual(len(pools), 3)
        self.assertEqual(pools[0].contexts.shape, (4, 128))
        self.assertEqual(pools[0].targets.shape, (4, 16))
        np.testing.assert_array_equal(pools[1].contexts, again[1].contexts)
        np.testing.assert_allclose(pools[2].contexts, 4 * pools[0].contexts)
        np.testing.assert_allclose(pools[2].targets, 4 * pools[0].targets)

    def test_kernelsynth_uses_official_bank_and_is_deterministic(self):
        self.assertEqual(len(_KERNEL_BANK), 33)
        first = load_kernelsynth_pool(
            2, series_length=24, horizon=8, seed=9, max_kernels=3)
        second = load_kernelsynth_pool(
            2, series_length=24, horizon=8, seed=9, max_kernels=3)
        self.assertEqual(first.contexts.shape, (2, 24))
        self.assertTrue(np.isfinite(first.contexts).all())
        np.testing.assert_array_equal(first.contexts, second.contexts)
        np.testing.assert_array_equal(first.targets, second.targets)

    def test_scalable_kernelsynth_supports_long_contexts(self):
        first = load_kernelsynth_pool(
            2, series_length=2048, horizon=16, seed=11, scalable=True)
        second = load_kernelsynth_pool(
            2, series_length=2048, horizon=16, seed=11, scalable=True)
        self.assertEqual(first.name, "kernelsynth_long_covariance_approx")
        self.assertEqual(first.contexts.shape, (2, 2048))
        self.assertTrue(np.isfinite(first.contexts).all())
        np.testing.assert_array_equal(first.contexts, second.contexts)


if __name__ == "__main__":
    unittest.main()
