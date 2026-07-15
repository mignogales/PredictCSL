import unittest

import numpy as np

from context_interpretability.data.synthetic_generators import (
    ControlSpec, gate_detectable, generate_control, verify_distant_information)


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


if __name__ == "__main__":
    unittest.main()
