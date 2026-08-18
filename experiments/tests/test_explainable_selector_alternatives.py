import unittest

import numpy as np

from experiments.explainable_selector_alternatives import (
    OrdinalQuantileTree,
    PiecewiseLinearTree,
    smooth_direct_risk,
)


class ExplainableSelectorAlternativesTests(unittest.TestCase):
    def test_smooth_direct_risk_penalizes_harm_boundary(self) -> None:
        values = smooth_direct_risk(np.asarray([-0.1, 0.0, 0.1]))
        self.assertTrue(np.all(np.diff(values) > 0.0))
        self.assertGreater(float(values[-1] - values[1]), 0.1)

    def test_ordinal_tree_returns_bounded_ordered_score(self) -> None:
        x = np.arange(200, dtype=np.float32)[:, None]
        y = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
        tree = OrdinalQuantileTree(4, 4, 8, 42).fit(x, y)
        prediction = tree.predict(x)
        self.assertTrue(np.all((prediction >= 0.0) & (prediction <= 1.0)))
        self.assertGreater(np.corrcoef(prediction, y)[0, 1], 0.9)

    def test_piecewise_linear_tree_fits_simple_rule(self) -> None:
        rng = np.random.RandomState(42)
        x = rng.normal(size=(400, 3)).astype(np.float32)
        # Give the regimes different means so an axis-aligned routing tree can
        # identify the split before the leaf equations recover the slopes.
        y = np.where(
            x[:, 0] < 0.0, -3.0 + 2.0 * x[:, 1], 3.0 - 3.0 * x[:, 2])
        tree = PiecewiseLinearTree(2, 40, 1e-4, 42).fit(x, y)
        prediction = tree.predict(x)
        self.assertLess(float(np.mean((prediction - y) ** 2)), 0.15)
        self.assertGreaterEqual(tree.get_n_leaves(), 2)


if __name__ == "__main__":
    unittest.main()
