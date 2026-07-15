import unittest

import numpy as np

from context_interpretability.metrics import forecast_metrics as fm
from context_interpretability.metrics import prediction_distance as pdist
from context_interpretability.metrics import statistics as st


class TestForecastMetrics(unittest.TestCase):
    def setUp(self):
        self.pred = np.array([[1.0, 2.0], [3.0, 5.0]])
        self.tgt = np.array([[1.0, 1.0], [1.0, 1.0]])

    def test_mae_mse(self):
        np.testing.assert_allclose(fm.mae(self.pred, self.tgt), [0.5, 3.0])
        np.testing.assert_allclose(fm.mse(self.pred, self.tgt), [0.5, 10.0])

    def test_smape_bounds(self):
        s = fm.smape(self.pred, self.tgt)
        self.assertTrue(np.all(s >= 0) and np.all(s <= 2))

    def test_mase_per_instance(self):
        ctx = np.array([[0., 1., 0., 1., 0., 1.]])
        # seasonal_error (m=1) = mean |diff| = 1.0
        m = fm.mase(np.array([[2.0]]), np.array([[0.0]]), context=ctx)
        np.testing.assert_allclose(m, [2.0])

    def test_crps_quantiles(self):
        q = np.zeros((1, 3, 4))
        q[0, 0], q[0, 1], q[0, 2] = -1.0, 0.0, 1.0
        crps = fm.crps_from_quantiles(q, [0.1, 0.5, 0.9], np.zeros((1, 4)))
        self.assertGreaterEqual(crps[0], 0.0)

    def test_dispatch(self):
        np.testing.assert_allclose(
            fm.compute_loss("mae", self.pred, self.tgt), [0.5, 3.0])
        with self.assertRaises(ValueError):
            fm.compute_loss("nope", self.pred, self.tgt)


class TestPredictionDistance(unittest.TestCase):
    def test_l1_and_normalized(self):
        a = np.array([[1.0, 1.0]])
        b = np.array([[2.0, 0.0]])
        np.testing.assert_allclose(pdist.l1_distance(a, b), [1.0])
        np.testing.assert_allclose(pdist.normalized_distance(a, b), [1.0],
                                   atol=1e-6)
        np.testing.assert_allclose(pdist.l1_distance(a, a), [0.0])


class TestStatistics(unittest.TestCase):
    def test_bootstrap_ci_contains_mean(self):
        v = np.random.default_rng(0).normal(2.0, 0.1, 200)
        lo, hi = st.bootstrap_ci(v, seed=0)
        self.assertLess(lo, 2.0)
        self.assertGreater(hi, 1.9)
        self.assertLess(hi - lo, 0.2)

    def test_paired_tests_detect_shift(self):
        d = np.random.default_rng(1).normal(0.5, 0.2, 60)
        self.assertLess(st.wilcoxon_signed_rank(d), 0.01)
        self.assertLess(st.paired_permutation_test(d, seed=1), 0.01)
        d0 = np.random.default_rng(2).normal(0.0, 1.0, 60)
        self.assertGreater(st.paired_permutation_test(d0, seed=2), 0.05)

    def test_benjamini_hochberg(self):
        p = [0.001, 0.01, 0.5, np.nan]
        rej, adj = st.benjamini_hochberg(p, alpha=0.05)
        self.assertTrue(rej[0] and rej[1])
        self.assertFalse(rej[2] or rej[3])
        self.assertTrue(np.isnan(adj[3]))
        self.assertTrue(np.all(np.diff(adj[:3][np.argsort(p[:3])]) >= 0))

    def test_effect_sizes(self):
        d = np.full(30, 1.0) + np.random.default_rng(3).normal(0, 0.1, 30)
        self.assertGreater(st.cohens_d_paired(d), 2.0)
        self.assertGreater(st.rank_biserial(d), 0.9)

    def test_spearman(self):
        x = np.arange(10.0)
        self.assertAlmostEqual(st.spearman(x, x ** 3), 1.0)
        self.assertAlmostEqual(st.spearman(x, -x), -1.0)

    def test_summary_keys(self):
        s = st.summarize_paired_effects(np.random.default_rng(4).normal(
            0.3, 0.1, 40), n_boot=200)
        for k in ("mean", "median", "std", "ci_low", "ci_high",
                  "prop_positive", "p_wilcoxon", "cohens_d"):
            self.assertIn(k, s)
        self.assertEqual(s["n"], 40)


if __name__ == "__main__":
    unittest.main()
