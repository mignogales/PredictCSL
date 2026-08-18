import unittest

import numpy as np

from experiments.evaluate_predictor_full_stats_gifteval import (
    _aggregate_mase,
    _per_sample,
    _summarize,
)


class TestPredictorFullStatsEvaluation(unittest.TestCase):
    def test_mase_uses_forecast_count_weighting(self):
        pred = np.array([[2.0, 4.0], [4.0, np.nan]])
        target = np.zeros_like(pred)
        metrics = _per_sample(pred, target, np.array([2.0, 1.0]))
        # Per-row MASE is [1.5, 4], weighted by valid counts [2, 1].
        self.assertAlmostEqual(_aggregate_mase(metrics), 7.0 / 3.0)

    def test_summary_relative_change_sign(self):
        rows = [{
            "selected_stats_normalized_mase": 1.0,
            "full_stats_normalized_mase": 0.9,
            "selected_stats_mase": 2.0,
            "full_stats_mase": 1.8,
        }]
        report = _summarize(rows, n_bootstrap=20, seed=1)
        self.assertAlmostEqual(report["full_stats_relative_change_pct"], -10.0)
        self.assertEqual(report["full_stats_better_cells"], 1)


if __name__ == "__main__":
    unittest.main()
