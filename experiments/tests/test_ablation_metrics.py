from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.gifteval_metric_version import METRIC_SUITE_VER
from experiments import test_window_ablation_gifteval_v5 as ablation


class AblationMetricTest(unittest.TestCase):
    def test_gluonts_mase_masks_zero_seasonal_denominators(self) -> None:
        abs_err = torch.tensor([[0.0, 2.0], [1.0, 1.0]])
        valid = torch.ones_like(abs_err, dtype=torch.bool)

        value = ablation.compute_mase_gluonts(
            abs_err, valid, np.array([0.0, 1.0]))

        self.assertEqual(value, 1.0)

    def test_mape_excludes_zero_targets(self) -> None:
        forecast = ablation.ForecastResult(
            median=torch.tensor([[5.0, 2.0]]))
        targets = torch.tensor([[0.0, 1.0]])

        metrics = ablation.compute_all_metrics(
            forecast, targets, seasonal_errors=np.array([1.0]))

        self.assertEqual(metrics["mape"], 1.0)
        self.assertEqual(metrics["_metric_suite_ver"], METRIC_SUITE_VER)

    def test_quantile_crps_does_not_divide_by_quantile_count_twice(self) -> None:
        quantiles = torch.zeros((1, 3, 2))
        targets = torch.ones((1, 2))

        value = ablation.crps_quantile_loss(
            quantiles, [0.1, 0.5, 0.9], targets)

        self.assertAlmostEqual(value, 1.0)

    def test_quantile_crps_ignores_missing_target_points(self) -> None:
        quantiles = torch.zeros((1, 3, 2))
        targets = torch.tensor([[1.0, 0.0]])
        valid = torch.tensor([[True, False]])

        value = ablation.crps_quantile_loss(
            quantiles, [0.1, 0.5, 0.9], targets, valid)

        self.assertAlmostEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()

