from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from experiments import build_context_length_dataset as build
from experiments import master_run_all as master
from experiments import predict_context_length as predictor
from experiments.compare_window_strategies_gifteval import _geomean


class MasterRecomputeConfigTest(unittest.TestCase):
    def test_requested_predictor_matrix(self) -> None:
        self.assertEqual(
            [v.name for v in master.VARIANTS],
            ["cheap", "cheap_cls", "mamba", "mamba_cls"],
        )
        self.assertTrue(all(v.skip_stages == ["1"] for v in master.VARIANTS))

    def test_model_aware_window_grids(self) -> None:
        self.assertEqual(build.window_grid_for_family("timesfm")[-2:],
                         [12288, 15360])
        self.assertEqual(build.window_grid_for_family("chronos2")[-1], 8192)
        self.assertEqual(build.window_grid_for_family("chronos_bolt")[-1], 2048)
        self.assertEqual(build.window_grid_for_family("sundial")[-1], 2560)

    def test_sanity_check_geomean_rules(self) -> None:
        self.assertAlmostEqual(_geomean(np.array([1.0, 4.0, np.nan])), 2.0)
        self.assertEqual(_geomean(np.array([0.0, 4.0])), 0.0)
        self.assertTrue(math.isnan(_geomean(np.array([-1.0, 4.0]))))


class SoftClassificationLossTest(unittest.TestCase):
    def test_top3_rank_weights_and_invalid_class_mask(self) -> None:
        old_objective = predictor.TRAINING_OBJECTIVE
        predictor.TRAINING_OBJECTIVE = "classification"
        try:
            logits = torch.tensor([[3.0, 2.0, 1.0, 9.0]], requires_grad=True)
            # Lower raw error is more accurate. Class 3 is unavailable.
            target = torch.tensor([[0.1, 0.2, 0.3, float("nan")]])
            recon = torch.zeros(1, 1, 1)
            mask = torch.zeros(1, 1, dtype=torch.bool)
            loss, task_loss, _ = predictor.compute_dual_loss(
                logits, recon, recon, mask, target, 1.0, 0.0)

            expected_soft = torch.tensor([[1.0, 0.5, 0.25]])
            expected = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, :3], expected_soft)
            self.assertTrue(torch.allclose(task_loss, expected))
            self.assertTrue(torch.allclose(loss, expected))

            loss.backward()
            self.assertEqual(float(logits.grad[0, 3]), 0.0)
        finally:
            predictor.TRAINING_OBJECTIVE = old_objective


if __name__ == "__main__":
    unittest.main()
