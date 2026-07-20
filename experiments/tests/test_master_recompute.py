from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

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
        self.assertEqual(build.window_grid_for_family("tirex")[-1], 8192)

    def test_tirex2_uses_dedicated_env(self) -> None:
        self.assertEqual(master.FAMILY_ENV["tirex"], "predictcsl-tirex")
        self.assertIn("predictcsl-tirex", master.ENVS_WITHOUT_MAMBA)

    def test_master_stage1_build_args(self) -> None:
        args = SimpleNamespace(
            stage1_batch_size=8,
            stage1_shard_size=50,
            stage1_windows=[32, 64, 128],
            stage1_n_series=1000,
        )
        self.assertEqual(
            master._stage1_build_args(args),
            [
                "--batch-size", "8",
                "--shard-size", "50",
                "--windows", "32", "64", "128",
                "--n-series", "1000",
            ],
        )

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


class PatchTSTFMCompatibilityTest(unittest.TestCase):
    def test_new_past_values_forward_signature(self) -> None:
        class NewGraniteAPI(torch.nn.Module):
            config = SimpleNamespace(context_length=8)

            def forward(self, past_values, prediction_length):
                self.seen = past_values
                forecast = torch.zeros(
                    past_values.shape[0], 99, prediction_length, 1,
                    device=past_values.device,
                )
                forecast[:, build.PATCHTST_FM_MEDIAN_QUANTILE_IDX, :, 0] = 2.0
                return (forecast,)

        model = NewGraniteAPI()
        result = build.predict_patchtst_fm(
            model, torch.ones(2, 5, 1), horizon=3, device="cpu")

        self.assertEqual(tuple(model.seen.shape), (2, 8))
        self.assertEqual(tuple(result.shape), (2, 3))
        self.assertTrue(torch.equal(result, torch.full((2, 3), 2.0)))

    def test_legacy_inputs_forward_signature(self) -> None:
        class LegacyAPI(torch.nn.Module):
            config = SimpleNamespace(context_length=8)

            def forward(self, inputs, prediction_length):
                forecast = torch.ones(
                    inputs.shape[0], 1, prediction_length,
                    device=inputs.device,
                )
                return (forecast,)

        result = build.predict_patchtst_fm(
            LegacyAPI(), torch.ones(1, 8, 1), horizon=2, device="cpu")

        self.assertTrue(torch.equal(result, torch.ones(1, 2)))


if __name__ == "__main__":
    unittest.main()
