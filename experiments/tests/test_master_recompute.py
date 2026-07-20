from __future__ import annotations

import math
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace

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

    def test_dedicated_envs_use_conda_activation(self) -> None:
        old_names = master._CONDA_ENV_NAMES
        master._CONDA_ENV_NAMES = {"predictcsl-legacy"}
        try:
            cmd = master._py(
                "predictcsl-legacy",
                "experiments.run_all",
                "--models",
                "Sundial-Base-128M",
            )
        finally:
            master._CONDA_ENV_NAMES = old_names

        self.assertEqual(cmd[:2], ["bash", "-lc"])
        self.assertIn("conda activate predictcsl-legacy", cmd[2])
        self.assertIn("exec python -m experiments.run_all", cmd[2])
        self.assertNotIn("conda run", cmd)

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


class TiRexCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_backend = os.environ.pop("PREDICTCSL_TIREX_BACKEND", None)

    def tearDown(self) -> None:
        if self.old_backend is not None:
            os.environ["PREDICTCSL_TIREX_BACKEND"] = self.old_backend
        else:
            os.environ.pop("PREDICTCSL_TIREX_BACKEND", None)

    def test_load_tirex_normalizes_indexed_cuda_device(self) -> None:
        calls = []
        fake_tirex2 = ModuleType("tirex2")
        fake_tirex2_model = ModuleType("tirex2.model")
        fake_tirex2_component = ModuleType("tirex2.model.component")
        fake_flashrnn_slstm = ModuleType("tirex2.model.component.flashrnn_slstm")

        def fake_load_model(model_id, device):
            calls.append((model_id, device, fake_flashrnn_slstm._flashrnn_backend(device)))
            return object()

        fake_tirex2.load_model = fake_load_model
        module_names = {
            "tirex2": fake_tirex2,
            "tirex2.model": fake_tirex2_model,
            "tirex2.model.component": fake_tirex2_component,
            "tirex2.model.component.flashrnn_slstm": fake_flashrnn_slstm,
        }
        old_modules = {name: sys.modules.get(name) for name in module_names}
        sys.modules.update(module_names)
        fake_tirex2_component.flashrnn_slstm = fake_flashrnn_slstm
        try:
            build.load_tirex("NX-AI/TiRex-2", "cuda:0")
            build.load_tirex("NX-AI/TiRex-2", "cpu")
        finally:
            for name, old_module in old_modules.items():
                if old_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_module

        self.assertEqual(
            calls,
            [
                ("NX-AI/TiRex-2", "cuda", "triton_fused"),
                ("NX-AI/TiRex-2", "cpu", "vanilla"),
            ],
        )
        self.assertEqual(fake_flashrnn_slstm._flashrnn_backend("cpu"), "vanilla")

    def test_tirex_cuda_uses_triton_backend(self) -> None:
        from experiments.tirex_compat import tirex_backend_for_device

        self.assertEqual(tirex_backend_for_device("cuda"), "triton_fused")

    def test_tirex_backend_can_be_overridden(self) -> None:
        from experiments.tirex_compat import tirex_backend_for_device

        os.environ["PREDICTCSL_TIREX_BACKEND"] = "vanilla"
        self.assertEqual(tirex_backend_for_device("cuda"), "vanilla")

    def test_tirex_rejects_invalid_backend(self) -> None:
        from experiments.tirex_compat import tirex_backend_for_device

        os.environ["PREDICTCSL_TIREX_BACKEND"] = "nope"
        with self.assertRaisesRegex(ValueError, "PREDICTCSL_TIREX_BACKEND"):
            tirex_backend_for_device("cuda")

    def test_tirex_long_horizon_is_forecast_in_chunks(self) -> None:
        from experiments.tirex_compat import forecast_tirex_medians

        fake_tirex2 = ModuleType("tirex2")

        class FakeTimeseries:
            def __init__(self, target, past_covariates, future_covariates):
                self.target = target

        class FakeModel:
            future_len = 3
            context_len = 5

            def __init__(self):
                self.context_lengths = []

            def forecast(self, series, prediction_length, output_type):
                self.context_lengths.append(
                    ([item.target.shape[-1] for item in series], prediction_length)
                )
                value = float(len(self.context_lengths))
                return [
                    np.full((1, 9, prediction_length), value, dtype=np.float32)
                    for _ in series
                ]

        fake_tirex2.TimeseriesType = FakeTimeseries
        old_tirex2 = sys.modules.get("tirex2")
        sys.modules["tirex2"] = fake_tirex2
        try:
            model = FakeModel()
            result = forecast_tirex_medians(model, torch.zeros(2, 4), 7)
        finally:
            if old_tirex2 is None:
                sys.modules.pop("tirex2", None)
            else:
                sys.modules["tirex2"] = old_tirex2

        np.testing.assert_array_equal(
            result,
            np.array([[1, 1, 1, 2, 2, 2, 3], [1, 1, 1, 2, 2, 2, 3]], dtype=np.float32),
        )
        self.assertEqual(model.context_lengths, [([4, 4], 3), ([5, 5], 3), ([5, 5], 1)])


if __name__ == "__main__":
    unittest.main()
