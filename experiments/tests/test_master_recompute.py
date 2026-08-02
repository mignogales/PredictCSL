from __future__ import annotations

import math
import json
import os
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
import torch

from experiments import build_context_length_dataset as build
from experiments import evaluate_instance_windows as instance_eval
from experiments import master_run_all as master
from experiments import models_config
from experiments import predict_context_length as predictor
from experiments.compare_window_strategies_gifteval import _geomean
from experiments.gifteval_mase import gluonts_leaderboard_mase
from experiments.test_window_ablation_gifteval_v5 import (
    ForecastResult, compute_per_sample_metrics)


class MasterRecomputeConfigTest(unittest.TestCase):
    def test_all_chronos2_variants_share_independent_recipe(self) -> None:
        chronos2 = {
            spec.display: spec.family
            for spec in models_config.CATALOG
            if spec.display in {
                "Chronos2-Small", "Chronos2-Base", "Chronos2-Synth",
            }
        }
        self.assertEqual(
            chronos2,
            {
                "Chronos2-Small": "chronos2",
                "Chronos2-Base": "chronos2",
                "Chronos2-Synth": "chronos2",
            },
        )
        from experiments.gifteval_inference_recipes import inference_recipe
        self.assertEqual(
            inference_recipe("chronos2"),
            "chronos2_univariate_no_cross_learning_v1",
        )

    def test_requested_predictor_matrix(self) -> None:
        self.assertEqual(
            [v.name for v in master.VARIANTS],
            ["cheap", "cheap_cls", "mamba", "mamba_cls"],
        )
        self.assertTrue(all(v.skip_stages == ["1"] for v in master.VARIANTS))
        self.assertEqual(
            [v.ablation_tree for v in master.VARIANTS],
            [
                "general_v3",
                "general_v3_classification",
                "general_v4",
                "general_v4_classification",
            ],
        )

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

    def test_patchtst_uses_leaderboard_compatible_env(self) -> None:
        self.assertEqual(
            master.FAMILY_ENV["patchtst_fm"], "predictcsl-patchtst")
        self.assertNotIn("predictcsl-patchtst", master.ENVS_WITHOUT_MAMBA)
        self.assertIn(
            "mamba-ssm==2.2.5",
            master.ENV_PREFLIGHTS["predictcsl-patchtst"],
        )
        self.assertEqual(
            master._resolve_groups(["PatchTST-FM-R1"]),
            {"predictcsl-patchtst": ["PatchTST-FM-R1"]},
        )

    def test_toto_uses_dedicated_env(self) -> None:
        self.assertEqual(master.FAMILY_ENV["toto"], "predictcsl-toto")
        self.assertEqual(
            master._resolve_groups(["Toto-2.0-313m"]),
            {"predictcsl-toto": ["Toto-2.0-313m"]},
        )

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

    def test_dedicated_variant_args_are_inside_activated_shell(self) -> None:
        old_names = master._CONDA_ENV_NAMES
        master._CONDA_ENV_NAMES = {"predictcsl-toto"}
        try:
            cmd = master._variant_cmd(
                "predictcsl-toto",
                master.VARIANTS[0],
                ["Toto-2.0-313m"],
                [],
                ["--force", "3"],
            )
        finally:
            master._CONDA_ENV_NAMES = old_names

        self.assertEqual(cmd[:2], ["bash", "-lc"])
        self.assertEqual(len(cmd), 3)
        self.assertIn("conda activate predictcsl-toto", cmd[2])
        self.assertIn("experiments.run_all_v3", cmd[2])
        self.assertIn("--models Toto-2.0-313m", cmd[2])
        self.assertIn("--force 3", cmd[2])

    def test_verbose_ablation_is_forwarded_to_variant(self) -> None:
        cmd = master._variant_cmd(
            None,
            master.VARIANTS[0],
            ["Chronos2-Small"],
            ["--verbose-ablation"],
            [],
        )
        self.assertIn("--verbose-ablation", cmd)

    def test_variant_can_be_dispatched_for_one_stage(self) -> None:
        cmd = master._variant_cmd(
            None,
            master.VARIANTS[0],
            ["Chronos2-Small"],
            [],
            [],
            only_stage="3",
        )
        self.assertIn("--only-stages", cmd)
        self.assertIn("3", cmd)
        self.assertNotIn("--skip-stages", cmd)

    def test_forecast_precompute_needs_no_predictor(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"PREDICTCSL_ABLATION_ROOT": "/tmp/predictcsl-ablation"},
        ):
            cmd = master._forecast_precompute_cmd(
                None, ["Chronos2-Small"], test=True)
        self.assertIn("--forecast-only", cmd)
        self.assertIn("--cache-root", cmd)
        self.assertIn("/tmp/predictcsl-ablation/general", cmd)
        self.assertIn("--test-datasets", cmd)
        self.assertNotIn("--predictor-dir", cmd)

    def test_chronos2_precompute_reuses_one_dataset_cache(self) -> None:
        groups = master._precompute_model_groups([
            "Chronos2-Small",
            "Moirai2-Small",
            "Chronos2-Synth",
            "Chronos2-Base",
            "TimesFM2.5-200M",
        ])
        self.assertEqual(groups, [
            ["Chronos2-Small", "Chronos2-Synth", "Chronos2-Base"],
            ["Moirai2-Small"],
            ["TimesFM2.5-200M"],
        ])

    def test_dedicated_stage1_args_are_inside_activated_shell(self) -> None:
        old_names = master._CONDA_ENV_NAMES
        master._CONDA_ENV_NAMES = {"predictcsl-legacy"}
        try:
            cmd = master._stage1_cmd(
                "predictcsl-legacy",
                ["Sundial-Base-128M"],
                ["-v"],
                [],
                ["--n-series", "200"],
            )
        finally:
            master._CONDA_ENV_NAMES = old_names

        self.assertEqual(cmd[:2], ["bash", "-lc"])
        self.assertEqual(len(cmd), 3)
        self.assertIn("conda activate predictcsl-legacy", cmd[2])
        self.assertIn("--models Sundial-Base-128M", cmd[2])
        self.assertIn("--build-args --n-series 200", cmd[2])

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


class PerInstanceWindowEvaluationTest(unittest.TestCase):
    def test_discovery_includes_complete_gifteval_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            compare_dir = os.path.join(
                root, "general_v3", "models", "TimesFM2.5-200M",
                "compare_real_vs_predicted",
            )
            os.makedirs(compare_dir)
            for dataset in ("Solar-W", "CarParts", "M4-Yearly"):
                path = os.path.join(
                    compare_dir,
                    f"compare_{dataset}_tshort_TimesFM2.5-200M.npz",
                )
                with open(path, "wb"):
                    pass

            cells = instance_eval.discover_cells(root, None)

        self.assertEqual(
            [(c.dataset, c.term) for c in cells],
            [("CarParts", "short"), ("M4-Yearly", "short"),
             ("Solar-W", "short")],
        )

    def test_predictor_choice_is_per_row_and_masks_unavailable_windows(self) -> None:
        windows = np.array([32, 64])
        errors = np.array([
            [1.0, 0.5],
            [2.0, np.nan],
            [np.nan, np.nan],
        ])
        scores = np.array([
            [0.0, 1.0],  # row 0 chooses 32
            [9.0, 0.0],  # row 1 wants 64, but it is unavailable
            [0.0, 1.0],  # no grid window -> full-native
        ])
        native = np.array([3.0, 4.0, 5.0])

        selected, selected_w, fallback = instance_eval._choose_scores(
            scores, errors, windows, native)

        np.testing.assert_allclose(selected, [1.0, 2.0, 5.0])
        np.testing.assert_array_equal(selected_w, [32, 32, -1])
        np.testing.assert_array_equal(fallback, [False, False, True])

    def test_fixed_window_is_capped_per_instance(self) -> None:
        windows = np.array([32, 64, 128])
        errors = np.array([
            [3.0, 2.0, 1.0],
            [4.0, 3.0, np.nan],
        ])
        native = np.array([9.0, 9.0])

        selected, selected_w, fallback = instance_eval._choose_capped_fixed(
            128, errors, windows, native)

        np.testing.assert_allclose(selected, [1.0, 3.0])
        np.testing.assert_array_equal(selected_w, [128, 64])
        self.assertFalse(fallback.any())

    def test_per_instance_mase_ignores_missing_horizon_points(self) -> None:
        forecast = ForecastResult(median=torch.tensor([
            [1.0, 5.0],
            [2.0, 3.0],
        ]))
        targets = torch.tensor([
            [3.0, float("nan")],
            [4.0, 7.0],
        ])
        metrics = compute_per_sample_metrics(
            forecast, targets, seasonal_errors=np.array([2.0, 2.0]))

        np.testing.assert_allclose(metrics["mae"], [2.0, 3.0])
        np.testing.assert_allclose(metrics["mase_gluonts"], [1.0, 1.5])


class GluonTSMaseVectorizedTest(unittest.TestCase):
    def test_global_point_aggregation_with_missing_labels(self) -> None:
        contexts = [np.array([0.0, 2.0, 4.0]),
                    np.array([0.0, 10.0, 20.0])]
        labels = np.array([[2.0, 4.0], [10.0, np.nan]])
        median = np.zeros((2, 2))

        value = gluonts_leaderboard_mase(
            median, [None, None], contexts, labels, "D")

        # Scaled valid point errors are [1, 2, 1], so GluonTS axis=None is 4/3.
        self.assertAlmostEqual(value, 4.0 / 3.0)

    def test_even_sample_forecast_uses_gluonts_upper_middle_index(self) -> None:
        samples = np.array([[[0.0], [1.0], [2.0], [100.0]]])
        value = gluonts_leaderboard_mase(
            np.array([[1.0]]), [None], [np.array([0.0, 1.0])],
            np.array([[2.0]]), "D", samples=samples)

        # GluonTS uses sorted_samples[round((S-1)*.5)] = index 2, prediction 2.
        self.assertEqual(value, 0.0)


class LeaderboardCheckpointTest(unittest.TestCase):
    def test_checkpoint_persists_stage_and_official_aggregation(self) -> None:
        from experiments import test_window_ablation_gifteval_v5 as ablation

        datasets = [
            ("dataset_one", "short", "Dataset-One", False),
            ("dataset_two", "long", "Dataset-Two", False),
        ]
        models = [("example/model", "example_family", "Example-Model")]
        naive = {
            ("dataset_one", "short"): 1.0,
            ("dataset_two", "long"): 2.0,
        }
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(ablation, "CACHE_ROOT", root), \
                mock.patch.object(
                    ablation, "published_naive_record",
                    side_effect=lambda name, term: {
                        "mase_gluonts_real": naive[(name, term)]
                    },
                ):
            for _name, term, display, _univariate in datasets:
                value = 2.0 if display == "Dataset-One" else 8.0
                ablation._save_result(
                    display, "Example-Model", term,
                    ablation.FULL_NATIVE_WINDOW,
                    {"mae": 1.0, "mse": 1.0, "rmse": 1.0,
                     "mase_gluonts_real": value,
                     "_mase_gluonts_real_standin": False},
                )
            path = ablation._write_leaderboard_parity_summary(
                models, datasets, root,
                checkpoint_stage="stage_3_gift_eval_ablation")
            with open(path) as f:
                payload = json.load(f)

        self.assertEqual(
            payload["last_checkpoint_stage"], "stage_3_gift_eval_ablation")
        record = payload["models"]["Example-Model"]
        self.assertTrue(record["complete"])
        self.assertAlmostEqual(record["geomean_normalized_mase"], math.sqrt(8.0))


class Chronos2IndependenceTest(unittest.TestCase):
    class FakePipeline:
        quantiles = [0.1, 0.5, 0.9]

        def __init__(self) -> None:
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            inputs = kwargs["inputs"]
            horizon = kwargs["prediction_length"]
            return torch.zeros(
                (inputs.shape[0], len(self.quantiles), horizon),
                dtype=torch.float32,
            )

    def test_stage1_disables_cross_learning(self) -> None:
        pipeline = self.FakePipeline()
        out = build.predict_chronos2(
            pipeline, torch.zeros((2, 32, 1)), horizon=4, device="cpu")
        self.assertEqual(tuple(out.shape), (2, 4))
        self.assertIs(pipeline.calls[0]["cross_learning"], False)

    def test_stage1_returns_median_on_requested_device(self) -> None:
        pipeline = self.FakePipeline()
        out = build.predict_chronos2(
            pipeline, torch.zeros((2, 32, 1)), horizon=4, device="meta")
        self.assertEqual(out.device.type, "meta")

    def test_gifteval_ablation_disables_cross_learning(self) -> None:
        from experiments import test_window_ablation_gifteval_v5 as ablation

        pipeline = self.FakePipeline()
        batches = [{
            "x": torch.zeros((2, 32, 1)),
            "y": torch.zeros((2, 4, 1)),
        }]
        forecast, _targets = ablation.predict_chronos2(
            pipeline, batches, horizon=4, device="cpu", batch_size=2)
        self.assertEqual(tuple(forecast.median.shape), (2, 4))
        self.assertIs(pipeline.calls[0]["cross_learning"], False)


class AblationDynamicBatchTest(unittest.TestCase):
    def test_deterministic_family_runs_once_without_autotune_probe(self) -> None:
        from experiments import test_window_ablation_gifteval_v5 as ablation

        calls = []

        def fake_forecast(
            _family, _handle, _model_id, batches, _width, horizon,
            _device, batch_size, **_kwargs,
        ):
            calls.append(batch_size)
            n = sum(batch["x"].shape[0] for batch in batches)
            zeros = torch.zeros((n, horizon))
            return ForecastResult(median=zeros), zeros

        batches = [{
            "x": torch.zeros((20, 1024, 1)),
            "y": torch.zeros((20, 4, 1)),
        }]
        with mock.patch.object(ablation, "_forecast_cell", fake_forecast):
            _fr, _targets, size = ablation._forecast_cell_dynamic(
                "flowstate", object(), "example/model", batches,
                1024, 4, "cuda", 2)

        self.assertEqual(calls, [2])
        self.assertEqual(size, 2)

    def test_sampling_family_does_not_probe(self) -> None:
        from experiments import test_window_ablation_gifteval_v5 as ablation

        calls = []

        def fake_forecast(
            _family, _handle, _model_id, batches, _width, horizon,
            _device, batch_size, **_kwargs,
        ):
            calls.append(batch_size)
            n = sum(batch["x"].shape[0] for batch in batches)
            zeros = torch.zeros((n, horizon))
            return ForecastResult(median=zeros), zeros

        batches = [{
            "x": torch.zeros((20, 1024, 1)),
            "y": torch.zeros((20, 4, 1)),
        }]
        with mock.patch.object(ablation, "_forecast_cell", fake_forecast):
            ablation._forecast_cell_dynamic(
                "sundial", object(), "example/model", batches,
                1024, 4, "cuda", 2)

        self.assertEqual(calls, [2])

    def test_short_context_uses_ceiling_scaled_batch(self) -> None:
        from experiments import test_window_ablation_gifteval_v5 as ablation

        calls = []

        def fake_forecast(
            _family, _handle, _model_id, batches, _width, horizon,
            _device, batch_size, **_kwargs,
        ):
            calls.append(batch_size)
            n = sum(batch["x"].shape[0] for batch in batches)
            zeros = torch.zeros((n, horizon))
            return ForecastResult(median=zeros), zeros

        batches = [{
            "x": torch.zeros((192, 384, 1)),
            "y": torch.zeros((192, 4, 1)),
        }]
        with mock.patch.object(ablation, "_forecast_cell", fake_forecast):
            _fr, _targets, size = ablation._forecast_cell_dynamic(
                "timesfm", object(), "example/model", batches,
                384, 4, "cuda", 32)

        self.assertEqual(calls, [96])
        self.assertEqual(size, 96)

    def test_oom_halves_batch_and_retries_without_caching(self) -> None:
        from experiments import test_window_ablation_gifteval_v5 as ablation

        calls = []

        def fake_forecast(
            _family, _handle, _model_id, batches, _width, horizon,
            _device, batch_size, **_kwargs,
        ):
            calls.append(batch_size)
            if batch_size > 4:
                raise torch.cuda.OutOfMemoryError("synthetic OOM")
            n = sum(batch["x"].shape[0] for batch in batches)
            zeros = torch.zeros((n, horizon))
            return ForecastResult(median=zeros), zeros

        batches = [{
            "x": torch.zeros((20, 256, 1)),
            "y": torch.zeros((20, 4, 1)),
        }]
        with mock.patch.object(ablation, "_forecast_cell", fake_forecast), \
                mock.patch.object(ablation, "_clear_accelerator_cache"):
            _fr, _targets, size = ablation._forecast_cell_dynamic(
                "flowstate", object(), "example/model", batches,
                256, 4, "cuda", 2)

        self.assertEqual(calls, [8, 4])
        self.assertEqual(size, 4)

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
    def test_patchtst_recipe_invalidates_padding_safe_v2_caches(self) -> None:
        from experiments.gifteval_inference_recipes import inference_recipe

        self.assertEqual(
            inference_recipe("patchtst_fm"),
            "official_patchtst_fm_wrapper_v3",
        )

    def test_legacy_official_api_keeps_inputs_on_cpu(self) -> None:
        class LegacyGraniteAPI(torch.nn.Module):
            def forward(self, inputs, prediction_length, quantile_levels):
                self.seen = inputs
                forecast = torch.zeros(
                    len(inputs), len(quantile_levels), prediction_length)
                forecast[:, build.PATCHTST_FM_MEDIAN_QUANTILE_IDX, :] = 3.0
                return SimpleNamespace(quantile_predictions=forecast)

        model = LegacyGraniteAPI()
        result = build.predict_patchtst_fm(
            model, torch.ones(2, 5, 1), horizon=3, device="cpu")

        self.assertTrue(all(row.device.type == "cpu" for row in model.seen))
        self.assertTrue(torch.equal(result, torch.full((2, 3), 3.0)))

    def test_official_quantile_head_and_list_input(self) -> None:
        class NewGraniteAPI(torch.nn.Module):
            def forward(self, past_values, prediction_length, quantile_levels):
                self.seen = past_values
                forecast = torch.zeros(
                    len(past_values), len(quantile_levels), prediction_length, 1,
                )
                forecast[:, build.PATCHTST_FM_MEDIAN_QUANTILE_IDX, :, 0] = 2.0
                return SimpleNamespace(quantile_outputs=forecast)

        model = NewGraniteAPI()
        result = build.predict_patchtst_fm(
            model, torch.ones(2, 5, 1), horizon=3, device="cpu")

        self.assertIsInstance(model.seen, list)
        self.assertEqual([tuple(x.shape) for x in model.seen], [(5,), (5,)])
        self.assertEqual(tuple(result.shape), (2, 3))
        self.assertTrue(torch.equal(result, torch.full((2, 3), 2.0)))
        self.assertFalse(result.requires_grad)

    def test_mean_imputes_missing_values(self) -> None:
        class API(torch.nn.Module):
            def forward(self, past_values, prediction_length, quantile_levels):
                self.seen = past_values
                forecast = torch.ones(
                    len(past_values), len(quantile_levels), prediction_length, 1)
                return SimpleNamespace(quantile_outputs=forecast)

        model = API()
        x = torch.tensor([[[1.0], [float("nan")], [3.0]]])
        result = build.predict_patchtst_fm(
            model, x, horizon=2, device="cpu")

        self.assertTrue(torch.equal(model.seen[0], torch.tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(result, torch.ones(1, 2)))


class TiRexCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_backend = os.environ.pop("PREDICTCSL_TIREX_BACKEND", None)

    def tearDown(self) -> None:
        if self.old_backend is not None:
            os.environ["PREDICTCSL_TIREX_BACKEND"] = self.old_backend
        else:
            os.environ.pop("PREDICTCSL_TIREX_BACKEND", None)

    def test_catalog_uses_published_gifteval_zero_shot_checkpoint(self) -> None:
        tirex = next(spec for spec in models_config.CATALOG
                     if spec.family == "tirex")
        self.assertEqual(tirex.model_id, "NX-AI/TiRex-2-gifteval-zs")

    def test_checkpoint_preflight_checks_only_the_small_config(self) -> None:
        from experiments.tirex_compat import require_tirex_checkpoint_access

        with mock.patch(
                "huggingface_hub.hf_hub_download",
                return_value="/cache/model-config.yaml") as download:
            path = require_tirex_checkpoint_access()

        self.assertEqual(path, "/cache/model-config.yaml")
        download.assert_called_once_with(
            repo_id="NX-AI/TiRex-2-gifteval-zs",
            filename="model-config.yaml",
            repo_type="model",
        )

    def test_checkpoint_preflight_explains_gated_access(self) -> None:
        from huggingface_hub.errors import GatedRepoError
        from experiments.tirex_compat import (
            TirexCheckpointAccessError,
            require_tirex_checkpoint_access,
        )

        with mock.patch(
                "huggingface_hub.hf_hub_download",
                side_effect=GatedRepoError("forbidden")):
            with self.assertRaises(TirexCheckpointAccessError) as raised:
                require_tirex_checkpoint_access()

        message = str(raised.exception)
        self.assertIn("TiREX checkpoint access denied", message)
        self.assertIn("huggingface-cli whoami", message)
        self.assertIn("HF_TOKEN", message)
        self.assertIn("Do not substitute NX-AI/TiRex-2", message)

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
                ("NX-AI/TiRex-2", "cuda", "vanilla"),
                ("NX-AI/TiRex-2", "cpu", "vanilla"),
            ],
        )
        self.assertEqual(fake_flashrnn_slstm._flashrnn_backend("cpu"), "vanilla")

    def test_tirex_cuda_uses_vanilla_backend(self) -> None:
        from experiments.tirex_compat import tirex_backend_for_device

        self.assertEqual(tirex_backend_for_device("cuda"), "vanilla")

    def test_tirex_backend_can_be_overridden(self) -> None:
        from experiments.tirex_compat import tirex_backend_for_device

        os.environ["PREDICTCSL_TIREX_BACKEND"] = "cuda_fused"
        self.assertEqual(tirex_backend_for_device("cuda"), "cuda_fused")

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

    def test_stage1_dynamic_batch_scales_with_context(self) -> None:
        self.assertEqual(build.batch_size_for_context(
            8, 8192, 500, True, 8192, 512), 8)
        self.assertEqual(build.batch_size_for_context(
            8, 2048, 500, True, 8192, 512), 32)
        self.assertEqual(build.batch_size_for_context(
            8, 32, 100, True, 8192, 512), 100)
        self.assertEqual(build.batch_size_for_context(
            8, 32, 100, False, 8192, 512), 8)

    def test_stage1_tirex_dynamic_batch_backs_off_after_oom(self) -> None:
        seen = []

        def fake_predict(_runner, x, horizon, _device):
            seen.append(len(x))
            if len(x) > 4:
                raise RuntimeError("CUDA out of memory")
            return torch.zeros(len(x), horizon)

        tuned = {}
        x = torch.zeros(10, 32, 1)
        with mock.patch.object(build, "_is_cuda", return_value=True), \
             mock.patch.object(build, "predict_tirex", side_effect=fake_predict), \
             mock.patch.object(torch.cuda, "empty_cache"):
            result = build._forecast_uniform(
                "tirex", object(), "unused", x, 32, 2, 8, "cuda:0",
                dynamic_batching=True, max_batch_size=16,
                tuned_batch_sizes=tuned)

        self.assertEqual(tuple(result.shape), (10, 2))
        self.assertEqual(seen[:2], [8, 4])
        self.assertEqual(tuned, {})


if __name__ == "__main__":
    unittest.main()
