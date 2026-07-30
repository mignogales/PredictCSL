from __future__ import annotations

import sys
import csv
import json
import os
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

from experiments import timesfm_gifteval
from experiments import run_all
from experiments import test_window_ablation_gifteval_v5 as ablation
from experiments.gifteval_inference_recipes import inference_recipe


class _FakeForecastConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeTimesFM:
    def __init__(self):
        self.model = SimpleNamespace(p=32, o=128)
        self.compiles = []
        self.inputs = []

    def compile(self, forecast_config):
        self.compiles.append(forecast_config)

    def forecast(self, horizon, inputs):
        self.inputs.append(inputs)
        # mean head plus q0.1..q0.9
        full = np.zeros((len(inputs), horizon, 10), dtype=np.float32)
        for q in range(9):
            full[:, :, q + 1] = q + 1
        return full[:, :, 0], full


class _FakeNonFiniteTimesFM(_FakeTimesFM):
    def forecast(self, horizon, inputs):
        mean, full = super().forecast(horizon, inputs)
        if len(inputs) > 1:
            full[1, 2, 5] = np.nan
        return mean, full


class TimesFMGiftEvalRecipeTest(unittest.TestCase):
    def tearDown(self) -> None:
        timesfm_gifteval._MODEL_CACHE.clear()

    def test_local_checkpoint_uses_from_pretrained(self) -> None:
        class FakeClass:
            loaded_from = None

            def load_checkpoint(self, path):
                raise AssertionError("abstract loader must not be called")

            @classmethod
            def from_pretrained(cls, path):
                cls.loaded_from = path
                return SimpleNamespace(source=path)

        fake_module = ModuleType("timesfm.timesfm_2p5.timesfm_2p5_torch")
        fake_module.TimesFM_2p5_200M_torch = FakeClass
        fake_package = ModuleType("timesfm.timesfm_2p5")
        fake_package.timesfm_2p5_torch = fake_module

        with mock.patch.dict(
                sys.modules,
                {
                    "timesfm.timesfm_2p5": fake_package,
                    "timesfm.timesfm_2p5.timesfm_2p5_torch": fake_module,
                }), mock.patch.dict(
                    os.environ, {"TIMESFM_2P5_CHECKPOINT": "/tmp/checkpoint"}):
            loaded = timesfm_gifteval.load_model("fake/model")

        self.assertEqual(FakeClass.loaded_from, "/tmp/checkpoint")
        self.assertEqual(loaded.source, "/tmp/checkpoint")

    def test_local_checkpoint_handles_hub_proxies_incompatibility(self) -> None:
        loaded = SimpleNamespace(calls=[])

        class FakeClass:
            WEIGHTS_FILENAME = "weights.safetensors"

            def __init__(self):
                self.model = SimpleNamespace(
                    load_checkpoint=lambda path, torch_compile: loaded.calls.append(
                        (path, torch_compile)))
                self.torch_compile = True

            def load_checkpoint(self, path):
                raise AssertionError("abstract loader must not be called")

            @classmethod
            def from_pretrained(cls, path):
                raise TypeError("unexpected keyword argument 'proxies'")

        fake_module = ModuleType("timesfm.timesfm_2p5.timesfm_2p5_torch")
        fake_module.TimesFM_2p5_200M_torch = FakeClass
        fake_package = ModuleType("timesfm.timesfm_2p5")
        fake_package.timesfm_2p5_torch = fake_module

        with tempfile.TemporaryDirectory() as checkpoint, mock.patch.dict(
                sys.modules,
                {
                    "timesfm.timesfm_2p5": fake_package,
                    "timesfm.timesfm_2p5.timesfm_2p5_torch": fake_module,
                }), mock.patch.dict(
                    os.environ, {"TIMESFM_2P5_CHECKPOINT": checkpoint}):
            timesfm_gifteval.load_model("fake/model")

        self.assertEqual(
            loaded.calls,
            [(os.path.join(checkpoint, "weights.safetensors"), True)],
        )

    def test_official_batch_compile_and_missing_value_handling(self) -> None:
        fake_timesfm = ModuleType("timesfm")
        fake_timesfm.configs = SimpleNamespace(ForecastConfig=_FakeForecastConfig)
        model = _FakeTimesFM()
        contexts = [
            np.arange(33, dtype=np.float32),
            np.array([1.0, np.nan, 3.0], dtype=np.float32),
            np.array([np.nan, np.nan], dtype=np.float32),
        ]

        with mock.patch.dict(sys.modules, {"timesfm": fake_timesfm}):
            result = timesfm_gifteval.forecast_quantiles(
                model, contexts, prediction_length=4, batch_size=2)

        self.assertEqual(result.shape, (3, 9, 4))
        np.testing.assert_allclose(result[:, 4, :], 5.0)
        self.assertEqual([cfg.max_context for cfg in model.compiles], [64, 32])
        self.assertEqual([cfg.max_horizon for cfg in model.compiles], [128, 128])
        self.assertEqual(
            [cfg.per_core_batch_size for cfg in model.compiles], [2, 1])
        self.assertTrue(np.isnan(model.inputs[0][1][1]))
        np.testing.assert_array_equal(model.inputs[1][0], np.zeros(2, dtype=np.float32))

    def test_compile_horizon_override(self) -> None:
        fake_timesfm = ModuleType("timesfm")
        fake_timesfm.configs = SimpleNamespace(ForecastConfig=_FakeForecastConfig)
        model = _FakeTimesFM()

        with mock.patch.dict(sys.modules, {"timesfm": fake_timesfm}):
            timesfm_gifteval.forecast_quantiles(
                model, [np.arange(15360, dtype=np.float32)],
                prediction_length=48, batch_size=1, max_horizon=128)

        self.assertEqual(model.compiles[0].max_context, 15360)
        self.assertEqual(model.compiles[0].max_horizon, 128)

    def test_nonfinite_forecast_reports_batch_and_rows(self) -> None:
        fake_timesfm = ModuleType("timesfm")
        fake_timesfm.configs = SimpleNamespace(ForecastConfig=_FakeForecastConfig)
        model = _FakeNonFiniteTimesFM()
        contexts = [np.arange(8, dtype=np.float32) for _ in range(3)]

        with mock.patch.dict(sys.modules, {"timesfm": fake_timesfm}):
            with self.assertRaisesRegex(
                    FloatingPointError,
                    r"batch=0 .*call_rows=\[1\] .*batch_rows=\[1\] "
                    r".*source_rows=\[501\].*compiled_max_horizon=128"):
                timesfm_gifteval.forecast_quantiles(
                    model, contexts, prediction_length=4, batch_size=2,
                    max_horizon=128, forecast_row_indices=[500, 501, 502])

    def test_compile_horizon_must_cover_requested_horizon(self) -> None:
        fake_timesfm = ModuleType("timesfm")
        fake_timesfm.configs = SimpleNamespace(ForecastConfig=_FakeForecastConfig)

        with mock.patch.dict(sys.modules, {"timesfm": fake_timesfm}):
            with self.assertRaisesRegex(ValueError, "must cover"):
                timesfm_gifteval.forecast_quantiles(
                    _FakeTimesFM(), [np.arange(8, dtype=np.float32)],
                    prediction_length=48, max_horizon=32)

    def test_old_timesfm_cache_is_rejected(self) -> None:
        for family in (
                "timesfm", "chronos2", "chronos_bolt", "moirai", "patchtst_fm",
                "sundial", "tirex"):
            with self.subTest(family=family):
                self.assertFalse(
                    ablation._inference_recipe_current({}, family))
                self.assertTrue(ablation._inference_recipe_current(
                    {"_inference_recipe": inference_recipe(family)}, family))

    def test_real_mase_standin_is_not_a_valid_forecast_cache(self) -> None:
        metrics = {
            "mae": 1.0,
            "mse": 1.0,
            "rmse": 1.0,
            "mase": 1.0,
            "mase_gluonts": 1.0,
            "mase_gluonts_real": 1.0,
            "smape": 1.0,
            "crps": 1.0,
            "_mase_gluonts_real_standin": True,
            "_inference_recipe": inference_recipe("chronos2"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            old_root = ablation.CACHE_ROOT
            ablation.CACHE_ROOT = tmp
            try:
                ablation._save_result(
                    "Example", "Chronos2-Small", "short", 32, metrics)
                self.assertFalse(ablation._result_cached_for_family(
                    "Example", "Chronos2-Small", "short", 32, "chronos2"))
            finally:
                ablation.CACHE_ROOT = old_root

    def test_stage3_done_marker_rejects_then_accepts_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_root = run_all.ABLATION_GENERAL
            run_all.ABLATION_GENERAL = tmp
            try:
                compare_dir = os.path.join(
                    tmp, "models", "TimesFM2.5-200M",
                    "compare_real_vs_predicted")
                os.makedirs(compare_dir)
                summary = os.path.join(compare_dir, "compare_summary.csv")
                row = {
                    "dataset_display": "Example",
                    "term": "short",
                    "inference_recipe": "old",
                }
                with open(summary, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)

                one_cell = [("example", "short", "Example", False)]
                with mock.patch.object(
                        run_all.datasets_config, "datasets_to_run",
                        return_value=one_cell):
                    done, _ = run_all._done_stage_3(
                        "timesfm", "TimesFM2.5-200M")
                self.assertFalse(done)

                row["inference_recipe"] = inference_recipe("timesfm")
                with open(summary, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                metrics_dir = os.path.join(
                    tmp, "datasets", "Example", "TimesFM2.5-200M",
                    "tshort", "wfull_native")
                os.makedirs(metrics_dir)
                with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
                    json.dump({
                        "_inference_recipe": inference_recipe("timesfm"),
                    }, f)

                with mock.patch.object(
                        run_all.datasets_config, "datasets_to_run",
                        return_value=one_cell):
                    done, _ = run_all._done_stage_3(
                        "timesfm", "TimesFM2.5-200M")
                self.assertTrue(done)
            finally:
                run_all.ABLATION_GENERAL = old_root


if __name__ == "__main__":
    unittest.main()
