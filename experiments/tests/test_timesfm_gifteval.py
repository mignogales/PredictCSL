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


class _FakeForecastConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeTimesFM:
    def __init__(self):
        self.model = SimpleNamespace(p=32)
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


class TimesFMGiftEvalRecipeTest(unittest.TestCase):
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
        self.assertEqual(
            [cfg.per_core_batch_size for cfg in model.compiles], [2, 1])
        self.assertTrue(np.isnan(model.inputs[0][1][1]))
        np.testing.assert_array_equal(model.inputs[1][0], np.zeros(2, dtype=np.float32))

    def test_old_timesfm_cache_is_rejected(self) -> None:
        self.assertFalse(ablation._inference_recipe_current({}, "timesfm"))
        self.assertTrue(ablation._inference_recipe_current(
            {"_timesfm_gifteval_recipe":
             timesfm_gifteval.TIMESFM_GIFTEVAL_RECIPE},
            "timesfm",
        ))
        self.assertTrue(ablation._inference_recipe_current({}, "chronos2"))

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
                    "timesfm_gifteval_recipe": "old",
                }
                with open(summary, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)

                done, _ = run_all._done_stage_3(
                    "timesfm", "TimesFM2.5-200M")
                self.assertFalse(done)

                row["timesfm_gifteval_recipe"] = (
                    timesfm_gifteval.TIMESFM_GIFTEVAL_RECIPE)
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
                        "_timesfm_gifteval_recipe":
                            timesfm_gifteval.TIMESFM_GIFTEVAL_RECIPE,
                    }, f)

                done, _ = run_all._done_stage_3(
                    "timesfm", "TimesFM2.5-200M")
                self.assertTrue(done)
            finally:
                run_all.ABLATION_GENERAL = old_root


if __name__ == "__main__":
    unittest.main()
