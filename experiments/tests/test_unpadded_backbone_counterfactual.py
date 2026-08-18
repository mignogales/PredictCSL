import unittest

import pandas as pd

from experiments.benchmark_unpadded_backbone_counterfactual import (
    execution_shape,
    requested_shapes,
    rollup,
)


class CounterfactualShapeTests(unittest.TestCase):
    def test_patchtst_uses_minimum_forecast_grid_and_native_cap(self):
        self.assertEqual(execution_shape("PatchTST-FM-R1", 128, 6), (16, 1))
        self.assertEqual(execution_shape("PatchTST-FM-R1", 8192, 720), (512, 1))

    def test_tirex_keeps_future_grid_and_tta(self):
        self.assertEqual(execution_shape("TiRex2", 1024, 96), (61, 2))
        self.assertEqual(execution_shape("TiRex2", 8192, 929), (285, 4))

    def test_requested_shapes_are_deduplicated(self):
        histogram = pd.DataFrame({
            "method": ["full_native", "balanced", "efficiency"],
            "window_size": [1024, 1024, 2048],
            "horizon": [96, 96, 96],
            "n_instances": [10, 10, 10],
        })
        self.assertEqual(
            requested_shapes(histogram, "TiRex2"), [(61, 2), (93, 2)])

    def test_rollup_uses_measured_batch_latency(self):
        histogram = pd.DataFrame({
            "method": ["full_native", "balanced", "efficiency", "max_efficiency"],
            "window_size": [2048, 1024, 1024, 1024],
            "horizon": [96, 96, 96, 96],
            "n_instances": [64, 64, 64, 64],
        })
        profile = pd.DataFrame({
            "tokens": [61, 93], "backbone_calls": [2, 2],
            "effective_batch_size": [32, 32], "median_ms": [10.0, 20.0],
        })
        summary = rollup(histogram, profile, "TiRex2")
        self.assertTrue((summary["speedup_x"] == 2.0).all())
        self.assertTrue((summary["valid_forecast"] == False).all())


if __name__ == "__main__":
    unittest.main()
