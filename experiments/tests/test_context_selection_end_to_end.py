import unittest

import pandas as pd

from experiments.summarize_context_selection_end_to_end import policy_timing


class ContextSelectionEndToEndTests(unittest.TestCase):

    @staticmethod
    def _histogram(method: str, window_size: int) -> pd.DataFrame:
        return pd.DataFrame([{
            "model": "Model", "dataset": "Example", "term": "short",
            "horizon": 64, "method": method, "window_size": window_size,
            "n_instances": 100, "cell_instances": 1000,
        }])

    @staticmethod
    def _timing() -> pd.DataFrame:
        common = {
            "model_short": "Model", "dataset_display": "Example",
            "term": "short", "horizon": 64, "n_timed_series": 20,
            "cuda_peak_allocated_gb": 3.0,
            "cuda_incremental_peak_allocated_gb": 1.0,
        }
        return pd.DataFrame([
            # CSV loading makes this column textual because the full-native row
            # below shares it with a string sentinel.
            {**common, "window_size": "256", "timing_kind": "numeric_window",
             "mean_s": 2.0},
            {**common, "window_size": "full_native", "timing_kind": "full_native",
             "mean_s": 4.0},
        ])

    def test_numeric_policy_scales_measured_seconds_per_series(self):
        result = policy_timing(
            self._histogram("balanced", 256), self._timing(),
            "Model", "balanced")
        self.assertAlmostEqual(result["estimated_tsfm_s"], 10.0)
        self.assertEqual(result["timing_coverage"], 1.0)

    def test_full_native_ignores_legacy_numeric_cap(self):
        result = policy_timing(
            self._histogram("full_native", 8192), self._timing(),
            "Model", "full_native")
        self.assertAlmostEqual(result["estimated_tsfm_s"], 20.0)
        self.assertEqual(result["timing_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
