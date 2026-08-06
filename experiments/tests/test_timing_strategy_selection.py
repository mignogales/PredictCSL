import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.benchmark_window_timing_gifteval import _selected_windows


class TimingStrategySelectionTests(unittest.TestCase):

    def test_unions_v3_and_v4_selected_windows(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root) / "Chronos2-Small"
            rows_v3 = pd.DataFrame([{
                "dataset_display": "Example", "term": "short",
                "full_window": 8192, "pred_window": 256,
            }])
            rows_v4 = pd.DataFrame([{
                "dataset_display": "Example", "term": "short",
                "full_window": 8192, "pred_window": 512,
            }])
            for name, frame in [
                ("strategy_comparison_v3", rows_v3),
                ("strategy_comparison_v4", rows_v4),
            ]:
                path = base / name
                path.mkdir(parents=True)
                frame.to_csv(path / "comparison.csv", index=False)

            selected = _selected_windows(
                root, "Chronos2-Small",
                ["strategy_comparison_v3", "strategy_comparison_v4"],
            )

            self.assertEqual(selected[("Example", "short")], {256, 512, 8192})


if __name__ == "__main__":
    unittest.main()
