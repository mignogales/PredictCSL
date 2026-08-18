import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.summarize_oracle_pareto_all_models import collect_frontiers


class SummarizeOracleParetoTests(unittest.TestCase):
    def test_collects_both_weightings_and_full_relative_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for suffix, weighting in (("", "cell"), ("_instance_weighted", "instances")):
                directory = root / f"Model{suffix}"
                directory.mkdir()
                pd.DataFrame({
                    "frontier_index": [0, 1, 2],
                    "normalized_mase": [0.8, 1.0, 1.02],
                    "flops_saved_pct": [10.0, 50.0, 70.0],
                    "selected_windows": ["{}", "{}", "{}"],
                }).to_csv(directory / "oracle_supported_frontier.csv", index=False)
                report = {
                    "n_cells": 2,
                    "n_supported_points": 3,
                    "full_normalized_mase": 1.0,
                    "unconstrained_oracle_normalized_mase": 0.8,
                    "unconstrained_oracle_flops_saved_pct": 10.0,
                    "maximum_supported_flops_saved_pct": 70.0,
                    "minimum_compute_normalized_mase": 1.02,
                    "flops_weighting": weighting,
                }
                (directory / "report.json").write_text(json.dumps(report))

            points, summary = collect_frontiers(root, ["Model"])

            self.assertEqual(set(points.flops_weighting), {"cell", "instances"})
            self.assertNotIn("selected_windows", points.columns)
            self.assertEqual(len(summary), 2)
            for value in summary.oracle_quality_gain_vs_full_pct:
                self.assertAlmostEqual(float(value), 20.0)
            self.assertTrue((
                summary.no_worse_than_full__flops_saved_pct == 50.0
            ).all())
            self.assertTrue((
                summary.within_2pct_of_full__flops_saved_pct == 70.0
            ).all())


if __name__ == "__main__":
    unittest.main()
