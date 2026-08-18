import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.oracle_pareto_frontier_gifteval import (
    Action,
    load_cell_actions,
    pareto_prune_actions,
    trace_supported_frontier,
)


class OracleParetoFrontierTests(unittest.TestCase):
    def test_within_cell_dominated_actions_are_removed(self):
        actions = [
            Action("a", "native", 1.0, 1.0, 10.0, True),
            Action("a", "short", 1.1, 1.0, 5.0),
            Action("a", "dominated", 1.2, 1.0, 8.0),
        ]
        kept = pareto_prune_actions(actions)
        self.assertEqual({action.window for action in kept}, {"native", "short"})

    def test_frontier_contains_accuracy_and_minimum_compute_endpoints(self):
        groups = [
            pareto_prune_actions([
                Action("a", "native", 1.0, 1.0, 10.0, True),
                Action("a", "short", 1.1, 1.0, 5.0),
            ]),
            pareto_prune_actions([
                Action("b", "native", 1.0, 1.0, 10.0, True),
                Action("b", "short", 1.2, 1.0, 2.0),
            ]),
        ]
        frontier = trace_supported_frontier(groups, full_flops=20.0)
        accuracy = frontier.loc[frontier.normalized_mase.idxmin()]
        cheapest = frontier.loc[frontier.total_flops.idxmin()]

        self.assertAlmostEqual(float(accuracy.normalized_mase), 1.0)
        self.assertAlmostEqual(float(accuracy.flops_saved_pct), 0.0)
        self.assertAlmostEqual(float(cheapest.total_flops), 7.0)
        self.assertAlmostEqual(float(cheapest.flops_saved_pct), 65.0)

    def test_returned_points_are_nondominated(self):
        groups = [
            [
                Action("a", "large", 1.0, 1.0, 10.0),
                Action("a", "medium", 1.03, 1.0, 7.0),
                Action("a", "small", 1.2, 1.0, 2.0),
            ],
            [
                Action("b", "large", 0.9, 1.0, 8.0),
                Action("b", "small", 1.1, 1.0, 3.0),
            ],
        ]
        frontier = trace_supported_frontier(groups, full_flops=18.0)
        for left_idx, left in frontier.iterrows():
            for right_idx, right in frontier.iterrows():
                if left_idx == right_idx:
                    continue
                dominates = (
                    right.total_flops <= left.total_flops
                    and right.normalized_mase <= left.normalized_mase
                    and (right.total_flops < left.total_flops
                         or right.normalized_mase < left.normalized_mase)
                )
                self.assertFalse(dominates)

    def test_single_grid_action_remains_available_to_diagnostic_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compare_dir = root / "models" / "Model" / "compare_real_vs_predicted"
            compare_dir.mkdir(parents=True)
            np.savez(
                compare_dir / "compare_Cell_tshort_Model.npz",
                window_grid=np.asarray([32]),
                real_curve_gluonts_real=np.asarray([0.8]),
            )
            comparison_path = root / "comparison.csv"
            fields = [
                "model_short", "model", "dataset_display", "term", "horizon",
                "n_instances", "naive_mase", "full_mase", "full_flops",
                "best_mase", "best_flops", "selection_eligible",
            ]
            with comparison_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "model_short": "Model",
                    "model": "unknown/model",
                    "dataset_display": "Cell",
                    "term": "short",
                    "horizon": 8,
                    "n_instances": 1,
                    "naive_mase": 1.0,
                    "full_mase": 1.0,
                    "full_flops": 100.0,
                    "best_mase": 0.8,
                    "best_flops": 40.0,
                    "selection_eligible": False,
                })

            groups, _, _ = load_cell_actions(
                str(root), str(comparison_path), "Model",
                "mase_gluonts_real", {}, "cell",
            )

            self.assertEqual({action.window for action in groups[0]}, {"native", "32"})
            self.assertAlmostEqual(min(action.mase for action in groups[0]), 0.8)


if __name__ == "__main__":
    unittest.main()
