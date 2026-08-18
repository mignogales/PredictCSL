import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.serieswise_oracle_pareto_gifteval import (
    CellActions,
    aggregate_policy,
    pareto_envelope,
    select_accuracy_oracle,
    select_minimum_compute,
    select_relative_tolerance,
)


def _cell() -> CellActions:
    return CellActions(
        key="toy/tshort",
        naive_mase=2.0,
        errors=np.asarray([
            [1.0, 0.8, 0.7],
            [2.0, 1.0, 0.9],
        ]),
        costs=np.asarray([
            [1.0, 2.0, 4.0],
            [1.0, 2.0, 4.0],
        ]),
        valid_counts=np.asarray([
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
        ]),
        windows=np.asarray([
            [32, 64, 128],
            [32, 64, 128],
        ]),
        selection_eligible=True,
        audit_path=Path("unused.npz"),
    )


class SerieswiseOracleParetoTests(unittest.TestCase):
    def test_accuracy_and_compute_endpoints(self) -> None:
        cell = _cell()
        np.testing.assert_array_equal(select_accuracy_oracle(cell), [2, 2])
        np.testing.assert_array_equal(select_minimum_compute(cell), [0, 0])

    def test_relative_tolerance_uses_cheapest_acceptable_action(self) -> None:
        cell = _cell()
        # In both rows the middle action is within 15% of the best and is the
        # cheapest acceptable choice.
        np.testing.assert_array_equal(
            select_relative_tolerance(cell, 0.15), [1, 1])

    def test_aggregate_uses_weighted_cell_mase_and_normalization(self) -> None:
        cell = _cell()
        point = aggregate_policy(
            [cell], [np.asarray([2, 2])], full_flops=8.0,
            family="test", parameter=0.0)
        expected_mase = (0.7 + 3 * 0.9) / 4
        self.assertAlmostEqual(point["geomean_mase"], expected_mase)
        self.assertAlmostEqual(point["normalized_mase"], expected_mase / 2.0)
        self.assertAlmostEqual(point["flops_saved_pct"], 0.0)

    def test_pareto_envelope_drops_dominated_points(self) -> None:
        points = pd.DataFrame([
            {"family": "a", "parameter": 0.0, "normalized_mase": 1.0,
             "total_flops": 10.0, "flops_saved_pct": 0.0},
            {"family": "b", "parameter": 0.0, "normalized_mase": 0.9,
             "total_flops": 8.0, "flops_saved_pct": 20.0},
            {"family": "c", "parameter": 0.0, "normalized_mase": 1.1,
             "total_flops": 7.0, "flops_saved_pct": 30.0},
            {"family": "dominated", "parameter": 0.0,
             "normalized_mase": 1.2, "total_flops": 9.0,
             "flops_saved_pct": 10.0},
        ])
        result = pareto_envelope(points)
        self.assertEqual(set(result["family"]), {"b", "c"})


if __name__ == "__main__":
    unittest.main()
