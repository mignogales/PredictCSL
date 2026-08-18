import unittest

import numpy as np

from experiments.improve_tree_distillation import (
    density_balanced_weight,
    empirical_cdf,
    policy_fidelity,
    validation_pareto_summary,
)


class ImprovedTreeDistillationTests(unittest.TestCase):
    def test_empirical_cdf_is_monotone(self) -> None:
        reference = np.asarray([3.0, 1.0, 2.0, 4.0])
        values = np.asarray([0.0, 1.0, 2.5, 5.0])
        result = empirical_cdf(reference, values)
        np.testing.assert_allclose(result, [0.0, 0.25, 0.5, 1.0])

    def test_density_weight_upweights_sparse_score_regions(self) -> None:
        score = np.asarray([0.0] * 100 + [1.0] * 4, dtype=np.float32)
        weight = density_balanced_weight(
            score, np.ones_like(score), bins=8, max_factor=4.0)
        self.assertGreater(float(weight[-1]), float(weight[0]))

    def test_policy_fidelity_is_perfect_for_monotone_transform(self) -> None:
        teacher = np.asarray([
            [0.1, 0.2, 0.3],
            [0.3, 0.2, 0.1],
        ], dtype=np.float32).ravel()
        student = 3.0 * teacher + 2.0
        data = {
            "validation_errors": np.ones((2, 3), dtype=np.float32),
            "windows": np.asarray([32, 64, 128]),
            "val_lengths": np.asarray([128, 128]),
            "task_series": np.asarray([0, 1]),
        }
        fidelity = policy_fidelity(
            teacher, student, data, quantiles=7)
        self.assertEqual(fidelity["mean_action_agreement"], 1.0)
        self.assertEqual(fidelity["mean_abs_log2_window_error"], 0.0)

    def test_validation_pareto_summary_uses_fixed_quality_budgets(self) -> None:
        profiles = {
            "dense_00": {"validation": {
                "mean_ratio": 1.0, "context_saved_pct": 10.0}},
            "dense_01": {"validation": {
                "mean_ratio": 1.005, "context_saved_pct": 40.0}},
            "dense_02": {"validation": {
                "mean_ratio": 1.02, "context_saved_pct": 70.0}},
        }
        summary = validation_pareto_summary(profiles)
        savings = summary["context_saved_by_mase_budget_pct"]
        self.assertEqual(savings["mase_plus_0pct"], 10.0)
        self.assertEqual(savings["mase_plus_0.5pct"], 40.0)
        self.assertEqual(savings["mase_plus_2pct"], 70.0)


if __name__ == "__main__":
    unittest.main()
