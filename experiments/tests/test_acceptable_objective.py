import unittest

import torch

from experiments.predict_context_length import (
    acceptable_decision_scores,
    acceptable_set_targets,
    acceptable_set_task_loss,
)


class AcceptableSetObjectiveTests(unittest.TestCase):
    def test_targets_include_all_near_oracle_windows(self):
        error = torch.tensor([[1.00, 1.02, 1.031, float("nan"), 1.03]])
        target, valid = acceptable_set_targets(error, rel_tolerance=0.03)
        self.assertTrue(torch.equal(
            target.bool(), torch.tensor([[True, True, False, False, True]])))
        self.assertTrue(torch.equal(
            valid, torch.tensor([[True, True, True, False, True]])))

    def test_targets_are_invariant_to_positive_error_scale(self):
        error = torch.tensor([[3.0, 3.05, 4.0], [8.0, 7.9, 8.1]])
        base, valid_base = acceptable_set_targets(error, rel_tolerance=0.03)
        scaled, valid_scaled = acceptable_set_targets(
            error * 41.0, rel_tolerance=0.03)
        self.assertTrue(torch.equal(base, scaled))
        self.assertTrue(torch.equal(valid_base, valid_scaled))

    def test_accuracy_first_policy_chooses_largest_credible_window(self):
        logits = torch.tensor([[3.0, 1.0, 0.1, -3.0]])
        scores = acceptable_decision_scores(logits, prob_threshold=0.5)
        self.assertEqual(int(scores.argmin(dim=1).item()), 2)

    def test_policy_falls_back_to_highest_probability(self):
        logits = torch.tensor([[-3.0, -0.2, -1.0]])
        scores = acceptable_decision_scores(logits, prob_threshold=0.6)
        self.assertEqual(int(scores.argmin(dim=1).item()), 1)

    def test_matching_logits_have_lower_loss(self):
        error = torch.tensor([[1.0, 1.02, 1.5, float("nan")]])
        matching = torch.tensor([[4.0, 4.0, -4.0, 2.0]])
        reversed_logits = -matching
        self.assertLess(
            float(acceptable_set_task_loss(
                matching, error, rel_tolerance=0.03)),
            float(acceptable_set_task_loss(
                reversed_logits, error, rel_tolerance=0.03)),
        )


if __name__ == "__main__":
    unittest.main()
