import unittest

import torch

from experiments.predict_context_length import pairwise_task_loss


class PairwiseTaskLossTests(unittest.TestCase):
    def test_correct_order_scores_better_than_reversed_scores(self):
        error = torch.tensor([[4.0, 2.0, 1.0, 3.0]])
        correct = torch.tensor([[2.0, 1.0, 0.0, 1.0]])
        reversed_scores = -correct

        self.assertLess(
            float(pairwise_task_loss(correct, error, tie_tolerance=0.0)),
            float(pairwise_task_loss(
                reversed_scores, error, tie_tolerance=0.0)),
        )

    def test_loss_is_invariant_to_positive_error_scale(self):
        error = torch.tensor([
            [1.0, 2.0, 1.5, 3.0],
            [5.0, 4.0, 8.0, 2.0],
        ])
        scores = torch.tensor([
            [0.1, 0.4, 0.2, 0.8],
            [0.7, 0.3, 1.0, 0.1],
        ])
        base = pairwise_task_loss(scores, error, tie_tolerance=0.0)
        scaled = pairwise_task_loss(scores, error * 37.0, tie_tolerance=0.0)
        self.assertTrue(torch.allclose(base, scaled, atol=1e-7, rtol=1e-7))

    def test_near_ties_are_ignored(self):
        scores = torch.tensor([[1.0, -4.0, 9.0]], requires_grad=True)
        error = torch.tensor([[1.000, 1.005, 0.999]])
        loss = pairwise_task_loss(scores, error, tie_tolerance=0.01)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(scores.grad, torch.zeros_like(scores)))

    def test_invalid_pairs_are_ignored(self):
        scores = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        error = torch.tensor([[1.0, float("nan"), 2.0, 1.0]])
        observed = pairwise_task_loss(scores, error, tie_tolerance=0.0)
        expected = pairwise_task_loss(
            scores[:, 2:], error[:, 2:], tie_tolerance=0.0)
        self.assertTrue(torch.allclose(observed, expected))


if __name__ == "__main__":
    unittest.main()
