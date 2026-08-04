import unittest

import numpy as np

from experiments.archive.heuristics.period_detection import (
    cadence_candidates,
    detect_period,
    window_similarity,
)


class PeriodDetectionTests(unittest.TestCase):
    @staticmethod
    def _reference_window_similarity(x, window):
        """Pre-vectorization implementation used to guard numerical parity."""
        n_chunks = x.size // window
        if n_chunks < 2:
            return float("-inf")
        chunks = x[-n_chunks * window:].reshape(n_chunks, window)
        scores = []
        for left, right in zip(chunks[:-1], chunks[1:]):
            finite = np.isfinite(left) & np.isfinite(right)
            if finite.sum() < 2:
                continue
            a = left[finite] - np.mean(left[finite])
            b = right[finite] - np.mean(right[finite])
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denominator > 1e-12:
                scores.append(float(np.dot(a, b) / denominator))
        return float(np.mean(scores)) if scores else float("-inf")

    def test_cadences_are_translated_to_samples(self):
        self.assertEqual(dict(cadence_candidates("S", 1000))["5T"], 300)
        self.assertEqual(dict(cadence_candidates("5T", 100))["H"], 12)
        monthly = dict(cadence_candidates("M", 30))
        self.assertEqual(monthly["Q"], 3)
        self.assertEqual(monthly["A"], 12)

    def test_detects_fundamental_instead_of_equal_scoring_multiple(self):
        pattern = np.random.RandomState(7).normal(size=300)
        period, label, _ = detect_period(np.tile(pattern, 8), "S", 1)
        self.assertEqual((period, label), (300.0, "5T"))

    def test_detects_daily_period_in_hourly_samples(self):
        period, label, _ = detect_period(
            np.tile(np.arange(24, dtype=float), 10), "H", 24)
        self.assertEqual((period, label), (24.0, "D"))

    def test_short_low_frequency_context_falls_back(self):
        period, label, _ = detect_period(
            np.array([1.0, np.nan, 2.0]), "A", 1)
        self.assertEqual((period, label), (1.0, "fallback_season"))

    def test_vectorized_similarity_matches_pairwise_reference(self):
        rng = np.random.RandomState(19)
        for length in (7, 31, 1000):
            for window in (1, 2, 3, 12, 97):
                if window > length:
                    continue
                x = rng.normal(size=length)
                x[rng.uniform(size=length) < 0.15] = np.nan
                expected = self._reference_window_similarity(x, window)
                actual = window_similarity(x, window)
                if np.isneginf(expected):
                    self.assertTrue(np.isneginf(actual), (length, window))
                else:
                    self.assertAlmostEqual(actual, expected, places=12)

    def test_scalar_chunks_are_rejected_without_pairwise_work(self):
        self.assertTrue(np.isneginf(window_similarity(np.arange(100_000.0), 1)))


if __name__ == "__main__":
    unittest.main()
