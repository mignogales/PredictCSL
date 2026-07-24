import unittest

import numpy as np

from experiments.period_detection import cadence_candidates, detect_period


class PeriodDetectionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
