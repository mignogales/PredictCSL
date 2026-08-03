from __future__ import annotations

import csv
import math
import os
import unittest

from experiments.gifteval_reference import (
    NORMALIZATION_REFERENCE,
    REFERENCE_DIR,
    leaderboard_dataset_key,
    published_naive_by_display,
    published_naive_record,
    published_seasonal_naive_mase,
)
from experiments import datasets_config


class GiftEvalReferenceTest(unittest.TestCase):
    def test_published_seasonal_naive_covers_all_official_configs(self) -> None:
        self.assertEqual(len(published_seasonal_naive_mase()), 97)
        self.assertEqual(len(published_naive_by_display()), 97)
        run_set = datasets_config.datasets_to_run()
        self.assertEqual(len(run_set), 97)
        self.assertIn("Solar-W", {row[2] for row in run_set})
        self.assertIn("CarParts", {row[2] for row in run_set})
        self.assertEqual(len(datasets_config.catalog()), 97)

    def test_dataset_aliases_match_leaderboard_csv_keys(self) -> None:
        self.assertEqual(
            leaderboard_dataset_key("car_parts_with_missing", "short"),
            "car_parts/M/short",
        )
        self.assertEqual(
            leaderboard_dataset_key("saugeenday/D", "short"),
            "saugeen/D/short",
        )
        self.assertEqual(
            leaderboard_dataset_key("m4_yearly", "short"),
            "m4_yearly/A/short",
        )

    def test_both_gluonts_views_use_the_published_value(self) -> None:
        record = published_naive_record("solar/W", "short")
        expected = published_seasonal_naive_mase()["solar/W/short"]
        self.assertEqual(record["mase_gluonts"], expected)
        self.assertEqual(record["mase_gluonts_real"], expected)
        self.assertEqual(record["_source"], "gift_eval_published_csv")
        self.assertEqual(
            NORMALIZATION_REFERENCE,
            "leaderboard_reference/seasonal_naive_all_results.csv",
        )

    def test_published_timesfm_headline_is_reproduced(self) -> None:
        path = os.path.join(REFERENCE_DIR, "timesfm-2.5_all_results.csv")
        with open(path, newline="") as f:
            model = {
                row["dataset"]: float(row["eval_metrics/MASE[0.5]"])
                for row in csv.DictReader(f)
            }
        naive = published_seasonal_naive_mase()
        score = math.exp(sum(
            math.log(model[key] / naive[key]) for key in model
        ) / len(model))
        self.assertAlmostEqual(score, 0.7050307775530733, places=14)

    def test_published_patchtst_headline_is_reproduced(self) -> None:
        path = os.path.join(REFERENCE_DIR, "patchtst-fm-r1_all_results.csv")
        with open(path, newline="") as f:
            model = {
                row["dataset"]: float(row["eval_metrics/MASE[0.5]"])
                for row in csv.DictReader(f)
            }
        naive = published_seasonal_naive_mase()
        self.assertEqual(len(model), 97)
        score = math.exp(sum(
            math.log(model[key] / naive[key]) for key in model
        ) / len(model))
        self.assertAlmostEqual(score, 0.7069359006427615, places=14)


if __name__ == "__main__":
    unittest.main()
