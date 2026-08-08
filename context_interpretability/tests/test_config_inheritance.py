from __future__ import annotations

import os
import tempfile
import unittest

import yaml

from context_interpretability.run_experiment import load_config


class TestConfigInheritance(unittest.TestCase):
    def test_extends_deep_merges_nested_mappings_and_replaces_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "base.yaml")
            child = os.path.join(tmp, "child.yaml")
            with open(base, "w") as f:
                yaml.safe_dump({
                    "scalar": 1,
                    "nested": {"keep": True, "replace": 2},
                    "grid": [1, 2, 3],
                }, f)
            with open(child, "w") as f:
                yaml.safe_dump({
                    "extends": "base.yaml",
                    "nested": {"replace": 9},
                    "grid": [8, 16],
                }, f)
            cfg = load_config(child, {"scalar": 7, "missing": None})
        self.assertEqual(cfg["scalar"], 7)
        self.assertEqual(cfg["nested"], {"keep": True, "replace": 9})
        self.assertEqual(cfg["grid"], [8, 16])

    def test_long_lag_followup_is_isolated_and_reaches_2048(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "configs",
            "experiments_long_lag.yaml")
        cfg = load_config(path, {})
        self.assertTrue(cfg["output_root"].endswith(
            "context_interpretability_long_lag"))
        self.assertEqual(
            cfg["synthetic_controls"]["distant_lags"],
            [128, 512, 1024, 2048])
        self.assertEqual(cfg["synthetic_controls"]["series_length"], 8192)
        self.assertIn(8192, cfg["synthetic_controls"]["context_lengths"])
        self.assertEqual(
            cfg["perturbation"]["methods"],
            ["block_mean", "permutation", "matched_block", "noise"])


if __name__ == "__main__":
    unittest.main()
