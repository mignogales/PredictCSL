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

    def test_long_lag_paper_comparison_matches_toto_sample_budget(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "configs",
            "experiments_long_lag_comparison.yaml")
        cfg = load_config(path, {})
        self.assertEqual(cfg["max_samples"], 48)
        self.assertEqual(cfg["synthetic_controls"]["n_instances"], 48)
        self.assertEqual(
            cfg["synthetic_controls"]["distant_lags"],
            [128, 512, 1024, 2048])
        self.assertEqual(cfg["synthetic_controls"]["series_length"], 8192)

    def test_dense_3k_followup_is_targeted_and_capped(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "configs",
            "experiments_long_lag_dense_3k.yaml")
        cfg = load_config(path, {})
        scfg = cfg["synthetic_controls"]
        self.assertTrue(cfg["output_root"].endswith(
            "context_interpretability_long_lag_dense_3k"))
        self.assertEqual(len(scfg["only_datasets"]), 4)
        self.assertEqual(scfg["run_methods"], [])
        self.assertEqual(scfg["sentinel_methods"], [])
        self.assertEqual(max(scfg["context_lengths"]), 3000)
        self.assertEqual(scfg["context_lengths"][-9:-1], [
            2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944])

    def test_forced_d2048_audit_is_targeted_and_isolated(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "configs",
            "experiments_long_lag_forced_d2048.yaml")
        cfg = load_config(path, {})
        self.assertTrue(cfg["output_root"].endswith(
            "context_interpretability_long_lag_forced_d2048"))
        self.assertEqual(cfg["synthetic_controls"]["only_datasets"], [
            "famB_linear_r8_d2048_s1_linear_n0.1"])
        self.assertTrue(cfg["perturbation"]["largest_context_only"])
        self.assertTrue(cfg["perturbation"]["only_forced_blocks"])
        self.assertTrue(cfg["perturbation"]["reuse_clean_cache_root"].endswith(
            "context_interpretability_long_lag"))

    def test_tirex_family_b_followup_is_core_and_causal_band_only(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "configs",
            "experiments_long_lag_tirex_family_b_targeted.yaml")
        cfg = load_config(path, {})
        self.assertTrue(cfg["output_root"].endswith(
            "context_interpretability_long_lag"))
        self.assertEqual(cfg["synthetic_controls"]["only_datasets"], [
            "famB_linear_r8_d128_s1_linear_n0.1",
            "famB_linear_r8_d512_s1_linear_n0.1",
            "famB_linear_r8_d1024_s1_linear_n0.1",
            "famB_linear_r8_d2048_s1_linear_n0.1",
        ])
        self.assertEqual(
            cfg["synthetic_controls"]["run_methods"], ["perturbation"])
        self.assertTrue(cfg["perturbation"]["largest_context_only"])
        self.assertTrue(cfg["perturbation"]["only_forced_blocks"])
        self.assertEqual(
            cfg["perturbation"]["cell_subdir"], "targeted_causal_band")

    def test_tirex_family_b_full_followup_keeps_exhaustive_perturbation(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "configs",
            "experiments_long_lag_tirex_family_b_full.yaml")
        cfg = load_config(path, {})
        self.assertEqual(cfg["synthetic_controls"]["only_datasets"], [
            "famB_linear_r8_d128_s1_linear_n0.1",
            "famB_linear_r8_d512_s1_linear_n0.1",
            "famB_linear_r8_d1024_s1_linear_n0.1",
            "famB_linear_r8_d2048_s1_linear_n0.1",
        ])
        self.assertEqual(
            cfg["synthetic_controls"]["run_methods"], ["perturbation"])
        self.assertFalse(cfg["perturbation"].get("largest_context_only", False))
        self.assertFalse(cfg["perturbation"].get("only_forced_blocks", False))
        self.assertNotIn("cell_subdir", cfg["perturbation"])


if __name__ == "__main__":
    unittest.main()
