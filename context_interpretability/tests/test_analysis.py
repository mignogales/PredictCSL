"""Analysis layer: aggregation pivots, significance, figures + hypothesis
report generated from a real (dummy-model) run tree."""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from context_interpretability.analysis import aggregate as agg
from context_interpretability.analysis import figures, hypotheses
from context_interpretability.analysis.significance import significance_table
from context_interpretability.experiments import (
    attention_masking, forecast_lens, integrated_gradients, perturbation)
from context_interpretability.schema import load_results
from context_interpretability.tests.dummy_adapter import (
    DummyAdapter, make_config, make_data)


def _build_run(tmp: str) -> str:
    """Small but complete run tree (exp0+1+3+5) under tmp/DummyModel."""
    run_dir = os.path.join(tmp, "DummyModel")
    adapter = DummyAdapter()
    data = make_data()
    cfg = make_config(tmp)
    perturbation.run(adapter, data, cfg,
                     os.path.join(run_dir, "exp1_perturbation"), seed=0)
    attention_masking.run(adapter, data, cfg,
                          os.path.join(run_dir, "exp0_attention_masking"),
                          seed=0)
    forecast_lens.run(adapter, data, cfg,
                      os.path.join(run_dir, "exp3_forecast_lens"), seed=0)
    integrated_gradients.run(
        adapter, data, cfg,
        os.path.join(run_dir, "exp5_integrated_gradients"), seed=0)
    return run_dir


class TestAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = _build_run(cls._tmp.name)
        cls.df = load_results(cls.run_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_collapse_seeds_reduces_rows(self):
        pert = self.df[self.df["method"] == "perturbation"]
        collapsed = agg.collapse_seeds(pert)
        self.assertLess(len(collapsed), len(pert))
        self.assertNotIn("seed", collapsed.columns)

    def test_heatmap_pivot_shape(self):
        piv = agg.heatmap_matrix(self.df[self.df["method"] == "perturbation"])
        self.assertEqual(list(piv.columns), [16, 32, 64])
        self.assertIn(0, piv.index)                     # lookback 0 present

    def test_profile_normalization_preserves_sign_and_reports_scale(self):
        prof = pd.Series([-2.0, 1.0, np.nan])
        normalized, max_abs = figures._normalize_profile(prof)
        self.assertEqual(max_abs, 2.0)
        np.testing.assert_allclose(normalized.iloc[:2], [-1.0, 0.5])
        self.assertTrue(np.isnan(normalized.iloc[2]))

    def test_perturbation_grid_uses_4x4_for_16_or_more(self):
        contexts = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
                    1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192]
        selected, nrows, ncols = figures._perturbation_grid_layout(contexts)
        self.assertEqual((nrows, ncols), (4, 4))
        self.assertEqual(len(selected), 16)
        self.assertNotIn(32, selected)
        self.assertNotIn(48, selected)
        self.assertEqual(selected[0], 64)
        self.assertEqual(selected[-1], 8192)

    def test_perturbation_grid_uses_largest_12_for_3x4(self):
        contexts = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
                    1024, 1536, 2048]
        selected, nrows, ncols = figures._perturbation_grid_layout(contexts)
        self.assertEqual((nrows, ncols), (3, 4))
        self.assertEqual(selected, contexts[1:])

    def test_sliced_profile_is_paired_against_full_context(self):
        rows = []
        for sample, loss16, loss32 in [("a", 3.0, 1.0),
                                       ("b", 6.0, 2.0)]:
            for W, loss in [(16, loss16), (32, loss32)]:
                rows.append({
                    "model": "m", "dataset": "d", "sample_id": sample,
                    "context_length": W, "lookback_start": 16,
                    "clean_loss": loss, "method": "attention_masking",
                })
        frame = pd.DataFrame(rows)
        profiles = figures._sliced_loss_delta_profiles(
            frame, frame[frame["context_length"] == 32])
        self.assertEqual(list(profiles[32].index), [16])
        self.assertAlmostEqual(profiles[32].iloc[0], 3.0)

    def test_lens_matrices(self):
        mats = agg.lens_matrices(self.df)
        self.assertEqual(set(mats),
                         {"error", "dist_to_full", "dist_to_same_norm"})
        self.assertEqual(mats["error"].shape[0], 3)     # 3 layers

    def test_significance_table(self):
        pert = self.df[self.df["method"] == "perturbation"]
        tab = significance_table(pert, n_boot=100)
        self.assertFalse(tab.empty)
        for col in ("mean", "ci_low", "ci_high", "p_wilcoxon", "p_adj",
                    "significant", "cohens_d"):
            self.assertIn(col, tab.columns)

    def test_instance_sufficient_context(self):
        mat = np.array([[3.0, 1.0, 1.01], [3.0, 2.0, 1.0]])
        suff = agg.instance_sufficient_context(mat, [16, 32, 64], 0.05)
        np.testing.assert_array_equal(suff, [32, 64])

    def test_figures_generate_all(self):
        figures.generate_all(self.run_dir)
        figdir = os.path.join(self.run_dir, "figures")
        names = os.listdir(figdir)
        self.assertTrue(any(n.startswith("02_masking") for n in names))
        self.assertTrue(any(n.startswith("03a_perturbation") for n in names))
        self.assertIn("03c_perturbation_profiles_log_y.png", names)
        self.assertTrue(any(n.startswith("05_lens") for n in names))
        self.assertTrue(any(n.startswith("06_ig") for n in names))
        self.assertTrue(any(n.startswith("09_cross_method") for n in names))

    def test_cross_model_masking_figure(self):
        root = os.path.dirname(self.run_dir)
        figures.generate_all_models(root, ["DummyModel"])
        self.assertTrue(os.path.exists(os.path.join(
            root, "figures",
            "02_masking_effect_vs_lookback_all_models.png")))

    def test_hypotheses_report(self):
        report = hypotheses.evaluate(self.run_dir)
        self.assertEqual(set(report["hypotheses"]),
                         {"H1", "H2", "H3", "H4", "H5"})
        for h in report["hypotheses"].values():
            self.assertIn(h["verdict"],
                          {"supporting", "contradicting", "mixed",
                           "insufficient_data"})
        # H4 needs exp4 output — absent here, must be insufficient not wrong
        self.assertEqual(report["hypotheses"]["H4"]["verdict"],
                         "insufficient_data")
        self.assertIn("ig_limitations", report)
        self.assertTrue(os.path.exists(
            os.path.join(self.run_dir, "hypotheses_report.md")))


if __name__ == "__main__":
    unittest.main()
