"""End-to-end runs of exp0/2/3/4/5 on the dummy adapter (CPU, seconds)."""

import json
import os
import tempfile
import unittest

import numpy as np

from context_interpretability.experiments import (
    activation_patching, attention_masking, forecast_lens,
    integrated_gradients, synthetic_controls)
from context_interpretability.schema import load_results
from context_interpretability.tests.dummy_adapter import (
    DummyAdapter, make_config, make_data)


class TestAttentionMaskingExp0(unittest.TestCase):
    def test_rows_and_semantics(self):
        adapter = DummyAdapter()
        data = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "exp0")
            attention_masking.run(adapter, data, make_config(tmp), out, seed=0)
            df = load_results(out)
            self.assertTrue((df["method"] == "attention_masking").all())
            w64 = df[df["context_length"] == 64]
            # hidden span rows: lookback_start = visible L, lookback_end = W
            self.assertEqual(sorted(w64["lookback_start"].unique().tolist()),
                             [16, 32])
            self.assertTrue((w64["lookback_end"] == 64).all())
            # masking more (smaller L) moves the prediction at least as much
            d16 = w64[w64["lookback_start"] == 16]["prediction_distance"].mean()
            d32 = w64[w64["lookback_start"] == 32]["prediction_distance"].mean()
            self.assertGreaterEqual(d16 + 1e-9, d32)


class TestActivationPatchingExp2(unittest.TestCase):
    def test_full_recovery_on_tokenwise_model(self):
        adapter = DummyAdapter()
        data = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "exp2")
            activation_patching.run(adapter, data, make_config(tmp), out,
                                    seed=0)
            df = load_results(out)
            bt = df[df["perturbation_type"] == "permutation/block_token"]
            self.assertFalse(bt.empty)
            # token-wise blocks + clean acts at the corrupted tokens ->
            # near-complete recovery at every layer (spec §5.6 R ~= 1)
            self.assertGreater(bt["recovery_score"].mean(), 0.95)
            ft = df[df["perturbation_type"] == "permutation/forecast_token"]
            self.assertFalse(ft.empty)      # dummy declares a readout token

    def test_recovery_score_identities(self):
        clean = np.array([[1.0, 1.0]])
        corr = np.array([[3.0, 3.0]])
        r_full = activation_patching.recovery_score(clean, clean, corr)
        r_none = activation_patching.recovery_score(corr, clean, corr)
        self.assertAlmostEqual(float(r_full[0]), 1.0, places=5)
        self.assertAlmostEqual(float(r_none[0]), 0.0, places=5)


class TestForecastLensExp3(unittest.TestCase):
    def test_last_layer_matches_and_saturation_written(self):
        adapter = DummyAdapter()
        data = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "exp3")
            forecast_lens.run(adapter, data, make_config(tmp), out, seed=0)
            # lens at the deepest layer must equal the normal forecast
            last = adapter.get_layer_names()[-1]
            w = data.window(64)
            np.testing.assert_allclose(adapter.forecast_from_layer(w, last),
                                       adapter.forecast(w), rtol=1e-5)
            df = load_results(out)
            deepest = df[df["layer"] == last]
            self.assertLess(deepest["prediction_distance_norm"].abs().max(),
                            1e-5)
            sat = os.path.join(out, "dummy", "saturation_layers.json")
            self.assertTrue(os.path.exists(sat))
            with open(sat) as f:
                self.assertIn("saturation_layer_by_context", json.load(f))


class TestIntegratedGradientsExp5(unittest.TestCase):
    def test_completeness_exact_on_linear_model(self):
        adapter = DummyAdapter(linear=True)
        x = np.random.default_rng(0).normal(0, 1, (3, 32)).astype(np.float32)
        bl = np.zeros_like(x)
        attr, comp = integrated_gradients.integrated_gradients(
            adapter, x, bl, steps=8, internal_batch_size=4,
            target_kind="mean_forecast")
        self.assertEqual(attr.shape, x.shape)
        self.assertLess(comp.max(), 1e-4)          # exact for a linear model

    def test_end_to_end_run(self):
        adapter = DummyAdapter()
        data = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "exp5")
            integrated_gradients.run(adapter, data, make_config(tmp), out,
                                     seed=0)
            df = load_results(out)
            self.assertTrue((df["method"] == "integrated_gradients").all())
            # per-sample block attributions are normalized (sum ~ <= 1)
            g = df[(df["context_length"] == 64)
                   & (df["perturbation_type"] == "ig/context_mean")]
            sums = g.groupby("sample_id")["attribution_score"].sum()
            self.assertTrue((sums <= 1.0 + 1e-6).all())
            done = json.load(open(os.path.join(out, "dummy", "w64",
                                               "done.json")))
            self.assertIn("convergence", done)
            self.assertIn("limitations", done)


class TestSyntheticControlsExp4(unittest.TestCase):
    def test_compact_design_and_sentinel_selection(self):
        cfg = make_config("unused")
        cfg["synthetic_controls"].update({
            "local_kinds": ["linear", "nonlinear", "seasonal"],
            "distant_lags": [32, 64, 128, 256],
            "dependency_strengths": [0.0, 0.5, 1.0],
            "noise_levels": [0.1],
            "noise_robustness": {
                "lag": 128, "strength": 1.0,
                "extra_noise_levels": [0.05, 0.25],
            },
            "run_methods": ["perturbation", "attention_masking"],
            "sentinel_methods": ["activation_patching",
                                 "integrated_gradients"],
            "sentinel_lags": [32, 256],
            "sentinel_strengths": [0.0, 1.0],
        })
        specs = synthetic_controls._control_specs(cfg["synthetic_controls"])
        self.assertEqual(len(specs), 18)
        self.assertEqual(sum(role == "core" for _spec, role in specs), 12)
        self.assertEqual(sum(role == "noise_robustness"
                             for _spec, role in specs), 2)
        sentinel = next((spec, role) for spec, role in specs
                        if spec.family == "B" and spec.distant_lag == 256
                        and spec.strength == 1.0 and role == "core")
        methods = synthetic_controls._methods_for_spec(
            *sentinel, cfg["synthetic_controls"])
        self.assertIn("integrated_gradients", methods)
        nonsentinel = next((spec, role) for spec, role in specs
                           if spec.family == "B" and spec.distant_lag == 64
                           and spec.strength == 0.5 and role == "core")
        methods = synthetic_controls._methods_for_spec(
            *nonsentinel, cfg["synthetic_controls"])
        self.assertNotIn("integrated_gradients", methods)

    def test_summary_and_oracle(self):
        adapter = DummyAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "exp4")
            path = synthetic_controls.run(adapter, make_config(tmp), out,
                                          seed=0)
            with open(path) as f:
                summ = json.load(f)
            fams = {c["spec"]["family"] for c in summ["controls"]}
            self.assertEqual(fams, {"A", "B", "C"})
            for c in summ["controls"]:
                self.assertIn("oracle", c)
                self.assertIn("sufficient_context", c)
                if c["spec"]["family"] == "B" and c["spec"]["strength"] >= 1.0:
                    self.assertTrue(c["oracle"]["distant_predictive"]
                                    or c["config_broken"])
            # perturbation results exist for the controls
            self.assertFalse(load_results(out).empty)


if __name__ == "__main__":
    unittest.main()
