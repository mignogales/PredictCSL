"""End-to-end runs of exp0/2/3/4/5 on the dummy adapter (CPU, seconds)."""

import json
import os
import tempfile
import unittest

import numpy as np

from context_interpretability.experiments import (
    activation_patching, attention_masking, forecast_lens,
    integrated_gradients, synthetic_controls, context_decomposition,
    predictor_contrast_saliency, tsfm_contrast_saliency)
from context_interpretability.schema import load_results
from context_interpretability.tests.dummy_adapter import (
    DummyAdapter, make_config, make_data)


class TestAttentionMaskingExp0(unittest.TestCase):
    def test_rows_and_semantics(self):
        adapter = DummyAdapter()
        data = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "exp0")
            config = make_config(tmp)
            config["attention_masking"] = {"metrics": ["mse", "mae"]}
            attention_masking.run(adapter, data, config, out, seed=0)
            df = load_results(out)
            self.assertTrue((df["method"] == "attention_masking").all())
            self.assertEqual(set(df["metric"]), {"mse", "mae"})
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


class TestPredictorContrastSaliencyExp6(unittest.TestCase):
    @staticmethod
    def _checkpoint(path):
        import torch
        from experiments.predict_context_length import build_predictor

        cfg = {
            "arch": "patchtst", "context_length": 64,
            "patch_length": 8, "d_model": 16,
            "num_hidden_layers": 1, "num_attention_heads": 4,
            "dropout": 0.0, "mask_ratio": 0.0,
            "n_windows": 3, "window_grid": [16, 32, 64],
            "n_horizons": 1, "horizon_grid": [8],
            "training_objective": "curve", "curve_metric": "mae",
        }
        model = build_predictor(cfg, 3, 1)
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "best_model.pt"))
        with open(os.path.join(path, "best_config.json"), "w") as f:
            json.dump(cfg, f)

    def test_curve_coordinates_share_one_fixed_input(self):
        import torch
        from experiments.predict_context_length import build_predictor

        cfg = {
            "arch": "patchtst", "context_length": 64,
            "patch_length": 8, "d_model": 16,
            "num_hidden_layers": 1, "num_attention_heads": 4,
            "dropout": 0.0, "mask_ratio": 0.0,
        }
        model = build_predictor(cfg, 3, 1).eval()
        x = torch.randn(2, 64)
        h = torch.zeros(2, dtype=torch.long)
        curve = model(x.unsqueeze(-1), h)[0]
        contrast = predictor_contrast_saliency.contrast_target(
            model, x, h, 0, 2)
        self.assertEqual(tuple(x.shape), (2, 64))
        torch.testing.assert_close(contrast, curve[:, 2] - curve[:, 0])

    def test_end_to_end_checkpoint_saliency(self):
        adapter = DummyAdapter()
        data = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "predictor")
            self._checkpoint(checkpoint)
            cfg = make_config(tmp)
            cfg["predictor_contrast_saliency"].update({
                "predictor_dir": checkpoint,
                "window_pairs": [[16, 64]],
                "steps": 8,
                "baselines": ["zero"],
            })
            out = os.path.join(tmp, "exp6")
            predictor_contrast_saliency.run(
                adapter, data, cfg, out, seed=0)
            df = load_results(out)
            self.assertFalse(df.empty)
            self.assertTrue(
                (df["method"] == "predictor_contrast_saliency").all())
            self.assertTrue(np.isfinite(df["attribution_score"]).all())
            cell = os.path.join(out, "dummy", "h8_L16_to_L64")
            with np.load(os.path.join(cell, "attr_zero.npz")) as z:
                self.assertEqual(z["attributions"].shape, (data.n, 64))
                self.assertTrue(np.isfinite(z["completeness_err"]).all())
            with open(os.path.join(cell, "done.json")) as f:
                done = json.load(f)
            self.assertEqual(done["contrast_semantics"],
                             "predicted_z_error(long)-predicted_z_error(short)")


class TestContextDecompositionExp7(unittest.TestCase):
    def test_tail_stat_prefix_preserves_visible_tail_and_stats(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(4, 64)).astype(np.float32)
        matched = context_decomposition.tail_stat_matched_prefix(x, 32)
        np.testing.assert_array_equal(matched[:, -32:], x[:, -32:])
        np.testing.assert_allclose(matched.mean(axis=1),
                                   x[:, -32:].mean(axis=1), atol=1e-6)
        np.testing.assert_allclose(matched.std(axis=1),
                                   x[:, -32:].std(axis=1), atol=1e-6)

    def test_end_to_end_mae_mse_differences(self):
        adapter = DummyAdapter()
        data = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "exp7")
            context_decomposition.run(
                adapter, data, make_config(tmp), out, seed=0)
            df = load_results(out)
            self.assertFalse(df.empty)
            self.assertEqual(set(df["metric"]), {"mae", "mse"})
            self.assertEqual(set(df["perturbation_type"]), {
                context_decomposition.VARIANT_FULL,
                context_decomposition.VARIANT_TAIL,
            })
            np.testing.assert_allclose(
                df["loss_delta"],
                df["intervened_loss"] - df["clean_loss"], atol=1e-7)
            cell = os.path.join(out, "dummy", "w64")
            self.assertTrue(os.path.isfile(
                os.path.join(cell, "predictions_L16.npz")))


class TestTSFMLossContrastSaliencyExp8(unittest.TestCase):
    def test_target_is_long_loss_minus_short_loss(self):
        import torch

        adapter = DummyAdapter(linear=True)
        data = make_data(n=3)
        x = torch.from_numpy(data.window(64)).requires_grad_(True)
        y = torch.from_numpy(data.targets)
        contrast, long_loss, short_loss = (
            tsfm_contrast_saliency.loss_contrast(
                adapter, x, y, short_length=16, metric="mse"))
        torch.testing.assert_close(contrast, long_loss - short_loss)
        self.assertEqual(tuple(contrast.shape), (3,))
        self.assertIsNotNone(contrast.grad_fn)

    def test_end_to_end_tsfm_saliency(self):
        adapter = DummyAdapter()
        data = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "exp8")
            tsfm_contrast_saliency.run(
                adapter, data, make_config(tmp), out, seed=0)
            df = load_results(out)
            self.assertFalse(df.empty)
            self.assertEqual(set(df["metric"]), {"mae", "mse"})
            self.assertTrue(
                (df["method"] == "tsfm_loss_contrast_saliency").all())
            np.testing.assert_allclose(
                df["loss_delta"],
                df["intervened_loss"] - df["clean_loss"], atol=1e-6)
            cell = os.path.join(out, "dummy", "mae_L16_to_L64")
            with np.load(os.path.join(cell, "attr_context_mean.npz")) as z:
                self.assertEqual(z["attributions"].shape, (data.n, 64))
                self.assertTrue(np.isfinite(z["completeness_err"]).all())


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
