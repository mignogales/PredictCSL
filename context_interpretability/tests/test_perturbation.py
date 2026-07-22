import os
import tempfile
import unittest

import numpy as np

from context_interpretability.experiments import perturbation as pert
from context_interpretability.schema import load_results
from context_interpretability.tests.dummy_adapter import (
    DummyAdapter, make_config, make_data)


class TestPerturbationPrimitives(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.win = rng.normal(0, 1, (4, 32)).astype(np.float32)
        self.blk = slice(8, 16)

    def _only_block_changed(self, out):
        np.testing.assert_array_equal(out[:, :8], self.win[:, :8])
        np.testing.assert_array_equal(out[:, 16:], self.win[:, 16:])
        self.assertEqual(out.shape, self.win.shape)

    def test_mean_replace(self):
        out = pert.block_mean_replace(self.win, self.blk)
        self._only_block_changed(out)
        for i in range(4):
            np.testing.assert_allclose(
                out[i, self.blk], self.win[i, self.blk].mean(), rtol=1e-5)

    def test_permutation_preserves_values(self):
        out = pert.within_block_permutation(self.win, self.blk, seed=1)
        self._only_block_changed(out)
        for i in range(4):
            np.testing.assert_allclose(np.sort(out[i, self.blk]),
                                       np.sort(self.win[i, self.blk]))
        out2 = pert.within_block_permutation(self.win, self.blk, seed=1)
        np.testing.assert_array_equal(out, out2)       # deterministic

    def test_matched_block(self):
        out = pert.matched_block_replace(self.win, self.blk, seed=2)
        self._only_block_changed(out)
        # replacement differs from the original block for random data
        self.assertTrue(np.any(out[:, self.blk] != self.win[:, self.blk]))

    def test_noise_scale_and_floor(self):
        out = pert.additive_noise(self.win, self.blk, 0.5, seed=3)
        self._only_block_changed(out)
        const = np.ones((2, 32), dtype=np.float32)
        out_c = pert.additive_noise(const, self.blk, 1.0, seed=3,
                                    std_floor=1e-3)
        self.assertTrue(np.any(out_c[:, self.blk] != 1.0))  # floor active


class TestPerturbationRunner(unittest.TestCase):
    def test_end_to_end_and_resume(self):
        adapter = DummyAdapter()
        data = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            out = os.path.join(tmp, "exp1")
            cells = pert.run(adapter, data, cfg, out, seed=0)
            self.assertEqual(len(cells), 3)             # W in {16, 32, 64}
            df = load_results(out)
            self.assertTrue((df["method"] == "perturbation").all())
            self.assertTrue(np.isfinite(df["loss_delta"]).all())
            self.assertTrue(np.isfinite(df["prediction_distance"]).all())
            # recency: recent-block effects exceed distant ones on average
            w64 = df[df["context_length"] == 64]
            recent = w64[w64["block_index"] == 0]["prediction_distance"].mean()
            distant = w64[w64["block_index"] == 7]["prediction_distance"].mean()
            self.assertGreater(recent, distant)
            # resume: a second run does no work (cells already done)
            n_rows = len(df)
            pert.run(adapter, data, cfg, out, seed=0)
            self.assertEqual(len(load_results(out)), n_rows)

    def test_interventions_are_grouped_without_changing_rows(self):
        class TrackingAdapter(DummyAdapter):
            def __init__(self):
                super().__init__()
                self.forecast_batch_sizes = []

            def forecast(self, contexts):
                self.forecast_batch_sizes.append(len(contexts))
                return super().forecast(contexts)

        adapter = TrackingAdapter()
        data = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)  # interventions_per_call == 3
            out = os.path.join(tmp, "exp1")
            pert.run(adapter, data, cfg, out, seed=0)
            # Clean-cache calls contain N samples; grouped intervention calls
            # contain up to 3*N and still produce the complete expected grid.
            self.assertIn(3 * data.n, adapter.forecast_batch_sizes)
            df = load_results(out)
            jobs = (2 + 4 + 8) * 7  # blocks per W * variants per block
            self.assertEqual(len(df), jobs * data.n)


if __name__ == "__main__":
    unittest.main()
