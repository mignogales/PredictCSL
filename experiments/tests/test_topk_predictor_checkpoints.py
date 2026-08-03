import json
import tempfile
import unittest
from pathlib import Path

from experiments.evaluate_topk_predictor_checkpoints import (
    package_trial,
    rank_trials,
)


class TopKPredictorCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.source = Path(self.tmp.name) / "source"
        trials = self.source / "trials"
        trials.mkdir(parents=True)
        (self.source / "best_config.json").write_text(json.dumps({
            "trial_idx": 2,
            "arch": "mamba",
            "window_grid": [32, 64],
            "horizon_grid": [16],
            "context_length": 64,
            "n_windows": 2,
            "n_horizons": 1,
        }))
        for idx, score in ((0, 0.3), (1, 0.2), (2, 0.2), (3, 0.4)):
            payload = {
                "trial_idx": idx,
                "val_regret": score,
                "val_win_acc": 0.1 * idx,
                "cfg": {"d_model": 32 + idx},
                "failed": False,
            }
            (trials / f"trial_{idx:03d}.json").write_text(json.dumps(payload))
            (trials / f"trial_{idx:03d}_best.pt").write_bytes(b"weights")

    def tearDown(self):
        self.tmp.cleanup()

    def test_selected_winner_stays_first_across_metric_tie(self):
        ranked = rank_trials(self.source, "val_regret", 3)
        self.assertEqual([r["trial_idx"] for r in ranked], [2, 1, 0])

    def test_package_uses_trial_weights_and_configuration(self):
        trial = rank_trials(self.source, "val_regret", 1)[0]
        package = Path(self.tmp.name) / "package"
        package_trial(self.source, trial, package, "val_regret", 1)
        self.assertTrue((package / "best_model.pt").is_symlink())
        cfg = json.loads((package / "best_config.json").read_text())
        self.assertEqual(cfg["trial_idx"], 2)
        self.assertEqual(cfg["d_model"], 34)
        self.assertEqual(cfg["val_regret"], 0.2)


if __name__ == "__main__":
    unittest.main()
