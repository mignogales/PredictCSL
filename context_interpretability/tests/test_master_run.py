from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from context_interpretability import master_run
from experiments import master_run_all as env_master


class TestInterpretabilityMaster(unittest.TestCase):
    def test_forwarded_args(self):
        args = SimpleNamespace(
            experiments=["exp0", "exp1"],
            source="gifteval",
            config=None,
            out="custom/results",
            horizon=64,
            batch_size=8,
            max_samples=32,
            seed=7,
            device="cuda:1",
        )
        self.assertEqual(master_run._forwarded_args(args), [
            "--experiments", "exp0", "exp1",
            "--source", "gifteval",
            "--out", "custom/results",
            "--horizon", "64",
            "--batch-size", "8",
            "--max-samples", "32",
            "--seed", "7",
            "--device", "cuda:1",
        ])

    def test_group_command_uses_dedicated_environment(self):
        old_names = env_master._CONDA_ENV_NAMES
        env_master._CONDA_ENV_NAMES = {"predictcsl-toto"}
        try:
            cmd = master_run._group_cmd(
                "predictcsl-toto", ["Toto-2.0-313m"],
                ["--experiments", "exp0", "exp1"])
        finally:
            env_master._CONDA_ENV_NAMES = old_names
        self.assertEqual(cmd[:2], ["bash", "-lc"])
        self.assertIn("conda activate predictcsl-toto", cmd[2])
        self.assertIn(
            "context_interpretability.run_experiment", cmd[2])
        self.assertIn("--models Toto-2.0-313m", cmd[2])

    def test_final_analysis_contains_all_selected_models(self):
        cmd = master_run._analysis_cmd(
            ["Sundial-Base-128M", "Toto-2.0-313m"], [])
        self.assertIn("--models", cmd)
        self.assertIn("Sundial-Base-128M", cmd)
        self.assertIn("Toto-2.0-313m", cmd)
        self.assertEqual(cmd[-1], "--analyze-only")

    def test_analyze_only_skips_model_environment_runs(self):
        with mock.patch.dict(os.environ, {
                "CONDA_DEFAULT_ENV": "predictcsl-main"}), \
             mock.patch.object(
                 env_master, "_resolve_groups",
                 return_value={
                     "predictcsl-toto": ["Toto-2.0-313m"],
                     "predictcsl-legacy": ["Sundial-Base-128M"],
                 }), \
             mock.patch.object(env_master, "_preflight_env") as preflight, \
             mock.patch.object(env_master, "_run") as run:
            master_run.main(["--analyze-only"])
        preflight.assert_not_called()
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][-1], "--analyze-only")

    def test_wrong_active_environment_is_rejected(self):
        with mock.patch.dict(os.environ, {
                "CONDA_DEFAULT_ENV": "predictcsl-toto"}):
            with self.assertRaises(SystemExit):
                master_run._require_main_environment()


if __name__ == "__main__":
    unittest.main()
