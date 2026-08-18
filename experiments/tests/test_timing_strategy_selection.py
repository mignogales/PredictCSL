import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import numpy as np

import experiments.benchmark_window_timing_gifteval as timing
from experiments.benchmark_window_timing_gifteval import (
    _run_coordinator,
    _selected_windows,
    _selection_csv_full_native,
    _selection_csv_windows,
    _stratified_indices,
)


class TimingStrategySelectionTests(unittest.TestCase):

    def test_unions_v3_and_v4_selected_windows(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root) / "Chronos2-Small"
            rows_v3 = pd.DataFrame([{
                "dataset_display": "Example", "term": "short",
                "full_window": 8192, "pred_window": 256,
            }])
            rows_v4 = pd.DataFrame([{
                "dataset_display": "Example", "term": "short",
                "full_window": 8192, "pred_window": 512,
            }])
            for name, frame in [
                ("strategy_comparison_v3", rows_v3),
                ("strategy_comparison_v4", rows_v4),
            ]:
                path = base / name
                path.mkdir(parents=True)
                frame.to_csv(path / "comparison.csv", index=False)

            selected = _selected_windows(
                root, "Chronos2-Small",
                ["strategy_comparison_v3", "strategy_comparison_v4"],
            )

            self.assertEqual(selected[("Example", "short")], {256, 512, 8192})

    def test_adds_filtered_risk_histogram_windows(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "selected_window_histograms.csv"
            pd.DataFrame([
                {
                    "model": "Chronos2-Small", "dataset": "Example",
                    "term": "short", "horizon": 64, "method": "balanced",
                    "window_size": 256, "n_instances": 7,
                },
                {
                    "model": "Chronos2-Small", "dataset": "Example",
                    "term": "short", "horizon": 64, "method": "aggressive",
                    "window_size": 128, "n_instances": 3,
                },
                {
                    "model": "Moirai2-Small", "dataset": "Example",
                    "term": "short", "horizon": 64, "method": "balanced",
                    "window_size": 512, "n_instances": 4,
                },
            ]).to_csv(path, index=False)

            selected, horizons = _selection_csv_windows(
                [str(path)], "Chronos2-Small", {"balanced"})

            self.assertEqual(selected[("Example", "short")], {256})
            self.assertEqual(horizons[("Example", "short")], 64)

    def test_full_native_is_not_misread_as_a_numeric_window(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "selected_window_histograms.csv"
            pd.DataFrame([
                {
                    "model": "Chronos2-Small", "dataset": "Example",
                    "term": "short", "horizon": 64, "method": "full_native",
                    "window_size": 8192, "n_instances": 10,
                },
                {
                    "model": "Chronos2-Small", "dataset": "Example",
                    "term": "short", "horizon": 64, "method": "balanced",
                    "window_size": 256, "n_instances": 10,
                },
            ]).to_csv(path, index=False)

            numeric, _ = _selection_csv_windows(
                [str(path)], "Chronos2-Small", {"full_native", "balanced"})
            native, horizons = _selection_csv_full_native(
                [str(path)], "Chronos2-Small", {"full_native", "balanced"})

            self.assertEqual(numeric[("Example", "short")], {256})
            self.assertEqual(native, {("Example", "short")})
            self.assertEqual(horizons[("Example", "short")], 64)

    def test_context_stratified_sample_spans_short_and_long_series(self):
        lengths = np.asarray([1, 2, 3, 100, 200, 300])
        selected = _stratified_indices(lengths, 3)
        self.assertEqual(lengths[selected].tolist(), [1, 3, 300])

    def test_old_or_differently_sampled_sidecars_are_not_reused(self):
        valid = {
            "timing_schema_version": timing.TIMING_SCHEMA_VERSION,
            "timing_kind": "numeric_window", "max_series": 256,
            "n_timed_series": 128, "n_repeats": 10,
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "timing.json"
            with patch.object(timing, "_timing_path", return_value=str(path)):
                path.write_text(json.dumps(valid))
                self.assertTrue(timing._timing_done(
                    "D", "M", "short", 256, 10, False,
                    "numeric_window", 256))
                self.assertFalse(timing._timing_done(
                    "D", "M", "short", 256, 10, False,
                    "numeric_window", 512))
                valid.pop("timing_schema_version")
                path.write_text(json.dumps(valid))
                self.assertFalse(timing._timing_done(
                    "D", "M", "short", 256, 10, False,
                    "numeric_window", 256))

    def test_coordinator_assigns_one_physical_gpu_per_dataset_shard(self):
        spawned = []

        class CompletedProcess:
            @staticmethod
            def wait():
                return 0

        def fake_popen(command, env):
            spawned.append((command, env.copy()))
            return CompletedProcess()

        args = SimpleNamespace()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}), \
                patch.object(timing.sys, "argv", ["timing", "--models", "M"]), \
                patch.object(timing.subprocess, "Popen", side_effect=fake_popen), \
                patch.object(timing, "run_timing") as aggregate:
            _run_coordinator(args, "cuda", n_gpus=2, n_visible=2)

        self.assertEqual(len(spawned), 2)
        self.assertEqual([env["CUDA_VISIBLE_DEVICES"] for _, env in spawned],
                         ["2", "3"])
        for shard_id, (command, _env) in enumerate(spawned):
            self.assertEqual(command[-4:], [
                "--shard-id", str(shard_id), "--num-shards", "2"])
        aggregate.assert_called_once_with(
            args, "cuda", shard_id=None, num_shards=1)

    def test_summary_only_stays_single_process_and_cpu(self):
        args = SimpleNamespace(summary_only=True)
        with patch.object(timing, "parse_args", return_value=args), \
                patch.object(timing.torch, "set_grad_enabled"), \
                patch.object(timing, "run_timing") as run_timing, \
                patch.object(timing, "_run_coordinator") as coordinator:
            timing.main()

        run_timing.assert_called_once_with(
            args, "cpu", shard_id=None, num_shards=1)
        coordinator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
