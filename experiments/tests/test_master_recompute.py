from __future__ import annotations

import math
import json
import os
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from experiments import build_context_length_dataset as build
from experiments import datasets_config
from experiments import evaluate_instance_windows as instance_eval
from experiments import master_run_all as master
from experiments import models_config
from experiments import predict_context_length as predictor
from experiments import run_all as run_all_orchestrator
from experiments import run_all_v3 as run_all_v3_orchestrator
from experiments.compare_window_strategies_gifteval import (
    _geomean,
    _instance_oracle_from_cache,
    _load_period_record,
    _mean_theoretical_flops_for_contexts,
    conservative_native_gate,
    compute_flops_savings,
    compute_summary_stats,
    load_strategy_records,
    discover_period_tree,
    theoretical_flops,
)
from experiments.gifteval_mase import gluonts_leaderboard_mase
from experiments.test_window_ablation_gifteval_v5 import (
    ForecastResult, compute_per_sample_metrics)


class MasterRecomputeConfigTest(unittest.TestCase):
    def test_master_period_tree_is_discovered_for_each_predictor_variant(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            period = os.path.join(root, "general_period", "models")
            os.makedirs(period)
            for variant in ("general_v3", "general_v3_classification", "general_v4"):
                self.assertEqual(
                    discover_period_tree(os.path.join(root, variant)),
                    os.path.join(root, "general_period"),
                )

    def test_variant_period_tree_takes_precedence_over_shared_master_tree(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            shared = os.path.join(root, "general_period", "models")
            variant = os.path.join(root, "general_v3_period", "models")
            os.makedirs(shared)
            os.makedirs(variant)
            self.assertEqual(
                discover_period_tree(os.path.join(root, "general_v3")),
                os.path.join(root, "general_v3_period"),
            )

    def test_long_period_mixture_covers_near_context_cycles(self) -> None:
        draws = build._sample_periods(np.random.RandomState(7), size=20_000)
        long_rate = float((draws > build.PERIOD_CORE_MAX).mean())
        self.assertAlmostEqual(long_rate, build.LONG_PERIOD_PROB, delta=0.02)
        self.assertGreater(draws.max(), 8_640)
        self.assertGreater(float((draws < 512).mean()), 0.4)
        self.assertEqual(
            build.synthetic_pool_signature()["period_max"], build.MAX_WINDOW)

    def test_master_force_stage1_regenerates_versioned_pool(self) -> None:
        args = SimpleNamespace(
            stage1_device="cuda",
            stage1_batch_size=None,
            stage1_shard_size=None,
            stage1_windows=None,
            stage1_n_series=None,
        )
        build_args = master._stage1_build_args(args)
        if master._stage_forced(["1"], "1"):
            build_args.append("--regenerate-pool")
        self.assertIn("--regenerate-pool", build_args)

    def test_stage1_explicit_cuda_does_not_fall_back_to_cpu(self) -> None:
        with mock.patch.object(build.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Refusing to silently run"):
                build.resolve_devices("cuda")
        with mock.patch.object(build.torch.cuda, "is_available", return_value=False):
            self.assertEqual(build.resolve_devices(None), ["cpu"])

    def test_all_chronos2_variants_share_independent_recipe(self) -> None:
        chronos2 = {
            spec.display: spec.family
            for spec in models_config.CATALOG
            if spec.display in {
                "Chronos2-Small", "Chronos2-Base", "Chronos2-Synth",
            }
        }
        self.assertEqual(
            chronos2,
            {
                "Chronos2-Small": "chronos2",
                "Chronos2-Base": "chronos2",
                "Chronos2-Synth": "chronos2",
            },
        )
        from experiments.gifteval_inference_recipes import inference_recipe
        self.assertEqual(
            inference_recipe("chronos2"),
            "chronos2_univariate_no_cross_learning_v1",
        )

    def test_requested_predictor_matrix(self) -> None:
        self.assertEqual(
            [v.name for v in master.VARIANTS],
            ["cheap", "mamba"],
        )
        self.assertTrue(all(v.skip_stages == ["1"] for v in master.VARIANTS))
        self.assertEqual(
            [v.ablation_tree for v in master.VARIANTS],
            [
                "general_v3",
                "general_v4",
            ],
        )
        by_name = {v.name: v for v in master.VARIANTS}
        self.assertIn("--cheap", by_name["mamba"].extra)
        self.assertEqual(
            predictor.HP_SPACE_MAMBA_CHEAP["patch_length"],
            predictor.HP_SPACE_PATCHTST_CHEAP["patch_length"],
        )
        self.assertEqual(
            predictor.HP_SPACE_MAMBA_CHEAP["d_model"],
            predictor.HP_SPACE_PATCHTST_CHEAP["d_model"],
        )
        self.assertEqual(
            predictor.HP_SPACE_MAMBA_CHEAP["num_hidden_layers"],
            predictor.HP_SPACE_PATCHTST_CHEAP["num_hidden_layers"],
        )

    def test_trial_resume_rejects_changed_search_config(self) -> None:
        current = predictor.TrialConfig(
            patch_length=64,
            d_model=128,
            num_hidden_layers=2,
            dropout=0.1,
            mask_ratio=0.3,
            learning_rate=1e-4,
            weight_decay=1e-4,
            arch="mamba",
            d_state=16,
            d_conv=4,
            expand=2,
        )
        cached = {
            "label_inference_recipe": "recipe-v1",
            "val_curve_mse": 0.5,
            "cfg": {**current.__dict__, "patch_length": 16},
        }
        self.assertFalse(predictor._cached_trial_is_compatible(
            cached, current, "recipe-v1"))
        cached["cfg"] = current.__dict__
        self.assertTrue(predictor._cached_trial_is_compatible(
            cached, current, "recipe-v1"))

    def test_old_unconstrained_mamba_artifact_is_not_done(self) -> None:
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
                run_all_orchestrator, "PREDICTOR_ROOT", root), mock.patch.dict(
                    os.environ,
                    {
                        "PREDICTCSL_PREDICTOR_ARCH": "mamba",
                        "PREDICTCSL_PREDICTOR_ROOT": root,
                        "PREDICTCSL_CHEAP_PREDICTOR": "1",
                        "PREDICTCSL_TRAINING_OBJECTIVE": "curve",
                    },
                    clear=False):
            run_dir = os.path.join(root, "Chronos2-Small")
            os.makedirs(run_dir)
            open(os.path.join(run_dir, "best_model.pt"), "wb").close()
            with open(os.path.join(run_dir, "best_config.json"), "w") as f:
                json.dump({
                    "arch": "mamba",
                    "training_objective": "curve",
                    "label_inference_recipe": (
                        "chronos2_univariate_no_cross_learning_v1"),
                }, f)

            done, reason = run_all_orchestrator._done_stage_2(
                "chronos2", "Chronos2-Small")

        self.assertFalse(done)
        self.assertIn("unconstrained Mamba", reason)

    def test_model_aware_window_grids(self) -> None:
        self.assertEqual(
            build.WINDOW_GRID[:len(build.BASE_WINDOW_GRID)],
            build.BASE_WINDOW_GRID,
        )
        self.assertEqual(build.window_grid_for_family("timesfm")[-2:],
                         [12288, 15360])
        self.assertEqual(build.window_grid_for_family("chronos2")[-1], 8192)
        self.assertEqual(build.window_grid_for_family("chronos_bolt")[-1], 2048)
        self.assertEqual(build.window_grid_for_family("sundial")[-1], 2880)
        self.assertNotIn(2880, build.window_grid_for_family("chronos2"))
        self.assertEqual(build.window_grid_for_family("tirex")[-1], 8192)

    def test_off_grid_cap_union_reuses_unaffected_old_shards(self) -> None:
        family = "chronos2"
        grid = build.window_grid_for_family(family)
        indices = [build.WINDOW_GRID.index(window) for window in grid]
        with tempfile.TemporaryDirectory() as model_dir:
            shard_dir = os.path.join(model_dir, "shards", "shard_000")
            os.makedirs(shard_dir)
            shape = (2, len(build.BASE_WINDOW_GRID), len(build.HORIZON_GRID))
            np.save(os.path.join(shard_dir, "curves_mae.npy"), np.ones(shape))
            np.save(os.path.join(shard_dir, "curves_mse.npy"), np.ones(shape) * 2)
            with open(os.path.join(shard_dir, "done.json"), "w") as handle:
                json.dump({
                    "start": 0,
                    "end": 2,
                    "window_indices": indices,
                    "horizon_grid": build.HORIZON_GRID,
                    "inference_recipe": build.inference_recipe(family),
                    "synthetic_pool_signature": build.synthetic_pool_signature(),
                }, handle)

            mae, mse, n_done = build.merge_shards(
                model_dir, 2, indices, family)

        self.assertEqual(n_done, 1)
        self.assertEqual(mae.shape, (2, len(grid), len(build.HORIZON_GRID)))
        np.testing.assert_allclose(mae, 1.0)
        np.testing.assert_allclose(mse, 2.0)

    def test_tirex2_uses_dedicated_env(self) -> None:
        self.assertEqual(master.FAMILY_ENV["tirex"], "predictcsl-tirex")
        self.assertIn("predictcsl-tirex", master.ENVS_WITHOUT_MAMBA)
        self.assertEqual(
            master.MAMBA_PREDICTOR_ENV_OVERRIDES["predictcsl-tirex"],
            master.MAIN_ENV,
        )

    def test_patchtst_uses_leaderboard_compatible_env(self) -> None:
        self.assertEqual(
            master.FAMILY_ENV["patchtst_fm"], "predictcsl-patchtst")
        self.assertNotIn("predictcsl-patchtst", master.ENVS_WITHOUT_MAMBA)
        self.assertIn(
            "mamba-ssm==2.2.5",
            master.ENV_PREFLIGHTS["predictcsl-patchtst"],
        )
        self.assertEqual(
            master._resolve_groups(["PatchTST-FM-R1"]),
            {"predictcsl-patchtst": ["PatchTST-FM-R1"]},
        )

    def test_toto_uses_dedicated_env(self) -> None:
        self.assertEqual(master.FAMILY_ENV["toto"], "predictcsl-toto")
        self.assertEqual(
            master._resolve_groups(["Toto-2.0-313m"]),
            {"predictcsl-toto": ["Toto-2.0-313m"]},
        )
        self.assertEqual(
            master.MAMBA_PREDICTOR_ENV_OVERRIDES["predictcsl-toto"],
            master.MAIN_ENV,
        )

    def test_toto_mamba_route_uses_main_env_and_cached_only_guard(self) -> None:
        mamba = next(v for v in master.VARIANTS if v.name == "mamba")
        route = master._variant_route("predictcsl-toto", mamba)
        self.assertEqual(
            route,
            (master.MAIN_ENV, ["--cached-only-ablation"]),
        )
        predictor_env, route_extra = route
        cmd = master._variant_cmd(
            predictor_env,
            mamba,
            ["Toto-2.0-313m"],
            [],
            route_extra,
        )

        self.assertEqual(cmd[0], sys.executable)
        self.assertIn("experiments.run_all_v4", cmd)
        self.assertIn("--cheap", cmd)
        self.assertIn("--cached-only-ablation", cmd)
        self.assertIn("Toto-2.0-313m", cmd)

    def test_tirex_mamba_route_uses_main_env_and_cached_only_guard(self) -> None:
        mamba = next(v for v in master.VARIANTS if v.name == "mamba")
        self.assertEqual(
            master._variant_route("predictcsl-tirex", mamba),
            (master.MAIN_ENV, ["--cached-only-ablation"]),
        )

    def test_dedicated_envs_use_conda_activation(self) -> None:
        old_names = master._CONDA_ENV_NAMES
        master._CONDA_ENV_NAMES = {"predictcsl-legacy"}
        try:
            cmd = master._py(
                "predictcsl-legacy",
                "experiments.run_all",
                "--models",
                "Sundial-Base-128M",
            )
        finally:
            master._CONDA_ENV_NAMES = old_names

        self.assertEqual(cmd[:2], ["bash", "-lc"])
        self.assertIn("conda activate predictcsl-legacy", cmd[2])
        self.assertIn("exec python -m experiments.run_all", cmd[2])
        self.assertNotIn("conda run", cmd)

    def test_dedicated_variant_args_are_inside_activated_shell(self) -> None:
        old_names = master._CONDA_ENV_NAMES
        master._CONDA_ENV_NAMES = {"predictcsl-toto"}
        try:
            cmd = master._variant_cmd(
                "predictcsl-toto",
                master.VARIANTS[0],
                ["Toto-2.0-313m"],
                [],
                ["--force", "3"],
            )
        finally:
            master._CONDA_ENV_NAMES = old_names

        self.assertEqual(cmd[:2], ["bash", "-lc"])
        self.assertEqual(len(cmd), 3)
        self.assertIn("conda activate predictcsl-toto", cmd[2])
        self.assertIn("experiments.run_all_v3", cmd[2])
        self.assertIn("--models Toto-2.0-313m", cmd[2])
        self.assertIn("--force 3", cmd[2])

    def test_verbose_ablation_is_forwarded_to_variant(self) -> None:
        cmd = master._variant_cmd(
            None,
            master.VARIANTS[0],
            ["Chronos2-Small"],
            ["--verbose-ablation"],
            [],
        )
        self.assertIn("--verbose-ablation", cmd)

    def test_variant_can_be_dispatched_for_one_stage(self) -> None:
        cmd = master._variant_cmd(
            None,
            master.VARIANTS[0],
            ["Chronos2-Small"],
            [],
            [],
            only_stage="3",
        )
        self.assertIn("--only-stages", cmd)
        self.assertIn("3", cmd)
        self.assertNotIn("--skip-stages", cmd)

    def test_forecast_precompute_needs_no_predictor(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"PREDICTCSL_ABLATION_ROOT": "/tmp/predictcsl-ablation"},
        ):
            cmd = master._forecast_precompute_cmd(
                None, ["Chronos2-Small"], test=True)
        self.assertIn("--forecast-only", cmd)
        self.assertIn("--cache-root", cmd)
        self.assertIn("/tmp/predictcsl-ablation/general", cmd)
        self.assertIn("--test-datasets", cmd)
        self.assertNotIn("--predictor-dir", cmd)

    def test_forecast_precompute_checkpoint_skips_completed_model(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            expected = len(datasets_config.datasets_to_run())
            with open(os.path.join(
                    root, "leaderboard_parity_summary.json"), "w") as f:
                json.dump({
                    "models": {
                        "Chronos2-Small": {
                            "complete": True,
                            "cells": expected,
                            "expected_cells": expected,
                            "inference_recipe": (
                                "chronos2_univariate_no_cross_learning_v1"),
                        },
                    },
                }, f)

            done, reason = master._forecast_precompute_done(
                root, "Chronos2-Small")

        self.assertTrue(done)
        self.assertIn(f"{expected}/{expected} cells", reason)

    def test_forecast_precompute_checkpoint_rejects_partial_model(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            expected = len(datasets_config.datasets_to_run())
            with open(os.path.join(
                    root, "leaderboard_parity_summary.json"), "w") as f:
                json.dump({
                    "models": {
                        "Chronos2-Small": {
                            "complete": False,
                            "cells": expected - 1,
                            "expected_cells": expected,
                            "inference_recipe": (
                                "chronos2_univariate_no_cross_learning_v1"),
                        },
                    },
                }, f)

            done, reason = master._forecast_precompute_done(
                root, "Chronos2-Small")

        self.assertFalse(done)
        self.assertIn("incomplete/stale cohort", reason)

    def test_forecast_precompute_checkpoint_rejects_stale_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            expected = len(datasets_config.datasets_to_run())
            with open(os.path.join(
                    root, "leaderboard_parity_summary.json"), "w") as f:
                json.dump({
                    "models": {
                        "Chronos2-Small": {
                            "complete": True,
                            "cells": expected,
                            "expected_cells": expected,
                            "inference_recipe": "old-recipe",
                        },
                    },
                }, f)

            done, reason = master._forecast_precompute_done(
                root, "Chronos2-Small")

        self.assertFalse(done)
        self.assertIn("stale model inference recipe", reason)

    def test_chronos2_precompute_reuses_one_dataset_cache(self) -> None:
        groups = master._precompute_model_groups([
            "Chronos2-Small",
            "Moirai2-Small",
            "Chronos2-Synth",
            "Chronos2-Base",
            "TimesFM2.5-200M",
        ])
        self.assertEqual(groups, [
            ["Chronos2-Small", "Chronos2-Synth", "Chronos2-Base"],
            ["Moirai2-Small"],
            ["TimesFM2.5-200M"],
        ])

    def test_dedicated_stage1_args_are_inside_activated_shell(self) -> None:
        old_names = master._CONDA_ENV_NAMES
        master._CONDA_ENV_NAMES = {"predictcsl-legacy"}
        try:
            cmd = master._stage1_cmd(
                "predictcsl-legacy",
                ["Sundial-Base-128M"],
                ["-v"],
                [],
                ["--n-series", "200"],
            )
        finally:
            master._CONDA_ENV_NAMES = old_names

        self.assertEqual(cmd[:2], ["bash", "-lc"])
        self.assertEqual(len(cmd), 3)
        self.assertIn("conda activate predictcsl-legacy", cmd[2])
        self.assertIn("--models Sundial-Base-128M", cmd[2])
        self.assertIn("--build-args --n-series 200", cmd[2])

    def test_master_stage1_build_args(self) -> None:
        args = SimpleNamespace(
            stage1_device="cuda",
            stage1_batch_size=8,
            stage1_shard_size=50,
            stage1_windows=[32, 64, 128],
            stage1_n_series=1000,
        )
        self.assertEqual(
            master._stage1_build_args(args),
            [
                "--device", "cuda",
                "--batch-size", "8",
                "--shard-size", "50",
                "--windows", "32", "64", "128",
                "--n-series", "1000",
            ],
        )

    def test_master_force_targets_only_requested_stage(self) -> None:
        self.assertFalse(master._stage_forced(None, "3"))
        self.assertTrue(master._stage_forced([], "3"))
        self.assertTrue(master._stage_forced(["3"], "3"))
        self.assertFalse(master._stage_forced(["2"], "3"))

    def test_sanity_check_geomean_rules(self) -> None:
        self.assertAlmostEqual(_geomean(np.array([1.0, 4.0, np.nan])), 2.0)
        self.assertEqual(_geomean(np.array([0.0, 4.0])), 0.0)
        self.assertTrue(math.isnan(_geomean(np.array([-1.0, 4.0]))))

    def test_summary_stats_accepts_current_model_specs(self) -> None:
        stats = compute_summary_stats(pd.DataFrame([{
            "model": "autogluon/chronos-2-small",
            "full_mase": 1.0,
            "best_mase": 0.8,
            "pred_mase": 0.9,
            "pred_clamped": False,
            "rel_gain_pred_over_full": 0.1,
            "delta_pred_vs_best": 0.1,
            "full_elapsed_s": 2.0,
            "best_elapsed_s": 1.0,
            "pred_elapsed_s": 1.5,
            "speedup_pred_vs_full": 4.0 / 3.0,
            "complexity_ratio_pred_vs_full": 0.75,
        }]))

        self.assertEqual(
            stats["inference_recipes"]["chronos2"],
            "chronos2_univariate_no_cross_learning_v1",
        )

    def test_strategy_comparison_uses_sanity_metrics_not_npz_copies(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            run_dir = os.path.join(root, "general_v3")
            compare_dir = os.path.join(
                run_dir, "models", "Chronos2-Small",
                "compare_real_vs_predicted",
            )
            os.makedirs(compare_dir)
            pd.DataFrame([{
                "dataset_display": "M4-Yearly",
                "term": "short",
                "model_short": "Chronos2-Small",
                "model": "autogluon/chronos-2-small",
                "horizon_real": 6,
                "n_instances": 10,
            }]).to_csv(os.path.join(compare_dir, "compare_summary.csv"), index=False)
            # Deliberately stale embedded curves: stage 4 must ignore these MASE
            # values while retaining predicted_mean as the window selector.
            np.savez_compressed(
                os.path.join(
                    compare_dir,
                    "compare_M4-Yearly_tshort_Chronos2-Small.npz",
                ),
                window_grid=np.array([32, 64, 8192]),
                real_curve_gluonts_real=np.array([9.0, 8.0, np.nan]),
                real_curve_gluonts=np.array([9.0, 8.0, np.nan]),
                predicted_mean=np.array([1.0, 0.0, 2.0]),
            )

            recipe = "chronos2_univariate_no_cross_learning_v1"
            common = {
                "_metric_suite_ver": 3,
                "_mase_gluonts_real_standin": False,
                "_inference_recipe": recipe,
                "elapsed_seconds": 1.0,
            }
            for window, mase in ((32, 0.8), (64, 1.1)):
                metric_dir = os.path.join(
                    run_dir, "datasets", "M4-Yearly", "Chronos2-Small",
                    "tshort", f"w{window}",
                )
                os.makedirs(metric_dir)
                with open(os.path.join(metric_dir, "metrics.json"), "w") as f:
                    json.dump({**common, "mase_gluonts_real": mase}, f)
            # An obsolete unsupported-window file may remain on disk. The
            # current Stage-3 curve marks it NaN, so Stage 4 must ignore it.
            stale_dir = os.path.join(
                run_dir, "datasets", "M4-Yearly", "Chronos2-Small",
                "tshort", "w8192",
            )
            os.makedirs(stale_dir)
            with open(os.path.join(stale_dir, "metrics.json"), "w") as f:
                json.dump({
                    **common,
                    "_metric_suite_ver": 0,
                    "mase_gluonts_real": 99.0,
                }, f)
            native_dir = os.path.join(
                run_dir, "datasets", "M4-Yearly", "Chronos2-Small",
                "tshort", "wfull_native",
            )
            os.makedirs(native_dir)
            with open(os.path.join(native_dir, "metrics.json"), "w") as f:
                json.dump({
                    **common,
                    "mase_gluonts_real": 0.7,
                    "_context_cap": 64,
                    "_mean_effective_context": 48.0,
                    "_n_width_groups": 2,
                }, f)

            result = load_strategy_records(
                run_dir, run_dir, {}, mase_metric="mase_gluonts_real")

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertAlmostEqual(row["full_mase"], 0.7)
        # The dataset oracle includes native-full as a candidate, so it cannot
        # report a negative improvement merely because every grid point loses.
        self.assertAlmostEqual(row["best_mase"], 0.7)
        self.assertAlmostEqual(row["pred_mase"], 1.1)
        self.assertTrue(row["best_is_full_native"])
        # The available dataset context (64) is retained; Chronos2's nominal
        # 8192 limit must never make a short dataset pretend to have more history.
        self.assertEqual(row["full_window"], 64)
        self.assertEqual(row["best_window"], 64)
        self.assertEqual(row["pred_window"], 64)
        self.assertEqual(row["full_baseline_source"], "full_native_sanity")

    def test_stage4_reads_metrics_from_active_run_tree(self) -> None:
        old_root = run_all_orchestrator.ABLATION_ROOT
        old_general = run_all_orchestrator.ABLATION_GENERAL
        old_subdir = run_all_orchestrator.STRATEGY_SUBDIR
        try:
            run_all_orchestrator.ABLATION_ROOT = "/tmp/ablation"
            run_all_orchestrator.ABLATION_GENERAL = "/tmp/ablation/general_v3"
            run_all_orchestrator.STRATEGY_SUBDIR = "strategy_comparison_v3"
            with mock.patch.object(
                    run_all_orchestrator, "_run", return_value=0) as run:
                run_all_orchestrator.stage_4_compare(
                    "Chronos2-Small", "chronos2", [])
        finally:
            run_all_orchestrator.ABLATION_ROOT = old_root
            run_all_orchestrator.ABLATION_GENERAL = old_general
            run_all_orchestrator.STRATEGY_SUBDIR = old_subdir

        cmd = run.call_args.args[0]
        self.assertNotIn("--cache-root", cmd)
        self.assertEqual(
            cmd[cmd.index("--run-dir") + 1],
            "/tmp/ablation/general_v3",
        )

    def test_stage4_rollup_is_restricted_to_selected_models(self) -> None:
        with mock.patch.object(
                run_all_orchestrator, "_run", return_value=0) as run:
            run_all_orchestrator.stage_4_rollup(
                [], models=["Chronos2-Small"])

        cmd = run.call_args.args[0]
        self.assertIn("--rollup-only", cmd)
        self.assertEqual(
            cmd[cmd.index("--models") + 1:],
            ["Chronos2-Small"],
        )

    def test_v3_output_root_configures_one_self_contained_tree(self) -> None:
        old_dataset = run_all_orchestrator.DATASET_ROOT
        old_ablation = run_all_orchestrator.ABLATION_ROOT
        old_logs = run_all_orchestrator.RUN_LOG_ROOT
        try:
            with mock.patch.dict(os.environ, {}, clear=False):
                root = run_all_v3_orchestrator._apply_output_root(
                    "logs/experiments/master_recompute/")
                self.assertEqual(root, "logs/experiments/master_recompute")
                self.assertEqual(
                    run_all_orchestrator.DATASET_ROOT,
                    "logs/experiments/master_recompute/context_length_dataset",
                )
                self.assertEqual(
                    run_all_orchestrator.ABLATION_ROOT,
                    "logs/experiments/master_recompute/window_ablation_gifteval",
                )
                self.assertEqual(
                    run_all_orchestrator.RUN_LOG_ROOT,
                    "logs/experiments/master_recompute/run_all_logs",
                )
        finally:
            run_all_orchestrator.DATASET_ROOT = old_dataset
            run_all_orchestrator.ABLATION_ROOT = old_ablation
            run_all_orchestrator.RUN_LOG_ROOT = old_logs


class PerInstanceWindowEvaluationTest(unittest.TestCase):
    def test_ineligible_stale_window_does_not_require_served_index(self) -> None:
        """Moirai-like stale caches outside Stage 3's mask are ignored."""
        with tempfile.TemporaryDirectory() as root:
            compare_dir = os.path.join(
                root, "general_v3", "models", "Moirai2-Small",
                "compare_real_vs_predicted")
            os.makedirs(compare_dir)
            anchor_path = os.path.join(
                compare_dir, "compare_Example_tshort_Moirai2-Small.npz")
            np.savez_compressed(
                anchor_path,
                window_grid=np.asarray([4096, 8192]),
                predicted_curves=np.zeros((2, 2)),
                real_curve_gluonts_real=np.asarray([1.0, np.nan]),
            )
            base = os.path.join(
                root, "general_v3", "datasets", "Example",
                "Moirai2-Small", "tshort")
            for window, values, served in (
                (4096, [1.0, 2.0], [0, 1]),
                ("full_native", [1.5, 2.5], [0, 1]),
            ):
                path = os.path.join(base, f"w{window}")
                os.makedirs(path)
                np.savez_compressed(
                    os.path.join(path, "per_sample_metrics.npz"),
                    mase_gluonts_real=np.asarray(values),
                    valid_count=np.ones(2, dtype=np.int32),
                    served_index=np.asarray(served, dtype=np.int32),
                    effective_context=np.asarray([4096, 4096]),
                )
            # Obsolete unsupported cache: deliberately unalignable and lacking
            # served_index. Its presence must have no effect on evaluation.
            stale = os.path.join(base, "w8192")
            os.makedirs(stale)
            np.savez_compressed(
                os.path.join(stale, "per_sample_metrics.npz"),
                mase_gluonts_real=np.asarray([99.0]),
                valid_count=np.ones(1, dtype=np.int32),
            )
            cell = instance_eval.Cell(
                "Moirai2-Small", "Example", "short", anchor_path)

            records, _audit = instance_eval.evaluate_cell(
                cell, root, os.path.join(root, "general_v3"))

        self.assertTrue(records)
        self.assertNotIn(99.0, [record["mase_gluonts"] for record in records])

    def test_comparable_curve_carries_forward_previous_window(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = os.path.join(
                root, "datasets", "Example", "Chronos2-Small", "tshort")

            def save(window, values, served):
                path = os.path.join(base, f"w{window}")
                os.makedirs(path)
                np.savez_compressed(
                    os.path.join(path, "per_sample_metrics.npz"),
                    mase_gluonts_real=np.asarray(values, dtype=float),
                    valid_count=np.ones(len(values), dtype=np.int32),
                    served_index=np.asarray(served, dtype=np.int32),
                )

            save(32, [3.0, 5.0], [0, 1])
            save(64, [1.0], [0])
            save("full_native", [2.0, 4.0], [0, 1])

            result = _instance_oracle_from_cache(
                root, "Example", "Chronos2-Small", "short",
                np.array([32, 64]), np.array([True, True]), 2,
                "mase_gluonts_real")

        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["comparable_curve"], [4.0, 3.0])
        self.assertAlmostEqual(result["instance_oracle_mase"], 2.5)
        self.assertTrue(result["instance_oracle_metric_exact"])

    def test_discovery_includes_complete_gifteval_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            compare_dir = os.path.join(
                root, "general_v3", "models", "TimesFM2.5-200M",
                "compare_real_vs_predicted",
            )
            os.makedirs(compare_dir)
            for dataset in ("Solar-W", "CarParts", "M4-Yearly"):
                path = os.path.join(
                    compare_dir,
                    f"compare_{dataset}_tshort_TimesFM2.5-200M.npz",
                )
                with open(path, "wb"):
                    pass

            cells = instance_eval.discover_cells(root, None)

        self.assertEqual(
            [(c.dataset, c.term) for c in cells],
            [("CarParts", "short"), ("M4-Yearly", "short"),
             ("Solar-W", "short")],
        )

    def test_predictor_choice_is_per_row_and_masks_unavailable_windows(self) -> None:
        windows = np.array([32, 64])
        errors = np.array([
            [1.0, 0.5],
            [2.0, np.nan],
            [np.nan, np.nan],
        ])
        scores = np.array([
            [0.0, 1.0],  # row 0 chooses 32
            [9.0, 0.0],  # row 1 wants 64, but it is unavailable
            [0.0, 1.0],  # no grid window -> full-native
        ])
        native = np.array([3.0, 4.0, 5.0])

        selected, selected_w, fallback = instance_eval._choose_scores(
            scores, errors, windows, native)

        np.testing.assert_allclose(selected, [1.0, 2.0, 5.0])
        np.testing.assert_array_equal(selected_w, [32, 32, -1])
        np.testing.assert_array_equal(fallback, [False, False, True])

    def test_predictor_can_choose_native_full_for_each_instance(self) -> None:
        windows = np.array([32, 64])
        errors = np.array([
            [2.0, 3.0],
            [4.0, np.nan],
        ])
        scores = np.array([
            [1.0, -2.0],  # largest-window score selects native full
            [0.0, -3.0],  # unavailable 64 score still represents native full
        ])
        native = np.array([1.0, 2.0])
        native_w = np.array([80, 48])

        selected, selected_w, selected_native = (
            instance_eval._choose_scores_with_native(
                scores, errors, windows, native, native_w))

        np.testing.assert_allclose(selected, [1.0, 2.0])
        np.testing.assert_array_equal(selected_w, [80, 48])
        np.testing.assert_array_equal(selected_native, [True, True])

    def test_instance_record_uses_gifteval_valid_count_weights(self) -> None:
        cell = instance_eval.Cell(
            "Chronos2-Small", "Example", "short", "unused.npz")
        record = instance_eval._record(
            cell=cell,
            method="predictor",
            error=np.array([1.0, 3.0]),
            window=np.array([32, 64]),
            fallback=np.array([False, False]),
            method_kind="predictor_instance",
            valid_count=np.array([1.0, 3.0]),
            metric_source="mase_gluonts_real",
        )

        self.assertAlmostEqual(record["mase_gluonts"], 2.5)
        self.assertTrue(record["mase_metric_exact"])
        self.assertEqual(record["policy_scope"], "instance_native")

    def test_fixed_window_is_capped_per_instance(self) -> None:
        windows = np.array([32, 64, 128])
        errors = np.array([
            [3.0, 2.0, 1.0],
            [4.0, 3.0, np.nan],
        ])
        native = np.array([9.0, 9.0])

        selected, selected_w, fallback = instance_eval._choose_capped_fixed(
            128, errors, windows, native)

        np.testing.assert_allclose(selected, [1.0, 3.0])
        np.testing.assert_array_equal(selected_w, [128, 64])
        self.assertFalse(fallback.any())

    def test_per_instance_mase_ignores_missing_horizon_points(self) -> None:
        forecast = ForecastResult(median=torch.tensor([
            [1.0, 5.0],
            [2.0, 3.0],
        ]))
        targets = torch.tensor([
            [3.0, float("nan")],
            [4.0, 7.0],
        ])
        metrics = compute_per_sample_metrics(
            forecast, targets, seasonal_errors=np.array([2.0, 2.0]))

        np.testing.assert_allclose(metrics["mae"], [2.0, 3.0])
        np.testing.assert_allclose(metrics["mase_gluonts"], [1.0, 1.5])
        np.testing.assert_allclose(metrics["mase_gluonts_real"], [1.0, 1.5])
        np.testing.assert_array_equal(metrics["valid_count"], [1, 2])

    def test_per_instance_real_mase_uses_gluonts_upper_middle_sample(self) -> None:
        forecast = ForecastResult(
            median=torch.tensor([[1.0]]),
            samples=torch.tensor([[[0.0], [1.0], [2.0], [100.0]]]),
        )
        metrics = compute_per_sample_metrics(
            forecast, torch.tensor([[2.0]]),
            seasonal_errors=np.array([1.0]))

        self.assertAlmostEqual(metrics["mase_gluonts"][0], 1.0)
        self.assertAlmostEqual(metrics["mase_gluonts_real"][0], 0.0)


class GluonTSMaseVectorizedTest(unittest.TestCase):
    def test_global_point_aggregation_with_missing_labels(self) -> None:
        contexts = [np.array([0.0, 2.0, 4.0]),
                    np.array([0.0, 10.0, 20.0])]
        labels = np.array([[2.0, 4.0], [10.0, np.nan]])
        median = np.zeros((2, 2))

        value = gluonts_leaderboard_mase(
            median, [None, None], contexts, labels, "D")

        # Scaled valid point errors are [1, 2, 1], so GluonTS axis=None is 4/3.
        self.assertAlmostEqual(value, 4.0 / 3.0)

    def test_even_sample_forecast_uses_gluonts_upper_middle_index(self) -> None:
        samples = np.array([[[0.0], [1.0], [2.0], [100.0]]])
        value = gluonts_leaderboard_mase(
            np.array([[1.0]]), [None], [np.array([0.0, 1.0])],
            np.array([[2.0]]), "D", samples=samples)

        # GluonTS uses sorted_samples[round((S-1)*.5)] = index 2, prediction 2.
        self.assertEqual(value, 0.0)


class LeaderboardCheckpointTest(unittest.TestCase):
    def test_checkpoint_persists_stage_and_official_aggregation(self) -> None:
        from experiments import test_window_ablation_gifteval_v5 as ablation

        datasets = [
            ("dataset_one", "short", "Dataset-One", False),
            ("dataset_two", "long", "Dataset-Two", False),
        ]
        models = [("example/model", "example_family", "Example-Model")]
        naive = {
            ("dataset_one", "short"): 1.0,
            ("dataset_two", "long"): 2.0,
        }
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(ablation, "CACHE_ROOT", root), \
                mock.patch.object(
                    ablation, "published_naive_record",
                    side_effect=lambda name, term: {
                        "mase_gluonts_real": naive[(name, term)]
                    },
                ):
            for _name, term, display, _univariate in datasets:
                value = 2.0 if display == "Dataset-One" else 8.0
                ablation._save_result(
                    display, "Example-Model", term,
                    ablation.FULL_NATIVE_WINDOW,
                    {"mae": 1.0, "mse": 1.0, "rmse": 1.0,
                     "mase_gluonts_real": value,
                     "_mase_gluonts_real_standin": False},
                )
            path = ablation._write_leaderboard_parity_summary(
                models, datasets, root,
                checkpoint_stage="stage_3_gift_eval_ablation")
            with open(path) as f:
                payload = json.load(f)

        self.assertEqual(
            payload["last_checkpoint_stage"], "stage_3_gift_eval_ablation")
        record = payload["models"]["Example-Model"]
        self.assertTrue(record["complete"])
        self.assertAlmostEqual(record["geomean_normalized_mase"], math.sqrt(8.0))


class Chronos2IndependenceTest(unittest.TestCase):
    class FakePipeline:
        quantiles = [0.1, 0.5, 0.9]

        def __init__(self) -> None:
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            inputs = kwargs["inputs"]
            horizon = kwargs["prediction_length"]
            return torch.zeros(
                (inputs.shape[0], len(self.quantiles), horizon),
                dtype=torch.float32,
            )

    def test_stage1_disables_cross_learning(self) -> None:
        pipeline = self.FakePipeline()
        out = build.predict_chronos2(
            pipeline, torch.zeros((2, 32, 1)), horizon=4, device="cpu")
        self.assertEqual(tuple(out.shape), (2, 4))
        self.assertIs(pipeline.calls[0]["cross_learning"], False)

    def test_stage1_returns_median_on_requested_device(self) -> None:
        pipeline = self.FakePipeline()
        out = build.predict_chronos2(
            pipeline, torch.zeros((2, 32, 1)), horizon=4, device="meta")
        self.assertEqual(out.device.type, "meta")

    def test_gifteval_ablation_disables_cross_learning(self) -> None:
        from experiments import test_window_ablation_gifteval_v5 as ablation

        pipeline = self.FakePipeline()
        batches = [{
            "x": torch.zeros((2, 32, 1)),
            "y": torch.zeros((2, 4, 1)),
        }]
        forecast, _targets = ablation.predict_chronos2(
            pipeline, batches, horizon=4, device="cpu", batch_size=2)
        self.assertEqual(tuple(forecast.median.shape), (2, 4))
        self.assertIs(pipeline.calls[0]["cross_learning"], False)


class AblationDynamicBatchTest(unittest.TestCase):
    def test_deterministic_family_runs_once_without_autotune_probe(self) -> None:
        from experiments import test_window_ablation_gifteval_v5 as ablation

        calls = []

        def fake_forecast(
            _family, _handle, _model_id, batches, _width, horizon,
            _device, batch_size, **_kwargs,
        ):
            calls.append(batch_size)
            n = sum(batch["x"].shape[0] for batch in batches)
            zeros = torch.zeros((n, horizon))
            return ForecastResult(median=zeros), zeros

        batches = [{
            "x": torch.zeros((20, 1024, 1)),
            "y": torch.zeros((20, 4, 1)),
        }]
        with mock.patch.object(ablation, "_forecast_cell", fake_forecast):
            _fr, _targets, size = ablation._forecast_cell_dynamic(
                "flowstate", object(), "example/model", batches,
                1024, 4, "cuda", 2)

        self.assertEqual(calls, [2])
        self.assertEqual(size, 2)

    def test_sampling_family_does_not_probe(self) -> None:
        from experiments import test_window_ablation_gifteval_v5 as ablation

        calls = []

        def fake_forecast(
            _family, _handle, _model_id, batches, _width, horizon,
            _device, batch_size, **_kwargs,
        ):
            calls.append(batch_size)
            n = sum(batch["x"].shape[0] for batch in batches)
            zeros = torch.zeros((n, horizon))
            return ForecastResult(median=zeros), zeros

        batches = [{
            "x": torch.zeros((20, 1024, 1)),
            "y": torch.zeros((20, 4, 1)),
        }]
        with mock.patch.object(ablation, "_forecast_cell", fake_forecast):
            ablation._forecast_cell_dynamic(
                "sundial", object(), "example/model", batches,
                1024, 4, "cuda", 2)

        self.assertEqual(calls, [2])

    def test_short_context_uses_ceiling_scaled_batch(self) -> None:
        from experiments import test_window_ablation_gifteval_v5 as ablation

        calls = []

        def fake_forecast(
            _family, _handle, _model_id, batches, _width, horizon,
            _device, batch_size, **_kwargs,
        ):
            calls.append(batch_size)
            n = sum(batch["x"].shape[0] for batch in batches)
            zeros = torch.zeros((n, horizon))
            return ForecastResult(median=zeros), zeros

        batches = [{
            "x": torch.zeros((192, 384, 1)),
            "y": torch.zeros((192, 4, 1)),
        }]
        with mock.patch.object(ablation, "_forecast_cell", fake_forecast):
            _fr, _targets, size = ablation._forecast_cell_dynamic(
                "timesfm", object(), "example/model", batches,
                384, 4, "cuda", 32)

        self.assertEqual(calls, [96])
        self.assertEqual(size, 96)

    def test_oom_halves_batch_and_retries_without_caching(self) -> None:
        from experiments import test_window_ablation_gifteval_v5 as ablation

        calls = []

        def fake_forecast(
            _family, _handle, _model_id, batches, _width, horizon,
            _device, batch_size, **_kwargs,
        ):
            calls.append(batch_size)
            if batch_size > 4:
                raise torch.cuda.OutOfMemoryError("synthetic OOM")
            n = sum(batch["x"].shape[0] for batch in batches)
            zeros = torch.zeros((n, horizon))
            return ForecastResult(median=zeros), zeros

        batches = [{
            "x": torch.zeros((20, 256, 1)),
            "y": torch.zeros((20, 4, 1)),
        }]
        with mock.patch.object(ablation, "_forecast_cell", fake_forecast), \
                mock.patch.object(ablation, "_clear_accelerator_cache"):
            _fr, _targets, size = ablation._forecast_cell_dynamic(
                "flowstate", object(), "example/model", batches,
                256, 4, "cuda", 2)

        self.assertEqual(calls, [8, 4])
        self.assertEqual(size, 4)

class PeriodStrategyAccountingTest(unittest.TestCase):
    def test_period_sidecar_exposes_per_series_windows(self) -> None:
        with tempfile.TemporaryDirectory() as compare_dir:
            stem = "period_Example_tshort_ExampleModel"
            with open(os.path.join(compare_dir, stem + ".json"), "w") as handle:
                json.dump({
                    "window_mean": 80.0,
                    "period_policy_version": 2,
                }, handle)
            np.savez_compressed(
                os.path.join(compare_dir, stem + "_win.npz"),
                windows=np.array([32, -1, 128], dtype=np.int64),
            )

            record = _load_period_record(
                compare_dir, "Example", "short", "ExampleModel", 2)

        np.testing.assert_array_equal(record["_windows"], [32, -1, 128])

    def test_period_flops_average_actions_not_mean_window(self) -> None:
        model = "autogluon/chronos-2-small"
        contexts = np.array([32, 8192], dtype=np.int64)
        actual = _mean_theoretical_flops_for_contexts(
            model, contexts, horizon=64, patch_sizes={})
        expected = np.mean([
            theoretical_flops(model, int(w), 64, {}) for w in contexts
        ])
        at_mean_window = theoretical_flops(
            model, int(contexts.mean()), 64, {})

        self.assertAlmostEqual(actual, expected)
        self.assertNotAlmostEqual(actual, at_mean_window)

    def test_period_uses_full_native_model_caps(self) -> None:
        from experiments.archive.heuristics.period_window_eval import _family_cap

        self.assertEqual(_family_cap("sundial", 128), 2880)
        self.assertEqual(_family_cap("toto", 128), 4096)
        self.assertLess(_family_cap("moirai", 1024), 8192)


class ConservativeStrategyTest(unittest.TestCase):
    def test_native_gate_requires_confident_practical_advantage(self) -> None:
        # Candidate index 0 is consistently one score unit better than full.
        curves = np.array([
            [0.0, 1.0],
            [0.1, 1.1],
            [-0.1, 0.9],
        ])
        shorten, mean, _se, lower = conservative_native_gate(
            curves, pred_idx=0, full_idx=1)
        self.assertTrue(shorten)
        self.assertAlmostEqual(mean, 1.0)
        self.assertGreater(lower, 0.1)

        uncertain = np.array([
            [0.0, 1.0],
            [2.0, 1.0],
        ])
        self.assertFalse(conservative_native_gate(
            uncertain, pred_idx=0, full_idx=1)[0])

    def test_flops_csv_names_own_primary_strategy_and_safe_row(self) -> None:
        frame = pd.DataFrame({
            "primary_strategy": ["pred_mamba_cls"],
            "n_instances": [2],
            "full_flops": [100.0],
            "pred_flops": [40.0],
            "safe_flops": [100.0],
            "best_flops": [30.0],
            "full_mase": [1.0],
            "pred_mase": [0.9],
            "safe_mase": [1.0],
            "best_mase": [0.8],
            # Own canonical columns also exist in comparison.csv; savings must
            # not duplicate the primary row.
            "pred_mamba_cls_flops": [40.0],
            "pred_mamba_cls_mase": [0.9],
        })
        rows = compute_flops_savings(frame)
        self.assertEqual(
            rows["strategy"].tolist(),
            ["pred_mamba_cls", "safe", "best"],
        )


class SoftClassificationLossTest(unittest.TestCase):
    def test_top3_rank_weights_and_invalid_class_mask(self) -> None:
        old_objective = predictor.TRAINING_OBJECTIVE
        predictor.TRAINING_OBJECTIVE = "classification"
        try:
            logits = torch.tensor([[3.0, 2.0, 1.0, 9.0]], requires_grad=True)
            # Lower raw error is more accurate. Class 3 is unavailable.
            target = torch.tensor([[0.1, 0.2, 0.3, float("nan")]])
            recon = torch.zeros(1, 1, 1)
            mask = torch.zeros(1, 1, dtype=torch.bool)
            loss, task_loss, _ = predictor.compute_dual_loss(
                logits, recon, recon, mask, target, 1.0, 0.0)

            expected_soft = torch.tensor([[1.0, 0.5, 0.25]])
            expected = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, :3], expected_soft)
            self.assertTrue(torch.allclose(task_loss, expected))
            self.assertTrue(torch.allclose(loss, expected))

            loss.backward()
            self.assertEqual(float(logits.grad[0, 3]), 0.0)
        finally:
            predictor.TRAINING_OBJECTIVE = old_objective


class RiskAwareLossTest(unittest.TestCase):
    def test_risk_trial_cache_tracks_loss_and_selection_signature(self) -> None:
        trial = predictor.TrialConfig(
            patch_length=64,
            d_model=128,
            num_hidden_layers=2,
            dropout=0.1,
            mask_ratio=0.3,
            learning_rate=1e-4,
            weight_decay=1e-4,
            num_attention_heads=4,
        )
        cached = {
            "label_inference_recipe": "recipe",
            "cfg": predictor.asdict(trial),
            "val_curve_mse": 1.0,
            "val_risk_score": 1.0,
            "risk_selection_signature": {"version": 0},
        }
        old_objective = predictor.TRAINING_OBJECTIVE
        predictor.TRAINING_OBJECTIVE = "risk"
        try:
            self.assertFalse(predictor._cached_trial_is_compatible(
                cached, trial, "recipe"))
            cached["risk_selection_signature"] = (
                predictor.risk_selection_signature())
            self.assertTrue(predictor._cached_trial_is_compatible(
                cached, trial, "recipe"))
        finally:
            predictor.TRAINING_OBJECTIVE = old_objective

    def test_risk_validation_selection_penalizes_tail_harm(self) -> None:
        class UnsafeModel(torch.nn.Module):
            mask_ratio = 0.0
            num_patches = 1

            def forward(self, x, horizon_idx, mask=None, valid_length=None):
                batch = x.shape[0]
                scores = torch.tensor([[-1.0, 0.0]], device=x.device).repeat(
                    batch, 1)
                patches = torch.zeros(batch, 1, 1, device=x.device)
                used_mask = torch.zeros(
                    batch, 1, dtype=torch.bool, device=x.device)
                return scores, patches, patches, used_mask

        old_objective = predictor.TRAINING_OBJECTIVE
        predictor.TRAINING_OBJECTIVE = "risk"
        try:
            metrics = predictor._evaluate(
                UnsafeModel(),
                x_val=torch.zeros(1, 1, 1),
                y_val_norm=torch.zeros(1, 2, 1),
                y_val_raw=torch.tensor([[[2.0], [1.0]]]),
                length_val=torch.ones(1, dtype=torch.long),
                batch_size=1,
                eval_seed=1,
            )

            self.assertAlmostEqual(metrics["val_regret"], 1.0)
            self.assertAlmostEqual(metrics["val_harm_p90"], 1.0)
            self.assertAlmostEqual(metrics["val_harmed_rate"], 1.0)
            self.assertGreater(metrics["val_risk_score"], metrics["val_regret"])
            self.assertEqual(
                predictor._active_selection_metric_key(), "val_risk_score")
        finally:
            predictor.TRAINING_OBJECTIVE = old_objective

    def test_full_optimal_curve_penalizes_harmful_shortening(self) -> None:
        raw = torch.tensor([[2.0, 1.5, 1.0]])
        calibrated = torch.log(raw / raw[:, -1:]).requires_grad_()
        unsafe = torch.tensor([[-2.0, 0.0, 1.0]], requires_grad=True)

        safe_loss = predictor.risk_aware_task_loss(calibrated, raw)
        unsafe_loss = predictor.risk_aware_task_loss(unsafe, raw)

        self.assertGreater(
            float(unsafe_loss.detach()), float(safe_loss.detach()))
        unsafe_loss.backward()
        self.assertTrue(torch.isfinite(unsafe.grad).all())

    def test_risk_objective_accepts_missing_windows(self) -> None:
        old_objective = predictor.TRAINING_OBJECTIVE
        predictor.TRAINING_OBJECTIVE = "risk"
        try:
            pred = torch.zeros(2, 3, requires_grad=True)
            raw = torch.tensor([
                [1.0, 2.0, float("nan")],
                [3.0, float("nan"), 2.0],
            ])
            recon = torch.zeros(2, 1, 1)
            mask = torch.zeros(2, 1, dtype=torch.bool)
            loss, task, _ = predictor.compute_dual_loss(
                pred, recon, recon, mask, raw, 1.0, 0.0)

            self.assertTrue(torch.isfinite(task))
            loss.backward()
            self.assertTrue(torch.isfinite(pred.grad).all())
        finally:
            predictor.TRAINING_OBJECTIVE = old_objective


class PatchTSTFMCompatibilityTest(unittest.TestCase):
    def test_patchtst_recipe_invalidates_padding_safe_v2_caches(self) -> None:
        from experiments.gifteval_inference_recipes import inference_recipe

        self.assertEqual(
            inference_recipe("patchtst_fm"),
            "official_patchtst_fm_wrapper_v3",
        )

    def test_legacy_official_api_keeps_inputs_on_cpu(self) -> None:
        class LegacyGraniteAPI(torch.nn.Module):
            def forward(self, inputs, prediction_length, quantile_levels):
                self.seen = inputs
                forecast = torch.zeros(
                    len(inputs), len(quantile_levels), prediction_length)
                forecast[:, build.PATCHTST_FM_MEDIAN_QUANTILE_IDX, :] = 3.0
                return SimpleNamespace(quantile_predictions=forecast)

        model = LegacyGraniteAPI()
        result = build.predict_patchtst_fm(
            model, torch.ones(2, 5, 1), horizon=3, device="cpu")

        self.assertTrue(all(row.device.type == "cpu" for row in model.seen))
        self.assertTrue(torch.equal(result, torch.full((2, 3), 3.0)))

    def test_official_quantile_head_and_list_input(self) -> None:
        class NewGraniteAPI(torch.nn.Module):
            def forward(self, past_values, prediction_length, quantile_levels):
                self.seen = past_values
                forecast = torch.zeros(
                    len(past_values), len(quantile_levels), prediction_length, 1,
                )
                forecast[:, build.PATCHTST_FM_MEDIAN_QUANTILE_IDX, :, 0] = 2.0
                return SimpleNamespace(quantile_outputs=forecast)

        model = NewGraniteAPI()
        result = build.predict_patchtst_fm(
            model, torch.ones(2, 5, 1), horizon=3, device="cpu")

        self.assertIsInstance(model.seen, list)
        self.assertEqual([tuple(x.shape) for x in model.seen], [(5,), (5,)])
        self.assertEqual(tuple(result.shape), (2, 3))
        self.assertTrue(torch.equal(result, torch.full((2, 3), 2.0)))
        self.assertFalse(result.requires_grad)

    def test_mean_imputes_missing_values(self) -> None:
        class API(torch.nn.Module):
            def forward(self, past_values, prediction_length, quantile_levels):
                self.seen = past_values
                forecast = torch.ones(
                    len(past_values), len(quantile_levels), prediction_length, 1)
                return SimpleNamespace(quantile_outputs=forecast)

        model = API()
        x = torch.tensor([[[1.0], [float("nan")], [3.0]]])
        result = build.predict_patchtst_fm(
            model, x, horizon=2, device="cpu")

        self.assertTrue(torch.equal(model.seen[0], torch.tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(result, torch.ones(1, 2)))


class TiRexCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_backend = os.environ.pop("PREDICTCSL_TIREX_BACKEND", None)

    def tearDown(self) -> None:
        if self.old_backend is not None:
            os.environ["PREDICTCSL_TIREX_BACKEND"] = self.old_backend
        else:
            os.environ.pop("PREDICTCSL_TIREX_BACKEND", None)

    def test_catalog_uses_published_gifteval_zero_shot_checkpoint(self) -> None:
        tirex = next(spec for spec in models_config.CATALOG
                     if spec.family == "tirex")
        self.assertEqual(tirex.model_id, "NX-AI/TiRex-2-gifteval-zs")

    def test_checkpoint_preflight_checks_only_the_small_config(self) -> None:
        from experiments.tirex_compat import require_tirex_checkpoint_access

        with mock.patch(
                "huggingface_hub.hf_hub_download",
                return_value="/cache/model-config.yaml") as download:
            path = require_tirex_checkpoint_access()

        self.assertEqual(path, "/cache/model-config.yaml")
        download.assert_called_once_with(
            repo_id="NX-AI/TiRex-2-gifteval-zs",
            filename="model-config.yaml",
            repo_type="model",
        )

    def test_checkpoint_preflight_explains_gated_access(self) -> None:
        from huggingface_hub.errors import GatedRepoError
        from experiments.tirex_compat import (
            TirexCheckpointAccessError,
            require_tirex_checkpoint_access,
        )

        with mock.patch(
                "huggingface_hub.hf_hub_download",
                side_effect=GatedRepoError("forbidden")):
            with self.assertRaises(TirexCheckpointAccessError) as raised:
                require_tirex_checkpoint_access()

        message = str(raised.exception)
        self.assertIn("TiREX checkpoint access denied", message)
        self.assertIn("huggingface-cli whoami", message)
        self.assertIn("HF_TOKEN", message)
        self.assertIn("Do not substitute NX-AI/TiRex-2", message)

    def test_checkpoint_preflight_unwraps_gated_403_cache_miss(self) -> None:
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
        from requests import Response
        from experiments.tirex_compat import (
            TirexCheckpointAccessError,
            require_tirex_checkpoint_access,
        )

        response = Response()
        response.status_code = 403
        response.url = (
            "https://huggingface.co/NX-AI/TiRex-2-gifteval-zs/"
            "resolve/main/model-config.yaml"
        )
        forbidden = HfHubHTTPError(
            "403: enable access to public gated repositories",
            response=response,
        )
        cache_miss = LocalEntryNotFoundError("not in local cache")
        cache_miss.__cause__ = forbidden

        with mock.patch(
                "huggingface_hub.hf_hub_download",
                side_effect=cache_miss):
            with self.assertRaises(TirexCheckpointAccessError) as raised:
                require_tirex_checkpoint_access()

        message = str(raised.exception)
        self.assertIn("fine-grained settings", message)
        self.assertIn("public gated repositories", message)

    def test_load_tirex_normalizes_indexed_cuda_device(self) -> None:
        calls = []
        fake_tirex2 = ModuleType("tirex2")
        fake_tirex2_model = ModuleType("tirex2.model")
        fake_tirex2_component = ModuleType("tirex2.model.component")
        fake_flashrnn_slstm = ModuleType("tirex2.model.component.flashrnn_slstm")

        def fake_load_model(model_id, device):
            calls.append((model_id, device, fake_flashrnn_slstm._flashrnn_backend(device)))
            return object()

        fake_tirex2.load_model = fake_load_model
        module_names = {
            "tirex2": fake_tirex2,
            "tirex2.model": fake_tirex2_model,
            "tirex2.model.component": fake_tirex2_component,
            "tirex2.model.component.flashrnn_slstm": fake_flashrnn_slstm,
        }
        old_modules = {name: sys.modules.get(name) for name in module_names}
        sys.modules.update(module_names)
        fake_tirex2_component.flashrnn_slstm = fake_flashrnn_slstm
        try:
            build.load_tirex("NX-AI/TiRex-2", "cuda:0")
            build.load_tirex("NX-AI/TiRex-2", "cpu")
        finally:
            for name, old_module in old_modules.items():
                if old_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_module

        self.assertEqual(
            calls,
            [
                ("NX-AI/TiRex-2", "cuda", "vanilla"),
                ("NX-AI/TiRex-2", "cpu", "vanilla"),
            ],
        )
        self.assertEqual(fake_flashrnn_slstm._flashrnn_backend("cpu"), "vanilla")

    def test_tirex_cuda_uses_vanilla_backend(self) -> None:
        from experiments.tirex_compat import tirex_backend_for_device

        self.assertEqual(tirex_backend_for_device("cuda"), "vanilla")

    def test_tirex_backend_can_be_overridden(self) -> None:
        from experiments.tirex_compat import tirex_backend_for_device

        os.environ["PREDICTCSL_TIREX_BACKEND"] = "cuda_fused"
        self.assertEqual(tirex_backend_for_device("cuda"), "cuda_fused")

    def test_tirex_rejects_invalid_backend(self) -> None:
        from experiments.tirex_compat import tirex_backend_for_device

        os.environ["PREDICTCSL_TIREX_BACKEND"] = "nope"
        with self.assertRaisesRegex(ValueError, "PREDICTCSL_TIREX_BACKEND"):
            tirex_backend_for_device("cuda")

    def test_tirex_long_horizon_is_forecast_in_chunks(self) -> None:
        from experiments.tirex_compat import forecast_tirex_medians

        fake_tirex2 = ModuleType("tirex2")

        class FakeTimeseries:
            def __init__(self, target, past_covariates, future_covariates):
                self.target = target

        class FakeModel:
            future_len = 3
            context_len = 5

            def __init__(self):
                self.context_lengths = []

            def forecast(self, series, prediction_length, output_type):
                self.context_lengths.append(
                    ([item.target.shape[-1] for item in series], prediction_length)
                )
                value = float(len(self.context_lengths))
                return [
                    np.full((1, 9, prediction_length), value, dtype=np.float32)
                    for _ in series
                ]

        fake_tirex2.TimeseriesType = FakeTimeseries
        old_tirex2 = sys.modules.get("tirex2")
        sys.modules["tirex2"] = fake_tirex2
        try:
            model = FakeModel()
            result = forecast_tirex_medians(model, torch.zeros(2, 4), 7)
        finally:
            if old_tirex2 is None:
                sys.modules.pop("tirex2", None)
            else:
                sys.modules["tirex2"] = old_tirex2

        np.testing.assert_array_equal(
            result,
            np.array([[1, 1, 1, 2, 2, 2, 3], [1, 1, 1, 2, 2, 2, 3]], dtype=np.float32),
        )
        self.assertEqual(model.context_lengths, [([4, 4], 3), ([5, 5], 3), ([5, 5], 1)])

    def test_stage1_dynamic_batch_scales_with_context(self) -> None:
        self.assertEqual(build.batch_size_for_context(
            8, 8192, 500, True, 8192, 512), 8)
        self.assertEqual(build.batch_size_for_context(
            8, 2048, 500, True, 8192, 512), 32)
        self.assertEqual(build.batch_size_for_context(
            8, 32, 100, True, 8192, 512), 100)
        self.assertEqual(build.batch_size_for_context(
            8, 32, 100, False, 8192, 512), 8)

    def test_stage1_tirex_dynamic_batch_backs_off_after_oom(self) -> None:
        seen = []

        def fake_predict(_runner, x, horizon, _device):
            seen.append(len(x))
            if len(x) > 4:
                raise RuntimeError("CUDA out of memory")
            return torch.zeros(len(x), horizon)

        tuned = {}
        x = torch.zeros(10, 32, 1)
        with mock.patch.object(build, "_is_cuda", return_value=True), \
             mock.patch.object(build, "predict_tirex", side_effect=fake_predict), \
             mock.patch.object(torch.cuda, "empty_cache"):
            result = build._forecast_uniform(
                "tirex", object(), "unused", x, 32, 2, 8, "cuda:0",
                dynamic_batching=True, max_batch_size=16,
                tuned_batch_sizes=tuned)

        self.assertEqual(tuple(result.shape), (10, 2))
        self.assertEqual(seen[:2], [8, 4])
        self.assertEqual(tuned, {})


if __name__ == "__main__":
    unittest.main()
