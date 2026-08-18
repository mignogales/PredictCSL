"""Regression tests for checkpoint-audited TSFM FLOPs models."""

import unittest

from experiments.compare_window_strategies_gifteval import (
    DEFAULT_PATCH_SIZES,
    MODEL_ARCH,
    resolve_model_arch,
    theoretical_flops,
)


class TheoreticalFlopsAuditTests(unittest.TestCase):
    def test_chronos_checkpoint_shapes_are_current(self) -> None:
        chronos2 = MODEL_ARCH["chronos2"]
        self.assertEqual(
            (chronos2.d_model, chronos2.d_ff, chronos2.patch_size,
             chronos2.n_enc_layers),
            (768, 3072, 16, 12),
        )
        bolt = MODEL_ARCH["chronos_bolt"]
        self.assertEqual(
            (bolt.d_model, bolt.d_ff, bolt.patch_size,
             bolt.n_enc_layers, bolt.n_dec_layers),
            (768, 3072, 16, 12, 12),
        )

    def test_tirex_native_grid_makes_cost_context_invariant(self) -> None:
        arch = resolve_model_arch("NX-AI/TiRex-2-gifteval-zs")
        self.assertEqual(
            (arch.d_model, arch.n_enc_layers, arch.output_quantiles),
            (512, 12, 9),
        )
        short = theoretical_flops(
            "NX-AI/TiRex-2-gifteval-zs", 1024, 96, DEFAULT_PATCH_SIZES)
        long = theoretical_flops(
            "NX-AI/TiRex-2-gifteval-zs", 8192, 96, DEFAULT_PATCH_SIZES)
        self.assertEqual(short, long)

    def test_tirex_long_horizon_repeats_full_tta_forecast(self) -> None:
        one_chunk = theoretical_flops(
            "NX-AI/TiRex-2-gifteval-zs", 8192, 928, DEFAULT_PATCH_SIZES)
        two_chunks = theoretical_flops(
            "NX-AI/TiRex-2-gifteval-zs", 8192, 929, DEFAULT_PATCH_SIZES)
        self.assertEqual(two_chunks, 2 * one_chunk)

    def test_moirai_long_horizon_counts_recursive_quantile_paths(self) -> None:
        one_call = theoretical_flops(
            "Salesforce/moirai-2.0-R-small", 1024, 64, DEFAULT_PATCH_SIZES)
        two_calls = theoretical_flops(
            "Salesforce/moirai-2.0-R-small", 1024, 65, DEFAULT_PATCH_SIZES)
        self.assertGreater(two_calls, 5 * one_call)

    def test_display_labels_resolve_checkpoint_specific_variants(self) -> None:
        small = resolve_model_arch("Chronos2-Small")
        base = resolve_model_arch("Chronos2-Base")
        self.assertEqual((small.d_model, small.n_enc_layers), (512, 6))
        self.assertEqual((base.d_model, base.n_enc_layers), (768, 12))
        self.assertLess(
            theoretical_flops("Chronos2-Small", 8192, 64, DEFAULT_PATCH_SIZES),
            theoretical_flops("Chronos2-Base", 8192, 64, DEFAULT_PATCH_SIZES),
        )

    def test_flowstate_uses_raw_steps_and_nonquadratic_ssm_cost(self) -> None:
        arch = resolve_model_arch(
            "ibm-granite/granite-timeseries-flowstate-r1")
        self.assertEqual(
            (arch.d_model, arch.d_ff, arch.n_enc_layers,
             arch.output_quantiles, arch.decoder_dim),
            (512, 1024, 6, 9, 256),
        )
        short = theoretical_flops(
            "FlowState-R1", 1024, 96, DEFAULT_PATCH_SIZES)
        long = theoretical_flops(
            "FlowState-R1", 4096, 96, DEFAULT_PATCH_SIZES)
        self.assertGreater(long / short, 3.5)
        self.assertLess(long / short, 5.0)

    def test_sundial_one_output_block_has_horizon_independent_cost(self) -> None:
        short_horizon = theoretical_flops(
            "thuml/sundial-base-128m", 2880, 16, DEFAULT_PATCH_SIZES)
        long_horizon = theoretical_flops(
            "thuml/sundial-base-128m", 2880, 720, DEFAULT_PATCH_SIZES)
        self.assertEqual(short_horizon, long_horizon)
        two_blocks = theoretical_flops(
            "thuml/sundial-base-128m", 2880, 721, DEFAULT_PATCH_SIZES)
        self.assertGreater(two_blocks, long_horizon)

    def test_bolt_long_horizon_counts_recursive_quantile_calls(self) -> None:
        one_block = theoretical_flops(
            "amazon/chronos-bolt-base", 2048, 64, DEFAULT_PATCH_SIZES)
        two_blocks = theoretical_flops(
            "amazon/chronos-bolt-base", 2048, 65, DEFAULT_PATCH_SIZES)
        self.assertGreater(two_blocks, 5 * one_block)

    def test_all_audited_costs_decrease_with_context(self) -> None:
        models = [
            "autogluon/chronos-2-synth",
            "amazon/chronos-bolt-base",
            "thuml/sundial-base-128m",
            "Datadog/Toto-2.0-313m",
        ]
        for model in models:
            with self.subTest(model=model):
                short = theoretical_flops(model, 256, 96, DEFAULT_PATCH_SIZES)
                long = theoretical_flops(model, 2048, 96, DEFAULT_PATCH_SIZES)
                self.assertLess(short, long)


if __name__ == "__main__":
    unittest.main()
