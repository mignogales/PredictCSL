import os
import tempfile
import unittest

import numpy as np
from torch import nn

from context_interpretability.adapters.base import (
    AdapterCapabilities, align_context_lengths, blocks_for_context, thin_blocks)
from context_interpretability.adapters.tsfm import TSFMAdapter
from context_interpretability.schema import (
    ResultsWriter, cell_done, load_results, make_row)


class TestSchema(unittest.TestCase):
    def test_make_row_defaults_and_validation(self):
        row = make_row(model="m", dataset="d", sample_id="s",
                       context_length=64, method="perturbation",
                       loss_delta=0.5)
        self.assertEqual(row["context_length"], 64)
        self.assertEqual(row["block_index"], -1)
        self.assertTrue(np.isnan(row["recovery_score"]))
        with self.assertRaises(KeyError):
            make_row(nonsense_column=1)

    def test_writer_roundtrip_and_done_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = os.path.join(tmp, "cell")
            w = ResultsWriter(cell)
            w.add(model="m", dataset="d", sample_id="s0", context_length=32,
                  method="perturbation", loss_delta=1.0)
            self.assertFalse(cell_done(cell))
            w.finalize({"context_length": 32})
            self.assertTrue(cell_done(cell))
            df = load_results(tmp)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "loss_delta"], 1.0)
            self.assertTrue(load_results(tmp, method="other").empty)


class TestContextAlignment(unittest.TestCase):
    def test_align_drops_and_aligns(self):
        pairs = align_context_lengths([32, 48, 64, 8192], patch_length=32,
                                      maximum=64)
        # 48 aligns down to 32 (duplicate, dropped); 8192 > max dropped
        self.assertEqual(pairs, [(32, 32), (64, 64)])

    def test_align_no_patch(self):
        pairs = align_context_lengths([10, 20], None, 15)
        self.assertEqual(pairs, [(10, 10)])

    def test_blocks_indexing(self):
        blocks = blocks_for_context(64, 16)
        self.assertEqual(len(blocks), 4)
        self.assertEqual(blocks[0].lookback_start, 0)   # most recent
        self.assertEqual(blocks[0].input_slice(64), slice(48, 64))
        self.assertEqual(blocks[3].input_slice(64), slice(0, 16))

    def test_partial_block_excluded_by_default(self):
        self.assertEqual(len(blocks_for_context(70, 16)), 4)
        self.assertEqual(len(blocks_for_context(70, 16, True)), 5)

    def test_thin_blocks(self):
        blocks = blocks_for_context(1024, 8)            # 128 blocks
        subset, thinned = thin_blocks(blocks, 16)
        self.assertTrue(thinned)
        self.assertLessEqual(len(subset), 16)
        self.assertEqual(subset[0].index, 0)            # recent kept dense
        self.assertEqual(subset[-1].index, 127)         # oldest kept
        subset2, thinned2 = thin_blocks(blocks[:10], 16)
        self.assertFalse(thinned2)
        self.assertEqual(len(subset2), 10)


class _NestedMLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.layers = nn.ModuleList([nn.Linear(4, 16), nn.Linear(16, 4)])


class _NestedStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.blocks = nn.ModuleList([_NestedMLPBlock() for _ in range(3)])


class TestTSFMLayerDiscovery(unittest.TestCase):
    def test_selects_residual_stack_not_nested_mlp_layers(self):
        adapter = TSFMAdapter(
            "unused", "patchtst_fm", "NestedModel",
            AdapterCapabilities(supports_forecast_lens=True), horizon=1,
            device="cpu", batch_size=1)
        adapter._backbone = _NestedStack()
        self.assertEqual(adapter.get_layer_names(), [
            "encoder.blocks.0", "encoder.blocks.1", "encoder.blocks.2"])


if __name__ == "__main__":
    unittest.main()
