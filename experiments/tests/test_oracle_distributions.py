import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from experiments.analyze_oracle_distributions import (
    _cell_size,
    _metric_vector,
    jensen_shannon,
    select_oracles,
)


class OracleDistributionTests(unittest.TestCase):

    def test_select_oracles_includes_native_and_uses_smallest_grid_on_tie(self):
        windows = np.asarray([32, 64])
        errors = np.asarray([
            [0.5, 0.5],
            [0.7, 0.6],
            [np.nan, 0.8],
        ])
        effective = np.asarray([
            [32, 64],
            [32, 64],
            [np.nan, 60],
        ])
        native_error = np.asarray([0.6, 0.4, 0.9])
        native_context = np.asarray([100, 100, 80])

        result = select_oracles(
            errors, windows, effective, native_error, native_context)

        np.testing.assert_array_equal(
            result["oracle_effective_context"], [32, 100, 60])
        np.testing.assert_array_equal(
            result["native_selected"], [False, True, False])
        np.testing.assert_allclose(
            result["oracle_fraction"], [0.32, 1.0, 0.75])

    def test_jensen_shannon_bounds_and_identity(self):
        left = np.asarray([0.5, 0.5, 0.0])
        right = np.asarray([0.0, 0.0, 1.0])
        self.assertEqual(jensen_shannon(left, left), 0.0)
        self.assertAlmostEqual(jensen_shannon(left, right), 1.0)

    def test_unaligned_legacy_action_is_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            aligned = Path(root) / "aligned.npz"
            legacy = Path(root) / "legacy.npz"
            np.savez_compressed(
                aligned,
                mase_gluonts_real=np.asarray([0.2, 0.3]),
                served_index=np.asarray([0, 1]),
            )
            np.savez_compressed(
                legacy,
                mase_gluonts_real=np.asarray([0.1]),
            )

            self.assertEqual(_cell_size([legacy, aligned]), 2)
            with warnings.catch_warnings(record=True) as caught:
                values, counts, effective, source = _metric_vector(
                    legacy, 2, "mase_gluonts_real")
            self.assertEqual(source, "unaligned")
            self.assertTrue(np.isnan(values).all())
            self.assertTrue((counts == 0).all())
            self.assertTrue(np.isnan(effective).all())
            self.assertEqual(len(caught), 1)


if __name__ == "__main__":
    unittest.main()
