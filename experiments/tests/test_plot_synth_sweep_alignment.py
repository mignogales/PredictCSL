import numpy as np

from experiments.plot_synth_sweep_alignment import (
    _log_interpolate,
    _tail_subset,
    _window_subset,
    load_alignment_cell,
)
from experiments.tests.test_plot_synth_sweep_results import _write_cell
from experiments.plot_synth_sweep_results import closest_agreement_window


def test_log_interpolate_does_not_extrapolate():
    result = _log_interpolate(
        np.array([1.0, 2.0]), np.array([2.0, 1.0]),
        np.array([0.5, 1.0, 2.0, 4.0]),
    )
    assert np.isnan(result[0])
    assert np.allclose(result[1:3], [2.0, 1.0])
    assert np.isnan(result[3])


def test_identical_normalized_bins_have_zero_alignment_std(tmp_path):
    cell = load_alignment_cell(_write_cell(tmp_path))
    assert np.nanmax(np.abs(cell.alignment_std)) < 1e-7
    assert np.nanmax(np.abs(cell.series_cv_median)) < 1e-7
    assert not cell.clamped


def test_terminal_duplicate_context_is_detected_as_clamped(tmp_path):
    path = _write_cell(tmp_path)
    with np.load(path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    arrays["contexts"][:, -1] = arrays["contexts"][:, -2]
    np.savez_compressed(path, **arrays)
    assert load_alignment_cell(path).clamped


def test_tail_subset_uses_unique_attainable_contexts():
    x, y = _tail_subset(
        np.array([1, 2, 3, 4, 4, 4, 4], dtype=float),
        np.array([5, 4, 3, 2, 2, 2, 2], dtype=float),
    )
    assert np.array_equal(x, [1, 2, 3, 4])
    assert np.array_equal(y, [5, 4, 3, 2])


def test_closest_agreement_window_limits_the_inset_to_three_points(tmp_path):
    path = _write_cell(tmp_path)
    with np.load(path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    arrays["contexts"][0] = [16, 32, 64, 64]
    arrays["contexts"][1] = [32, 64, 128, 256]
    np.savez_compressed(path, **arrays)
    cell = load_alignment_cell(path)
    minimum, centre, maximum = closest_agreement_window([
        (np.array([0.5, 1, 2, 4], dtype=float),
         np.array([1.8, 1.2, 1.0, 1.2], dtype=float)),
        (np.array([0.5, 1, 2, 4], dtype=float),
         np.array([1.4, 1.1, 1.0, 1.3], dtype=float)),
    ])
    assert (minimum, centre, maximum) == (1.0, 2.0, 4.0)
    first_x, _ = _window_subset(
        cell.actual_ratios[0], cell.relative_bin_curves[0], minimum, maximum)
    second_x, _ = _window_subset(
        cell.actual_ratios[1], cell.relative_bin_curves[1], minimum, maximum)
    assert np.array_equal(first_x, [1, 2])
    assert np.array_equal(second_x, [1, 2, 4])
