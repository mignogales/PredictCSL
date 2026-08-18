from pathlib import Path

import numpy as np

from experiments.plot_synth_sweep_results import (
    closest_agreement_window,
    collect_results,
    coverage_rows,
    summarize_cell,
)
from experiments.synth_param_sweeps import RATIO_GRID, resolve_ratios


def _write_cell(root: Path) -> Path:
    cell = root / "Model-A" / "period"
    cell.mkdir(parents=True)
    # 2 bins, 3 series, 4 ratios, 2 horizons. The h=64 curve has its first
    # within-2%-of-best point at ratio 2 and its minimum at ratio 4.
    curves = np.array([
        [[[4, 4], [2.06, 2.06], [2.02, 2.02], [2, 2]]] * 3,
        [[[8, 8], [4.12, 4.12], [4.04, 4.04], [4, 4]]] * 3,
    ], dtype=np.float32)
    np.savez_compressed(
        cell / "results.npz",
        curves_mae=curves,
        curves_mse=curves,
        naive_mae=np.full((2, 3, 2), 4.0, dtype=np.float32),
        contexts=np.array([[16, 32, 64, 128], [32, 64, 128, 256]]),
        ratios=np.array([0.5, 1, 2, 4], dtype=np.float32),
        norms=np.array([32, 64], dtype=np.float32),
        horizons=np.array([64, 64]),
        eval_horizons=np.array([16, 64]),
    )
    (cell / "done.json").write_text("{}\n")
    return cell / "results.npz"


def test_summarize_cell(tmp_path):
    summary, ratios, relative = summarize_cell(_write_cell(tmp_path))
    assert summary.model == "Model-A"
    assert summary.experiment == "period"
    assert summary.evaluation_horizon == 64
    assert summary.best_nominal_ratio == 4
    assert summary.saturation_ratio_2pct == 2
    assert np.isclose(summary.short_context_penalty_pct, 100)
    assert np.isclose(summary.longest_context_penalty_pct, 0)
    assert ratios.shape == relative.shape == (4,)


def test_coverage_distinguishes_partial_cells(tmp_path):
    _write_cell(tmp_path)
    rows = coverage_rows(tmp_path, ["Model-A", "Model-B"], ["period"])
    assert [row["status"] for row in rows] == ["complete", "missing"]


def test_collect_results_records_clamped_cell(tmp_path):
    path = _write_cell(tmp_path)
    with np.load(path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    arrays["contexts"][0] = [16, 32, 64, 64]
    np.savez_compressed(path, **arrays)
    _, curves = collect_results(tmp_path)
    curve = curves[("Model-A", "period")]
    assert curve.clamped


def test_closest_agreement_window_uses_the_minimum_spread_point():
    window = closest_agreement_window([
        (np.array([1, 2, 4, 8, 16], dtype=float),
         np.array([1.5, 1.2, 1.0, 1.2, 1.5], dtype=float)),
        (np.array([1, 2, 4, 8, 16], dtype=float),
         np.array([1.7, 1.3, 1.0, 1.3, 1.7], dtype=float)),
    ])
    assert window == (2.0, 4.0, 16.0)


def test_only_long_tail_sweeps_extend_the_ratio_grid():
    assert resolve_ratios("period", test=False) == RATIO_GRID
    assert resolve_ratios("delay", test=False)[-4:] == [24.0, 32.0, 48.0, 64.0]
    assert resolve_ratios("horizon", test=True) == [0.5, 1.0, 2.0, 4.0]
