import numpy as np

from experiments.analyze_gifteval_lengths import (
    summarize_lengths,
    threshold_coverage,
)


def test_summarize_lengths():
    result = summarize_lengths(
        [100, 200, 300, 400], ge_name="x", term="short", display="X")
    assert result["n_instances"] == 4
    assert result["min"] == 100
    assert result["median"] == 250
    assert result["mean"] == 250
    assert result["max"] == 400


def test_threshold_coverage_is_strictly_greater_than_gate():
    cells = [
        ({}, np.asarray([100, 200])),
        ({}, np.asarray([200, 400])),
    ]
    result = threshold_coverage(cells, [200]).iloc[0]
    assert result["cells_mean_gt"] == 1
    assert result["instances_gt"] == 1
    assert result["instances_total"] == 4
