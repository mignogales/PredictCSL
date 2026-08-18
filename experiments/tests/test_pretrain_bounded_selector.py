import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pretrain_bounded_selector import (
    bounded_risk_loss,
    iter_univariate_targets,
    rolling_cutoffs,
    source_to_split,
    standardize_example,
)


def test_iter_univariate_targets_supports_both_shapes():
    assert len(list(iter_univariate_targets([1, 2, 3]))) == 1
    channels = list(iter_univariate_targets([[1, 2], [3, 4]]))
    assert [channel for channel, _ in channels] == [0, 1]
    np.testing.assert_array_equal(channels[1][1], [3, 4])


def test_rolling_cutoffs_are_valid_and_include_latest():
    cutoffs = rolling_cutoffs(10_000, 8192, 192, 3)
    assert cutoffs[0] >= 8192
    assert cutoffs[-1] == 9808
    assert cutoffs == sorted(set(cutoffs))
    assert rolling_cutoffs(8300, 8192, 192, 2) == []


def test_standardize_example_imputes_context_but_not_future():
    values = np.arange(30, dtype=np.float32)
    values[3] = np.nan
    values[25] = np.nan
    result = standardize_example(
        values, cutoff=20, max_window=20, max_horizon=10,
        min_context_observed=0.8, min_future_observed=0.8)
    assert result is not None
    context, future = result
    assert np.isfinite(context).all()
    assert context[3] == 0.0
    assert np.isnan(future[5])


def test_bounded_risk_loss_rewards_correct_ordering():
    error = torch.tensor([[2.0, 1.0, 1.5], [1.5, 2.0, 1.0]])
    target = torch.log(error / error[:, -1:])
    wrong = -target
    assert bounded_risk_loss(target, error) < bounded_risk_loss(wrong, error)


def test_source_split_must_be_unique():
    splits = {"train": ["a"], "val": ["b"], "test": ["c"]}
    assert source_to_split("b", splits) == "val"
