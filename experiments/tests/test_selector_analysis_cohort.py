import pandas as pd

from experiments.compare_window_strategies_gifteval import compute_summary_stats


def test_selector_analysis_excludes_one_action_cells_only_from_nested_cohort():
    rows = []
    for name, eligible, pred in [
        ("Regular", True, 0.8),
        ("CarParts", False, 2.0),
        ("Solar-W", False, 3.0),
    ]:
        rows.append({
            "dataset_display": name,
            "term": "short",
            "selection_eligible": eligible,
            "full_mase": 1.0,
            "best_mase": 0.7,
            "pred_mase": pred,
            "pred_clamped": False,
            "rel_gain_pred_over_full": 1.0 - pred,
            "delta_pred_vs_best": pred - 0.7,
            "full_elapsed_s": 2.0,
            "best_elapsed_s": 1.0,
            "pred_elapsed_s": 1.5,
            "speedup_pred_vs_full": 4.0 / 3.0,
            "complexity_ratio_pred_vs_full": 0.75,
        })

    stats = compute_summary_stats(pd.DataFrame(rows))

    # The official/headline cohort remains complete.
    assert stats["headline_aggregation"]["cohort_size"] == 3
    assert stats["pred_mase"]["n"] == 3

    selector = stats["selector_analysis"]
    assert selector["cohort_size"] == 1
    assert selector["excluded_cells"] == ["CarParts/short", "Solar-W/short"]
    assert abs(selector["pred_mase"]["relative_gain_vs_full"] - 0.2) < 1e-12
