import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.summarize_selector_pareto import (
    collect,
    with_tirex_unpadded_counterfactual,
)


def _report(path: Path, model: str, points: list[tuple[float, float]]) -> None:
    aggregate = {}
    for index, (quality, saving) in enumerate(points):
        aggregate[f"dense_{index:03d}"] = {
            "geomean_cell_mase_ratio": 1.0 + quality / 100.0,
            "theoretical_flops_saved_pct": saving,
            "instance_harm5_rate": 0.01,
            "coverage": 0.5,
            "n_cells": 2,
            "n_instances": 10,
        }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model": model, "aggregate": aggregate}))


class SelectorParetoTests(unittest.TestCase):
    def test_marks_selector_and_union_fronts_and_quality_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extra = root / "extra"
            compact = root / "compact"
            _report(
                extra / "m" / "full_score" / "real_evaluation.json",
                "Model", [(0.0, 10.0), (0.5, 50.0), (1.0, 40.0)])
            _report(
                compact / "m" / "full_score" / "real_evaluation.json",
                "Model", [(0.0, 20.0), (0.5, 45.0), (1.0, 60.0)])

            points, summary, aggregate = collect(
                extra, compact, [("Model", "m")])

            extra_dominated = points[
                (points.selector == "ExtraTrees")
                & (points.flops_saved_pct == 40.0)
            ].iloc[0]
            self.assertFalse(bool(extra_dominated.on_selector_pareto))
            extra_at_zero = summary[
                summary.selector == "ExtraTrees"
            ].iloc[0]["max_flops_saved_at_mase_plus_0pct"]
            compact_at_zero = summary[
                summary.selector == "Depth-8 tree"
            ].iloc[0]["max_flops_saved_at_mase_plus_0pct"]
            self.assertEqual(float(extra_at_zero), 10.0)
            self.assertEqual(float(compact_at_zero), 20.0)
            self.assertEqual(set(aggregate.selector), {"ExtraTrees", "Depth-8 tree"})

            relabelled, _summary, _aggregate = collect(
                extra, compact, [("Model", "m")],
                compact_label="Rank-distilled depth-8 tree")
            self.assertEqual(
                set(relabelled.selector),
                {"ExtraTrees", "Rank-distilled depth-8 tree"})

    def test_tirex_unpadded_counterfactual_uses_selected_context_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extra = root / "extra"
            compact = root / "compact"
            for selector_root in (extra, compact):
                report = selector_root / "tirex" / "full_score" / "real_evaluation.json"
                _report(report, "TiRex2", [(0.0, 0.0), (0.5, 0.0)])
                pd.DataFrame([
                    {
                        "dataset": "d", "term": "short", "horizon": 720,
                        "method": "full_native", "window_size": 8192,
                        "n_instances": 10,
                    },
                    {
                        "dataset": "d", "term": "short", "horizon": 720,
                        "method": "dense_000", "window_size": 8192,
                        "n_instances": 10,
                    },
                    {
                        "dataset": "d", "term": "short", "horizon": 720,
                        "method": "dense_001", "window_size": 1024,
                        "n_instances": 10,
                    },
                ]).to_csv(report.parent / "selected_window_histograms.csv", index=False)

            points, _summary, _aggregate = collect(
                extra, compact, [("TiRex2", "tirex")])
            counterfactual, summary, _aggregate = (
                with_tirex_unpadded_counterfactual(points))
            expected = 100.0 * (1.0 - 61.0 / 285.0)
            selected = counterfactual[
                counterfactual.method == "dense_001"
            ].flops_saved_pct
            self.assertTrue(all(abs(value - expected) < 1e-9 for value in selected))
            self.assertTrue(counterfactual.tirex_unpadded_counterfactual.all())
            self.assertTrue(all(
                abs(value - expected) < 1e-9
                for value in summary["max_flops_saved_at_mase_plus_0.5pct"]
            ))


if __name__ == "__main__":
    unittest.main()
