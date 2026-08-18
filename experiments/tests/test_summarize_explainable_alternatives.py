import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.summarize_explainable_alternatives import summarize


class SummarizeExplainableAlternativesTests(unittest.TestCase):
    def test_writes_selected_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "example"
            model.mkdir()
            (model / "explainable_alternatives_screen.json").write_text(json.dumps({
                "model": "Example",
                "best_candidate": {
                    "candidate": "model_tree_rank_d4_a1",
                    "type": "piecewise_linear_tree",
                    "artifact_bytes": 123,
                    "validation_pareto": {
                        "mean_context_saved_across_budgets_pct": 42.0,
                        "context_saved_by_mase_budget_pct": {
                            "mase_plus_0pct": 10.0,
                            "mase_plus_1pct": 74.0,
                        },
                    },
                },
            }))
            output = root / "summary.csv"
            rows = summarize(root, output)
            self.assertEqual(rows[0]["candidate"], "model_tree_rank_d4_a1")
            with output.open(newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(written[0]["model"], "Example")


if __name__ == "__main__":
    unittest.main()
