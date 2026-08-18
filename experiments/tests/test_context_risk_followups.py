import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.summarize_context_risk_ablations import is_dominated
from experiments.summarize_context_risk_multiseed import summarize


class ContextRiskFollowupTests(unittest.TestCase):

    def test_pareto_direction_uses_lower_mase_and_higher_saving(self):
        weak = {"geomean_mase_ratio": 1.01, "flops_saved_pct": 20.0}
        strong = {"geomean_mase_ratio": 1.00, "flops_saved_pct": 30.0}
        self.assertTrue(is_dominated(weak, [weak, strong]))
        self.assertFalse(is_dominated(strong, [weak, strong]))

    def test_multiseed_cluster_summary_writes_intervals(self):
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            runs = []
            for seed, delta in [(11, 0.0), (22, 0.01)]:
                rows = []
                for dataset, ratio in [("A", 0.99 + delta), ("B", 1.01 + delta)]:
                    rows.append({
                        "dataset": dataset,
                        "method": "balanced",
                        "cell_mase_ratio": ratio,
                        "n_instances": 10,
                        "harm5_rate": 0.1,
                        "coverage": 0.5,
                        "selected_theoretical_macs": 80.0,
                        "native_theoretical_macs": 100.0,
                    })
                path = root / f"seed_{seed}.csv"
                pd.DataFrame(rows).to_csv(path, index=False)
                runs.append((seed, path))

            report = summarize(runs, root / "out", repeats=100, seed=7)

            self.assertEqual(report["seeds"], [11, 22])
            self.assertTrue((root / "out" / "cluster_bootstrap_summary.csv").exists())
            summary = pd.read_csv(root / "out" / "cluster_bootstrap_summary.csv")
            self.assertIn("geomean_mase_ratio", set(summary["metric"]))
            self.assertTrue((summary["ci95_high"] >= summary["ci95_low"]).all())


if __name__ == "__main__":
    unittest.main()
