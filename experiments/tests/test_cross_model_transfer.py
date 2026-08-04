import argparse
import tempfile
import unittest
from pathlib import Path

from experiments.evaluate_cross_model_transfer import (
    compare_command,
    ensure_datasets_link,
    overlay_command,
    transfer_tree,
)


class CrossModelTransferTests(unittest.TestCase):

    def test_link_and_cached_only_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "general"
            (canonical / "datasets").mkdir(parents=True)
            (canonical / "results.csv").write_text("model_short,dataset_display,term,window_size\n")
            source = root / "Chronos2-Small"
            source.mkdir()
            (source / "best_model.pt").write_bytes(b"weights")
            (source / "best_config.json").write_text("{}")
            tree = transfer_tree(root / "transfers", source, "Chronos2-Base")
            ensure_datasets_link(tree, canonical)
            self.assertTrue((tree / "datasets").is_symlink())
            args = argparse.Namespace(
                source_predictor_dir=str(source), canonical_run_dir=str(canonical),
                short_context_mode="pad", device="cpu", no_plots=True,
            )
            overlay = overlay_command(args, tree, "Chronos2-Base")
            self.assertIn("--cached-only", overlay)
            self.assertIn("--preloaded-results-csv", overlay)
            self.assertIn("Chronos2-Base", overlay)
            self.assertIn("Chronos2-Base", compare_command(tree, "Chronos2-Base"))


if __name__ == "__main__":
    unittest.main()
