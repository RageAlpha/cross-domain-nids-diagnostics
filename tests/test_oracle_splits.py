import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.oracle import create_splits as splits
from experiments.oracle import run_target_oracle as runner


class CorrectedOracleProtocolTests(unittest.TestCase):
    def test_plurality_repair_and_tie_break(self):
        # session A rows 0,1,2 has train plurality; session B rows 3,4 ties val/test,
        # so the lowest row (3) selects validation. Row 5 is a dropped duplicate.
        historical = np.array([0, 0, 2, 1, 2, 0], dtype=np.uint8)
        sorted_retained = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        starts = np.array([0, 3], dtype=np.int64)
        lowest = np.array([0, 3], dtype=np.int64)
        corrected, diagnostics = splits.plurality_repair(
            historical, sorted_retained, starts, lowest, rows=6
        )
        self.assertEqual(corrected.tolist(), [0, 0, 0, 1, 1, 3])
        self.assertEqual(diagnostics["session_groups_crossing_corrected_partitions"], 0)
        self.assertEqual(diagnostics["tied_session_groups"], 1)

    def test_corrected_validation(self):
        historical = np.array([0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.uint8)
        corrected = np.array([0, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3], dtype=np.uint8)
        labels = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
        retained = np.arange(11, dtype=np.int64)
        result = splits.validate_corrected(corrected, historical, labels, retained, 0)
        self.assertEqual(result["drop_rows"], 1)
        self.assertEqual(result["retained_rows"], 11)
        self.assertEqual(result["session_groups_crossing_corrected_partitions"], 0)

    def test_runner_loads_locked_drop_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "oracle__cicids2018__seed_42.npz"
            codes = np.array([0, 0, 1, 2, 3], dtype=np.uint8)
            labels = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
            np.savez_compressed(path, split_codes=codes, binary_labels=labels)
            manifest = pd.DataFrame([{
                "target": "CICIDS2018",
                "seed": 42,
                "split_map_path": str(path.resolve()),
                "split_map_sha256": runner.hash_file(path),
                "train_rows": 2,
                "val_rows": 1,
                "test_rows": 1,
                "drop_rows": 1,
            }])
            loaded = runner.load_corrected_map(root, manifest, "CICIDS2018", 42)
            self.assertEqual(loaded["sizes"], {"train": 2, "val": 1, "test": 1, "drop": 1})
            self.assertEqual(loaded["retained_target_rows"], 4)


if __name__ == "__main__":
    unittest.main()
