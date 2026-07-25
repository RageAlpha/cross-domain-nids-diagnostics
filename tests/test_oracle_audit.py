import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cross_domain_nids import oracle_split_audit as audit


class OraclePartitionAuditTests(unittest.TestCase):
    def test_normalization_and_hash_reproducibility(self):
        frame = pd.DataFrame({
            "FLOW_START_MILLISECONDS": [1, 1],
            "FLOW_END_MILLISECONDS": [2, 2],
            "IPV4_SRC_ADDR": ["a", "a"],
            "L4_SRC_PORT": [3, 3],
            "IPV4_DST_ADDR": ["b", "b"],
            "L4_DST_PORT": [4, 4],
            "PROTOCOL": [6, 6],
            "L7_PROTO": [np.nan, 0.0],
            "Label": [1, 1],
            "Attack": ["x", "x"],
        })
        normalized = audit.normalize_chunk(frame)
        first, second = audit.double_hash(normalized)
        self.assertEqual(first[0], first[1])
        self.assertEqual(second[0], second[1])

    def test_overlap_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # groups: rows 0/1 shared train-test; rows 2/3 shared train-val; row 4 test only
            h1 = audit.open_array(root / "h1.npy", np.uint64, 5)
            h2 = audit.open_array(root / "h2.npy", np.uint64, 5)
            h1[:] = [10, 10, 20, 20, 30]
            h2[:] = [1, 1, 2, 2, 3]
            h1.flush(); h2.flush()
            codes = audit.open_array(root / "codes.npy", np.uint8, 5)
            codes[:] = [0, 2, 0, 1, 2]
            codes.flush()
            order, starts, structure = audit.group_layout(root / "h1.npy", root / "h2.npy", 5, "synthetic", "Synthetic")
            summary, pairs = audit.summarize_assignment(order, starts, root / "codes.npy", 5, "Synthetic", 42, "synthetic")
            self.assertEqual(structure["unique_groups"], 3)
            self.assertEqual(summary["cross_partition_groups"], 2)
            self.assertEqual(summary["test_rows_with_train_or_val_identity"], 1)
            self.assertEqual(summary["val_rows_with_train_identity"], 1)
            train_test = next(row for row in pairs if row["left_partition"] == "train" and row["right_partition"] == "test")
            self.assertEqual(train_test["shared_groups"], 1)
            del h1, h2, codes, order, starts

    def test_disjoint_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h1 = audit.open_array(root / "h1.npy", np.uint64, 3)
            h2 = audit.open_array(root / "h2.npy", np.uint64, 3)
            h1[:] = [10, 20, 30]; h2[:] = [1, 2, 3]
            h1.flush(); h2.flush()
            codes = audit.open_array(root / "codes.npy", np.uint8, 3)
            codes[:] = [0, 1, 2]; codes.flush()
            order, starts, _ = audit.group_layout(root / "h1.npy", root / "h2.npy", 3, "synthetic", "Synthetic")
            summary, _ = audit.summarize_assignment(order, starts, root / "codes.npy", 3, "Synthetic", 42, "synthetic")
            self.assertTrue(summary["test_independent"])
            self.assertTrue(summary["all_partitions_independent"])
            del h1, h2, codes, order, starts


if __name__ == "__main__":
    unittest.main()
