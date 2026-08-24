from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from audit_imagenet_code_corpus import count_unique_codes


class ImageNetCodeAuditTests(unittest.TestCase):
    def test_counts_complete_and_partial_codebooks(self):
        complete = np.array([[0, 1], [2, 3]], dtype=np.int16)
        partial = np.array([[0, 1], [1, 2]], dtype=np.int16)

        self.assertEqual(count_unique_codes(complete, codebook_size=4), 4)
        self.assertEqual(count_unique_codes(partial, codebook_size=4), 3)

    def test_rejects_late_out_of_range_code(self):
        codes = np.zeros((4097, 4), dtype=np.int16)
        codes[0] = [0, 1, 2, 3]
        codes[-1, 0] = -1

        with self.assertRaisesRegex(ValueError, "outside"):
            count_unique_codes(codes, codebook_size=4)


if __name__ == "__main__":
    unittest.main()
