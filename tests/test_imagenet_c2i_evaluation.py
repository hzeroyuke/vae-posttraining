from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
LLAMAGEN_ROOT = PROJECT_ROOT / "third_party" / "LlamaGen"
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(LLAMAGEN_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from generate_llamagen_c2i_fid import (
    batch_indices,
    batch_labels,
    create_npz_from_sample_folder,
    sampling_seed,
)
from summarize_imagenet_c2i_metrics import parse_metrics


class ImageNetC2IEvaluationTests(unittest.TestCase):
    def test_rank_indices_cover_global_batch_once(self):
        observed = []
        for rank in range(4):
            observed.extend(batch_indices(3, 2, 4, rank).tolist())
        self.assertEqual(sorted(observed), list(range(24, 32)))

    def test_labels_and_sampling_seeds_are_reproducible(self):
        first = batch_labels(7, 16, 1000, 123)
        second = batch_labels(7, 16, 1000, 123)
        other = batch_labels(8, 16, 1000, 123)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(sampling_seed(123, 7, 2), sampling_seed(123, 7, 2))
        self.assertNotEqual(sampling_seed(123, 7, 2), sampling_seed(123, 7, 3))

    def test_metric_parser_requires_official_metric_set(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.txt"
            path.write_text(
                "\n".join(
                    (
                        "Inception Score: 100.5",
                        "FID: 3.25",
                        "sFID: 4.5",
                        "Precision: 0.75",
                        "Recall: 0.5",
                    )
                ),
                encoding="utf-8",
            )
            self.assertEqual(parse_metrics(path)["FID"], 3.25)
            path.write_text(json.dumps({"FID": 3.25}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing metrics"):
                parse_metrics(path)

    def test_fid_npz_packaging_is_complete_and_removes_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_dir = root / "png"
            sample_dir.mkdir()
            expected = np.stack(
                (
                    np.full((4, 4, 3), 17, dtype=np.uint8),
                    np.full((4, 4, 3), 231, dtype=np.uint8),
                )
            )
            for index, array in enumerate(expected):
                Image.fromarray(array, mode="RGB").save(sample_dir / f"{index:06d}.png")
            output = root / "samples.npz"
            create_npz_from_sample_folder(sample_dir, output, 2, 4, False)

            with np.load(output) as archive:
                self.assertTrue(np.array_equal(archive["arr_0"], expected))
            self.assertFalse((root / "samples.npy").exists())
            self.assertFalse((root / "samples.partial.npz").exists())


if __name__ == "__main__":
    unittest.main()
