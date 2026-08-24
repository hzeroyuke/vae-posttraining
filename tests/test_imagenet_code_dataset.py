from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


LLAMAGEN_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "LlamaGen"
sys.path.insert(0, str(LLAMAGEN_ROOT))

from dataset.imagenet_code import ConsolidatedImageNetCodeDataset


class ImageNetCodeDatasetTests(unittest.TestCase):
    def test_dual_crop_sampling_selects_scale_then_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codes = np.empty((1, 2, 10, 4), dtype=np.int16)
            for group in range(2):
                for crop in range(10):
                    codes[0, group, crop] = group * 100 + crop
            np.save(root / "codes.npy", codes)
            np.save(root / "labels.npy", np.array([17], dtype=np.int16))

            dataset = ConsolidatedImageNetCodeDataset(root)
            self.assertEqual(dataset.augmentations_per_image, 20)
            self.assertTrue(dataset.flip)

            with patch(
                "dataset.imagenet_code.torch.rand", return_value=torch.tensor([0.25])
            ), patch(
                "dataset.imagenet_code.torch.randint", return_value=torch.tensor(3)
            ):
                selected, label = dataset[0]

            self.assertTrue(torch.equal(selected, torch.tensor([103, 103, 103, 103])))
            self.assertTrue(torch.equal(label, torch.tensor([17])))

    def test_epoch_cache_is_deterministic_and_avoids_online_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codes = np.empty((4, 2, 3, 4), dtype=np.int16)
            for image in range(4):
                for group in range(2):
                    for crop in range(3):
                        codes[image, group, crop] = image * 100 + group * 10 + crop
            np.save(root / "codes.npy", codes)
            np.save(root / "labels.npy", np.arange(4, dtype=np.int16))

            indices = np.array([3, 1], dtype=np.int64)
            first = ConsolidatedImageNetCodeDataset(root)
            second = ConsolidatedImageNetCodeDataset(root)
            cache_bytes = first.prepare_epoch(
                indices, seed=20260812, epoch=7, chunk_size=1
            )
            second.prepare_epoch(indices, seed=20260812, epoch=7, chunk_size=2)

            self.assertGreater(cache_bytes, first._epoch_codes.nbytes)
            self.assertTrue(np.array_equal(first._epoch_codes, second._epoch_codes))
            with patch(
                "dataset.imagenet_code.torch.rand",
                side_effect=AssertionError("online crop sampling should not run"),
            ):
                selected, label = first[3]
            self.assertEqual(tuple(selected.shape), (4,))
            self.assertTrue(torch.equal(label, torch.tensor([3])))
            with self.assertRaisesRegex(IndexError, "not prepared"):
                first[0]


if __name__ == "__main__":
    unittest.main()
