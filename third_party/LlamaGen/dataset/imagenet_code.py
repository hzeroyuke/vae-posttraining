from __future__ import annotations

import mmap
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ConsolidatedImageNetCodeDataset(Dataset):
    """Memory-mapped ImageNet codes written by extract_codes_c2i_parquet."""

    def __init__(self, code_path: str | Path):
        root = Path(code_path)
        self.codes = np.load(root / "codes.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy", mmap_mode="r")
        # Distributed shuffling makes row accesses effectively random. Disable
        # the filesystem's large sequential readahead on the backing mappings;
        # this avoids turning each 1.1 KiB crop read into a much larger disk IO.
        for array in (self.codes, self.labels):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and hasattr(mapping, "madvise"):
                mapping.madvise(mmap.MADV_RANDOM)
        if self.codes.ndim not in (2, 3, 4):
            raise ValueError(
                f"Expected codes [N,T], [N,A,T], or [N,G,A,T], got {self.codes.shape}"
            )
        if self.labels.ndim != 1 or len(self.codes) != len(self.labels):
            raise ValueError(
                f"Code/label shape mismatch: codes={self.codes.shape}, labels={self.labels.shape}"
            )
        if self.codes.ndim == 4:
            self.augmentation_groups = int(self.codes.shape[1])
            self.augmentations_per_group = int(self.codes.shape[2])
            self.augmentations_per_image = (
                self.augmentation_groups * self.augmentations_per_group
            )
        elif self.codes.ndim == 3:
            self.augmentation_groups = 1
            self.augmentations_per_group = int(self.codes.shape[1])
            self.augmentations_per_image = self.augmentations_per_group
        else:
            self.augmentation_groups = 1
            self.augmentations_per_group = 1
            self.augmentations_per_image = 1
        self.flip = self.augmentations_per_image > 1
        self.aug_feature_dir = None
        self._epoch_codes = None
        self._epoch_labels = None
        self._epoch_positions = None

    def __len__(self) -> int:
        return len(self.codes)

    def prepare_epoch(
        self,
        indices,
        *,
        seed: int,
        epoch: int,
        chunk_size: int = 16_384,
    ) -> int:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("Epoch indices must be a non-empty one-dimensional array")
        if indices.min() < 0 or indices.max() >= len(self):
            raise IndexError("Epoch indices are outside the dataset")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Epoch indices must be unique within a distributed rank")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(epoch) * 1_000_003)
        if self.codes.ndim == 4:
            if self.augmentation_groups != 2:
                raise ValueError(
                    f"Expected two crop-range groups, got {self.codes.shape}"
                )
            groups = (torch.rand(len(indices), generator=generator) < 0.5).numpy()
            groups = groups.astype(np.int64, copy=False)
            variants = torch.randint(
                self.augmentations_per_group,
                (len(indices),),
                generator=generator,
            ).numpy()
        elif self.codes.ndim == 3:
            groups = None
            variants = torch.randint(
                self.augmentations_per_group,
                (len(indices),),
                generator=generator,
            ).numpy()
        else:
            groups = None
            variants = None

        positions = np.full(len(self), -1, dtype=np.int32)
        positions[indices] = np.arange(len(indices), dtype=np.int32)
        cache = np.empty(
            (len(indices), int(self.codes.shape[-1])),
            dtype=self.codes.dtype,
        )
        ordered_positions = np.argsort(indices, kind="stable")
        ordered_indices = indices[ordered_positions]
        mapping = getattr(self.codes, "_mmap", None)
        if mapping is not None and hasattr(mapping, "madvise"):
            # Every row is selected once over the two distributed ranks. Scan
            # those rows in file order so the local/NAS filesystem can readahead
            # efficiently instead of serving hundreds of thousands of seeks.
            mapping.madvise(mmap.MADV_SEQUENTIAL)
        row_bytes = int(np.prod(self.codes.shape[1:]) * self.codes.dtype.itemsize)
        source_block_rows = max(1, (64 * 1024 * 1024) // max(row_bytes, 1))
        try:
            for source_start in range(0, len(self.codes), source_block_rows):
                source_stop = min(source_start + source_block_rows, len(self.codes))
                start = np.searchsorted(ordered_indices, source_start, side="left")
                stop = np.searchsorted(ordered_indices, source_stop, side="left")
                if start == stop:
                    continue
                cache_positions = ordered_positions[start:stop]
                source_indices = ordered_indices[start:stop] - source_start
                # The contiguous source block is at most 64 MiB. Advanced
                # indexing copies only the selected crop rows into `cache`.
                source_block = self.codes[source_start:source_stop]
                if self.codes.ndim == 4:
                    selected = source_block[
                        source_indices,
                        groups[cache_positions],
                        variants[cache_positions],
                    ]
                elif self.codes.ndim == 3:
                    selected = source_block[
                        source_indices,
                        variants[cache_positions],
                    ]
                else:
                    selected = source_block[source_indices]
                cache[cache_positions] = selected
                del selected, source_block
                if mapping is not None and hasattr(mapping, "madvise"):
                    # The copied epoch cache is the only data needed by
                    # training; do not retain clean source pages between blocks.
                    mapping.madvise(mmap.MADV_DONTNEED)
        finally:
            if mapping is not None and hasattr(mapping, "madvise"):
                mapping.madvise(mmap.MADV_RANDOM)

        # Keep labels alongside the epoch code cache.  Reading one label from
        # the NAS-backed mapping for every sample otherwise turns training into
        # millions of tiny random reads and can stall a DDP rank independently.
        epoch_labels = np.asarray(self.labels[indices], dtype=np.int64).copy()
        self._epoch_codes = cache
        self._epoch_labels = epoch_labels
        self._epoch_positions = positions
        return int(cache.nbytes + epoch_labels.nbytes + positions.nbytes)

    def __getitem__(self, index: int):
        if self._epoch_codes is not None:
            position = int(self._epoch_positions[index])
            if position < 0:
                raise IndexError(f"Dataset index {index} was not prepared for this rank")
            codes = self._epoch_codes[position]
            codes = np.array(codes, dtype=np.int64, copy=True)
            label = np.asarray([self._epoch_labels[position]], dtype=np.int64)
            return torch.from_numpy(codes), torch.from_numpy(label)

        codes = self.codes[index]
        if codes.ndim == 3:
            if codes.shape[0] != 2:
                raise ValueError(f"Expected two crop-range groups, got {codes.shape}")
            # Match upstream CustomDataset: choose the optional 1.05 range with
            # probability 0.5, then choose one of that range's ten crops.
            group = 1 if torch.rand(1) < 0.5 else 0
            variant = torch.randint(codes.shape[1], (1,)).item()
            codes = codes[group, variant]
        elif codes.ndim == 2:
            variant = torch.randint(codes.shape[0], (1,)).item()
            codes = codes[variant]
        codes = np.array(codes, dtype=np.int64, copy=True)
        label = np.asarray([self.labels[index]], dtype=np.int64)
        return torch.from_numpy(codes), torch.from_numpy(label)


def build_imagenet_code(args):
    root = Path(args.code_path)
    if not (root / "codes.npy").is_file() or not (root / "labels.npy").is_file():
        raise FileNotFoundError(
            f"Missing consolidated ImageNet codes under {root}; run extract_codes_c2i_parquet.py first"
        )
    return ConsolidatedImageNetCodeDataset(root)
