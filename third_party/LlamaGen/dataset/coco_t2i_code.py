from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CocoT2ICodeDataset(Dataset):
    def __init__(self, code_path: str, text_feature_path: str, train_count: int):
        self.code_path = Path(code_path)
        self.text_feature_path = Path(text_feature_path)
        self.codes = np.load(self.code_path / "codes.npy", mmap_mode="r")
        shard_ids_path = self.text_feature_path / "shard_ids.npy"
        self.sharded_features = shard_ids_path.is_file()
        if self.sharded_features:
            self.lengths = np.load(self.text_feature_path / "lengths.npy", mmap_mode="r")
            self.shard_ids = np.load(shard_ids_path, mmap_mode="r")
            self.shard_offsets = np.load(
                self.text_feature_path / "shard_offsets.npy", mmap_mode="r"
            )
            self.embedding_shards = {
                int(shard_id): np.load(
                    self.text_feature_path
                    / "shards"
                    / f"rank{int(shard_id):02d}_embeddings.npy",
                    mmap_mode="r",
                )
                for shard_id in np.unique(self.shard_ids)
            }
            self.embeddings = None
            self.offsets = None
            text_count = len(self.lengths)
        else:
            self.embeddings = np.load(
                self.text_feature_path / "text_embeddings.npy", mmap_mode="r"
            )
            self.offsets = np.load(self.text_feature_path / "offsets.npy", mmap_mode="r")
            self.lengths = None
            self.shard_ids = None
            self.shard_offsets = None
            self.embedding_shards = None
            text_count = len(self.offsets) - 1
        image_indices_path = self.text_feature_path / "image_indices.npy"
        self.image_indices = (
            np.load(image_indices_path, mmap_mode="r") if image_indices_path.is_file() else None
        )
        if self.codes.ndim != 3:
            raise ValueError(f"Expected codes [N,K,T], got {self.codes.shape}")
        if self.image_indices is None and text_count != len(self.codes):
            raise ValueError("Code and text feature counts differ without an image-index map")
        if self.image_indices is not None:
            if len(self.image_indices) != text_count:
                raise ValueError("Text feature and image-index counts differ")
            if self.image_indices.min() < 0 or self.image_indices.max() >= len(self.codes):
                raise ValueError("Text feature image index is outside the code array")
        if not 0 < train_count <= text_count:
            raise ValueError(f"Invalid train_count={train_count} for {text_count} text records")
        self.train_count = train_count
        self.variants_per_image = int(self.codes.shape[1])
        self.flip = False
        self.augmentations_per_image = self.variants_per_image

    def __len__(self) -> int:
        return self.train_count

    def __getitem__(self, index: int):
        variant = torch.randint(self.variants_per_image, (1,)).item()
        image_index = int(self.image_indices[index]) if self.image_indices is not None else index
        codes = np.array(self.codes[image_index, variant], dtype=np.int64, copy=True)
        if self.sharded_features:
            shard_id = int(self.shard_ids[index])
            start = int(self.shard_offsets[index])
            length = int(self.lengths[index])
            embedding = np.array(
                self.embedding_shards[shard_id][start : start + length],
                dtype=np.float32,
                copy=True,
            )
        else:
            start, end = int(self.offsets[index]), int(self.offsets[index + 1])
            embedding = np.array(self.embeddings[start:end], dtype=np.float32, copy=True)
        return torch.from_numpy(codes), torch.from_numpy(embedding)


def collate_coco_t2i(batch, max_text_length: int = 120):
    codes = torch.stack([item[0] for item in batch])
    embeddings = torch.zeros(len(batch), max_text_length, 2048, dtype=torch.float32)
    text_mask = torch.zeros(len(batch), max_text_length, dtype=torch.bool)
    for index, (_, embedding) in enumerate(batch):
        length = min(max_text_length, len(embedding))
        embeddings[index, -length:] = embedding[:length]
        text_mask[index, -length:] = True

    input_length = max_text_length + codes.shape[1] - 1
    causal = torch.tril(torch.ones(input_length, input_length, dtype=torch.bool))
    attention_mask = causal.unsqueeze(0).repeat(len(batch), 1, 1)
    attention_mask[:, :, :max_text_length] &= text_mask[:, None, :]
    diagonal = torch.eye(input_length, dtype=torch.bool).unsqueeze(0)
    attention_mask |= diagonal
    valid = torch.ones(len(batch), dtype=torch.long)
    return codes, embeddings, attention_mask.unsqueeze(1), valid


def read_prompts(manifest: str, start: int, count: int) -> list[dict]:
    records = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line]
    return records[start : start + count]
