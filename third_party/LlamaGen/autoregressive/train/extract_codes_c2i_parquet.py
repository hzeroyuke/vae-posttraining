from __future__ import annotations

import argparse
from bisect import bisect_right
from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from dataset.augmentation import center_crop_arr
from vqvae_rl.models.vendor.llamagen_vq_model import VQ_models
from vqvae_rl.models.llamagen_tokenizer import load_llamagen_checkpoint


class ImageNetParquetDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: int, crop_range: float, ten_crop: bool):
        data_root = root / "data" if (root / "data").is_dir() else root
        self.files = sorted(data_root.glob(f"{split}-*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No {split!r} parquet shards under {data_root}")
        self.row_groups = []
        self.offsets = [0]
        for file_index, path in enumerate(self.files):
            parquet = pq.ParquetFile(path)
            for row_group in range(parquet.num_row_groups):
                rows = parquet.metadata.row_group(row_group).num_rows
                self.row_groups.append((file_index, row_group))
                self.offsets.append(self.offsets[-1] + rows)
        self.image_size = image_size
        self.crop_size = round(image_size * crop_range) if ten_crop else image_size
        self.ten_crop = ten_crop
        self._cache_key = None
        self._cache_table = None

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index: int):
        group_index = bisect_right(self.offsets, index) - 1
        file_index, row_group = self.row_groups[group_index]
        cache_key = (file_index, row_group)
        if cache_key != self._cache_key:
            parquet = pq.ParquetFile(self.files[file_index], memory_map=True)
            self._cache_table = parquet.read_row_group(row_group, columns=["image", "label"])
            self._cache_key = cache_key
        row_index = index - self.offsets[group_index]
        record = self._cache_table["image"][row_index].as_py()
        image = Image.open(BytesIO(record["bytes"])).convert("RGB")
        image = center_crop_arr(image, self.crop_size)
        if self.ten_crop:
            crops = transforms.TenCrop(self.image_size)(image)
            tensor = torch.stack([transforms.ToTensor()(crop) for crop in crops])
            tensor = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                inplace=True,
            )(tensor)
        else:
            tensor = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                inplace=True,
            )(transforms.ToTensor()(image))
        return tensor, int(self._cache_table["label"][row_index].as_py()), index


class ContiguousDistributedSampler(torch.utils.data.Sampler[int]):
    """Partition a sorted subset into contiguous ranges for row-group locality."""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        self.num_replicas = num_replicas
        self.rank = rank
        total = len(dataset)
        self.start = (total * rank) // num_replicas
        self.end = (total * (rank + 1)) // num_replicas

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self):
        return self.end - self.start


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_sha256(checkpoint: Path, code_path: Path) -> str:
    """Reuse a sibling corpus hash for the same immutable checkpoint path."""
    checkpoint_text = str(checkpoint)
    for manifest_path in sorted(code_path.parent.glob("*/manifest.json")):
        if manifest_path.parent == code_path:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        digest = manifest.get("vq_checkpoint_sha256")
        if (
            manifest.get("vq_checkpoint") == checkpoint_text
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest.lower())
        ):
            return digest.lower()
    return sha256(checkpoint)


def load_existing_indices(code_path: Path, sample_count: int) -> np.ndarray:
    """Return the sorted sample indices already materialized in shard parts."""
    seen = np.zeros(sample_count, dtype=np.bool_)
    for part_path in sorted((code_path / "shards").glob("rank*.part*.npz")):
        try:
            with np.load(part_path) as shard:
                indices = np.asarray(shard["indices"], dtype=np.int64)
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            # A worker can be interrupted while replacing a shard. The final
            # consolidation still verifies every sample through ``seen``.
            continue
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= sample_count):
            raise ValueError(f"Invalid indices in {part_path}")
        seen[indices] = True
    return np.flatnonzero(~seen).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--code-path", type=Path, required=True)
    parser.add_argument("--vq-ckpt", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-range", type=float, default=1.1)
    parser.add_argument("--ten-crop", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--part-batches", type=int, default=256,
                        help="number of batches buffered per NAS shard part")
    parser.add_argument("--global-seed", type=int, default=20260812)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="resume from existing shard parts and encode only missing samples")
    parser.add_argument("--indices-file", type=Path, default=None,
                        help="optional .npy file of original sample indices to encode")
    parser.add_argument("--shard-rank", type=int, default=-1,
                        help="shard filename rank, useful for an extra independent worker")
    parser.add_argument("--shard-only", action="store_true",
                        help="write shard parts without consolidating or deleting them")
    parser.add_argument("--contiguous", action="store_true",
                        help="partition sorted indices contiguously across ranks")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.global_seed + rank)
    if rank == 0:
        if args.code_path.exists() and any(args.code_path.iterdir()) and not (args.resume or args.indices_file is not None):
            raise RuntimeError(f"Fresh code path required: {args.code_path}")
        (args.code_path / "shards").mkdir(parents=True, exist_ok=True)
    dist.barrier()
    base_dataset = ImageNetParquetDataset(args.data_path, args.split, args.image_size, args.crop_range, args.ten_crop)
    sample_count = min(args.max_samples, len(base_dataset)) if args.max_samples else len(base_dataset)
    if args.indices_file is not None:
        selected = np.asarray(np.load(args.indices_file), dtype=np.int64)
        if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= sample_count):
            raise ValueError(f"Invalid indices file: {args.indices_file}")
        dataset = Subset(base_dataset, selected.tolist())
    elif args.resume:
        remaining_path = args.code_path / "resume_remaining.npy"
        if rank == 0:
            remaining = load_existing_indices(args.code_path, sample_count)
            np.save(remaining_path, remaining)
        dist.barrier()
        remaining = np.load(remaining_path, mmap_mode="r")
        dataset = Subset(base_dataset, remaining.tolist())
    else:
        dataset = Subset(base_dataset, range(sample_count)) if args.max_samples else base_dataset
    sampler_cls = ContiguousDistributedSampler if args.contiguous else DistributedSampler
    sampler = sampler_cls(dataset, num_replicas=world_size, rank=rank, **({} if args.contiguous else {
        "shuffle": False, "drop_last": False, "seed": args.global_seed,
    }))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    # Consolidation-only resume runs have no samples left to encode; avoid a
    # needless checkpoint read from NAS before merging the existing shards.
    model = None
    if len(sampler):
        model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8, codebook_l2_norm=True, codebook_show_usage=False)
        load_llamagen_checkpoint(model, str(args.vq_ckpt))
        model = model.to(device).eval()
    code_parts, label_parts, index_parts = [], [], []
    shard_rank = rank if args.shard_rank < 0 else args.shard_rank
    existing_parts = sorted((args.code_path / "shards").glob(f"rank{shard_rank:02d}.part*.npz"))
    part_index = 0
    if existing_parts:
        part_index = max(int(path.stem.rsplit("part", 1)[1]) for path in existing_parts) + 1

    def flush_part() -> None:
        nonlocal part_index, code_parts, label_parts, index_parts
        if not code_parts:
            return
        np.savez(
            args.code_path / "shards" / f"rank{shard_rank:02d}.part{part_index:05d}.npz",
            codes=np.concatenate(code_parts),
            labels=np.concatenate(label_parts),
            indices=np.concatenate(index_parts),
        )
        part_index += 1
        code_parts, label_parts, index_parts = [], [], []

    with torch.inference_mode():
        for batch_index, (images, labels, indices) in enumerate(loader):
            if args.ten_crop:
                batch, crops = images.shape[:2]
                flat = images.flatten(0, 1).to(device, non_blocking=True)
                _, _, info = model.encode(flat)
                codes = info[-1].reshape(batch, crops, -1)
            else:
                images = images.to(device, non_blocking=True)
                flat = torch.cat([images, torch.flip(images, dims=[-1])], dim=0)
                _, _, info = model.encode(flat)
                encoded = info[-1].reshape(2, images.shape[0], -1)
                codes = encoded.permute(1, 0, 2).contiguous()
            code_parts.append(codes.cpu().numpy().astype(np.int16))
            label_parts.append(labels.numpy().astype(np.int16))
            index_parts.append(indices.numpy().astype(np.int64))
            if rank == 0 and (batch_index + 1) % 100 == 0:
                print(f"rank0 processed {(batch_index + 1) * args.batch_size:,}/{len(sampler):,}", flush=True)
            if len(code_parts) >= args.part_batches:
                flush_part()
    flush_part()
    if args.shard_only:
        dist.barrier()
        dist.destroy_process_group()
        return
    dist.barrier()
    if rank == 0:
        part_paths = sorted((args.code_path / "shards").glob("rank*.part*.npz"))
        if not part_paths:
            raise RuntimeError("No code shard parts were written")
        first_path = None
        for candidate in part_paths:
            try:
                with np.load(candidate) as first:
                    code_shape = (sample_count,) + first["codes"].shape[1:]
                    code_dtype = first["codes"].dtype
                first_path = candidate
                break
            except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                continue
        if first_path is None:
            raise RuntimeError("No readable code shard parts were written")
        codes = np.lib.format.open_memmap(args.code_path / "codes.npy", mode="w+", dtype=code_dtype, shape=code_shape)
        labels = np.lib.format.open_memmap(args.code_path / "labels.npy", mode="w+", dtype=np.int16, shape=(sample_count,))
        seen = np.zeros(sample_count, dtype=bool)
        for part_path in part_paths:
            try:
                with np.load(part_path) as shard:
                    indices = shard["indices"].astype(np.int64, copy=False)
                    keep = ~seen[indices]
                    if np.any(keep):
                        selected = indices[keep]
                        codes[selected] = shard["codes"][keep]
                        labels[selected] = shard["labels"][keep]
                        seen[selected] = True
            except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                continue
        codes.flush(); labels.flush()
        if not np.all(seen):
            raise RuntimeError(f"Incomplete indices: {(~seen).sum()} missing from {sample_count}")
        for part_path in part_paths:
            part_path.unlink()
        if (args.code_path / "resume_remaining.npy").exists():
            (args.code_path / "resume_remaining.npy").unlink()
        manifest = {"data_path": str(args.data_path), "split": args.split, "sample_count": sample_count, "augmentations_per_image": int(codes.shape[1]) if codes.ndim == 3 else 1,
                    "tokens_per_image": int(codes.shape[-1]), "image_size": args.image_size, "ten_crop": args.ten_crop, "crop_range": args.crop_range,
                    "vq_checkpoint": str(args.vq_ckpt), "vq_checkpoint_sha256": checkpoint_sha256(args.vq_ckpt, args.code_path), "world_size": world_size, "global_seed": args.global_seed}
        (args.code_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2), flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
