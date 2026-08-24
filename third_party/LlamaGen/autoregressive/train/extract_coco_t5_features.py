from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from language.t5 import T5Embedder


class ManifestPrompts(Dataset):
    def __init__(self, manifest: Path):
        self.records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return (
            str(record["prompt"]),
            int(record["index"]),
            int(record.get("image_index", record["index"])),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--t5-root", type=Path, required=True)
    parser.add_argument("--t5-model-type", default="flan-t5-xl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=120)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--global-seed", type=int, default=20260726)
    parser.add_argument("--sharded-only", action="store_true")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.global_seed + rank)
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise RuntimeError(f"Fresh output required: {args.output}")
        (args.output / "shards").mkdir(parents=True, exist_ok=True)
    dist.barrier()

    dataset = ManifestPrompts(args.manifest)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    embedder = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=str(args.t5_root),
        dir_or_name=args.t5_model_type,
        torch_dtype=dtype,
        model_max_length=args.max_length,
    )

    embedding_parts = []
    length_parts = []
    index_parts = []
    image_index_parts = []
    processed = 0
    with torch.no_grad():
        for prompts, indices, image_indices in loader:
            cleaned = [embedder.text_preprocessing(prompt) for prompt in prompts]
            tokens = embedder.tokenizer(
                cleaned,
                max_length=args.max_length,
                padding=True,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            outputs = embedder.model(
                input_ids=tokens["input_ids"].to(device),
                attention_mask=tokens["attention_mask"].to(device),
            )["last_hidden_state"]
            lengths = tokens["attention_mask"].sum(dim=1).to(torch.int16)
            for embedding, length in zip(outputs, lengths.tolist()):
                embedding_parts.append(embedding[:length].float().cpu().numpy())
            length_parts.append(lengths.numpy())
            index_parts.append(indices.numpy().astype(np.int64, copy=False))
            image_index_parts.append(image_indices.numpy().astype(np.int64, copy=False))
            processed += len(indices)
            if rank == 0 and processed % 1000 < len(indices):
                print(f"rank0 processed {processed}/{len(sampler)}", flush=True)

    np.save(args.output / "shards" / f"rank{rank:02d}_embeddings.npy", np.concatenate(embedding_parts))
    np.save(args.output / "shards" / f"rank{rank:02d}_lengths.npy", np.concatenate(length_parts))
    np.save(args.output / "shards" / f"rank{rank:02d}_indices.npy", np.concatenate(index_parts))
    np.save(
        args.output / "shards" / f"rank{rank:02d}_image_indices.npy",
        np.concatenate(image_index_parts),
    )
    dist.barrier()
    if args.sharded_only:
        if rank == 0:
            print(
                json.dumps(
                    {
                        "sample_count": len(dataset),
                        "world_size": world_size,
                        "storage": "raw_shards_pending_index_finalization",
                    },
                    indent=2,
                ),
                flush=True,
            )
        dist.destroy_process_group()
        return
    if rank == 0:
        global_lengths = np.zeros(len(dataset), dtype=np.int16)
        global_image_indices = np.zeros(len(dataset), dtype=np.int64)
        locations: dict[int, tuple[int, int, int]] = {}
        for item in range(world_size):
            lengths = np.load(args.output / "shards" / f"rank{item:02d}_lengths.npy")
            indices = np.load(args.output / "shards" / f"rank{item:02d}_indices.npy")
            image_indices = np.load(
                args.output / "shards" / f"rank{item:02d}_image_indices.npy"
            )
            cursor = 0
            for index, image_index, length in zip(
                indices.tolist(), image_indices.tolist(), lengths.tolist()
            ):
                if index not in locations:
                    locations[index] = (item, cursor, length)
                    global_lengths[index] = length
                    global_image_indices[index] = image_index
                cursor += length
        if sorted(locations) != list(range(len(dataset))):
            raise RuntimeError("Merged T5 feature indices are incomplete")
        offsets = np.zeros(len(dataset) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(global_lengths, dtype=np.int64)
        output = np.lib.format.open_memmap(
            args.output / "text_embeddings.npy",
            mode="w+",
            dtype=np.float32,
            shape=(int(offsets[-1]), 2048),
        )
        shard_arrays = {
            item: np.load(args.output / "shards" / f"rank{item:02d}_embeddings.npy", mmap_mode="r")
            for item in range(world_size)
        }
        for index in range(len(dataset)):
            item, start, length = locations[index]
            output[offsets[index] : offsets[index + 1]] = shard_arrays[item][start : start + length]
        output.flush()
        np.save(args.output / "offsets.npy", offsets)
        np.save(args.output / "image_indices.npy", global_image_indices)
        metadata = {
            "manifest": str(args.manifest),
            "sample_count": len(dataset),
            "embedding_rows": int(offsets[-1]),
            "embedding_dim": 2048,
            "embedding_dtype": "float32",
            "max_length": args.max_length,
            "t5_root": str(args.t5_root),
            "t5_model_type": args.t5_model_type,
            "world_size": world_size,
            "global_seed": args.global_seed,
        }
        (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
