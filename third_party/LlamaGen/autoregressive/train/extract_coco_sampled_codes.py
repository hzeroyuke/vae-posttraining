from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from dataset.augmentation import center_crop_arr
from vqvae_rl.models.llamagen_tokenizer import StochasticLlamaGenTokenizer


class ManifestImages(Dataset):
    def __init__(self, manifest: Path, image_size: int, crop_mode: str):
        self.records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        self.image_size = image_size
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            if self.crop_mode == "center-crop":
                image = center_crop_arr(image, self.image_size)
            else:
                image = image.resize(
                    (self.image_size, self.image_size), Image.Resampling.BICUBIC
                )
            array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(127.5).sub(1.0)
        return tensor, int(record["index"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vq-ckpt", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-mode", choices=("resize", "center-crop"), default="resize")
    parser.add_argument("--samples-per-image", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=20260726)
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

    dataset = ManifestImages(args.manifest, args.image_size, args.crop_mode)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    model = StochasticLlamaGenTokenizer(
        checkpoint=str(args.vq_ckpt),
        codebook_size=16384,
        codebook_embed_dim=8,
        codebook_l2_norm=True,
        freeze_encoder=True,
        freeze_decoder=True,
        freeze_codebook=True,
    ).to(device).eval()

    code_parts = []
    index_parts = []
    processed = 0
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            logits = model.encode_logits(images)
            batch, vocab, height, width = logits.shape
            flat = logits.permute(0, 2, 3, 1).reshape(-1, vocab).float()
            sampled = Categorical(logits=flat / args.temperature).sample((args.samples_per_image,))
            codes = sampled.view(args.samples_per_image, batch, height, width).permute(1, 0, 2, 3)
            code_parts.append(codes.reshape(batch, args.samples_per_image, -1).to(torch.int16).cpu().numpy())
            index_parts.append(indices.numpy().astype(np.int64, copy=False))
            processed += batch
            if rank == 0 and processed % 1000 < batch:
                print(f"rank0 processed {processed}/{len(sampler)}", flush=True)

    np.save(args.output / "shards" / f"rank{rank:02d}_codes.npy", np.concatenate(code_parts))
    np.save(args.output / "shards" / f"rank{rank:02d}_indices.npy", np.concatenate(index_parts))
    dist.barrier()
    if rank == 0:
        codes_parts = []
        indices_parts = []
        for item in range(world_size):
            codes_parts.append(np.load(args.output / "shards" / f"rank{item:02d}_codes.npy", mmap_mode="r"))
            indices_parts.append(np.load(args.output / "shards" / f"rank{item:02d}_indices.npy"))
        codes = np.concatenate(codes_parts)
        indices = np.concatenate(indices_parts)
        order = np.argsort(indices, kind="stable")
        codes, indices = codes[order], indices[order]
        keep = np.concatenate([[True], indices[1:] != indices[:-1]])
        codes, indices = codes[keep], indices[keep]
        if not np.array_equal(indices, np.arange(len(dataset))):
            raise RuntimeError("Merged code indices are incomplete")
        np.save(args.output / "codes.npy", codes)
        metadata = {
            "manifest": str(args.manifest),
            "sample_count": len(dataset),
            "samples_per_image": args.samples_per_image,
            "tokens_per_image": int(codes.shape[-1]),
            "temperature": args.temperature,
            "code_dtype": str(codes.dtype),
            "vq_checkpoint": str(args.vq_ckpt),
            "vq_checkpoint_sha256": sha256(args.vq_ckpt),
            "world_size": world_size,
            "global_seed": args.global_seed,
            "crop_mode": args.crop_mode,
            "sampling": "categorical_policy_no_selection",
        }
        (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
