from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

from autoregressive.models.generate import generate
from autoregressive.models.gpt import GPT_models
from vqvae_rl.models.llamagen_tokenizer import extract_state_dict
from vqvae_rl.models.vendor.llamagen_vq_model import VQ_models


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def batch_indices(
    batch_index: int,
    per_rank_batch_size: int,
    world_size: int,
    rank: int,
) -> np.ndarray:
    global_batch_size = per_rank_batch_size * world_size
    start = batch_index * global_batch_size
    return start + rank + np.arange(per_rank_batch_size, dtype=np.int64) * world_size


def batch_labels(
    batch_index: int,
    global_batch_size: int,
    num_classes: int,
    global_seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(global_seed + batch_index * 1_000_003)
    return torch.randint(0, num_classes, (global_batch_size,), generator=generator)


def sampling_seed(global_seed: int, batch_index: int, rank: int) -> int:
    return global_seed + batch_index * 1_000_003 + rank * 10_007 + 17


def checkpoint_model_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("GPT checkpoint must be a mapping")
    for name in ("model", "module", "state_dict"):
        value = checkpoint.get(name)
        if isinstance(value, dict):
            return value
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("GPT checkpoint has no supported model state")


def atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def create_npz_from_sample_folder(
    sample_dir: Path,
    output_path: Path,
    num_samples: int,
    image_size: int,
    keep_staging_npy: bool,
) -> None:
    staging = output_path.with_suffix(".npy")
    temporary_npz = output_path.with_name(f"{output_path.stem}.partial.npz")
    images = np.lib.format.open_memmap(
        staging,
        mode="w+",
        dtype=np.uint8,
        shape=(num_samples, image_size, image_size, 3),
    )
    for index in tqdm(range(num_samples), desc="Packing FID samples"):
        path = sample_dir / f"{index:06d}.png"
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if array.shape != (image_size, image_size, 3):
            raise ValueError(f"Unexpected image shape in {path}: {array.shape}")
        images[index] = array
    images.flush()
    np.savez(temporary_npz, arr_0=images)
    os.replace(temporary_npz, output_path)
    del images
    if not keep_staging_npy:
        staging.unlink()


def write_labels(
    path: Path,
    num_samples: int,
    global_batch_size: int,
    num_classes: int,
    global_seed: int,
) -> None:
    total_samples = math.ceil(num_samples / global_batch_size) * global_batch_size
    labels = np.empty(total_samples, dtype=np.int16)
    for batch_index, start in enumerate(range(0, total_samples, global_batch_size)):
        labels[start : start + global_batch_size] = batch_labels(
            batch_index, global_batch_size, num_classes, global_seed
        ).numpy()
    np.save(path, labels[:num_samples])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-ckpt", type=Path, required=True)
    parser.add_argument("--vq-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpt-model", choices=sorted(GPT_models), default="GPT-XL")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--image-size-eval", type=int, default=256)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.75)
    parser.add_argument("--cfg-interval", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--per-rank-batch-size", type=int, default=4)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--skip-npz", action="store_true")
    parser.add_argument("--keep-staging-npy", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed c2i sampling requires CUDA")
    if args.image_size % args.downsample_size:
        raise ValueError("image-size must be divisible by downsample-size")
    if args.num_fid_samples <= 0 or args.per_rank_batch_size <= 0:
        raise ValueError("sample counts and batch size must be positive")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.set_grad_enabled(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "png"
    if rank == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier(device_ids=[local_rank])

    vq_checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    vq_state = extract_state_dict(vq_checkpoint)
    vq_model = VQ_models["VQ-16"](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_l2_norm=True,
        codebook_show_usage=False,
    )
    missing, unexpected = vq_model.load_state_dict(vq_state, strict=False)
    if missing or set(unexpected) - {"quantize.codebook_used"}:
        raise RuntimeError(
            f"Incompatible VQ checkpoint: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    vq_model.eval().requires_grad_(False).to(device)
    del vq_checkpoint, vq_state

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size**2,
        num_classes=args.num_classes,
        cls_token_num=1,
        model_type="c2i",
    ).to(device=device, dtype=dtype)
    gpt_checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
    checkpoint_steps = int(gpt_checkpoint.get("steps", -1))
    missing, unexpected = gpt_model.load_state_dict(
        checkpoint_model_state(gpt_checkpoint), strict=False
    )
    allowed_missing = {name for name in missing if name.endswith("freqs_cis")}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            f"Incompatible GPT checkpoint: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    gpt_model.eval().requires_grad_(False)
    del gpt_checkpoint
    if args.compile:
        gpt_model = torch.compile(gpt_model, mode="reduce-overhead", fullgraph=True)

    global_batch_size = args.per_rank_batch_size * world_size
    total_samples = math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size
    iterations = total_samples // global_batch_size
    iterator = (
        tqdm(range(iterations), desc="Generating c2i samples")
        if rank == 0
        else range(iterations)
    )
    for batch_index in iterator:
        indices = batch_indices(
            batch_index, args.per_rank_batch_size, world_size, rank
        )
        requested = indices < args.num_fid_samples
        paths = [sample_dir / f"{index:06d}.png" for index in indices[requested]]
        if paths and all(path.is_file() for path in paths):
            continue

        labels = batch_labels(
            batch_index, global_batch_size, args.num_classes, args.global_seed
        )[rank::world_size].to(device)
        torch.cuda.manual_seed(sampling_seed(args.global_seed, batch_index, rank))
        codes = generate(
            gpt_model,
            labels,
            latent_size**2,
            cfg_scale=args.cfg_scale,
            cfg_interval=args.cfg_interval,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_logits=True,
        )
        images = vq_model.decode_code(
            codes.reshape(-1),
            [len(labels), args.codebook_embed_dim, latent_size, latent_size],
        )
        if args.image_size_eval != args.image_size:
            images = F.interpolate(
                images,
                size=(args.image_size_eval, args.image_size_eval),
                mode="bicubic",
            )
        arrays = (
            torch.clamp(127.5 * images + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to(device="cpu", dtype=torch.uint8)
            .numpy()
        )
        for local_index, global_index in enumerate(indices):
            if global_index >= args.num_fid_samples:
                continue
            path = sample_dir / f"{global_index:06d}.png"
            if not path.exists():
                Image.fromarray(arrays[local_index], mode="RGB").save(path)

    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        png_count = sum(1 for _ in sample_dir.glob("*.png"))
        if png_count != args.num_fid_samples:
            raise RuntimeError(
                f"Expected {args.num_fid_samples} PNG files, found {png_count}"
            )
        labels_path = args.output_dir / "labels.npy"
        write_labels(
            labels_path,
            args.num_fid_samples,
            global_batch_size,
            args.num_classes,
            args.global_seed,
        )
        npz_path = args.output_dir / "samples.npz"
        if not args.skip_npz and not npz_path.exists():
            create_npz_from_sample_folder(
                sample_dir,
                npz_path,
                args.num_fid_samples,
                args.image_size_eval,
                args.keep_staging_npy,
            )
        manifest = {
            "complete": not args.skip_npz,
            "gpt_checkpoint": str(args.gpt_ckpt.resolve()),
            "gpt_checkpoint_steps": checkpoint_steps,
            "vq_checkpoint": str(args.vq_ckpt.resolve()),
            "gpt_model": args.gpt_model,
            "gpt_type": "c2i",
            "image_size": args.image_size,
            "image_size_eval": args.image_size_eval,
            "latent_size": latent_size,
            "num_classes": args.num_classes,
            "codebook_size": args.codebook_size,
            "codebook_embed_dim": args.codebook_embed_dim,
            "cfg_scale": args.cfg_scale,
            "cfg_interval": args.cfg_interval,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "num_fid_samples": args.num_fid_samples,
            "global_seed": args.global_seed,
            "world_size": world_size,
            "per_rank_batch_size": args.per_rank_batch_size,
            "sampling_scheme": "paired_batch_seed_v1",
            "samples_npz": str(npz_path.resolve()) if not args.skip_npz else None,
            "labels_npy": str(labels_path.resolve()),
        }
        atomic_json_write(args.output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2), flush=True)
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
