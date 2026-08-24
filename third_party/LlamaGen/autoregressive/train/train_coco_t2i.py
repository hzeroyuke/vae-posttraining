from __future__ import annotations

import argparse
from contextlib import nullcontext
from glob import glob
import inspect
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from autoregressive.models.gpt import GPT_models
from dataset.coco_t2i_code import CocoT2ICodeDataset, collate_coco_t2i
from utils.distributed import init_distributed_mode
from utils.logger import create_logger


def create_optimizer(model, weight_decay, learning_rate, betas, logger):
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    decay = [value for value in parameters.values() if value.dim() >= 2]
    no_decay = [value for value in parameters.values() if value.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    logger.info(
        f"optimizer tensors: decay={len(decay)}, no_decay={len(no_decay)}, fused={fused}"
    )
    return torch.optim.AdamW(
        groups, lr=learning_rate, betas=betas, **({"fused": True} if fused else {})
    )


def main(args) -> None:
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    init_distributed_mode(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    effective_micro_world = world_size * args.gradient_accumulation_steps
    if args.global_batch_size % effective_micro_world:
        raise ValueError(
            "global batch must be divisible by world size times gradient accumulation steps"
        )
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{args.gpt_model}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        wandb_run = None
        if args.wandb_project:
            import wandb

            wandb_run = wandb.init(
                entity=args.wandb_entity or None,
                project=args.wandb_project,
                name=args.wandb_name or args.gpt_model,
                id=args.wandb_run_id or None,
                resume="allow" if args.wandb_run_id else None,
                config=vars(args),
            )
    else:
        logger = create_logger(None)
        wandb_run = None
    logger.info(f"{args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={world_size}")

    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size**2,
        cls_token_num=args.cls_token_num,
        model_type="t2i",
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    optimizer = create_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)
    dataset = CocoT2ICodeDataset(args.code_path, args.text_feature_path, args.train_count)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size // effective_micro_world,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_coco_t2i,
    )
    logger.info(
        f"Dataset contains {len(dataset):,} caption records with "
        f"{dataset.variants_per_image} sampled-code variants per image"
    )
    logger.info(
        f"micro batch per rank={args.global_batch_size // effective_micro_world}, "
        f"gradient accumulation={args.gradient_accumulation_steps}, "
        f"effective global batch={args.global_batch_size}"
    )
    if len(loader) % args.gradient_accumulation_steps:
        raise ValueError(
            f"loader length {len(loader)} is not divisible by gradient accumulation "
            f"{args.gradient_accumulation_steps}"
        )

    if args.save_initial_checkpoint:
        if rank == 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "steps": 0, "args": args},
                f"{checkpoint_dir}/0000000.pt",
            )
            logger.info(f"Saved initial checkpoint to {checkpoint_dir}/0000000.pt")
        dist.barrier()

    model = DDP(model, device_ids=[args.gpu], gradient_as_bucket_view=True)
    model.train()
    dtype = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.mixed_precision
    ]
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision == "fp16")
    train_steps = 0
    running_loss = 0.0
    log_steps = 0
    start_time = time.time()
    stop = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        accumulated_loss = 0.0
        for micro_step, (codes, conditioning, attention_mask, valid) in enumerate(loader, start=1):
            codes = codes.to(device, non_blocking=True)
            conditioning = conditioning.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            update_step = micro_step % args.gradient_accumulation_steps == 0
            sync_context = nullcontext() if update_step else model.no_sync()
            with sync_context:
                with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                    _, loss = model(
                        cond_idx=conditioning,
                        idx=codes[:, :-1],
                        targets=codes,
                        mask=attention_mask,
                        valid=valid,
                    )
                scaler.scale(loss / args.gradient_accumulation_steps).backward()
            accumulated_loss += loss.item()
            if not update_step:
                continue
            if args.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_steps += 1
            running_loss += accumulated_loss / args.gradient_accumulation_steps
            accumulated_loss = 0.0
            log_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                average = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(average)
                average = average.item() / world_size
                speed = log_steps / elapsed
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {average:.4f}, Train Steps/Sec: {speed:.2f}"
                )
                if rank == 0 and wandb_run is not None:
                    wandb_run.log(
                        {"train/loss": average, "train/steps_per_sec": speed, "train/epoch": epoch},
                        step=train_steps,
                    )
                running_loss = 0.0
                log_steps = 0
                start_time = time.time()

            should_save = train_steps % args.ckpt_every == 0
            if args.max_steps > 0 and train_steps == args.max_steps:
                should_save = True
            if should_save:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args,
                    }
                    path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, path)
                    logger.info(f"Saved checkpoint to {path}")
                dist.barrier()
            if args.max_steps > 0 and train_steps >= args.max_steps:
                stop = True
                break
        if stop:
            logger.info(f"Reached max_steps={args.max_steps}")
            break
    logger.info("Done!")
    if rank == 0 and wandb_run is not None:
        wandb_run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", required=True)
    parser.add_argument("--text-feature-path", required=True)
    parser.add_argument("--train-count", type=int, default=50000)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--gpt-model", default="GPT-B", choices=list(GPT_models))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--cls-token-num", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=78000)
    parser.add_argument("--global-batch-size", type=int, default=192)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--global-seed", type=int, default=20260726)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--save-initial-checkpoint", action="store_true")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-run-id", default="")
    main(parser.parse_args())
