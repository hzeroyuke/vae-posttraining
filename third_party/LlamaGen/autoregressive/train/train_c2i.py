# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from copy import deepcopy
import os
import time
import inspect
import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models
from autoregressive.train.checkpoint_resume import resolve_resume_geometry


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def atomic_torch_save(payload, path):
    """Publish checkpoints only after torch.save has completed successfully."""
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def capture_rng_state(device):
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng_state(state, device):
    if not state:
        return
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], device)


class FixedIndexSampler(Sampler[int]):
    """Yield a precomputed sampler suffix without touching its skipped prefix."""

    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)

    def __iter__(self):
        return (int(index) for index in self.indices)

    def __len__(self):
        return int(len(self.indices))


def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer



#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    init_distributed_mode(args)
    effective_world = dist.get_world_size() * args.gradient_accumulation_steps
    assert args.global_batch_size % effective_world == 0, (
        "Global batch must be divisible by world size times gradient accumulation"
    )
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
        if args.gpt_ckpt:
            checkpoint_path = Path(args.gpt_ckpt).resolve()
            experiment_path = checkpoint_path.parent.parent
            if (
                checkpoint_path.parent.name != "checkpoints"
                or experiment_path.parent != Path(args.results_dir).resolve()
            ):
                raise ValueError(
                    "--gpt-ckpt must be under the requested results directory so "
                    "resumed checkpoints can be written in place"
                )
            experiment_dir = str(experiment_path)
            experiment_name = experiment_path.name
        else:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            experiment_name = f"{experiment_index:03d}-{model_string_name}"
            experiment_dir = f"{args.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
        wandb_run = None
        if args.wandb_project:
            import wandb

            wandb_run = wandb.init(
                entity=args.wandb_entity or None,
                project=args.wandb_project,
                name=args.wandb_name or model_string_name,
                id=args.wandb_run_id or None,
                resume="allow" if args.wandb_run_id else None,
                config=vars(args),
            )

    else:
        logger = create_logger(None)
        wandb_run = None

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    if args.init_from and args.gpt_ckpt:
        raise ValueError("--init-from and --gpt-ckpt are mutually exclusive")
    if args.init_from:
        checkpoint = torch.load(
            args.init_from, map_location="cpu", mmap=True, weights_only=False
        )
        source_key = "ema" if args.init_from_ema and "ema" in checkpoint else "model"
        if source_key not in checkpoint:
            raise KeyError(f"Checkpoint {args.init_from} has no {source_key!r} state")
        model.load_state_dict(checkpoint[source_key], strict=True)
        if args.ema:
            ema.load_state_dict(checkpoint[source_key], strict=True)
        del checkpoint
        logger.info(
            f"Initialized model weights from {args.init_from} ({source_key}); "
            "optimizer and adaptation step are reset"
        )

    if args.dataset == "imagenet_paired_code" and not args.replay_code_path:
        raise ValueError("imagenet_paired_code requires --replay-code-path")
    if args.dataset != "imagenet_paired_code" and args.replay_code_path:
        raise ValueError("--replay-code-path requires --dataset imagenet_paired_code")
    if args.kl_weight > 0 and not args.teacher_ckpt:
        raise ValueError("--kl-weight > 0 requires --teacher-ckpt")

    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # Setup data:
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // effective_world),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    flip_info = 'with' if dataset.flip else 'without'
    if hasattr(dataset, "augmentations_per_image"):
        aug_info = dataset.augmentations_per_image
    else:
        aug_info = 10 if 'ten_crop' in dataset.feature_dir else 1
        aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
    logger.info(f"Dataset contains {len(dataset):,} images ({args.code_path}) "
                f"{flip_info} flip augmentation and {aug_info} crop augmentation")

    # Prepare models for training:
    resume_rng_state = None
    resume_micro_steps = 0
    if args.gpt_ckpt:
        # Resume checkpoints live on NAS.  mmap makes torch.load touch optimizer
        # tensors through many random page faults, which can stall for minutes;
        # a regular sequential load is faster and remains safe for the 2-rank run.
        checkpoint = torch.load(
            args.gpt_ckpt, map_location="cpu", mmap=False, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = int(checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0]))
        updates_per_epoch = len(loader) // args.gradient_accumulation_steps
        if updates_per_epoch <= 0:
            raise ValueError("Data loader has fewer micro-batches than one accumulation window")
        start_epoch = int(checkpoint.get("epoch", train_steps // updates_per_epoch))
        resume_geometry = resolve_resume_geometry(
            checkpoint,
            fallback_micro_steps=(
                (train_steps % updates_per_epoch) * args.gradient_accumulation_steps
            ),
            current_world_size=dist.get_world_size(),
            current_gradient_accumulation=args.gradient_accumulation_steps,
            current_global_batch_size=args.global_batch_size,
        )
        resume_micro_steps = resume_geometry.micro_steps
        if resume_micro_steps > len(loader):
            raise ValueError(
                f"Checkpoint resume position {resume_micro_steps} exceeds "
                f"the current loader length {len(loader)}"
            )
        rng_states = checkpoint.get("rng_states")
        world_size_changed = resume_geometry.source_world_size != dist.get_world_size()
        if rng_states is not None and not world_size_changed and rank < len(rng_states):
            resume_rng_state = rng_states[rank]
        if resume_geometry.saved_micro_steps != resume_micro_steps:
            logger.info(
                "Rescaled checkpoint epoch position from "
                f"{resume_geometry.saved_micro_steps} to {resume_micro_steps} rank-local "
                f"micro-batches ({resume_geometry.consumed_samples_in_epoch:,} global samples)"
            )
        if world_size_changed:
            logger.info(
                "Checkpoint world size changed from "
                f"{resume_geometry.source_world_size} to {dist.get_world_size()}; "
                "starting fresh rank-local RNG streams from the configured seed"
            )
        del checkpoint
        logger.info(
            f"Resume training from checkpoint: {args.gpt_ckpt}; "
            f"steps={train_steps}, epoch={start_epoch}, "
            f"micro_steps_in_epoch={resume_micro_steps}"
        )
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema and not args.init_from:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    if args.save_initial_checkpoint:
        if train_steps != 0:
            raise ValueError("--save-initial-checkpoint is only valid for a fresh run")
        initial_rng_states = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(capture_rng_state(device), initial_rng_states, dst=0)
        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "steps": 0,
                "args": args,
                "epoch": 0,
                "micro_steps_in_epoch": 0,
                "rng_states": initial_rng_states,
            }
            if args.ema:
                checkpoint["ema"] = ema.state_dict()
            checkpoint_path = f"{checkpoint_dir}/0000000.pt"
            atomic_torch_save(checkpoint, checkpoint_path)
            logger.info(f"Saved initial checkpoint to {checkpoint_path}")
        dist.barrier()

    teacher = None
    if args.kl_weight > 0:
        teacher = deepcopy(model).to(device)
        teacher_checkpoint = torch.load(
            args.teacher_ckpt, map_location="cpu", mmap=True, weights_only=False
        )
        teacher_source = "ema" if args.teacher_from_ema and "ema" in teacher_checkpoint else "model"
        if teacher_source not in teacher_checkpoint:
            raise KeyError(f"Checkpoint {args.teacher_ckpt} has no {teacher_source!r} state")
        teacher.load_state_dict(teacher_checkpoint[teacher_source], strict=True)
        del teacher_checkpoint
        requires_grad(teacher, False)
        teacher.train()
        logger.info(
            f"Frozen KL teacher loaded from {args.teacher_ckpt} ({teacher_source}); "
            "teacher/student replay forwards share dropout RNG"
        )

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0

    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_new_ce = 0
    running_replay_ce = 0
    running_kl = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    stop_training = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        # ImageNet size is not divisible by the effective batch. Do not carry a
        # partial accumulation across sampler epochs; every optimizer update
        # should represent exactly global_batch_size images.
        epoch_resume_micro_steps = resume_micro_steps if epoch == start_epoch else 0
        sampler.set_epoch(epoch)
        resume_indices = None
        sampler_indices = None
        if epoch_resume_micro_steps:
            sampler_indices = np.fromiter(
                iter(sampler), dtype=np.int64, count=len(sampler)
            )
            # The skipped prefix is already represented by the checkpoint;
            # avoid reading it again during a geometry-aware resume.
            resume_prefix = epoch_resume_micro_steps * loader.batch_size
            resume_indices = sampler_indices[resume_prefix:]
            if not len(resume_indices):
                raise ValueError(
                    "Checkpoint resume position leaves no samples in the current epoch"
                )
        if args.preload_epoch_codes:
            if not hasattr(dataset, "prepare_epoch"):
                raise TypeError(f"Dataset {type(dataset).__name__} cannot preload epoch codes")
            preload_start = time.time()
            if sampler_indices is None:
                sampler_indices = np.fromiter(
                    iter(sampler), dtype=np.int64, count=len(sampler)
                )
            preload_indices = resume_indices if resume_indices is not None else sampler_indices
            cache_bytes = dataset.prepare_epoch(
                preload_indices,
                seed=args.global_seed * dist.get_world_size() + rank,
                epoch=epoch,
            )
            dist.barrier()
            if rank == 0:
                logger.info(
                    f"Preloaded epoch {epoch} rank-local codes "
                    f"({cache_bytes / (1024 ** 3):.2f} GiB per rank) in "
                    f"{time.time() - preload_start:.1f}s"
                )
        logger.info(f"Beginning epoch {epoch}...")
        if epoch == start_epoch and resume_rng_state is not None and epoch_resume_micro_steps == 0:
            restore_rng_state(resume_rng_state, device)
        if epoch_resume_micro_steps and resume_indices is not None:
            resume_loader = DataLoader(
                dataset,
                batch_size=loader.batch_size,
                shuffle=False,
                sampler=FixedIndexSampler(resume_indices),
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            iterator = iter(resume_loader)
            if resume_rng_state is not None:
                # Dataset-side random crop selection begins with the first
                # suffix batch when epoch preloading is disabled.
                restore_rng_state(resume_rng_state, device)
        else:
            iterator = iter(loader)
        if epoch_resume_micro_steps and resume_indices is None:
            for _ in range(epoch_resume_micro_steps):
                next(iterator)
            if resume_rng_state is not None:
                # Skipping batches above reproduces their dataset-side random
                # choices; restore the checkpoint stream before the first new batch.
                restore_rng_state(resume_rng_state, device)
        micro_steps = epoch_resume_micro_steps
        resume_micro_steps = 0
        resume_rng_state = None
        for batch in iterator:
            paired_training = args.dataset == "imagenet_paired_code"
            if paired_training:
                x, replay_x, y = batch
                replay_x = replay_x.to(device, non_blocking=True)
            else:
                x, y = batch
                replay_x = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]
            if paired_training:
                replay_indices = replay_x.reshape(replay_x.shape[0], -1)
                if replay_indices.shape != z_indices.shape:
                    raise ValueError(
                        f"New/replay token shapes differ: {z_indices.shape} != {replay_indices.shape}"
                    )
                # Avoid a redundant DDP all-reduce: the replay backward below synchronizes
                # the gradients accumulated by this new-code backward as well.
                with model.no_sync():
                    with torch.cuda.amp.autocast(dtype=ptdtype):
                        _, new_ce = model(
                            cond_idx=c_indices,
                            idx=z_indices[:, :-1],
                            targets=z_indices,
                        )
                        new_objective = args.new_loss_weight * new_ce
                scaler.scale(new_objective / args.gradient_accumulation_steps).backward()

                teacher_logits = None
                if teacher is not None:
                    rng_state = torch.cuda.get_rng_state(device)
                    with torch.no_grad(), torch.cuda.amp.autocast(dtype=ptdtype):
                        teacher_logits, _ = teacher(
                            cond_idx=c_indices,
                            idx=replay_indices[:, :-1],
                            targets=None,
                        )
                    torch.cuda.set_rng_state(rng_state, device)

                with torch.cuda.amp.autocast(dtype=ptdtype):
                    replay_logits, replay_ce = model(
                        cond_idx=c_indices,
                        idx=replay_indices[:, :-1],
                        targets=replay_indices,
                    )
                    if teacher_logits is not None:
                        temperature = args.kl_temperature
                        kl_loss = F.kl_div(
                            F.log_softmax(replay_logits / temperature, dim=-1),
                            F.softmax(teacher_logits / temperature, dim=-1),
                            reduction="batchmean",
                        ) * (temperature ** 2 / replay_indices.shape[1])
                    else:
                        kl_loss = replay_ce.new_zeros(())
                    replay_objective = (
                        args.replay_loss_weight * replay_ce
                        + args.kl_weight * kl_loss
                    )
                scaler.scale(replay_objective / args.gradient_accumulation_steps).backward()
                loss = new_objective.detach() + replay_objective.detach()
                del replay_logits, teacher_logits
            else:
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    _, raw_loss = model(cond_idx=c_indices, idx=z_indices[:, :-1], targets=z_indices)
                loss = raw_loss / args.gradient_accumulation_steps
                sync_context = model.no_sync() if (micro_steps + 1) % args.gradient_accumulation_steps else nullcontext()
                with sync_context:
                    scaler.scale(loss).backward()
                new_ce = raw_loss.detach()
                replay_ce = loss.new_zeros(())
                kl_loss = loss.new_zeros(())
            should_step = (micro_steps + 1) % args.gradient_accumulation_steps == 0
            if should_step:
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                train_steps += 1
            if args.ema and should_step:
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)

            # Log loss values:
            running_loss += (new_objective.detach() + replay_objective.detach()).item() if paired_training else raw_loss.item()
            running_new_ce += new_ce.item()
            running_replay_ce += replay_ce.item()
            running_kl += kl_loss.item()
            log_steps += 1
            micro_steps += 1
            if should_step and train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = (log_steps / args.gradient_accumulation_steps) / (end_time - start_time)
                # Reduce loss history over all processes:
                averages = torch.tensor(
                    [
                        running_loss / log_steps,
                        running_new_ce / log_steps,
                        running_replay_ce / log_steps,
                        running_kl / log_steps,
                    ],
                    device=device,
                )
                dist.all_reduce(averages, op=dist.ReduceOp.SUM)
                averages = (averages / dist.get_world_size()).tolist()
                avg_loss, avg_new_ce, avg_replay_ce, avg_kl = averages
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"New CE: {avg_new_ce:.4f}, Replay CE: {avg_replay_ce:.4f}, "
                    f"KL: {avg_kl:.6f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if rank == 0 and wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": avg_loss,
                            "train/new_ce": avg_new_ce,
                            "train/replay_ce": avg_replay_ce,
                            "train/teacher_kl": avg_kl,
                            "train/steps_per_sec": steps_per_sec,
                            "train/epoch": epoch,
                        },
                        step=train_steps,
                    )
                # Reset monitoring variables:
                running_loss = 0
                running_new_ce = 0
                running_replay_ce = 0
                running_kl = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            should_save = should_step and train_steps % args.ckpt_every == 0 and train_steps > 0
            if args.max_steps > 0 and train_steps == args.max_steps:
                should_save = True
            if should_save:
                gathered_rng_states = [None] * dist.get_world_size() if rank == 0 else None
                dist.gather_object(capture_rng_state(device), gathered_rng_states, dst=0)
                if rank == 0:
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args,
                        "epoch": epoch,
                        "micro_steps_in_epoch": micro_steps,
                        "rng_states": gathered_rng_states,
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        atomic_torch_save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                    if not args.no_cloud_save:
                        cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                        atomic_torch_save(checkpoint, cloud_checkpoint_path)
                        logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()

            if args.max_steps > 0 and train_steps >= args.max_steps:
                stop_training = True
                break
        if stop_training:
            logger.info(f"Reached max_steps={args.max_steps}")
            break
        if micro_steps % args.gradient_accumulation_steps:
            optimizer.zero_grad(set_to_none=True)
            logger.info(
                f"Dropped {micro_steps % args.gradient_accumulation_steps} incomplete "
                "micro-batches at epoch boundary"
            )

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    if rank == 0 and wandb_run is not None:
        wandb_run.finish()
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--replay-code-path", type=str, default=None, help="paired official-tokenizer codes")
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--save-initial-checkpoint", action="store_true", help="save the exact step-0 model for cross-run audits")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--init-from", type=str, default=None, help="initialize model weights only and reset optimizer/step")
    parser.add_argument("--init-from-ema", action="store_true", help="use EMA weights from --init-from when available")
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="frozen base GPT for replay logit KL")
    parser.add_argument("--teacher-from-ema", action="store_true")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many steps; 0 disables the limit")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--new-loss-weight", type=float, default=1.0)
    parser.add_argument("--replay-loss-weight", type=float, default=0.0)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument(
        "--preload-epoch-codes",
        action="store_true",
        help="preselect one code crop per rank and epoch into host memory",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-run-id", type=str, default="")
    parser.add_argument("--no-cloud-save", action="store_true", help="disable duplicate checkpoint writes to cloud_save_path")
    args = parser.parse_args()
    main(args)
