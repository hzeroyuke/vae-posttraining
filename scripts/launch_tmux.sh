#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SESSION="${SESSION:-vae_posttraining}"
CONFIG="${CONFIG:-configs/coco_vae_pickscore_1k.yaml}"
GPUS="${CUDA_VISIBLE_DEVICES:-6}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhaoyuke/test/vqvae/.conda-env/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/zhaoyuke/test/vqvae/.conda-env/bin/torchrun}"
LOG_FILE="${LOG_FILE:-/mnt/data/zyk/vae_posttraining/${SESSION}.log}"
NPROC="${NPROC_PER_NODE:-1}"
RESUME_ARG=""
if [ -n "${RESUME:-}" ]; then
  RESUME_ARG="--resume ${RESUME}"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists"
  exit 1
fi

tmux new-session -d -s "$SESSION" "bash -lc '
set -euo pipefail
cd "$PWD"
export PYTHONPATH="$PWD/.deps:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HOME="/mnt/data/zyk/hf_home"
export HF_HUB_CACHE="/mnt/data/zyk/hf_cache"
export TRANSFORMERS_CACHE="/mnt/data/zyk/hf_cache"
export TORCH_HOME="${TORCH_HOME_OVERRIDE:-/mnt/data/zyk/torch_cache}"
export WANDB_ENTITY="yukezhao37-zhejiang-university"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [ "$NPROC" -gt 1 ]; then
  exec -a vae_posttraining.train "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC" -m vae_posttraining.train --config "$CONFIG" $RESUME_ARG >> "$LOG_FILE" 2>&1
else
  exec -a vae_posttraining.train "$PYTHON_BIN" -m vae_posttraining.train --config "$CONFIG" $RESUME_ARG >> "$LOG_FILE" 2>&1
fi
'"
echo "Started tmux session '$SESSION' on CUDA_VISIBLE_DEVICES=$GPUS"
echo "Log: $LOG_FILE"
