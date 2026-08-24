#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh

NAME="${1:?usage: train_imagenet_xl_one.sh NAME GPUS [CHECKPOINT]}"
GPUS="${2:?usage: train_imagenet_xl_one.sh NAME GPUS [CHECKPOINT]}"
CHECKPOINT="${3:-}"
ROOT="${IMAGENET_EXPERIMENT_ROOT:-$PROJECT_ROOT/outputs/imagenet_xl_384}"
TRAINING_ROOT="$ROOT/training"
MAX_STEPS="${IMAGENET_MAX_STEPS:-78000}"
GLOBAL_BATCH="${IMAGENET_GLOBAL_BATCH_SIZE:-256}"
RDZV_CONF="${IMAGENET_RDZV_CONF:-heartbeat=60,keep_alive_interval=30,keep_alive_max_attempt=10}"

case "$NAME" in
  official)
    RUN_ID="llamagen-imagenet-xl-official-78k"
    ACCUMULATION="${IMAGENET_BASE_GRAD_ACCUMULATION:-32}"
    ;;
  reward)
    RUN_ID="llamagen-imagenet-xl-reward-78k"
    ACCUMULATION="${IMAGENET_REWARD_GRAD_ACCUMULATION:-32}"
    ;;
  *)
    echo "NAME must be official or reward, got: $NAME" >&2
    exit 2
    ;;
esac

test -f "$ROOT/codes/$NAME/manifest.json"
mkdir -p "$ROOT/logs" "$TRAINING_ROOT/$NAME" "$ROOT/wandb"
export WANDB_DIR="$ROOT/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"

checkpoint_args=(--save-initial-checkpoint)
if [[ -n "$CHECKPOINT" ]]; then
  test -f "$CHECKPOINT"
  checkpoint_args=(--gpt-ckpt "$CHECKPOINT")
elif [[ -e "$TRAINING_ROOT/$NAME/results" ]]; then
  echo "Fresh run refused because results already exist: $TRAINING_ROOT/$NAME/results" >&2
  exit 2
fi

preload_args=()
if [[ "${IMAGENET_PRELOAD_EPOCH_CODES:-1}" == 1 ]]; then
  preload_args=(--preload-epoch-codes)
fi

CUDA_VISIBLE_DEVICES="$GPUS" exec ionice -c2 -n0 torchrun --standalone --rdzv-conf "$RDZV_CONF" \
  --nproc_per_node="$(gpu_count "$GPUS")" \
  third_party/LlamaGen/autoregressive/train/train_c2i.py \
  --code-path "$ROOT/codes/$NAME" --cloud-save-path "$TRAINING_ROOT/$NAME/cloud" \
  --results-dir "$TRAINING_ROOT/$NAME/results" --dataset imagenet_code --gpt-model GPT-XL \
  --gpt-type c2i --image-size 384 --downsample-size 16 --num-classes 1000 \
  --vocab-size 16384 --cls-token-num 1 --epochs 300 --max-steps "$MAX_STEPS" \
  --lr 1e-4 --weight-decay 0.05 --beta1 0.9 --beta2 0.95 --max-grad-norm 1.0 \
  --dropout-p 0.1 --token-dropout-p 0.1 --global-batch-size "$GLOBAL_BATCH" \
  --gradient-accumulation-steps "$ACCUMULATION" --global-seed 20260812 \
  --num-workers "${IMAGENET_TRAIN_WORKERS:-0}" "${preload_args[@]}" \
  --mixed-precision bf16 --no-compile \
  --log-every 100 --ckpt-every 10000 "${checkpoint_args[@]}" \
  --no-cloud-save --wandb-project "$WANDB_PROJECT" --wandb-entity "${WANDB_ENTITY:-}" \
  --wandb-name "$RUN_ID" --wandb-run-id "$RUN_ID"
