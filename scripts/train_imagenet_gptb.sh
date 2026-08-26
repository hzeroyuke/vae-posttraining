#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh

CODE_PATH="${IMAGENET_GPTB_CODE_PATH:-$PROJECT_ROOT/outputs/imagenet_gptb_384/codes/official}"
GPUS="${IMAGENET_GPTB_GPUS:-0,1,2,3,4,5,6,7}"
OUTPUT_ROOT="${IMAGENET_GPTB_OUTPUT_ROOT:-$OUTPUT_ROOT/imagenet_gptb_384}"
TRAINING_ROOT="$OUTPUT_ROOT/training"
RESULTS_DIR="$TRAINING_ROOT/gptb/results"
RUN_ID="${IMAGENET_GPTB_RUN_ID:-llamagen-imagenet-gptb-384}"
IMAGE_SIZE="${IMAGENET_GPTB_IMAGE_SIZE:-384}"
DOWNSAMPLE_SIZE="${IMAGENET_GPTB_DOWNSAMPLE_SIZE:-16}"
GLOBAL_BATCH_SIZE="${IMAGENET_GPTB_GLOBAL_BATCH_SIZE:-256}"
GRAD_ACCUMULATION="${IMAGENET_GPTB_GRAD_ACCUMULATION:-1}"
MAX_STEPS="${IMAGENET_GPTB_MAX_STEPS:-78000}"
EPOCHS="${IMAGENET_GPTB_EPOCHS:-300}"
NUM_WORKERS="${IMAGENET_GPTB_NUM_WORKERS:-0}"
LOG_EVERY="${IMAGENET_GPTB_LOG_EVERY:-100}"
CKPT_EVERY="${IMAGENET_GPTB_CKPT_EVERY:-10000}"

test -f "$CODE_PATH/manifest.json"
test -f "$CODE_PATH/codes.npy"
test -f "$CODE_PATH/labels.npy"
python - "$CODE_PATH/manifest.json" "$IMAGE_SIZE" "$DOWNSAMPLE_SIZE" <<'PY'
import json
import sys

manifest_path, image_size, downsample_size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
expected_tokens = (image_size // downsample_size) ** 2
if manifest.get("image_size") not in (None, image_size):
    raise SystemExit(
        f"Code manifest image_size={manifest['image_size']} does not match {image_size}"
    )
if manifest.get("tokens_per_image") not in (None, expected_tokens):
    raise SystemExit(
        f"Code manifest tokens_per_image={manifest['tokens_per_image']} does not match {expected_tokens}"
    )
PY
test ! -e "$RESULTS_DIR"
mkdir -p "$OUTPUT_ROOT/logs" "$TRAINING_ROOT/gptb" "$OUTPUT_ROOT/wandb"
export WANDB_DIR="$OUTPUT_ROOT/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"

NPROC="$(gpu_count "$GPUS")"
if [ "$NPROC" -lt 2 ]; then
  echo "GPT-B multi-GPU training requires at least two GPUs; got: $GPUS" >&2
  exit 2
fi
if [ $((GLOBAL_BATCH_SIZE % (NPROC * GRAD_ACCUMULATION))) -ne 0 ]; then
  echo "Global batch must be divisible by world size times gradient accumulation." >&2
  exit 2
fi

preload_args=()
if [[ "${IMAGENET_GPTB_PRELOAD_EPOCH_CODES:-0}" == 1 ]]; then
  preload_args+=(--preload-epoch-codes)
fi

CUDA_VISIBLE_DEVICES="$GPUS" exec torchrun --standalone \
  --nproc_per_node="$NPROC" \
  third_party/LlamaGen/autoregressive/train/train_c2i.py \
  --code-path "$CODE_PATH" \
  --cloud-save-path "$TRAINING_ROOT/gptb/cloud" \
  --results-dir "$RESULTS_DIR" \
  --dataset imagenet_code \
  --gpt-model GPT-B \
  --gpt-type c2i \
  --image-size "$IMAGE_SIZE" \
  --downsample-size "$DOWNSAMPLE_SIZE" \
  --num-classes 1000 \
  --vocab-size 16384 \
  --cls-token-num 1 \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --beta1 0.9 \
  --beta2 0.95 \
  --max-grad-norm 1.0 \
  --dropout-p 0.1 \
  --token-dropout-p 0.1 \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUMULATION" \
  --global-seed 20260812 \
  --num-workers "$NUM_WORKERS" \
  --mixed-precision bf16 \
  --no-compile \
  --log-every "$LOG_EVERY" \
  --ckpt-every "$CKPT_EVERY" \
  --save-initial-checkpoint \
  --no-cloud-save \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  --wandb-name "$RUN_ID" \
  --wandb-run-id "$RUN_ID"
