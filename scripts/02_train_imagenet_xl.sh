#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh

IMAGENET_ROOT="${IMAGENET_EXPERIMENT_ROOT:-$PROJECT_ROOT/outputs/imagenet_xl_384}"
BASE_GPUS="${IMAGENET_BASE_GPUS:-1,3}"
REWARD_GPUS="${IMAGENET_REWARD_GPUS:-4,5}"
ROOT="$IMAGENET_ROOT/training"
MAX_STEPS="${IMAGENET_MAX_STEPS:-78000}"
GLOBAL_BATCH="${IMAGENET_GLOBAL_BATCH_SIZE:-256}"
BASE_ACCUM="${IMAGENET_BASE_GRAD_ACCUMULATION:-32}"
REWARD_ACCUM="${IMAGENET_REWARD_GRAD_ACCUMULATION:-32}"

test -f "$IMAGENET_ROOT/codes/official/manifest.json"
test -f "$IMAGENET_ROOT/codes/reward/manifest.json"
test ! -e "$ROOT/official/results"
test ! -e "$ROOT/reward/results"
mkdir -p "$IMAGENET_ROOT/logs" "$ROOT/official" "$ROOT/reward"
export WANDB_DIR="$IMAGENET_ROOT/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"

train_one() {
  local gpus="$1" name="$2" run_id="$3" accum="$4"
  local accumulation_var
  if [ "$name" = official ]; then
    accumulation_var=IMAGENET_BASE_GRAD_ACCUMULATION
  else
    accumulation_var=IMAGENET_REWARD_GRAD_ACCUMULATION
  fi
  env IMAGENET_EXPERIMENT_ROOT="$IMAGENET_ROOT" \
    IMAGENET_MAX_STEPS="$MAX_STEPS" \
    IMAGENET_GLOBAL_BATCH_SIZE="$GLOBAL_BATCH" \
    "$accumulation_var=$accum" \
    bash scripts/train_imagenet_xl_one.sh "$name" "$gpus"
}

train_one "$BASE_GPUS" official llamagen-imagenet-xl-official-78k "$BASE_ACCUM" > "$IMAGENET_ROOT/logs/train_official.log" 2>&1 &
left_pid=$!
train_one "$REWARD_GPUS" reward llamagen-imagenet-xl-reward-78k "$REWARD_ACCUM" > "$IMAGENET_ROOT/logs/train_reward.log" 2>&1 &
right_pid=$!

for _ in $(seq 1 600); do
  left="$ROOT/official/results/000-GPT-XL/checkpoints/0000000.pt"
  right="$ROOT/reward/results/000-GPT-XL/checkpoints/0000000.pt"
  if [ -f "$left" ] && [ -f "$right" ]; then break; fi
  sleep 2
done
test -f "$ROOT/official/results/000-GPT-XL/checkpoints/0000000.pt"
test -f "$ROOT/reward/results/000-GPT-XL/checkpoints/0000000.pt"
python scripts/audit_matching_initial_checkpoints.py \
  --left "$ROOT/official/results/000-GPT-XL/checkpoints/0000000.pt" \
  --right "$ROOT/reward/results/000-GPT-XL/checkpoints/0000000.pt" \
  --output "$IMAGENET_ROOT/initial_checkpoint_audit.json"
status=0
if ! wait "$left_pid"; then status=1; fi
if ! wait "$right_pid"; then status=1; fi
exit "$status"
