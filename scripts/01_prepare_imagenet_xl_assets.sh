#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh

DATA_PATH="${IMAGENET_PARQUET_ROOT:-$PROJECT_ROOT/data/imagenet-1k}"
BASE_VQ="${OFFICIAL_VQ:-$PROJECT_ROOT/models/vq_ds16_c2i.pt}"
REWARD_VQ="${REWARD_VQ:-$PROJECT_ROOT/models/reward_vq_ds16_c2i.pt}"
IMAGENET_ROOT="${IMAGENET_EXPERIMENT_ROOT:-$PROJECT_ROOT/outputs/imagenet_xl_384}"
CODE_GPUS="${IMAGENET_CODE_GPUS:-1,3}"
REWARD_CODE_GPUS="${IMAGENET_REWARD_CODE_GPUS:-4,5,7}"
BASE_BATCH="${IMAGENET_BASE_CODE_BATCH_SIZE:-4}"
REWARD_BATCH="${IMAGENET_REWARD_CODE_BATCH_SIZE:-4}"
MAX_SAMPLES="${IMAGENET_MAX_SAMPLES:-0}"
RESUME="${IMAGENET_RESUME:-0}"
CONTIGUOUS="${IMAGENET_CODE_CONTIGUOUS:-0}"

test -d "$DATA_PATH/data"
test -f "$BASE_VQ"
test -f "$REWARD_VQ"
if [ "$RESUME" != "1" ]; then
  test ! -e "$IMAGENET_ROOT/codes"
fi
mkdir -p "$IMAGENET_ROOT/logs" "$IMAGENET_ROOT/codes"

extract_one() {
  local gpu_list="$1" name="$2" ckpt="$3" batch_size="$4" crop_range="$5"
  local max_args=()
  local resume_args=()
  local contiguous_args=()
  if [ "$MAX_SAMPLES" -gt 0 ]; then max_args+=(--max-samples "$MAX_SAMPLES"); fi
  if [ "$RESUME" = "1" ]; then resume_args+=(--resume); fi
  if [ "$CONTIGUOUS" = "1" ]; then contiguous_args+=(--contiguous); fi
  CUDA_VISIBLE_DEVICES="$gpu_list" torchrun --standalone --nproc_per_node="$(gpu_count "$gpu_list")" \
    third_party/LlamaGen/autoregressive/train/extract_codes_c2i_parquet.py \
    --data-path "$DATA_PATH" --split train --code-path "$IMAGENET_ROOT/codes/$name" \
    --vq-ckpt "$ckpt" --image-size 384 --crop-range "$crop_range" --ten-crop \
    --batch-size "$batch_size" --num-workers "${IMAGENET_CODE_WORKERS:-2}" \
    --part-batches "${IMAGENET_CODE_PART_BATCHES:-256}" \
    --global-seed 20260812 "${max_args[@]}" "${resume_args[@]}" "${contiguous_args[@]}"
}

extract_one "$CODE_GPUS" official_110 "$BASE_VQ" "$BASE_BATCH" 1.1 > "$IMAGENET_ROOT/logs/extract_official_110.log" 2>&1 &
left_pid=$!
extract_one "$REWARD_CODE_GPUS" reward_110 "$REWARD_VQ" "$REWARD_BATCH" 1.1 > "$IMAGENET_ROOT/logs/extract_reward_110.log" 2>&1 &
right_pid=$!
status=0
if ! wait "$left_pid"; then status=1; fi
if ! wait "$right_pid"; then status=1; fi
if [ "$status" -ne 0 ]; then exit "$status"; fi

extract_one "$CODE_GPUS" official_105 "$BASE_VQ" "$BASE_BATCH" 1.05 > "$IMAGENET_ROOT/logs/extract_official_105.log" 2>&1 &
left_pid=$!
extract_one "$REWARD_CODE_GPUS" reward_105 "$REWARD_VQ" "$REWARD_BATCH" 1.05 > "$IMAGENET_ROOT/logs/extract_reward_105.log" 2>&1 &
right_pid=$!
status=0
if ! wait "$left_pid"; then status=1; fi
if ! wait "$right_pid"; then status=1; fi
if [ "$status" -ne 0 ]; then exit "$status"; fi

test ! -e "$IMAGENET_ROOT/codes/official"
test ! -e "$IMAGENET_ROOT/codes/reward"
python scripts/combine_imagenet_code_variants.py \
  --primary "$IMAGENET_ROOT/codes/official_110" \
  --secondary "$IMAGENET_ROOT/codes/official_105" \
  --output "$IMAGENET_ROOT/codes/official"
python scripts/combine_imagenet_code_variants.py \
  --primary "$IMAGENET_ROOT/codes/reward_110" \
  --secondary "$IMAGENET_ROOT/codes/reward_105" \
  --output "$IMAGENET_ROOT/codes/reward"

for name in official reward; do
  python scripts/audit_imagenet_code_corpus.py --code-path "$IMAGENET_ROOT/codes/$name" \
    --output "$IMAGENET_ROOT/codes/$name/audit.json" > "$IMAGENET_ROOT/logs/audit_$name.log"
done
echo "ImageNet GPT-XL code assets ready under $IMAGENET_ROOT"
