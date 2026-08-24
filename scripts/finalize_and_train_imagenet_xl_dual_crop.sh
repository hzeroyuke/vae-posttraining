#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh

ROOT="${IMAGENET_EXPERIMENT_ROOT:-$PROJECT_ROOT/outputs/imagenet_xl_384}"
DATA="${IMAGENET_PARQUET_ROOT:-$PROJECT_ROOT/data/imagenet-1k}"
OFFICIAL_VQ="${OFFICIAL_VQ:-$PROJECT_ROOT/models/vq_ds16_c2i.pt}"
REWARD_VQ="${REWARD_VQ:-$PROJECT_ROOT/models/reward_vq_ds16_c2i.pt}"
POLL_SECONDS="${IMAGENET_FINALIZE_POLL_SECONDS:-60}"
TRAIN_GPU_MAX_USED_MB="${IMAGENET_TRAIN_GPU_MAX_USED_MB:-4096}"
AUTO_SELECT_TRAIN_GPUS="${IMAGENET_AUTO_SELECT_TRAIN_GPUS:-1}"
PY="${PYTHON_BIN:-$CONDA_PREFIX/bin/python}"
LOG="$ROOT/finalize_dual_crop.log"
last_official_restart=0
last_reward_restart=0

mkdir -p "$ROOT/logs"

extract_complete() {
  local name="$1"
  test -f "$ROOT/codes/${name}_105/codes.npy" \
    && test -f "$ROOT/codes/${name}_105/labels.npy" \
    && test -f "$ROOT/codes/${name}_105/manifest.json"
}

extract_alive() {
  local name="$1"
  pgrep -f -- "python -u third_party/LlamaGen/autoregressive/train/extract_codes_c2i_parquet.py .*--code-path $ROOT/codes/${name}_105" >/dev/null
}

recover_extract() {
  local name="$1" gpus="$2" checkpoint="$3" now="$4" last_restart session
  if [[ "$name" == official ]]; then
    last_restart=$last_official_restart
  else
    last_restart=$last_reward_restart
  fi
  if extract_complete "$name" || extract_alive "$name" || (( now - last_restart < 600 )); then
    return 0
  fi
  session="llamagen_imagenet_xl_${name}_codes_105_recovery"
  if tmux has-session -t "$session" 2>/dev/null; then
    return 0
  fi
  tmux new-session -d -s "$session" \
    "cd '$PWD' && source scripts/common.sh && export PYTHONPATH=third_party/LlamaGen:\$PYTHONPATH && CUDA_VISIBLE_DEVICES='$gpus' '$CONDA_PREFIX/bin/torchrun' --standalone --nproc_per_node=2 third_party/LlamaGen/autoregressive/train/extract_codes_c2i_parquet.py --data-path '$DATA' --split train --code-path '$ROOT/codes/${name}_105' --vq-ckpt '$checkpoint' --image-size 384 --crop-range 1.05 --ten-crop --batch-size 4 --num-workers 2 --part-batches 256 --global-seed 20260812 --resume > '$ROOT/logs/extract_${name}_105_recovery.log' 2>&1"
  printf '%s restarted %s crop-range 1.05 extraction on GPUs %s\n' \
    "$(date '+%F %T')" "$name" "$gpus" >> "$LOG"
  if [[ "$name" == official ]]; then
    last_official_restart=$now
  else
    last_reward_restart=$now
  fi
}

select_training_gpu_pairs() {
  local -a candidates=()
  local now
  while true; do
    mapfile -t candidates < <(
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, -v maximum="$TRAIN_GPU_MAX_USED_MB" '
            {
              gsub(/[[:space:]]/, "", $1)
              gsub(/[[:space:]]/, "", $2)
              if (($2 + 0) <= maximum) print ($2 + 0), $1
            }
          ' \
        | sort -n -k1,1 -k2,2 \
        | head -n 4 \
        | awk '{print $2}'
    )
    if (( ${#candidates[@]} == 4 )); then
      printf '%s,%s %s,%s\n' \
        "${candidates[0]}" "${candidates[1]}" \
        "${candidates[2]}" "${candidates[3]}"
      return 0
    fi
    now="$(date '+%F %T')"
    printf '%s waiting for four training GPUs below %s MiB used; found %s\n' \
      "$now" "$TRAIN_GPU_MAX_USED_MB" "${#candidates[@]}" >> "$LOG"
    sleep "$POLL_SECONDS"
  done
}

printf '%s waiting for both crop-range 1.05 corpora\n' "$(date '+%F %T')" >> "$LOG"
while ! extract_complete official || ! extract_complete reward; do
  now="$(date +%s)"
  recover_extract official "${IMAGENET_BASE_GPUS:-2,3}" "$OFFICIAL_VQ" "$now"
  recover_extract reward "${IMAGENET_REWARD_GPUS:-0,7}" "$REWARD_VQ" "$now"
  sleep "$POLL_SECONDS"
done
printf '%s crop-range 1.05 corpora complete\n' "$(date '+%F %T')" >> "$LOG"

for name in official reward; do
  "$PY" scripts/audit_imagenet_code_corpus.py \
    --code-path "$ROOT/codes/${name}_110" \
    --output "$ROOT/codes/${name}_110/audit.json" \
    >> "$LOG" 2>&1
  "$PY" scripts/audit_imagenet_code_corpus.py \
    --code-path "$ROOT/codes/${name}_105" \
    --output "$ROOT/codes/${name}_105/audit.json" \
    >> "$LOG" 2>&1
done

test ! -e "$ROOT/codes/official"
test ! -e "$ROOT/codes/reward"
"$PY" scripts/combine_imagenet_code_variants.py \
  --primary "$ROOT/codes/official_110" \
  --secondary "$ROOT/codes/official_105" \
  --output "$ROOT/codes/official" >> "$LOG" 2>&1
"$PY" scripts/combine_imagenet_code_variants.py \
  --primary "$ROOT/codes/reward_110" \
  --secondary "$ROOT/codes/reward_105" \
  --output "$ROOT/codes/reward" >> "$LOG" 2>&1

for name in official reward; do
  "$PY" scripts/audit_imagenet_code_corpus.py \
    --code-path "$ROOT/codes/$name" \
    --output "$ROOT/codes/$name/audit.json" \
    >> "$LOG" 2>&1
done
printf '%s dual-crop corpora combined and audited\n' "$(date '+%F %T')" >> "$LOG"

TRAIN_BASE_GPUS="${IMAGENET_BASE_GPUS:-2,3}"
TRAIN_REWARD_GPUS="${IMAGENET_REWARD_GPUS:-6,7}"
if [[ "$AUTO_SELECT_TRAIN_GPUS" == 1 ]]; then
  read -r TRAIN_BASE_GPUS TRAIN_REWARD_GPUS < <(select_training_gpu_pairs)
fi
printf '%s selected training GPUs official=%s reward=%s (max pre-use %s MiB)\n' \
  "$(date '+%F %T')" "$TRAIN_BASE_GPUS" "$TRAIN_REWARD_GPUS" \
  "$TRAIN_GPU_MAX_USED_MB" >> "$LOG"
printf '{\n  "official_gpus": "%s",\n  "reward_gpus": "%s",\n  "max_used_mb": %s\n}\n' \
  "$TRAIN_BASE_GPUS" "$TRAIN_REWARD_GPUS" "$TRAIN_GPU_MAX_USED_MB" \
  > "$ROOT/training_gpu_selection.json"

IMAGENET_EXPERIMENT_ROOT="$ROOT" \
IMAGENET_MAX_STEPS="${IMAGENET_MAX_STEPS:-78000}" \
IMAGENET_BASE_GPUS="$TRAIN_BASE_GPUS" \
IMAGENET_REWARD_GPUS="$TRAIN_REWARD_GPUS" \
  bash scripts/02_train_imagenet_xl.sh > "$ROOT/train_dual_crop_tmux.log" 2>&1 &
training_launcher=$!

for _ in $(seq 1 600); do
  if test -f "$ROOT/initial_checkpoint_audit.json"; then
    break
  fi
  if ! kill -0 "$training_launcher" 2>/dev/null; then
    wait "$training_launcher"
  fi
  sleep 2
done
test -f "$ROOT/initial_checkpoint_audit.json"

IMAGENET_EXPERIMENT_ROOT="$ROOT" \
IMAGENET_MAX_STEPS="${IMAGENET_MAX_STEPS:-78000}" \
IMAGENET_BASE_GPUS="$TRAIN_BASE_GPUS" \
IMAGENET_REWARD_GPUS="$TRAIN_REWARD_GPUS" \
  bash scripts/monitor_imagenet_xl_training.sh

wait "$training_launcher" || true
printf '%s final dual-crop GPT-XL audit passed\n' "$(date '+%F %T')" >> "$LOG"
