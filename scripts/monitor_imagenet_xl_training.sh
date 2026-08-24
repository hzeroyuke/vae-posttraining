#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh

ROOT="${IMAGENET_EXPERIMENT_ROOT:-$PROJECT_ROOT/outputs/imagenet_xl_384}"
EXPECTED_STEPS="${IMAGENET_MAX_STEPS:-78000}"
MONITOR_TASKS="${IMAGENET_MONITOR_TASKS:-both}"
OFFICIAL_GPUS="${IMAGENET_BASE_GPUS:-1,3}"
REWARD_GPUS="${IMAGENET_REWARD_GPUS:-4,5}"
BASE_ACCUMULATION="${IMAGENET_BASE_GRAD_ACCUMULATION:-32}"
REWARD_ACCUMULATION="${IMAGENET_REWARD_GRAD_ACCUMULATION:-32}"
PRELOAD_EPOCH_CODES="${IMAGENET_PRELOAD_EPOCH_CODES:-1}"
OFFICIAL="$ROOT/training/official/results/000-GPT-XL/checkpoints/$(printf '%07d' "$EXPECTED_STEPS").pt"
REWARD="$ROOT/training/reward/results/000-GPT-XL/checkpoints/$(printf '%07d' "$EXPECTED_STEPS").pt"
LOG="$ROOT/monitor.log"
EVAL_SESSION="${IMAGENET_EVAL_TMUX_SESSION:-llamagen_imagenet_xl_evaluation}"
FID_SAMPLES="${IMAGENET_FID_SAMPLES:-50000}"
GPU_RESERVATION_SESSION="${IMAGENET_GPU_RESERVATION_SESSION:-}"
FINAL_AUDIT="$ROOT/final_checkpoint_audit.json"
EVAL_SUMMARY="$ROOT/evaluation/c2i_$(printf '%07d' "$EXPECTED_STEPS")_fid${FID_SAMPLES}/summary.json"
last_official_restart=0
last_reward_restart=0
last_eval_restart=0
RANDOM_IO_STEP=10000

case "$MONITOR_TASKS" in
  both)
    MONITOR_OFFICIAL=1
    MONITOR_REWARD=1
    ;;
  official)
    MONITOR_OFFICIAL=1
    MONITOR_REWARD=0
    ;;
  reward)
    MONITOR_OFFICIAL=0
    MONITOR_REWARD=1
    ;;
  *)
    echo "IMAGENET_MONITOR_TASKS must be both, official, or reward; got: $MONITOR_TASKS" >&2
    exit 2
    ;;
esac

process_alive() {
  local name="$1"
  pgrep -f -- "python -u third_party/LlamaGen/autoregressive/train/train_c2i.py --code-path $ROOT/codes/$name" >/dev/null
}

release_gpu_reservation() {
  [[ -z "$GPU_RESERVATION_SESSION" ]] && return 0
  if tmux has-session -t "$GPU_RESERVATION_SESSION" 2>/dev/null; then
    tmux kill-session -t "$GPU_RESERVATION_SESSION"
    printf '%s monitor: released GPU reservation session %s\n' \
      "$(date '+%F %T')" "$GPU_RESERVATION_SESSION" >> "$LOG"
    sleep 3
  fi
}

latest_checkpoint() {
  local name="$1" candidate
  while IFS= read -r candidate; do
    if validate_checkpoint "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(
    find "$ROOT/training/$name/results/000-GPT-XL/checkpoints" \
      -maxdepth 1 -type f -name '*.pt' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | cut -d' ' -f2-
  )
}

validate_checkpoint() {
  local checkpoint="$1"
  python - "$checkpoint" <<'PY'
import sys

import torch

path = sys.argv[1]
try:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    assert isinstance(checkpoint.get("model"), dict)
    assert isinstance(checkpoint.get("optimizer"), dict)
    assert int(checkpoint["steps"]) >= 0
except Exception as error:
    print(f"invalid checkpoint {path}: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

recover_one() {
  local name="$1" gpus="$2" now="$3" checkpoint session last_restart accumulation_var accumulation
  if [[ "$name" == official ]]; then
    last_restart=$last_official_restart
    accumulation_var=IMAGENET_BASE_GRAD_ACCUMULATION
    accumulation=$BASE_ACCUMULATION
  else
    last_restart=$last_reward_restart
    accumulation_var=IMAGENET_REWARD_GRAD_ACCUMULATION
    accumulation=$REWARD_ACCUMULATION
  fi
  if process_alive "$name" || (( now - last_restart < 600 )); then
    return 0
  fi
  checkpoint="$(latest_checkpoint "$name")"
  if [[ -z "$checkpoint" ]]; then
    printf '%s monitor: no checkpoint available for %s recovery\n' "$(date '+%F %T')" "$name" >> "$LOG"
    return 0
  fi
  session="llamagen_imagenet_xl_${name}_recovery"
  if tmux has-session -t "$session" 2>/dev/null; then
    return 0
  fi
  release_gpu_reservation
  tmux new-session -d -s "$session" \
    "cd '$PWD' && IMAGENET_EXPERIMENT_ROOT='$ROOT' IMAGENET_MAX_STEPS='$EXPECTED_STEPS' $accumulation_var='$accumulation' IMAGENET_PRELOAD_EPOCH_CODES='$PRELOAD_EPOCH_CODES' bash scripts/train_imagenet_xl_one.sh '$name' '$gpus' '$checkpoint' >> '$ROOT/logs/train_${name}_recovery.log' 2>&1"
  printf '%s monitor: started %s recovery from %s on GPUs %s\n' "$(date '+%F %T')" "$name" "$checkpoint" "$gpus" >> "$LOG"
  if [[ "$name" == official ]]; then
    last_official_restart=$now
  else
    last_reward_restart=$now
  fi
}

terminate_training_processes() {
  local name="$1" pattern pid
  pattern="third_party/LlamaGen/autoregressive/train/train_c2i.py --code-path $ROOT/codes/$name"
  while read -r pid; do
    [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
  done < <(pgrep -f -- "$pattern" || true)
  sleep 3
  while read -r pid; do
    [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
  done < <(pgrep -f -- "$pattern" || true)
}

restart_after_random_io_patch() {
  local name="$1" session marker checkpoint
  marker="$ROOT/.${name}_random_io_restart"
  checkpoint="$ROOT/training/$name/results/000-GPT-XL/checkpoints/$(printf '%07d' "$RANDOM_IO_STEP").pt"
  [[ -f "$marker" || ! -f "$checkpoint" ]] && return 0
  if ! validate_checkpoint "$checkpoint"; then
    printf '%s monitor: checkpoint %s is still incomplete; deferring %s restart\n' \
      "$(date '+%F %T')" "$checkpoint" "$name" >> "$LOG"
    return 0
  fi
  session="llamagen_imagenet_xl_${name}_recovery"
  printf '%s monitor: checkpoint %s ready; restarting %s once to apply epoch cache and mmap random-read advice\n' \
    "$(date '+%F %T')" "$checkpoint" "$name" >> "$LOG"
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux send-keys -t "$session:0" C-c || true
    sleep 8
    tmux kill-session -t "$session" 2>/dev/null || true
  fi
  if process_alive "$name"; then
    terminate_training_processes "$name"
  fi
  : > "$marker"
}

printf '%s monitor: started mode=%s official_gpus=%s reward_gpus=%s official_accum=%s reward_accum=%s expected_steps=%s\n' \
  "$(date '+%F %T')" "$MONITOR_TASKS" "$OFFICIAL_GPUS" "$REWARD_GPUS" \
  "$BASE_ACCUMULATION" "$REWARD_ACCUMULATION" "$EXPECTED_STEPS" >> "$LOG"

while true; do
  if [[ "$MONITOR_TASKS" == official && -f "$OFFICIAL" ]]; then
    if validate_checkpoint "$OFFICIAL"; then
      date '+%F %T monitor: official final checkpoint validation passed' >> "$LOG"
      exit 0
    fi
  elif [[ "$MONITOR_TASKS" == reward && -f "$REWARD" ]]; then
    if validate_checkpoint "$REWARD"; then
      date '+%F %T monitor: reward final checkpoint validation passed' >> "$LOG"
      exit 0
    fi
  elif [[ "$MONITOR_TASKS" == both && -f "$OFFICIAL" && -f "$REWARD" ]]; then
    if [[ ! -f "$FINAL_AUDIT" ]]; then
      date '+%F %T monitor: final checkpoint paths found; attempting audit' >> "$LOG"
      if python scripts/audit_imagenet_xl_training.py \
        --experiment-root "$ROOT" \
        --expected-steps "$EXPECTED_STEPS" >> "$LOG" 2>&1; then
        date '+%F %T monitor: final checkpoint audit passed' >> "$LOG"
      else
        date '+%F %T monitor: final checkpoint audit not ready; retrying' >> "$LOG"
      fi
    fi
    if [[ -f "$FINAL_AUDIT" ]]; then
      if [[ "${IMAGENET_AUTO_EVALUATE:-1}" != 1 ]]; then
        exit 0
      fi
      if [[ -f "$EVAL_SUMMARY" ]]; then
        date '+%F %T monitor: final c2i evaluation passed' >> "$LOG"
        exit 0
      fi
      now="$(date +%s)"
      if ! tmux has-session -t "$EVAL_SESSION" 2>/dev/null \
          && (( now - last_eval_restart >= 600 )); then
        tmux new-session -d -s "$EVAL_SESSION" \
          "cd '$PWD' && IMAGENET_EXPERIMENT_ROOT='$ROOT' IMAGENET_MAX_STEPS='$EXPECTED_STEPS' IMAGENET_FID_SAMPLES='$FID_SAMPLES' bash scripts/evaluate_imagenet_xl_c2i.sh > '$ROOT/logs/evaluate_c2i_session.log' 2>&1"
        printf '%s monitor: started final c2i evaluation in %s\n' \
          "$(date '+%F %T')" "$EVAL_SESSION" >> "$LOG"
        last_eval_restart=$now
      fi
      sleep 60
      continue
    fi
  fi
  now="$(date +%s)"
  if (( MONITOR_OFFICIAL )); then
    restart_after_random_io_patch official
    recover_one official "$OFFICIAL_GPUS" "$now"
  fi
  if (( MONITOR_REWARD )); then
    restart_after_random_io_patch reward
    recover_one reward "$REWARD_GPUS" "$now"
  fi
  sleep 60
done
