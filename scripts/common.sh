#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$PROJECT_ROOT/.env" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
fi
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/activate_env.sh"

export PYTHONPATH="$PROJECT_ROOT/third_party/LlamaGen:$PROJECT_ROOT/src:$PROJECT_ROOT/scripts:${PYTHONPATH:-}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
export TOKENIZER_RUN_DIR="${TOKENIZER_RUN_DIR:-$OUTPUT_ROOT/tokenizer/seed3101}"
export RL_VQ="${RL_VQ:-$OUTPUT_ROOT/tokenizer/pickscore_seed3101_step1000_vq.pt}"
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$OUTPUT_ROOT/gptxl_full_coco}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-vqvae-rl-llamagen-pickscore}"
export TOKENIZER_WANDB_RUN_ID="${TOKENIZER_WANDB_RUN_ID:-vq-tokenizer-pickscore-seed3101-1k-clean}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"

require_env() {
  local name
  for name in "$@"; do
    if [ -z "${!name:-}" ]; then
      echo "Required environment variable is unset: $name" >&2
      exit 2
    fi
  done
}

gpu_count() {
  local value="${1//[[:space:]]/}"
  local items
  IFS=',' read -r -a items <<< "$value"
  echo "${#items[@]}"
}
