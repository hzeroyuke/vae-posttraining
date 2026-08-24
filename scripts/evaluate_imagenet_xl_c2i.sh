#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh

ROOT="${IMAGENET_EXPERIMENT_ROOT:-$PROJECT_ROOT/outputs/imagenet_xl_384}"
EXPECTED_STEPS="${IMAGENET_MAX_STEPS:-78000}"
NUM_SAMPLES="${IMAGENET_FID_SAMPLES:-50000}"
OFFICIAL_GPUS="${IMAGENET_BASE_EVAL_GPUS:-2,7}"
REWARD_GPUS="${IMAGENET_REWARD_EVAL_GPUS:-1,3}"
PER_RANK_BATCH="${IMAGENET_EVAL_PER_RANK_BATCH_SIZE:-4}"
ASSET_ROOT="${IMAGENET_EVAL_ASSET_ROOT:-$PROJECT_ROOT/outputs/evaluation_assets}"
REFERENCE="${IMAGENET_FID_REFERENCE:-$ASSET_ROOT/VIRTUAL_imagenet256_labeled.npz}"
FID_ENV="${IMAGENET_FID_ENV:-$HOME/.cache/vqvae-gptxl-scaling/llamagen-c2i-fid}"
FID_PYTHON="${IMAGENET_FID_PYTHON:-$FID_ENV/bin/python}"
EVALUATOR="$PROJECT_ROOT/third_party/LlamaGen/evaluations/c2i/evaluator.py"
EVALUATION_ROOT="$ROOT/evaluation/c2i_$(printf '%07d' "$EXPECTED_STEPS")_fid${NUM_SAMPLES}"

test -f "$ROOT/final_checkpoint_audit.json"
test -f "$EVALUATOR"
mkdir -p "$ROOT/logs" "$EVALUATION_ROOT" "$ASSET_ROOT/inception"
if [[ ! -f "$REFERENCE" || ! -x "$FID_PYTHON" ]]; then
  IMAGENET_EXPERIMENT_ROOT="$ROOT" \
    IMAGENET_EVAL_ASSET_ROOT="$ASSET_ROOT" \
    IMAGENET_FID_REFERENCE="$REFERENCE" \
    IMAGENET_FID_ENV="$FID_ENV" \
    bash scripts/prepare_imagenet_c2i_evaluation.sh
fi
test -f "$REFERENCE"
test -x "$FID_PYTHON"

manifest_value() {
  local manifest="$1" key="$2"
  python - "$manifest" "$key" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

sample_one() {
  local name="$1" gpus="$2" vq checkpoint output
  local compile_args=()
  vq="$(manifest_value "$ROOT/codes/$name/manifest.json" vq_checkpoint)"
  checkpoint="$ROOT/training/$name/results/000-GPT-XL/checkpoints/$(printf '%07d' "$EXPECTED_STEPS").pt"
  output="$EVALUATION_ROOT/$name"
  test -f "$vq"
  test -f "$checkpoint"
  if [[ -f "$output/samples.npz" && -f "$output/manifest.json" ]]; then
    return 0
  fi
  if [[ "${IMAGENET_EVAL_COMPILE:-0}" == 1 ]]; then
    compile_args=(--compile)
  fi
  CUDA_VISIBLE_DEVICES="$gpus" torchrun --standalone \
    --nproc_per_node="$(gpu_count "$gpus")" \
    scripts/generate_llamagen_c2i_fid.py \
    --gpt-ckpt "$checkpoint" --vq-ckpt "$vq" --output-dir "$output" \
    --gpt-model GPT-XL --precision bf16 --image-size 384 --image-size-eval 256 \
    --downsample-size 16 --num-classes 1000 --codebook-size 16384 \
    --codebook-embed-dim 8 --cfg-scale 1.75 --cfg-interval -1 \
    --temperature 1.0 --top-k 0 --top-p 1.0 \
    --per-rank-batch-size "$PER_RANK_BATCH" --num-fid-samples "$NUM_SAMPLES" \
    --global-seed 0 "${compile_args[@]}"
}

sample_one official "$OFFICIAL_GPUS" > "$ROOT/logs/evaluate_c2i_generate_official.log" 2>&1 &
official_pid=$!
sample_one reward "$REWARD_GPUS" > "$ROOT/logs/evaluate_c2i_generate_reward.log" 2>&1 &
reward_pid=$!
status=0
if ! wait "$official_pid"; then status=1; fi
if ! wait "$reward_pid"; then status=1; fi
if [[ "$status" -ne 0 ]]; then exit "$status"; fi

for name in official reward; do
  metrics="$EVALUATION_ROOT/$name/metrics.txt"
  if [[ ! -f "$metrics" ]]; then
    (
      cd "$ASSET_ROOT/inception"
      CUDA_VISIBLE_DEVICES="${IMAGENET_FID_GPU:-}" TF_CPP_MIN_LOG_LEVEL=2 \
        "$FID_PYTHON" "$EVALUATOR" "$REFERENCE" \
        "$EVALUATION_ROOT/$name/samples.npz"
    ) | tee "$metrics"
  fi
done

python scripts/summarize_imagenet_c2i_metrics.py \
  --evaluation-root "$EVALUATION_ROOT" \
  --reference "$REFERENCE" \
  --output "$EVALUATION_ROOT/summary.json"
