#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh

ROOT="${IMAGENET_EXPERIMENT_ROOT:-$PROJECT_ROOT/outputs/imagenet_xl_384}"
PY="${PYTHON_BIN:-$CONDA_PREFIX/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

coverage() {
  "$PY" - "$ROOT/codes/$1" <<'PY'
import sys
import zipfile
from pathlib import Path
import numpy as np

code_path = Path(sys.argv[1])
seen = np.zeros(1_281_167, dtype=bool)
for part in (code_path / "shards").glob("rank*.part*.npz"):
    try:
        with np.load(part) as shard:
            seen[shard["indices"]] = True
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        # A worker may still be finishing an npz write; retry next poll.
        continue
print(int(seen.sum()))
PY
}

mkdir -p "$ROOT/logs"
while true; do
  official_110="$(coverage official_110)"
  reward_110="$(coverage reward_110)"
  official_105="$(coverage official_105)"
  reward_105="$(coverage reward_105)"
  echo "$(date '+%F %T') coverage official_110=$official_110 reward_110=$reward_110 official_105=$official_105 reward_105=$reward_105"
  if [ "$official_110" -eq 1281167 ] && [ "$reward_110" -eq 1281167 ] \
    && [ "$official_105" -eq 1281167 ] && [ "$reward_105" -eq 1281167 ]; then
    break
  fi
  sleep "${IMAGENET_MERGE_POLL_SECONDS:-60}"
done

extract_merge() {
  local name="$1" ckpt="$2" crop_range="$3"
  CUDA_VISIBLE_DEVICES="${IMAGENET_MERGE_GPU:-2}" PYTHONPATH=third_party/LlamaGen \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=1 \
    third_party/LlamaGen/autoregressive/train/extract_codes_c2i_parquet.py \
    --data-path "${IMAGENET_PARQUET_ROOT:-$PROJECT_ROOT/data/imagenet-1k}" \
    --split train --code-path "$ROOT/codes/$name" --vq-ckpt "$ckpt" \
    --image-size 384 --crop-range "$crop_range" --ten-crop --batch-size 1 \
    --num-workers 1 --part-batches 1 --global-seed 20260812 --resume \
    > "$ROOT/logs/merge_${name}.log" 2>&1
}

extract_merge official_110 "${OFFICIAL_VQ:-$PROJECT_ROOT/models/vq_ds16_c2i.pt}" 1.1
extract_merge reward_110 "${REWARD_VQ:-$PROJECT_ROOT/models/reward_vq_ds16_c2i.pt}" 1.1
extract_merge official_105 "${OFFICIAL_VQ:-$PROJECT_ROOT/models/vq_ds16_c2i.pt}" 1.05
extract_merge reward_105 "${REWARD_VQ:-$PROJECT_ROOT/models/reward_vq_ds16_c2i.pt}" 1.05

test ! -e "$ROOT/codes/official"
test ! -e "$ROOT/codes/reward"
"$PY" scripts/combine_imagenet_code_variants.py \
  --primary "$ROOT/codes/official_110" --secondary "$ROOT/codes/official_105" \
  --output "$ROOT/codes/official"
"$PY" scripts/combine_imagenet_code_variants.py \
  --primary "$ROOT/codes/reward_110" --secondary "$ROOT/codes/reward_105" \
  --output "$ROOT/codes/reward"

"$PY" scripts/audit_imagenet_code_corpus.py --code-path "$ROOT/codes/official" --output "$ROOT/codes/official/audit.json"
"$PY" scripts/audit_imagenet_code_corpus.py --code-path "$ROOT/codes/reward" --output "$ROOT/codes/reward/audit.json"
echo "ImageNet code assets merged and audited under $ROOT"
