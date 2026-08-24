#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh

ROOT="${IMAGENET_EXPERIMENT_ROOT:-$PROJECT_ROOT/outputs/imagenet_xl_384}"
ASSET_ROOT="${IMAGENET_EVAL_ASSET_ROOT:-$PROJECT_ROOT/outputs/evaluation_assets}"
REFERENCE="${IMAGENET_FID_REFERENCE:-$ASSET_ROOT/VIRTUAL_imagenet256_labeled.npz}"
FID_ENV="${IMAGENET_FID_ENV:-$HOME/.cache/vqvae-gptxl-scaling/llamagen-c2i-fid}"
REFERENCE_URL="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"

mkdir -p "$ASSET_ROOT/inception" "$ROOT/logs" "$(dirname "$FID_ENV")"
if [[ ! -f "$REFERENCE" ]]; then
  partial="$REFERENCE.partial"
  curl --fail --location --retry 8 --retry-delay 10 --continue-at - \
    --output "$partial" "$REFERENCE_URL"
  mv "$partial" "$REFERENCE"
fi

if [[ ! -x "$FID_ENV/bin/python" ]]; then
  python3 -m venv "$FID_ENV"
fi
"$FID_ENV/bin/python" -m pip --isolated install \
  --index-url https://pypi.org/simple --upgrade pip
"$FID_ENV/bin/python" -m pip --isolated install \
  --index-url https://pypi.org/simple \
  "numpy==1.26.4" "scipy>=1.13,<1.16" "tensorflow==2.19.1" \
  "pydantic<3" "requests>=2.31" "tqdm>=4.66"

"$FID_ENV/bin/python" - <<'PY'
import numpy
import scipy
import tensorflow

print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("tensorflow", tensorflow.__version__)
PY
(
  cd "$ASSET_ROOT/inception"
  "$FID_ENV/bin/python" - "$PROJECT_ROOT/third_party/LlamaGen/evaluations/c2i/evaluator.py" <<'PY'
import runpy
import sys

import tensorflow.compat.v1 as tf

module = runpy.run_path(sys.argv[1])
module["_download_inception_model"]()
with open(module["INCEPTION_V3_PATH"], "rb") as handle:
    graph = tf.GraphDef()
    graph.ParseFromString(handle.read())
if not graph.node:
    raise RuntimeError("Downloaded Inception graph has no nodes")
print("inception_graph_nodes", len(graph.node))
PY
)
sha256sum "$REFERENCE" | tee "$ASSET_ROOT/VIRTUAL_imagenet256_labeled.sha256"
sha256sum "$ASSET_ROOT/inception/classify_image_graph_def.pb" \
  | tee "$ASSET_ROOT/inception/classify_image_graph_def.sha256"
printf '%s\n' "$REFERENCE" > "$ROOT/imagenet_fid_reference.txt"
printf '%s\n' "$FID_ENV/bin/python" > "$ROOT/imagenet_fid_python.txt"
