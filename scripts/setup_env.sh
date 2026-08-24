#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

export PIP_NO_EXTRA_INDEX=1

if [ -d .venv ] || [ -d .conda-env ]; then
  echo "An environment already exists; remove it explicitly to rebuild." >&2
  exit 2
fi

if "$PYTHON_BIN" -m venv .venv >/tmp/vqvae_gptxl_venv.log 2>&1; then
  source .venv/bin/activate
else
  cat /tmp/vqvae_gptxl_venv.log
  rm -rf .venv
  if ! command -v conda >/dev/null 2>&1; then
    echo "python venv failed and conda is unavailable." >&2
    exit 1
  fi
  conda create -y -p "$PWD/.conda-env" python=3.12 pip
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$PWD/.conda-env"
fi

python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install --upgrade pip setuptools wheel

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e .

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda", torch.version.cuda)
    print("device0", torch.cuda.get_device_name(0))
PY
