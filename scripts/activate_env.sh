#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -n "${VQVAE_ENV_ROOT:-}" ] && [ -f "$VQVAE_ENV_ROOT/bin/activate" ]; then
  source "$VQVAE_ENV_ROOT/bin/activate"
elif [ -n "${VQVAE_ENV_ROOT:-}" ] && [ -x "$VQVAE_ENV_ROOT/bin/python" ]; then
  export PATH="$VQVAE_ENV_ROOT/bin:$PATH"
  export CONDA_PREFIX="$VQVAE_ENV_ROOT"
elif [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  source "$ROOT_DIR/.venv/bin/activate"
elif [ -f "$ROOT_DIR/.conda-env/bin/activate" ]; then
  source "$ROOT_DIR/.conda-env/bin/activate"
elif [ -x "$ROOT_DIR/.conda-env/bin/python" ]; then
  export PATH="$ROOT_DIR/.conda-env/bin:$PATH"
  export CONDA_PREFIX="$ROOT_DIR/.conda-env"
else
  echo "No environment found. Run: bash scripts/setup_env.sh" >&2
  exit 1
fi
