# Train LlamaGen GPT-B with the Reward VAE

This guide reproduces the ImageNet class-conditional (c2i) GPT-B experiment on a
new machine. It trains GPT-B from scratch using codes produced by the reward-tuned
VQ-16 tokenizer. It does not require a pretrained GPT-B checkpoint or the official
VAE.

The default recipe uses 384px images, VQ downsampling 16, and 576 visual tokens per
image. GPT-B is the 111M-parameter LlamaGen transformer. The reward checkpoint and
the generated code corpus must be kept separate from source code and credentials.

## 1. Clone and install

```bash
git clone https://github.com/hzeroyuke/vae-posttraining.git
cd vae-posttraining

# Option A: create the repository environment (CUDA 12.8 PyTorch wheels)
bash scripts/setup_env.sh
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  export VQVAE_ENV_ROOT="$PWD/.conda-env"
  source scripts/activate_env.sh
fi

# Option B: use a cluster-managed CUDA environment
export VQVAE_ENV_ROOT=/path/to/your/cuda-environment
source scripts/activate_env.sh
python -m pip install -r requirements.txt
python -m pip install -e .
```

The GPT-B launcher accepts `VQVAE_ENV_ROOT` when the environment lives outside the
repository. It must provide CUDA-enabled PyTorch, `torchrun`, NumPy, Pillow, and
PyArrow. Check the installation before starting a long run:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())
PY
```

## 2. Download the reward VAE

The published reward checkpoint is:

```text
hzeroyuke/vae-posttraining/reward_vae_vq_ds16_c2i_seed3101_step1000.pt
SHA256: 55c06b0ab914f9f40b24eba07033f7c452eecb6233562baf27e69121282434ea
```

Using the Hugging Face mirror for download:

```bash
MODEL_ROOT="$PWD/models"
mkdir -p "$MODEL_ROOT"
export HF_ENDPOINT=https://hf-mirror.com
hf download hzeroyuke/vae-posttraining \
  reward_vae_vq_ds16_c2i_seed3101_step1000.pt \
  --repo-type model --local-dir "$MODEL_ROOT"
export REWARD_VAE="$MODEL_ROOT/reward_vae_vq_ds16_c2i_seed3101_step1000.pt"
sha256sum "$REWARD_VAE"
```

If the mirror is unavailable, unset `HF_ENDPOINT` and use the official endpoint:

```bash
unset HF_ENDPOINT
hf download hzeroyuke/vae-posttraining \
  reward_vae_vq_ds16_c2i_seed3101_step1000.pt \
  --repo-type model --local-dir "$MODEL_ROOT"
```

The repository is public, so this download does not require a token. A token is only
needed for gated ImageNet access; enter it interactively with `hf auth login` and do
not put it in a script or `.env` file tracked by git.

## 3. Prepare ImageNet parquet data

The encoder expects this layout:

```text
$IMAGENET_PARQUET_ROOT/
  data/
    train-00000.parquet
    train-00001.parquet
    ...
```

Every parquet row must contain an `image` column with encoded image bytes and a
`label` column with the integer ImageNet-1k class id. The full train split contains
1,281,167 images and 1,000 classes. The repository does not redistribute ImageNet or
convert the original JPEG archives to parquet.

Obtain ImageNet through the [official ImageNet download page](https://image-net.org/download-images.php)
or the gated [ILSVRC/imagenet-1k dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k),
then use your local parquet conversion pipeline. Verify the resulting directory:

```bash
export IMAGENET_PARQUET_ROOT=/path/to/imagenet-1k-parquet
test -d "$IMAGENET_PARQUET_ROOT/data"
find "$IMAGENET_PARQUET_ROOT/data" -name 'train-*.parquet' | head
```

For a smoke test, the extractor also accepts `--max-samples 1024`; do not use that
small corpus for final metrics.

## 4. Encode the reward VAE

The commands below generate both crop-range groups used by the ImageNet recipe, then
combine them into one reward code corpus. Set `CODE_GPUS` to GPUs that are available
on the new machine.

```bash
export VQVAE_ENV_ROOT=/path/to/your/cuda-environment  # omit if already activated
source scripts/common.sh

export IMAGENET_PARQUET_ROOT=/path/to/imagenet-1k-parquet
export REWARD_VAE=/path/to/models/reward_vae_vq_ds16_c2i_seed3101_step1000.pt
export IMAGENET_EXPERIMENT_ROOT="$PWD/outputs/imagenet_gptb_reward_384"
export CODE_GPUS=0,1,2,3
export CODE_ROOT="$IMAGENET_EXPERIMENT_ROOT/codes"
TORCHRUN="$(command -v torchrun)"
PYTHON="$(command -v python)"
mkdir -p "$IMAGENET_EXPERIMENT_ROOT/logs" "$CODE_ROOT"

extract_reward() {
  local name="$1" crop_range="$2"
  local max_samples_args=()
  if [[ "${IMAGENET_MAX_SAMPLES:-0}" -gt 0 ]]; then
    max_samples_args+=(--max-samples "$IMAGENET_MAX_SAMPLES")
  fi
  CUDA_VISIBLE_DEVICES="$CODE_GPUS" "$TORCHRUN" --standalone \
    --nproc_per_node="$(gpu_count "$CODE_GPUS")" \
    third_party/LlamaGen/autoregressive/train/extract_codes_c2i_parquet.py \
    --data-path "$IMAGENET_PARQUET_ROOT" --split train \
    --code-path "$CODE_ROOT/$name" --vq-ckpt "$REWARD_VAE" \
    --image-size 384 --crop-range "$crop_range" --ten-crop \
    --batch-size "${IMAGENET_REWARD_CODE_BATCH_SIZE:-4}" \
    --num-workers "${IMAGENET_CODE_WORKERS:-2}" \
    --part-batches "${IMAGENET_CODE_PART_BATCHES:-256}" \
    --global-seed 20260812 "${max_samples_args[@]}"
}

extract_reward reward_110 1.1 \
  > "$IMAGENET_EXPERIMENT_ROOT/logs/extract_reward_110.log" 2>&1
extract_reward reward_105 1.05 \
  > "$IMAGENET_EXPERIMENT_ROOT/logs/extract_reward_105.log" 2>&1

"$PYTHON" scripts/combine_imagenet_code_variants.py \
  --primary "$CODE_ROOT/reward_110" \
  --secondary "$CODE_ROOT/reward_105" \
  --output "$CODE_ROOT/reward"
"$PYTHON" scripts/audit_imagenet_code_corpus.py \
  --code-path "$CODE_ROOT/reward" \
  --output "$CODE_ROOT/reward/audit.json"
```

The final code array has shape `[N, 2, 10, 576]`: two crop ranges, ten crops per
range, and 576 tokens per crop. The manifest records the reward VAE SHA256. If an
extraction is interrupted, rerun the corresponding command with `--resume` after
preserving its code directory.

## 5. Train GPT-B on reward codes

The launcher validates the code manifest before starting and refuses to overwrite an
existing result directory. GPT-B is initialized from scratch; no GPT checkpoint is
downloaded.

```bash
export IMAGENET_GPTB_CODE_PATH="$CODE_ROOT/reward"
export IMAGENET_GPTB_GPUS=0,1,2,3,4,5,6,7
export IMAGENET_GPTB_OUTPUT_ROOT="$IMAGENET_EXPERIMENT_ROOT"
export IMAGENET_GPTB_GLOBAL_BATCH_SIZE=256
export IMAGENET_GPTB_GRAD_ACCUMULATION=1
export IMAGENET_GPTB_MAX_STEPS=78000

bash scripts/train_imagenet_gptb.sh
```

For a smoke test, use a separate output root and set `IMAGENET_GPTB_MAX_STEPS=2`.
The global batch must be divisible by the number of visible GPUs times gradient
accumulation. To train on a 256px corpus, encode with `--image-size 256`, set
`IMAGENET_GPTB_IMAGE_SIZE=256`, and use a manifest with 256 tokens per image.

Training outputs are written under:

```text
$IMAGENET_GPTB_OUTPUT_ROOT/training/gptb/results/000-GPT-B/
  checkpoints/0000000.pt
  checkpoints/00010000.pt
  log.txt
```

## 6. Verify the run

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts third_party/LlamaGen
bash -n scripts/*.sh
tail -f "$IMAGENET_GPTB_OUTPUT_ROOT/training/gptb/results/000-GPT-B/log.txt"
```

The step-0 checkpoint confirms the model geometry, while later checkpoints contain
the optimizer state needed for continuation. Keep the VAE, code corpus, and output
directory immutable for an exact rerun.
