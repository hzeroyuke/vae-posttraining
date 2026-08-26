# LlamaGen GPT-XL on ImageNet

Chinese translation: [README_zh.md](README_zh.md).

## English

This repository contains the ImageNet class-conditional (c2i) training pipeline for
the LlamaGen GPT-XL prior. It encodes ImageNet-1k images with a VQ-16 tokenizer and
trains matched GPT-XL models from the resulting discrete image codes. The default
experiment compares two tokenizers:

- official: the released LlamaGen VQ-16 checkpoint;
- reward: a separately supplied reward-tuned VQ-16 checkpoint.

Datasets, model weights, generated samples, and checkpoints are not included in git.

### Protocol

| Item | Default |
| --- | --- |
| Task | ImageNet class-conditional generation (c2i) |
| Dataset | ImageNet-1k, 1,281,167 training images, 1,000 classes |
| Input/code resolution | 384px input, VQ-16, 24 x 24 = 576 tokens |
| Codebook | 16,384 entries |
| Data augmentation | Ten-crop at crop ranges 1.10 and 1.05; one group and one crop are sampled per image and epoch |
| Prior | LlamaGen GPT-XL, 774,699,520 parameters |
| Optimizer | AdamW, learning rate 1e-4, betas (0.9, 0.95), weight decay 0.05 |
| Regularization | Residual/FFN dropout 0.1, token dropout 0.1, gradient clipping 1.0 |
| Global batch | 256 images; default two-GPU training uses micro-batch 4 and accumulation 32 |
| Precision | BF16 |
| Seed | 20260812 |
| Default endpoint | 78,000 optimizer steps; checkpoints every 10,000 steps |

The script keeps --epochs 300 as the upstream epoch limit, but the default
--max-steps 78000 intentionally stops much earlier. Therefore the default run is a
matched 78k-step experiment, not a reproduction of the full 300-epoch published
LlamaGen-XL result.

### Requirements

- Linux with CUDA, NCCL, and enough GPU memory for GPT-XL;
- Python 3.10 or newer;
- a CUDA-enabled PyTorch installation;
- local ImageNet parquet shards and both VQ checkpoints;
- sufficient local/NAS storage for encoded codes, logs, checkpoints, and evaluation samples.

The repository helper creates a virtual environment and installs PyTorch from the CUDA
12.8 wheel index:

```bash
bash scripts/setup_env.sh
```

If PyTorch is already managed by your cluster environment, install the Python
dependencies and the package manually instead:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Data and checkpoints

The extraction script expects parquet files below a data/ directory:

```text
<IMAGENET_PARQUET_ROOT>/data/train-*.parquet
```

Each row must contain image bytes and an integer ImageNet label. Set the paths for the
dataset, VQ checkpoints, and experiment output before running:

```bash
export IMAGENET_PARQUET_ROOT=/path/to/imagenet-1k
export OFFICIAL_VQ=/path/to/vq_ds16_c2i.pt
export REWARD_VQ=/path/to/reward_vq_ds16_c2i.pt
export IMAGENET_EXPERIMENT_ROOT=/path/to/runs/imagenet_xl_384

# Optional: log to Weights & Biases. The default is offline logging.
export WANDB_MODE=offline
export WANDB_PROJECT=llamagen-imagenet-xl
```

Do not commit these paths, credentials, or model files. A local .env file is loaded
automatically by scripts/common.sh and should remain untracked.

### External resources

Download the upstream LlamaGen source and official checkpoints from these locations:

- [FoundationVision/LlamaGen](https://github.com/FoundationVision/LlamaGen): upstream model and sampling code;
- [official VQ-16 checkpoint](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt): set this file as OFFICIAL_VQ;
- [reward VQ-16 checkpoint](https://huggingface.co/hzeroyuke/vae-posttraining/blob/main/reward_vae_vq_ds16_c2i_seed3101_step1000.pt): use this file as the GPT-B reward tokenizer;
- [published LlamaGen-XL checkpoint](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_XL_384.pt): optional for inference/reference only, not required for this repository's from-scratch GPT-XL training.

The GPT-B reward experiment trains the prior from scratch, so no pretrained GPT-B
checkpoint is required. The complete download, code extraction, and training sequence
is documented in [docs/GPTB_REWARD_TRAINING.md](docs/GPTB_REWARD_TRAINING.md).

For example, download the official VAE on a new machine with:

```bash
MODEL_ROOT=/path/to/models
mkdir -p "$MODEL_ROOT"
curl -L --fail --retry 8 \
  -o "$MODEL_ROOT/vq_ds16_c2i.pt" \
  https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt
export OFFICIAL_VQ="$MODEL_ROOT/vq_ds16_c2i.pt"
```

Obtain ImageNet-1k from either the [official ImageNet download page](https://image-net.org/download-images.php)
(registration and the ImageNet license are required) or the gated
[ILSVRC/imagenet-1k dataset on Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k).
For the Hugging Face route, accept the dataset terms in the browser and authenticate
on the new machine with `hf auth login` before downloading.
The extraction stage does not download ImageNet and does not unpack JPEG archives.
It expects parquet shards under IMAGENET_PARQUET_ROOT/data/train-*.parquet, with an
image column containing image bytes and a label column containing the integer class id.
If you download the original ImageNet archive, convert it to that schema before running
scripts/01_prepare_imagenet_xl_assets.sh; this repository does not include that converter.

The reward VAE is experiment-specific and is not an upstream LlamaGen release. Copy the
exported reward checkpoint from the machine where the reward-tuned tokenizer was trained,
or regenerate it with your own tokenizer post-training procedure, then set REWARD_VQ.
The paired scripts require both VQ checkpoints, and the final audit verifies that their
checkpoint hashes are different.

The FID reference batch is downloaded automatically by
scripts/prepare_imagenet_c2i_evaluation.sh from the
[OpenAI ImageNet reference file](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz).

### Training

Run the two expensive stages explicitly when you want an auditable boundary between
code extraction and model training:

```bash
bash scripts/01_prepare_imagenet_xl_assets.sh
bash scripts/02_train_imagenet_xl.sh
```

Stage 1 extracts both crop ranges for both tokenizers, combines them into
codes/{official,reward}, and writes corpus audits. Stage 2 starts the matched
official and reward GPT-XL jobs and writes an initial step-0 checkpoint for comparison.

To train a single GPT-B c2i prior on an existing ImageNet code corpus, use the
multi-GPU launcher below. It defaults to the same 384px/576-token geometry as the
GPT-XL experiment; set `IMAGENET_GPTB_IMAGE_SIZE=256` for a 256px corpus. The code
directory must contain `codes.npy`, `labels.npy`, and `manifest.json`.

```bash
VQVAE_ENV_ROOT=/path/to/cuda-environment \
IMAGENET_GPTB_CODE_PATH=/path/to/imagenet_codes \
IMAGENET_GPTB_GPUS=0,1,2,3 \
IMAGENET_GPTB_OUTPUT_ROOT=/path/to/runs/imagenet_gptb_384 \
bash scripts/train_imagenet_gptb.sh
```

The script uses GPT-B (about 111M parameters), `vocab_size=16384`, c2i conditioning,
BF16, and NCCL DDP. `VQVAE_ENV_ROOT` is optional when the repository has its own
`.venv` or `.conda-env`; it selects a cluster-managed CUDA environment otherwise.
For a bounded smoke test, set `IMAGENET_GPTB_MAX_STEPS=2` and use a separate output
directory.

For the complete new-machine procedure that downloads the reward VAE, prepares the
ImageNet parquet code corpus, and trains GPT-B on reward codes, see
[docs/GPTB_REWARD_TRAINING.md](docs/GPTB_REWARD_TRAINING.md).

For a long run in a detached tmux session:

```bash
bash scripts/run_imagenet_xl_tmux.sh
tmux attach -t llamagen_imagenet_xl
```

Useful resource overrides include:

```bash
export IMAGENET_CODE_GPUS=0,1
export IMAGENET_REWARD_CODE_GPUS=2,3
export IMAGENET_BASE_GPUS=0,1
export IMAGENET_REWARD_GPUS=2,3
export IMAGENET_CODE_WORKERS=2
export IMAGENET_CODE_PART_BATCHES=256
export IMAGENET_PRELOAD_EPOCH_CODES=1
```

For a bounded smoke test, use a separate output directory and lower the step limit:

```bash
IMAGENET_MAX_SAMPLES=1024 \
IMAGENET_MAX_STEPS=10 \
IMAGENET_EXPERIMENT_ROOT=/tmp/imagenet_xl_smoke \
bash scripts/01_prepare_imagenet_xl_assets.sh

IMAGENET_MAX_STEPS=10 \
IMAGENET_EXPERIMENT_ROOT=/tmp/imagenet_xl_smoke \
bash scripts/02_train_imagenet_xl.sh
```

The smoke run checks wiring only and cannot pass the full-corpus audit.

### Resume and monitoring

Each single-model job can resume from a complete checkpoint. Keep the same code path,
world size, global batch, and accumulation settings used by the original run:

```bash
bash scripts/train_imagenet_xl_one.sh \
  official 0,1 \
  "$IMAGENET_EXPERIMENT_ROOT/training/official/results/000-GPT-XL/checkpoints/00010000.pt"
```

scripts/monitor_imagenet_xl_training.sh validates checkpoints, restarts failed jobs,
and can launch the final paired evaluation. Set IMAGENET_MONITOR_TASKS to official,
reward, or both. The optional finalize_and_train_imagenet_xl_dual_crop.sh helper is
intended for recovery workflows that need automatic GPU selection and crop retries.

### Evaluation

The evaluation follows the LlamaGen c2i protocol: 50,000 samples per system at 256px,
classifier-free guidance scale 1.75, temperature 1.0, and no top-k truncation.
Prepare the isolated TensorFlow evaluator and ImageNet reference batch first:

```bash
bash scripts/prepare_imagenet_c2i_evaluation.sh
```

Then evaluate both final checkpoints:

```bash
bash scripts/evaluate_imagenet_xl_c2i.sh
```

The summary is written to:

```text
$IMAGENET_EXPERIMENT_ROOT/evaluation/c2i_00078000_fid50000/summary.json
```

The evaluator reports Inception Score, FID, sFID, Precision, and Recall. Generation is
resumable at the PNG/sample boundary, and both systems use the same class and sampling
seed schedule.

### Outputs

```text
$IMAGENET_EXPERIMENT_ROOT/
  codes/
    official/{codes.npy,labels.npy,manifest.json,audit.json}
    reward/{codes.npy,labels.npy,manifest.json,audit.json}
  training/
    official/results/000-GPT-XL/checkpoints/{0000000,00010000,...,00078000}.pt
    reward/results/000-GPT-XL/checkpoints/{0000000,00010000,...,00078000}.pt
  initial_checkpoint_audit.json
  final_checkpoint_audit.json
  evaluation/c2i_00078000_fid50000/summary.json
```

### Repository layout

- scripts/01_prepare_imagenet_xl_assets.sh: ImageNet encoding, dual-crop merge, and corpus audit;
- scripts/02_train_imagenet_xl.sh: paired GPT-XL training launcher;
- scripts/train_imagenet_xl_one.sh: single-run training and checkpoint resume;
- scripts/train_imagenet_gptb.sh: configurable multi-GPU GPT-B c2i training launcher;
- docs/GPTB_REWARD_TRAINING.md: end-to-end Reward VAE to GPT-B reproduction guide;
- scripts/monitor_imagenet_xl_training.sh: training monitoring, recovery, and final evaluation scheduling;
- scripts/evaluate_imagenet_xl_c2i.sh: sampling and FID metric calculation;
- third_party/LlamaGen/: minimal vendored LlamaGen dependency and ImageNet data implementation.

### Verification

These checks do not start a training job:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts third_party/LlamaGen
bash -n scripts/*.sh
```

For a completed full-data run, the final audit verifies corpus shape, codebook usage,
matched step-0 initialization, GPT-XL parameter count, training arguments, and loadable
optimizer state:

```bash
python scripts/audit_imagenet_xl_training.py \
  --experiment-root "$IMAGENET_EXPERIMENT_ROOT" \
  --expected-steps 78000
```

### License and attribution

The vendored LlamaGen subset is under
[third_party/LlamaGen/LICENSE](third_party/LlamaGen/LICENSE) and records its upstream
commit in [third_party/LlamaGen/UPSTREAM.md](third_party/LlamaGen/UPSTREAM.md). Please
also follow the licenses and terms of ImageNet,
PyTorch, and any downloaded checkpoint or evaluator dependency.
