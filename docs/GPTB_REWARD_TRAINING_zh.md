# 使用 Reward VAE 训练 LlamaGen GPT-B

本文档说明如何在另一台机器上复现 ImageNet c2i GPT-B 实验。训练目标是使用
Reward VAE 生成的离散图像 code，从零训练 GPT-B。训练不需要预训练 GPT-B checkpoint，
也不需要官方 Base VAE。

默认配置为 384px 输入、VQ 下采样倍数 16、每张图 576 个 visual token。GPT-B 是
约 1.11 亿参数的 LlamaGen Transformer。Reward checkpoint 和生成的 code corpus
应与源码、凭据分开保存。

## 1. 克隆仓库并安装环境

```bash
git clone https://github.com/hzeroyuke/vae-posttraining.git
cd vae-posttraining

# 方案 A：创建仓库环境（安装 CUDA 12.8 PyTorch wheel）
bash scripts/setup_env.sh
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  export VQVAE_ENV_ROOT="$PWD/.conda-env"
  source scripts/activate_env.sh
fi

# 方案 B：使用集群统一管理的 CUDA 环境
export VQVAE_ENV_ROOT=/path/to/your/cuda-environment
source scripts/activate_env.sh
python -m pip install -r requirements.txt
python -m pip install -e .
```

GPT-B 启动脚本支持通过 `VQVAE_ENV_ROOT` 选择仓库外部的 CUDA 环境。该环境需要提供
支持 CUDA 的 PyTorch、`torchrun`、NumPy、Pillow 和 PyArrow。长时间运行前先检查：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())
PY
```

## 2. 下载 Reward VAE

发布的 Reward checkpoint：

```text
hzeroyuke/vae-posttraining/reward_vae_vq_ds16_c2i_seed3101_step1000.pt
SHA256: 55c06b0ab914f9f40b24eba07033f7c452eecb6233562baf27e69121282434ea
```

使用 Hugging Face 镜像下载：

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

如果镜像不可用，取消 `HF_ENDPOINT` 后使用官方站点：

```bash
unset HF_ENDPOINT
hf download hzeroyuke/vae-posttraining \
  reward_vae_vq_ds16_c2i_seed3101_step1000.pt \
  --repo-type model --local-dir "$MODEL_ROOT"
```

该模型仓库是公开的，下载不需要 token。只有访问 gated ImageNet 数据集时才需要
token；请使用 `hf auth login` 交互式输入，不要把 token 写入脚本或提交到 git。

## 3. 准备 ImageNet parquet 数据

编码脚本要求以下目录结构：

```text
$IMAGENET_PARQUET_ROOT/
  data/
    train-00000.parquet
    train-00001.parquet
    ...
```

每一行必须有 `image` 列（编码后的图片字节）和 `label` 列（ImageNet-1k 整数类别）。
完整训练集包含 1,281,167 张图片和 1,000 个类别。仓库不分发 ImageNet，也不包含将
原始 JPEG 压缩包转换成 parquet 的脚本。

请从 [ImageNet 官方下载页面](https://image-net.org/download-images.php) 或 gated 的
[ILSVRC/imagenet-1k 数据集](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
获取数据，再使用你自己的 parquet 转换流程。检查目录：

```bash
export IMAGENET_PARQUET_ROOT=/path/to/imagenet-1k-parquet
test -d "$IMAGENET_PARQUET_ROOT/data"
find "$IMAGENET_PARQUET_ROOT/data" -name 'train-*.parquet' | head
```

冒烟测试时可以给编码器传入 `--max-samples 1024`，但小数据不能用于最终指标。

## 4. 使用 Reward VAE 提取 code

下面的命令提取 ImageNet 训练协议所需的两组 crop range，再合并成一个 Reward code
corpus。请把 `CODE_GPUS` 改成新机器上可用的 GPU。

```bash
export VQVAE_ENV_ROOT=/path/to/your/cuda-environment  # 已激活环境时可省略
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

最终 code 数组形状为 `[N, 2, 10, 576]`：两个 crop range、每组十个 crop、每个 crop
576 个 token。manifest 会记录 Reward VAE 的 SHA256。如果编码中断，保留对应目录，
给相应的提取命令增加 `--resume` 后重新运行。

## 5. 在 Reward code 上训练 GPT-B

启动脚本会先校验 code manifest，并拒绝覆盖已有结果目录。GPT-B 从零初始化，不需要
下载 GPT checkpoint。

```bash
export IMAGENET_GPTB_CODE_PATH="$CODE_ROOT/reward"
export IMAGENET_GPTB_GPUS=0,1,2,3,4,5,6,7
export IMAGENET_GPTB_OUTPUT_ROOT="$IMAGENET_EXPERIMENT_ROOT"
export IMAGENET_GPTB_GLOBAL_BATCH_SIZE=256
export IMAGENET_GPTB_GRAD_ACCUMULATION=1
export IMAGENET_GPTB_MAX_STEPS=78000

bash scripts/train_imagenet_gptb.sh
```

冒烟测试请使用独立输出目录，并设置 `IMAGENET_GPTB_MAX_STEPS=2`。global batch 必须能被
可见 GPU 数量乘以梯度累积步数整除。如果使用 256px code，需要编码时设置
`--image-size 256`，训练时设置 `IMAGENET_GPTB_IMAGE_SIZE=256`，并确保 manifest 中
每张图片是 256 个 token。

训练输出位置：

```text
$IMAGENET_GPTB_OUTPUT_ROOT/training/gptb/results/000-GPT-B/
  checkpoints/0000000.pt
  checkpoints/00010000.pt
  log.txt
```

## 6. 验证训练

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts third_party/LlamaGen
bash -n scripts/*.sh
tail -f "$IMAGENET_GPTB_OUTPUT_ROOT/training/gptb/results/000-GPT-B/log.txt"
```

step-0 checkpoint 用于确认模型几何配置，后续 checkpoint 包含继续训练所需的 optimizer
state。为了精确复现实验，请保持 VAE、code corpus 和输出目录不变。
