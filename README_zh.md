# LlamaGen GPT-XL 在 ImageNet 上的训练

English version: [README.md](README.md).

本仓库包含 LlamaGen GPT-XL 在 ImageNet 上进行类别条件生成（c2i）训练的完整流程：
使用 VQ-16 tokenizer 将 ImageNet-1k 编码为离散图像 token，再训练 GPT-XL 先验模型。
默认同时运行两条可匹配的实验：

- official：官方发布的 LlamaGen VQ-16 checkpoint；
- reward：由外部提供的 reward-tuned VQ-16 checkpoint。

数据集、模型权重、生成图片和训练 checkpoint 均不会提交到 git。

### 训练协议

| 项目 | 默认值 |
| --- | --- |
| 任务 | ImageNet 类别条件生成（c2i） |
| 数据集 | ImageNet-1k，1,281,167 张训练图片，1,000 个类别 |
| 输入/编码分辨率 | 输入 384px，VQ-16，24 x 24 = 576 个 token |
| Codebook | 16,384 个条目 |
| 数据增强 | crop range 1.10 和 1.05 的十裁剪；每个 epoch 为每张图片随机选择一个组和一个 crop |
| 先验模型 | LlamaGen GPT-XL，774,699,520 个参数 |
| 优化器 | AdamW，学习率 1e-4，betas (0.9, 0.95)，weight decay 0.05 |
| 正则化 | Residual/FFN dropout 0.1，token dropout 0.1，梯度裁剪 1.0 |
| Global batch | 256 张图片；默认双卡训练时 micro-batch 为 4、累积 32 次 |
| 精度 | BF16 |
| 随机种子 | 20260812 |
| 默认终点 | 78,000 个 optimizer step；每 10,000 step 保存一次 checkpoint |

脚本保留上游的 --epochs 300 作为 epoch 上限，但默认的 --max-steps 78000 会更早停止。
因此默认运行是匹配的 78k-step 实验，并不是 LlamaGen-XL 已发表的完整 300-epoch 结果复现。

### 环境要求

- Linux、CUDA、NCCL，以及足够显存运行 GPT-XL；
- Python 3.10 或更高版本；
- 支持 CUDA 的 PyTorch；
- 本地 ImageNet parquet 分片和两个 VQ checkpoint；
- 足够的本地或 NAS 空间，用于 codes、日志、checkpoint 和评估样本。

仓库脚本会创建虚拟环境，并从 CUDA 12.8 wheel 源安装 PyTorch：

```bash
bash scripts/setup_env.sh
```

如果集群已经统一管理 PyTorch，也可以手动安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 数据与 checkpoint

编码脚本默认从以下位置读取 parquet：

```text
<IMAGENET_PARQUET_ROOT>/data/train-*.parquet
```

每一行需要包含图片字节和整数形式的 ImageNet label。运行前设置数据集、VQ checkpoint 和输出目录：

```bash
export IMAGENET_PARQUET_ROOT=/path/to/imagenet-1k
export OFFICIAL_VQ=/path/to/vq_ds16_c2i.pt
export REWARD_VQ=/path/to/reward_vq_ds16_c2i.pt
export IMAGENET_EXPERIMENT_ROOT=/path/to/runs/imagenet_xl_384

# 可选：记录到 Weights & Biases；默认离线记录。
export WANDB_MODE=offline
export WANDB_PROJECT=llamagen-imagenet-xl
```

不要提交这些路径、凭据或模型文件。scripts/common.sh 会自动加载本地 .env，该文件应保持未跟踪状态。

### 外部资源获取

官方 LlamaGen 源码和权重可以从以下位置获取：

- [FoundationVision/LlamaGen](https://github.com/FoundationVision/LlamaGen)：上游模型和采样代码；
- [官方 VQ-16 checkpoint](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt)：下载后设置为 OFFICIAL_VQ；
- [Reward VQ-16 checkpoint](https://huggingface.co/hzeroyuke/vae-posttraining/blob/main/reward_vae_vq_ds16_c2i_seed3101_step1000.pt)：GPT-B Reward 实验使用此文件；
- [已发布的 LlamaGen-XL checkpoint](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_XL_384.pt)：只用于推理或结果参考，本仓库从头训练 GPT-XL 时不需要它。

GPT-B Reward 实验会从零训练 prior，不需要下载预训练 GPT-B checkpoint。完整的下载、
code 提取和训练步骤见 [docs/GPTB_REWARD_TRAINING_zh.md](docs/GPTB_REWARD_TRAINING_zh.md)。

例如，在新机器上可以这样下载官方 VAE：

```bash
MODEL_ROOT=/path/to/models
mkdir -p "$MODEL_ROOT"
curl -L --fail --retry 8 \
  -o "$MODEL_ROOT/vq_ds16_c2i.pt" \
  https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt
export OFFICIAL_VQ="$MODEL_ROOT/vq_ds16_c2i.pt"
```

ImageNet-1k 可以从 [ImageNet 官方下载页面](https://image-net.org/download-images.php) 获取（需要注册并遵守 ImageNet 使用许可），
也可以使用 Hugging Face 上需要授权的
[ILSVRC/imagenet-1k 数据集](https://huggingface.co/datasets/ILSVRC/imagenet-1k)。使用 Hugging Face 方案时，先在网页上接受数据集条款，
再在新机器上执行 `hf auth login` 完成认证。编码阶段不会自动下载 ImageNet，也不会直接解压 JPEG 压缩包。
它要求 IMAGENET_PARQUET_ROOT/data/train-*.parquet 下存在 parquet 分片，其中 image 列包含图片字节，label 列包含整数类别编号。
如果下载的是原始 ImageNet 压缩包，需要先转换成这个 schema，再运行 scripts/01_prepare_imagenet_xl_assets.sh；本仓库没有附带该转换脚本。

reward VAE 是本实验专用的 checkpoint，不是上游 LlamaGen 发布的公开权重。请从训练 reward tokenizer 的原机器复制导出的 checkpoint，
或者使用自己的 tokenizer post-training 流程重新生成，然后设置 REWARD_VQ。成对训练脚本要求两个 VQ checkpoint 都存在，最终审计还会检查它们的 checkpoint hash 不同。

FID reference batch 会由 scripts/prepare_imagenet_c2i_evaluation.sh 自动从
[OpenAI ImageNet reference 文件](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) 下载。

### 训练

如果希望在数据编码和模型训练之间保留可审计的边界，分别执行两个阶段：

```bash
bash scripts/01_prepare_imagenet_xl_assets.sh
bash scripts/02_train_imagenet_xl.sh
```

第一阶段为两个 tokenizer 提取两种 crop range，合并为 codes/{official,reward}，并写入数据审计文件。
第二阶段同时启动 official 和 reward 两个匹配的 GPT-XL 训练，并保存 step-0 初始 checkpoint 供比较。

如果要在已有的 ImageNet code corpus 上训练单路 GPT-B c2i 先验，使用下面的多卡脚本。
脚本默认采用与 GPT-XL 实验相同的 384px/576-token 几何配置；如果 code corpus 是
256px 编码，可以设置 `IMAGENET_GPTB_IMAGE_SIZE=256`。code 目录必须已经包含
`codes.npy`、`labels.npy` 和 `manifest.json`。

```bash
VQVAE_ENV_ROOT=/path/to/cuda-environment \
IMAGENET_GPTB_CODE_PATH=/path/to/imagenet_codes \
IMAGENET_GPTB_GPUS=0,1,2,3 \
IMAGENET_GPTB_OUTPUT_ROOT=/path/to/runs/imagenet_gptb_384 \
bash scripts/train_imagenet_gptb.sh
```

脚本使用 GPT-B（约 1.11 亿参数）、`vocab_size=16384`、c2i 条件、BF16 和 NCCL DDP。
如果仓库内已有 `.venv` 或 `.conda-env`，则 `VQVAE_ENV_ROOT` 可以省略；在集群统一管理
CUDA 环境时，可以用它指定外部环境。冒烟测试可设置 `IMAGENET_GPTB_MAX_STEPS=2`，
并使用独立的输出目录。

从另一台机器下载 Reward VAE、准备 ImageNet parquet code corpus，并在 Reward code
上训练 GPT-B 的完整流程见
[docs/GPTB_REWARD_TRAINING_zh.md](docs/GPTB_REWARD_TRAINING_zh.md)。

长时间运行可以放在 tmux 中：

```bash
bash scripts/run_imagenet_xl_tmux.sh
tmux attach -t llamagen_imagenet_xl
```

常用资源配置如下：

```bash
export IMAGENET_CODE_GPUS=0,1
export IMAGENET_REWARD_CODE_GPUS=2,3
export IMAGENET_BASE_GPUS=0,1
export IMAGENET_REWARD_GPUS=2,3
export IMAGENET_CODE_WORKERS=2
export IMAGENET_CODE_PART_BATCHES=256
export IMAGENET_PRELOAD_EPOCH_CODES=1
```

冒烟测试应使用独立输出目录，并降低样本数和 step 数；它只用于检查流程连通性：

```bash
IMAGENET_MAX_SAMPLES=1024 \
IMAGENET_MAX_STEPS=10 \
IMAGENET_EXPERIMENT_ROOT=/tmp/imagenet_xl_smoke \
bash scripts/01_prepare_imagenet_xl_assets.sh

IMAGENET_MAX_STEPS=10 \
IMAGENET_EXPERIMENT_ROOT=/tmp/imagenet_xl_smoke \
bash scripts/02_train_imagenet_xl.sh
```

冒烟数据不满足全量 corpus audit 的预期，不能用于最终结果验证。

### 恢复与监控

单模型训练可以从完整 checkpoint 恢复。恢复时应保持原来的 code path、world size、global batch 和梯度累积配置：

```bash
bash scripts/train_imagenet_xl_one.sh \
  official 0,1 \
  "$IMAGENET_EXPERIMENT_ROOT/training/official/results/000-GPT-XL/checkpoints/00010000.pt"
```

scripts/monitor_imagenet_xl_training.sh 会校验 checkpoint、重启失败任务，并可在最终 checkpoint 就绪后启动成对评估。
通过 IMAGENET_MONITOR_TASKS=official、reward 或 both 选择监控范围。可选的
scripts/finalize_and_train_imagenet_xl_dual_crop.sh 用于需要自动选择 GPU 和重试 crop 提取的恢复场景。

### 评估

评估遵循 LlamaGen c2i 协议：每个系统生成 50,000 张 256px 图片，CFG scale 为 1.75，
temperature 为 1.0，不使用 top-k 截断。先准备独立的 TensorFlow evaluator 和 ImageNet reference batch：

```bash
bash scripts/prepare_imagenet_c2i_evaluation.sh
```

然后评估两个最终 checkpoint：

```bash
bash scripts/evaluate_imagenet_xl_c2i.sh
```

汇总文件位置：

```text
$IMAGENET_EXPERIMENT_ROOT/evaluation/c2i_00078000_fid50000/summary.json
```

评估器会输出 Inception Score、FID、sFID、Precision 和 Recall。生成过程支持在 PNG/样本边界恢复，
两个系统使用相同的类别和采样随机种子计划。

### 代码结构

- scripts/01_prepare_imagenet_xl_assets.sh：ImageNet 编码、双 crop 合并和 corpus audit；
- scripts/02_train_imagenet_xl.sh：启动两路 GPT-XL 训练；
- scripts/train_imagenet_xl_one.sh：单路训练或 checkpoint 恢复；
- scripts/train_imagenet_gptb.sh：可配置的多卡 GPT-B c2i 训练脚本；
- docs/GPTB_REWARD_TRAINING_zh.md：从 Reward VAE 到 GPT-B 的完整复现实验指南；
- scripts/monitor_imagenet_xl_training.sh：训练监控、恢复和最终评估调度；
- scripts/evaluate_imagenet_xl_c2i.sh：采样和 FID 指标计算；
- third_party/LlamaGen/：保留许可证的最小 LlamaGen 依赖和 ImageNet 数据实现。

### 验证

下面的检查不会启动训练任务：

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts third_party/LlamaGen
bash -n scripts/*.sh
```

完整数据训练结束后，最终审计会检查 corpus shape、codebook 使用情况、匹配的 step-0 初始化、
GPT-XL 参数量、训练参数和可加载的 optimizer state：

```bash
python scripts/audit_imagenet_xl_training.py \
  --experiment-root "$IMAGENET_EXPERIMENT_ROOT" \
  --expected-steps 78000
```

### 许可证与致谢

vendored LlamaGen 子集的许可证在 [third_party/LlamaGen/LICENSE](third_party/LlamaGen/LICENSE)，上游 commit 记录在
[third_party/LlamaGen/UPSTREAM.md](third_party/LlamaGen/UPSTREAM.md)。同时请遵守 ImageNet、PyTorch 以及下载的 checkpoint 和 evaluator 依赖的许可证与使用条款。
