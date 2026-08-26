# VAE Post-training with Continuous GRPO

## 1. 实验目标

构建一套针对连续潜变量 VAE 的强化学习后训练 codebase。核心目标不是只优化 decoder，而是训练整个 VAE，使其同时满足：

- 在固定 held-out 数据上，重建图像的 PickScore 稳定提升。
- encoder 学到更有语义、对上层连续生成模型更友好的 latent space。
- MSE、PSNR、LPIPS、FID 等重建与生成质量指标不退化。
- latent 通道数、空间尺寸、缩放方式和边缘分布仍兼容 SiT-XL/2。
- 冻结 SiT 直接替换 VAE decoder 时有收益；重新适配 SiT 后，候选 encoder 的 latent 也比原始 latent 更容易建模。

首个正式基线采用：

- VAE：`stabilityai/sd-vae-ft-mse`。
- 上层基座：官方 class-conditional SiT-XL/2，分辨率 256x256。
- latent 接口：4 通道、32x32，SiT 空间缩放系数 `s=0.18215`。
- reward：`laion/CLIP-ViT-H-14-laion2B-s32B-b79K` 与 `yuvalkirstain/PickScore_v1`。
- 数据：ImageNet-1K，文本为 `a photo of a {class_name}`，与官方 SiT 的类别条件一致。
- 硬件：GPU 0-3。

## 2. 核心假设

原始 VAE encoder 后验为：

`q_phi(u|x) = N(mu_phi(x), diag(sigma_post_phi(x)^2))`

SD-VAE 的 prior KL 权重通常很小，`logvar` 又会被 clamp，导致 `sigma_post` 相对 latent 尺度非常小。直接从 `q_phi` 采样多个 latent，组内图像和 reward 差异可能小于 reward model 本身的数值噪声，无法形成有效 advantage。

因此把 VAE 后验与 RL 探索策略解耦。在 SiT 使用的缩放 latent 空间中定义：

`m_phi(x) = s * mu_phi(x)`

`pi_phi(z|x) = N(m_phi(x), diag(sigma_expl^2))`

`sigma_expl,c = rho * Std_data,h,w(m_ref,c)`

其中 `m_ref` 来自冻结的初始 VAE，`rho` 首轮扫描 `0.10 / 0.20 / 0.30`，额外保留 `rho=0.05` 作为近确定性对照。`sigma_expl` 是训练开始前统计并固定的每通道尺度，不等于 VAE 的 `sigma_post`，也不由 encoder 的 `logvar` 产生。

采样与解码为：

`epsilon_i ~ N(0, I)`

`z_i = m_phi(x) + sigma_expl * epsilon_i`

`x_hat_i = D_psi(z_i / s)`

行为策略 log-prob 为：

`log pi_phi(z_i|x) = -0.5 * sum_j[((z_ij-m_ij)/sigma_expl,j)^2 + 2 log sigma_expl,j + log(2*pi)]`

实现 score-function gradient 时，必须用 `log_prob(z_i.detach())`。如果不 detach，而让 `z_i=m_phi+sigma_expl*epsilon_i` 的重参数化路径进入同一个 log-prob，`z_i-m_phi` 对均值的梯度会抵消，GRPO 梯度将错误地变成零或接近零。

## 3. 三类 KL 必须分开

实验中存在三个含义不同的 KL，配置、日志和代码命名不得混用。

### 3.1 VAE prior KL

`KL_prior = KL(q_phi(u|x) || N(0,I))`

它维护 VAE 本身的生成语义，使用原始 SD-VAE 训练配方中的权重，首轮默认 `1e-6`。该项继续使用 encoder 的真实 `mu` 和 `logvar`，不能替换为 `sigma_expl`。

### 3.2 Posterior reference KL

`KL_post_ref = KL(q_phi(u|x) || q_ref(u|x))`

它用两个对角高斯的闭式解约束 VAE 的真实 posterior：

`KL_post_ref = 0.5 * sum_j[log(sigma_ref,j^2/sigma_phi,j^2) + (sigma_phi,j^2 + (mu_phi,j-mu_ref,j)^2)/sigma_ref,j^2 - 1]`

该项防止整个 encoder 训练时 `logvar` 和 latent 分布漂移。`q_ref` 来自冻结的初始 VAE。

### 3.3 RL behavior-policy KL

`KL_policy = KL(pi_phi(z|x) || pi_ref(z|x))`

候选和 reference 使用相同的固定 `sigma_expl`，因此：

`KL_policy = 0.5 * sum_j[(m_phi,j-m_ref,j)^2 / sigma_expl,j^2]`

该项是 GRPO 的 trust-region anchor，只约束行为策略均值。训练和 W&B 同时记录总 KL 与每 latent element 的平均 KL，避免 4x32x32 的维度使数值不可比较。

## 4. 整个 VAE 的梯度路由

单独使用 `-A log pi(z|x)` 只能更新产生 `m_phi` 的 encoder。整个 VAE 仍然训练，但 reward 与重建必须使用严格分离的两条梯度路径：PickScore 只能进入 encoder 的 GRPO，decoder 只能接收重建相关损失。

### 4.1 Encoder mean 的 GRPO 路径

每张输入图像采样 `G` 个行为 latent，计算：

`A_i = stopgrad((r_i - mean_group(r)) / (std_group(r) + 1e-6))`

`L_grpo = -mean_i[A_i * sum_j(log pi_phi(z_i.detach()|x)_j)]`

这里对 action/latent 维度必须求和，因为它们共同构成一个联合高斯动作；只在 group 和 source-image batch 上取均值。为便于不同 latent 分辨率之间比较，日志可额外记录除以 latent 维数后的 per-element 数值，但优化目标不得用 per-element 均值替代联合 log-prob。

默认每批 rollout 只做一次 on-policy 更新，不复用旧 rollout，不使用 PPO ratio。联合高斯有 4096 个维度，多 epoch 的联合 importance ratio 很容易溢出或被 clip 饱和；只有单步版本稳定后才增加 `pi_old` 与 clipped ratio 实验。

RL loss 只允许更新 encoder trunk 和 `quant_conv` 的 mean 输出。`sigma_expl` 固定，VAE `logvar` 不接收 RL 梯度。

### 4.2 完整 VAE 的重建路径

使用 VAE 自身 posterior 的重参数化样本，而不是行为策略样本：

`u_post ~ q_phi(u|x)`

`x_rec = D_psi(u_post)`

`L_vae = lambda_l1 L1(x_rec,x) + lambda_perc LPIPS(x_rec,x) + lambda_prior KL_prior + lambda_post KL_post_ref`

该路径训练 encoder、mean/logvar 输出、`quant_conv`、`post_quant_conv` 和全部 decoder block。可在重建稳定后加入原始 VAE 风格的 PatchGAN loss，但 GAN 不作为首个 smoke run 的必要条件。

### 4.3 Decoder 的重建约束

禁止把 PickScore 的可微图像梯度传入 decoder。行为 latent 在进入 decoder 前与 encoder 计算图断开，PickScore 结果随后再次 detach，仅用于构造 GRPO advantage。

Decoder 只接收 L1、LPIPS、行为重建 anchor，以及可选的 PatchGAN/feature-matching 等重建分布约束。若使用冻结 SiT-XL/2 的 latent replay pool，也只能用 reference reconstruction、LPIPS 或其他非 reward 保真目标约束 decoder。

最终目标为：

`L_total = lambda_grpo L_grpo + beta_policy KL_policy + L_vae + L_decoder_reconstruction`

所有模块均参与训练，但不同损失的梯度边界必须明确：

| 模块 | GRPO PickScore | VAE 重建/先验 KL | Posterior ref KL | Decoder 重建约束 |
| --- | --- | --- | --- | --- |
| encoder trunk | 是 | 是 | 是 | 否 |
| mean head | 是 | 是 | 是 | 否 |
| logvar head | 否 | 是 | 是 | 否 |
| post-quant conv | 否 | 是 | 否 | 是 |
| decoder 全部参数 | 否 | 是 | 否 | 是 |

## 5. Reward 与约束

组内 advantage 使用 shaped reward：

`r_i = PickScore(c,x_hat_i) - alpha_l1 * hinge(L1_i,tau_l1) - alpha_lpips * hinge(LPIPS_i,tau_lpips)`

其中 `hinge(v,tau)=max(0,v-tau)`，阈值由初始 VAE 在固定验证集上的 p90 指标确定。训练必须同时记录未经 shaping 的原始 PickScore；最终结论只能基于原始 held-out PickScore，不能用 shaped reward 代替。

为了减少 reward hacking：

- PickScore 仅作为训练 reward 和主目标。
- ImageReward、CLIPScore、AES、MSE、PSNR、SSIM、LPIPS、DISTS、FID、precision/recall 作为独立评测。
- checkpoint 不能按训练 reward 单独挑选，必须先通过重建和 latent compatibility gate。
- reward model 使用 `eval()` 且参数 `requires_grad=False`；其输出在 advantage 中 detach，禁止 PickScore 图像梯度进入 decoder。

## 6. 数据与固定评测集合

- 训练集：ImageNet-1K train，256x256 crop，类别名转为固定英文 prompt。
- 重建验证：固定 10,000 张 ImageNet validation 图像，保存 index manifest。
- 冻结 SiT 验证：固定 class id、初始噪声、ODE/SDE 设置、CFG scale 和 seed；candidate 与 base 解码完全相同的 SiT latent。
- Fresh 验证：使用训练和 checkpoint 选择阶段从未出现的新 seed，防止对固定 latent replay 过拟合。
- 所有 latent pool 保存生成配置、VAE/SiT checkpoint hash、数据 index、prompt、seed 和缩放系数。

## 7. 分阶段实验

### Phase 0：基线复现与噪声标定

在不训练参数的情况下完成：

1. 复现初始 VAE 的 deterministic mean reconstruction、native posterior reconstruction 和 SiT-XL/2 生成结果。
2. 对完全相同的图像重复运行 PickScore，测量 FP32/BF16、不同 batch shape 下的 scorer 数值噪声。
3. 统计每通道 `m_ref` 的 mean/std、真实 posterior sigma、`logvar` clamp 命中率和 `sigma_post/std(m_ref)`。
4. 在 1,024 张固定图像上，对 `rho=0.05/0.10/0.20/0.30` 各采样 `G=8`，记录组内 PickScore std、top-bottom reward gap、pairwise LPIPS、L1/MSE 和 action norm。

选择最小的、同时满足以下条件的 `rho`：

- 组内 reward std 至少是 scorer 重复计算噪声 std 的 3 倍。
- 至少 90% 的 group 有非零 reward 排序。
- p90 LPIPS 和 MSE 不超过初始 VAE 验证分布的 1.25 倍。
- 无 NaN、decoder 饱和或明显离开自然图像流形的样本。

若 `rho=0.30` 仍没有有效 reward spread，则停止 GRPO 主实验，不能通过提高学习率掩盖探索失败。

### Phase 1：梯度路由 smoke test

使用 32 张图像、`G=4`、10 个 optimizer step，逐项反向并断言：

- `L_grpo`：encoder mean 路径梯度非零；logvar head、post-quant conv 和 decoder 梯度严格为零。
- PickScore reward 单独反向时：encoder mean 路径梯度非零；decoder 梯度严格为零。
- `L_vae`：encoder、mean/logvar、post-quant conv 和 decoder 梯度均非零。
- `KL_policy` 在 candidate 等于 reference 时小于 `1e-8`，扰动 mean 后与手算闭式结果一致。
- `KL_post_ref` 在 candidate 等于 reference 时小于 `1e-8`。
- 同一 group 必须位于同一 GPU；advantage 只能在该 group 内归一化，不能跨输入图像或跨 rank 混合。

随后进行 100-step smoke training，检查 reward、KL、重建指标、梯度范数和显存是否稳定。

### Phase 2：1,000-step 对照与消融

固定数据顺序、增强、seed、batch 和总 decode 数，比较：

| 实验 | 探索 | Encoder GRPO | Decoder 重建 | KL anchors | 目的 |
| --- | --- | --- | --- | --- | --- |
| A | 无 | 无 | 无 | prior | 完整 VAE 重建训练 control |
| B | native posterior | 是 | 是 | 全部 | 验证尖锐 posterior 的失败假设 |
| C | decoupled | 是 | 是 | 全部 | whole-VAE 主实验，reward 仅更新 encoder |
| D | decoupled | 是 | 是 | 无 policy KL | 验证 trust region 必要性，仅作消融 |
| E | decoupled | 无 | 是 | posterior ref | 纯重建 control |

主实验 C 使用 3 个 seed。建议初始配置：

- 每卡 source batch `1`，group size `G=8`；显存不足时降为 `G=4` 并保持所有实验一致。
- gradient accumulation `4-8`，BF16 VAE/PickScore forward，log-prob、advantage、KL 和 reward reduction 强制 FP32。
- encoder trunk LR `1e-6`，mean head LR `3e-6`，logvar head LR `1e-7`，decoder LR `1e-6`。
- AdamW，weight decay `0.01`，100-step warmup，global grad norm clip `1.0`。
- VAE EMA `0.9999`，每 100 step 保存并固定评测；训练 1,000 step 后通过 gate 才扩展到 10,000 step。
- `beta_policy` 用 target-KL controller 调整，扫描每-element target `1e-3 / 3e-3 / 1e-2`。
- 原始 VAE prior KL 默认 `1e-6`，不得随 `beta_policy` 联动。

### Phase 3：冻结 SiT 直接替换

冻结 SiT-XL/2，对同一批 SiT latent 分别使用 `D_ref` 和候选 `D_psi` 解码。该实验只验证 decoder 是否改善且仍兼容原 SiT，不能证明 encoder latent space 更好。

评测两套集合：

- 固定 latent：每类固定 seed，用于 checkpoint 曲线与 paired bootstrap。
- fresh latent：全新 seed，只在候选确定后运行一次。

报告 PickScore、ImageReward、CLIPScore、AES、FID-50K、IS、precision/recall，以及逐样本 base/candidate 对照图。

### Phase 4：验证 encoder latent space

“更好的 latent space”必须通过重新学习上层生成模型验证，因为冻结 SiT 的直接替换只经过 decoder，不使用候选 encoder。

1. 用相同 ImageNet 图像分别缓存 base VAE 和 candidate VAE 的 deterministic mean latent，保持相同 scaling、数据顺序和增强。
2. 从相同随机初始化分别训练两个 SiT-B/2 pilot，使用完全相同的 optimizer、step、batch 和随机种子。
3. 比较 held-out interpolant/velocity loss、达到同一 FID 所需 step、固定算力下的 FID/PickScore，以及 latent class linear probe。
4. pilot 至少两个 seed 一致后，再进行 SiT-XL/2 matched-compute 实验；资源不足时可从同一官方 SiT checkpoint 做 base/candidate 双分支适配，但必须明确这是 continuation，不是 from-scratch 结论。

只有 candidate latent 在 matched-compute SiT 训练中表现更好，才能声称 encoder latent space 改善。

## 8. 通过门槛

候选 checkpoint 必须同时满足：

- 固定重建集 raw PickScore 的 paired bootstrap 95% CI 下界大于 0，提升至少为 reward 数值噪声的 3 倍。
- MSE 和 LPIPS 相对初始 VAE 退化不超过 1%，PSNR 下降不超过 0.1 dB；任何一项超限即失败。
- 每通道 latent mean/std 相对变化不超过 2%，`logvar` clamp 命中率不增加超过 1 个百分点。
- `KL_policy` 不超过预设 target 的 2 倍，且没有持续单向增长。
- 冻结 SiT fresh latent 上 PickScore 的 paired 95% CI 下界大于 0，FID-50K 不退化超过 2%。
- 三 seed 中至少两个通过全部门槛，aggregate 结果通过；不得只报告最好 seed。
- “latent space 改善”结论额外要求 Phase 4 matched-compute SiT pilot 的收敛速度或最终生成指标显著优于 base latent。

## 9. 必须记录与审计的内容

W&B entity 使用 `yukezhao37-zhejiang-university`，API key 仅通过环境变量传入：

```bash
export WANDB_API_KEY=...
```

每个 run 必须记录：

- raw/shaped reward、group reward std、advantage mean/std、top-bottom gap。
- `sigma_post`、`sigma_expl`、每通道 latent mean/std、logvar clamp rate。
- 三类 KL 的 total 与 per-element 值。
- 每个损失对各模块的梯度范数，以及全局 grad norm。
- MSE、L1、PSNR、SSIM、LPIPS、DISTS 和固定样本图。
- 配置全文、Git commit/diff、依赖版本、模型 hash、数据 manifest 和所有随机种子。
- base/candidate 的相同 latent、相同 prompt 并排解码结果。

实验用 tmux 启动。大型数据集、模型缓存、latent pool、checkpoint 和评测产物保存到：

```text
/mnt/data/zyk
```

资源访问较慢时使用：

```bash
export http_proxy=http://127.0.0.1:8848
export https_proxy=http://127.0.0.1:8848
```
