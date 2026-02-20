# Latent CycleGAN (SD1.5 VAE Latents)

这个项目实现了一个“在 SD1.5 VAE 潜空间（latent）上做 CycleGAN 风格迁移”的训练脚本。

与经典 CycleGAN 的区别：
- 输入不是 RGB 图片，而是通过 SD1.5 的 VAE 压缩得到的 latent（`.pt` 文件）。
- 网络结构仍遵循 CycleGAN：两个生成器 `G: A→B`、`F: B→A`，两个判别器 `D_A`、`D_B`（PatchGAN）。
- 使用 **LSGAN**（MSE）作为对抗损失，使训练更稳定。

模型实现细节（本仓库当前版本）：
- 生成器：轻量 ResNet（带 InstanceNorm，保持经典 CycleGAN 风格）。
- 判别器：PatchGAN，并在卷积层上使用 **谱归一化（Spectral Normalization）** 来约束 Lipschitz（替代原本的 InstanceNorm）。

> 重要：本代码支持你的设定——latent `.pt` **没有进行 0.18215 缩放**。该信息通过配置项 `data.latents_scaled` 控制，默认 `false`。

---

## 整体流程（Train / Eval 在做什么）

这个仓库有两条主流程：

1) **Train（训练）**：
- **输入**：两域（A/B）无配对的 latent 数据（`.pt`），每个文件是一个 VAE latent（形状 `[4,H,W]`）。
- **做的事**：在 latent 空间训练 CycleGAN（`G: A→B`、`F: B→A`，以及 `D_A/D_B`）。
- **输出**：checkpoint（包含模型与优化器等状态）、训练日志（JSONL）、可视化图片（把 latent 解码成 RGB 便于肉眼观察）。

2) **Eval（评测）**：
- **输入**：
  - checkpoint（从中读取 `G/F` 权重以及训练配置 `cfg`）
  - 两域的 **RGB 测试图片目录**（`eval.testA_dir`、`eval.testB_dir`）
  - 一个可用的 SD VAE（用于 RGB⇄latent）
- **做的事**：
  - RGB 测试图 → VAE 编码成 latent
  - latent 上跑 `G/F` 得到“生成 latent”
  - 生成 latent → VAE 解码回 RGB
  - 计算指标（LPIPS/CLIP），并可选保存对比图
- **输出**：metrics.csv（逐图）、summary.json（均值汇总）、可选生成图像，以及 CLIP cache（加速用）。

---

## 代码结构

- `train_latent_cyclegan.py`
  - 读取 YAML 配置
  - 加载 A/B 两域 latent `.pt`
  - 构建轻量版 CycleGAN（ResNet 生成器 + PatchGAN 判别器）
  - 训练循环：固定宽度进度条 + 每个 epoch 写 JSONL 日志
  - 每隔 N 个 epoch（默认 5）进行可视化（VAE 解码 latent→RGB，保存对比图）
- `configs/example.yaml`
  - 全量示例配置（数据路径、训练超参、日志与可视化输出位置等）
- `requirements.txt`
  - 依赖列表

---

## 数据格式（输入）

你需要准备两套“无配对”的 latent 数据：

- `data.a_dir`：域 A 的 `.pt` 文件目录（可递归扫描子目录）
- `data.b_dir`：域 B 的 `.pt` 文件目录（可递归扫描子目录）

每个 `.pt` 文件支持以下两种内容：

1) 直接保存 `torch.Tensor`
- 形状为 `[4, H, W]` 或 `[1, 4, H, W]`

2) 保存为 `dict`
- 至少包含 `{"latent": <torch.Tensor>}`，其中 tensor 形状同上

训练时默认使用“无配对”采样：A 域与 B 域分别 shuffle，各取一个 batch，符合 CycleGAN 的无监督设定。

---

## 损失函数

- 对抗损失：LSGAN（MSE）
  - 生成器希望 `D( fake ) → 1`
  - 判别器希望 `D( real ) → 1` 且 `D( fake ) → 0`
- Cycle Consistency：L1
  - `A → G(A) → F(G(A)) ≈ A`
  - `B → F(B) → G(F(B)) ≈ B`

总损失为：

\[
L = L_{gan}(G) + L_{gan}(F) + \lambda_{cyc} L_{cyc}
\]

其中 `lambda_cyc` 从配置读取（默认 10）。

---

## 安装依赖

建议使用你当前的 conda 环境（例如 `aivenv`）：

```bash
pip install -r requirements.txt
```

---

## 配置说明

训练完全由 YAML 配置驱动。你可以从 `configs/example.yaml` 复制并修改。

关键字段：

- `data.a_dir` / `data.b_dir`
  - 两个域的 `.pt` 数据目录
- `data.latents_scaled`
  - `false`：表示你的 latent **没有**做 0.18215 缩放（你的场景默认如此）
  - `true`：表示 latent 已被缩放，可视化解码时会先除以 `vae_scaling_factor`
- `data.latent_divisor`
  - 训练时会先做：`model_latent = raw_latent / latent_divisor`
  - 可视化解码前会做：`raw_latent = model_latent * latent_divisor`
  - 用途：让进入网络（尤其是 `tanh` 输出/输入范围）更接近 ±1 的数值尺度
- `train.*`
  - `epochs`、`batch_size`、`lr`、`amp`、`lambda_cyc` 等
- `logging.log_dir` / `logging.log_file`
  - JSONL 日志输出位置
- `visualization.*`
  - `every_epochs`：每隔多少个 epoch 可视化一次（默认 5）
  - `out_dir`：可视化图片输出目录
  - `vae_model_name_or_path`：用于解码的 VAE（来自 diffusers）
  - `vae_subfolder`：如果是 `runwayml/stable-diffusion-v1-5`，通常为 `vae`

---

## 运行训练

```bash
python train_latent_cyclegan.py --config configs/example.yaml
```

### Train 的输入是什么

- `config.data.a_dir`：域 A 的 latent `.pt` 目录（递归扫描）
- `config.data.b_dir`：域 B 的 latent `.pt` 目录（递归扫描）
- 每个 `.pt` 文件可以是：
  - 直接保存的 `torch.Tensor`（`[4,H,W]` 或 `[1,4,H,W]`）
  - 或保存为 `dict` 且包含 `{"latent": Tensor}`

此外，训练会用到两个数值尺度相关配置：
- `data.latents_scaled`：是否使用 diffusers 约定的 0.18215 缩放（你的数据通常是 `false`）
- `data.latent_divisor`：训练前把 raw latent 除以该值，让进入网络的数值尺度更“温和”（例如默认 10.0）

### Train 训练时做了什么

训练在 latent 空间实现标准 CycleGAN：

1) **采样方式（无配对）**
- A/B 两个 DataLoader 各自 shuffle
- 每个 step 取一个 A batch + 一个 B batch（它们不是同一张图的配对）

2) **前向路径（在 latent 上）**
- `fake_b = G(real_a)`
- `fake_a = F(real_b)`
- `rec_a = F(fake_b)`（cycle）
- `rec_b = G(fake_a)`（cycle）
- `id_b = G(real_b)`、`id_a = F(real_a)`（identity，可通过 `lambda_id` 控制）

3) **损失**
- GAN：LSGAN（MSE），让判别器输出更稳定
- Cycle：L1（`rec_a≈real_a`，`rec_b≈real_b`）
- Identity：L1（可选）
- 判别器 loss 额外乘以 `d_loss_scale`（默认 0.5，常见实践）

4) **Replay Buffer（判别器假样本缓冲）**
- `train.fake_buffer_size > 0` 时启用
- 判别器训练时，部分假样本来自历史缓冲（减少模式震荡，属于经典 CycleGAN trick）

5) **AMP 与学习率调度**
- `train.amp: true` 且 CUDA 可用时启用 AMP（`GradScaler` + autocast）
- `lr_constant_epochs` + `lr_decay_epochs` 做线性衰减 schedule（按 epoch 调一次）

checkpoint 保存格式（当前版本）包含：
- `G` / `F`：两个生成器的 `state_dict`
- `D_A` / `D_B`：两个判别器的 `state_dict`
- `opt_G` / `opt_D`：优化器状态
- `sched_G` / `sched_D`：学习率调度器状态
- `scaler`：AMP scaler 状态
- `cfg`：完整训练配置（用于 eval/可视化复用）

### Train 的输出是什么

训练会写出三类主要产物：

1) **Checkpoints**（默认在 `outputs/model/`）
- `epoch_XXXX.pt`：每个 epoch 保存一次
- `last.pt`：每个 epoch 也会覆盖更新一份“最新 checkpoint”，方便断点续训

2) **日志**
- `outputs/logs/train_log.jsonl`：每个 epoch 一行 JSON 汇总

3) **可视化（可选）**
- `outputs/vis/epoch_XXXX.png`
- 训练脚本会（按 `visualization.every_epochs`）加载 VAE，把一小批固定 latent 解码成 RGB 并拼图，便于观察趋势

### Train 的断点续训怎么起作用

断点续训由以下配置控制（见 `train.*`）：

- `train.resume: true`：开启续训
- `train.resume_path`：可选，指定要从哪个 checkpoint 续训；为空则默认使用 `{checkpoint_dir}/last.pt`
- `train.resume_restore_rng: true`：是否恢复随机数状态（python/torch/cuda），开启后更接近“完全可复现地接着跑”

续训时脚本会从 checkpoint 恢复：
- 模型参数：`G/F/D_A/D_B`
- 优化器状态：`opt_G/opt_D`（包括动量等）
- 学习率调度器：`sched_G/sched_D`
- AMP scaler：`scaler`
- （可选）随机数状态：`rng`

恢复完成后：
- `start_epoch = loaded_epoch + 1`，从下一个 epoch 继续训练
- 因为包含优化器/调度器状态，学习率、动量等都会从“中断前”的状态继续

训练过程中：
- 每个 epoch 会打印 `Epoch x/y`
- epoch 内会显示固定宽度进度条，并在右侧实时更新 loss（`G/D/cyc`）

---

## 输出（日志 / 可视化）

### 1) JSONL 训练日志

默认输出到：
- `outputs/logs/train_log.jsonl`

每一行是一条 JSON，代表一个 epoch 的汇总信息，例如包含：
- `epoch`、`time_sec`
- `loss_G`、`loss_D`
- `loss_gan_G`、`loss_gan_F`
- `loss_cyc`
- `loss_D_A`、`loss_D_B`

### 2) 可视化图片

默认每 5 个 epoch 生成一张对比图，输出到：
- `outputs/vis/epoch_XXXX.png`

图像网格布局为（每个样本一行，4 列）：
- `A`、`G(A)`、`B`、`F(B)`

可视化仅用于观察训练趋势，解码依赖 diffusers 加载的 VAE。

---

## 常见问题

1) 运行时报缺少 `pyyaml` / `diffusers` / `torchvision`
- 重新执行 `pip install -r requirements.txt`

2) 可视化阶段提示 `vae_model_name_or_path not set`
- 在配置的 `visualization.vae_model_name_or_path` 填写可用的 VAE 或 SD1.5 模型路径/名称

3) latent 形状不匹配
- 需要是 `[4,H,W]` 或 `[1,4,H,W]`（或 dict 里 `latent` 满足该形状）

4) 关于 0.18215 缩放
- 如果你的 latent 文件“没有缩放”（你的设定），保持 `data.latents_scaled: false`
- 如果你确认 latent 经过了缩放，再改为 `true` 并确保 `visualization.vae_scaling_factor` 正确

---

## 评测（LPIPS / CLIP，A→B 与 B→A）

仓库提供了一个独立评测脚本：
- `eval_latent_cyclegan.py`

它会：
- 从 checkpoint 加载生成器 `G` / `F`
- 用 SD1.5 VAE 将测试图片编码到 latent
- 做 A→B（用 `G`）和 B→A（用 `F`）生成
- 输出逐图指标到 CSV，并写一个 summary.json（按方向汇总）

### Eval 的输入是什么

- checkpoint：
  - 读取 `G/F` 权重
  - 读取训练时保存的 `cfg`（尤其是 `data.latents_scaled`、`data.latent_divisor`、`visualization.vae_*`）
- 两域 RGB 测试图片目录：`eval.testA_dir` 与 `eval.testB_dir`
- 一个可用的 SD VAE：用于 `RGB -> latent -> RGB`

注意：Eval 用的是“图片作为输入”，不是 `.pt` latent 数据。原因是评测需要落在可感知的 RGB 空间（LPIPS/CLIP 都是图像指标），因此必须进行 VAE 的 encode/decode。

### Eval 评测时做了什么（逐张图）

以 A→B 为例（B→A 同理）：

1) **读入并预处理 RGB**
- 从 `testA_dir` 读入图片，resize 到 `eval.image_size`，并归一化到 `[-1,1]`

2) **RGB 编码到 latent**
- `latent = VAE.encode(image).latent_dist.sample()`
- 如果 `latents_scaled: true`，会乘 `vae_scaling_factor` 以对齐训练时 latent 的尺度约定
- 如果 `latent_divisor != 1`：
  - 输入生成器前：`lat_in = latent / latent_divisor`
  - 生成器输出后：`lat_out = gen(lat_in) * latent_divisor`

3) **latent 解码回 RGB**
- 如果 `latents_scaled: true`，解码前会除以 `vae_scaling_factor`
- `img_gen = VAE.decode(lat_out)`，并把范围映射回 `[0,1]`

4) **计算指标（content + FID）**
- content 指标：生成图与源图之间
  - `content_lpips`：LPIPS( gen, src )（越小越接近）
  - `content_clip`：CLIP cosine( gen, src )（越大越接近）

- 风格指标：**FID（clean-fid）**
  - FID 是“集合级”指标：
    - A→B：比较“生成 B 集合”与“真实 B 测试集”的 Inception 特征分布距离
    - B→A：比较“生成 A 集合”与“真实 A 测试集”的 Inception 特征分布距离
  - 需要先对参考测试集预计算 `fid_stats.npz`（见下文）。

5) **可选保存对比图**
- `eval.save_images: true` 时，会保存“源图 | 生成图”的拼图，并附上指标表格

拼图的指标表格为 **三行一列**：
- `content_lpips`
- `content_clip`
- `fid`

### 测试集目录约定

默认读取（见 `configs/example.yaml` 的 `eval.testA_dir/testB_dir`）：
- `datasets/testA`（域 A 的测试图片）
- `datasets/testB`（域 B 的测试图片）

也可以用命令行参数覆盖（更适合临时切换数据集）。

### 运行示例

```bash
python eval_latent_cyclegan.py --config configs/example.yaml
```

你也可以用命令行参数覆盖 config（例如临时换一个 checkpoint）：

```bash
python eval_latent_cyclegan.py --config configs/example.yaml --checkpoint outputs/model/epoch_0050.pt
```

或者覆盖测试集目录：

```bash
python eval_latent_cyclegan.py --config configs/example.yaml --testA datasets/testA --testB datasets/testB
```

### FID：参考集预计算（clean-fid）

FID 需要对参考集（通常是测试集目录）预先计算均值/协方差（$
\mu,\Sigma$）。本仓库提供脚本：

- `precompute_fid_vectors.py`

对每个测试集目录各跑一次：

```bash
python precompute_fid_vectors.py datasets/testA
python precompute_fid_vectors.py datasets/testB
```

输出：
- `datasets/testA/vectors/`：每张图的 Inception feature（文件名与图片尽量对应，后缀为 `.npy`）
- `datasets/testA/fid_stats.npz`：参考集统计（mu/sigma/count/dim）

（`testB` 同理）

### 单张图片推理 + 自动算分（A→B 与 B→A）

如果你想对“指定的一张图片”直接跑推理，并自动计算与 `eval_latent_cyclegan.py` 一致的两个指标：
`content_lpips / content_clip`（每个方向一套），并可选计算 `fid`（需要参考集 stats），可以使用：

- `eval_single_image.py`

最简单的用法（给 1 张图，同时跑 A→B 与 B→A）：

```bash
python eval_single_image.py --config configs/example.yaml --image datasets/testA/00010.jpg
```

如果希望同时计算 FID（需要你先跑过上面的 `precompute_fid_vectors.py`），给出参考集路径：

```bash
python eval_single_image.py --config configs/example.yaml \
  --image datasets/testA/00010.jpg \
  --testA datasets/testA --testB datasets/testB
```

如果你希望更严格地按域来跑（A→B 用 A 图，B→A 用 B 图），分别传入两张：

```bash
python eval_single_image.py --config configs/example.yaml \
  --imageA datasets/testA/00010.jpg \
  --imageB "datasets/testB/2014-08-01 17_41_55.jpg"
```

输出默认写到：
- `outputs/eval_single/{checkpoint_stem}/{image_tag}/metrics.csv`
- `outputs/eval_single/{checkpoint_stem}/{image_tag}/summary.json`

同时会保存：
- `.../gen/`：纯生成图（用于 FID 计算）
- `.../fid/`：生成集的 mu/sigma
- `.../images/`：带指标表格的“源图 | 生成图”拼图

### 关于 VAE（评测必需）

`eval_latent_cyclegan.py` 会把 RGB 测试图编码到 SD VAE latent，再在 latent 上跑 `G/F`，最后再解码回 RGB。
因此必须能拿到 VAE 的 `model_name_or_path`：
- 推荐直接在训练配置里设置 `visualization.vae_model_name_or_path` / `visualization.vae_subfolder`（checkpoint 会保存到 `cfg`，eval 会自动读取）
- 或在 eval 时用参数显式指定：

```bash
python eval_latent_cyclegan.py --config configs/example.yaml \
  --vae_model runwayml/stable-diffusion-v1-5 --vae_subfolder vae
```

### 关于 style 指标（已移除）

当前版本已移除：
- `style_lpips`
- `style_clip`

评测脚本只保留 content 指标（`content_lpips/content_clip`），并新增 FID 作为“风格分布相似度”的集合级度量。

输出位置：
- `outputs/eval/{checkpoint_stem}/metrics.csv`：逐图指标
- `outputs/eval/{checkpoint_stem}/summary.json`：按方向的平均指标

提示：CSV 不能“直接显示图片”，但会记录 `src_path` / `gen_path`，你可以点开这些路径查看生成结果。

生成图像：
- 纯生成图（用于 FID）：
  - `outputs/eval/{checkpoint_stem}/gen/A2B/*.png`
  - `outputs/eval/{checkpoint_stem}/gen/B2A/*.png`
- 拼图（可选，带三行指标表格）：
  - `outputs/eval/{checkpoint_stem}/images/A2B/*.png`
  - `outputs/eval/{checkpoint_stem}/images/B2A/*.png`

FID 输出：
- `outputs/eval/{checkpoint_stem}/fid/A2B_gen_stats.npz`
- `outputs/eval/{checkpoint_stem}/fid/B2A_gen_stats.npz`

依赖说明：
- LPIPS 需要额外安装：`pip install lpips`
- CLIP 需要 `transformers`（已在 requirements 中）
- FID 需要 `clean-fid`（已在 requirements 中）

---

## 变更摘要（最近几次迭代）

- 指标调整：删除 `style_lpips/style_clip`，仅保留 `content_lpips/content_clip`。
- 新增 FID：使用 clean-fid。
  - 增加参考集预计算脚本 `precompute_fid_vectors.py`（保存 per-image vectors + `fid_stats.npz`）。
  - eval 时对生成图集合计算 mu/sigma 并输出 FID，同时把指标表格改为三行一列（content_lpips、content_clip、fid）。
- 判别器更新：PatchGAN 判别器用卷积层谱归一化替代 InstanceNorm（其余训练流程不变）。
