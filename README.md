# Latent CycleGAN (SD1.5 VAE Latents)

这个项目实现了一个“在 SD1.5 VAE 潜空间（latent）上做 CycleGAN 风格迁移”的训练脚本。

与经典 CycleGAN 的区别：
- 输入不是 RGB 图片，而是通过 SD1.5 的 VAE 压缩得到的 latent（`.pt` 文件）。
- 网络结构仍遵循 CycleGAN：两个生成器 `G: A→B`、`F: B→A`，两个判别器 `D_A`、`D_B`（PatchGAN）。
- 使用 **LSGAN**（MSE）作为对抗损失，使训练更稳定。

> 重要：本代码支持你的设定——latent `.pt` **没有进行 0.18215 缩放**。该信息通过配置项 `data.latents_scaled` 控制，默认 `false`。

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
- 对每张图输出 3 类指标到 CSV，并写一个 summary.json

### 测试集目录约定

默认读取：
- `dataset/testA`（域 A 的测试图片）
- `dataset/testB`（域 B 的测试图片）

你也可以用参数覆盖。

### 运行示例

```bash
python eval_latent_cyclegan.py --config configs/example.yaml
```

你也可以用命令行参数覆盖 config（例如临时换一个 checkpoint）：

```bash
python eval_latent_cyclegan.py --config configs/example.yaml --checkpoint outputs/model/epoch_0200.pt
```

输出位置：
- `outputs/eval/epoch_0200/metrics.csv`：逐图指标
- `outputs/eval/epoch_0200/summary.json`：按方向的平均指标

提示：CSV 不能“直接显示图片”，但会记录 `src_path` / `gen_path`，你可以点开这些路径查看生成结果。

生成图像（可选）：
- `outputs/eval/epoch_0200/images/A2B/*.png`
- `outputs/eval/epoch_0200/images/B2A/*.png`

依赖说明：
- LPIPS 需要额外安装：`pip install lpips`
- CLIP 需要 `transformers`（已在 requirements 中）
