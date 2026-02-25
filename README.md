# Latent StarGAN v2 (Multi-domain Style Transfer in SD1.5 VAE Latent Space)

本实现面向 `B x 4 x 32 x 32` 的 SD1.5 VAE latent tensor（**原始 unscaled latent**，没有乘 `0.18215`），基于 StarGAN v2 思路实现多域风格迁移。

## 文件说明

### `model.py`
实现核心网络：
- `AdaIN`：`InstanceNorm2d(affine=False)` + `Linear(style_dim -> 2C)` 注入风格；`fc` 权重/偏置零初始化，前向为 `(1+gamma)*norm(x)+beta`，确保初始为恒等映射。
- `AdaINResBlock`：两次 `3x3 Conv`，每次卷积前做 AdaIN + LeakyReLU。
- `Generator`：不做空间下采样，编码到高通道后串联 `n_res_blocks`，再解码回 4 通道。
- `MappingNetwork`：共享 MLP + `N` 个域分支，按 `y` 选择目标域风格向量。
- `StyleEncoder`：轻量卷积主干 + GAP + `N` 个域分支，按 `y` 提取风格向量。
- `Discriminator`：PatchGAN，多分支输出 `[B, N, H', W']`，`forward(x, y)` 内部 `gather` 出对应域通道。

**主要输入输出**
- `Generator(x, s)`：`x [B,4,H,W]` + `s [B,style_dim]` -> `x_fake [B,4,H,W]`
- `MappingNetwork(z, y)`：`z [B,latent_dim]`, `y [B]` -> `s [B,style_dim]`
- `StyleEncoder(x_ref, y)`：`x_ref [B,4,H,W]`, `y [B]` -> `s [B,style_dim]`
- `Discriminator(x, y)`：`x [B,4,H,W]`, `y [B]` -> `logits [B,1,H',W']`

### `dataset.py`
实现多域 latent 读取与两阶段均匀采样：
- 支持 `.pt` / `.npy`。
- 初始化时根据显存可用量判断是否预加载到 GPU（`preload_to_gpu=true` 时优先尝试）。
- `set_epoch(epoch)` 预生成整轮采样索引：
  1) 等概率采样源域；
  2) 等概率采样目标域（与源域不同）；
  3) 分别在对应域内等概率采样文件索引。
- `__getitem__` 返回：
  - `content`
  - `target_style`
  - `target_style_id`
  - `source_style_id`

### `loss.py`
纯函数损失，不做模型前向：
- `d_hinge_loss(real_logits, fake_logits)`
- `g_adversarial_loss(fake_logits)`
- `style_reconstruction_loss(pred_style, target_style)`
- `diversity_sensitive_loss(fake_1, fake_2)`
- `cycle_consistency_loss(reconstructed, original)`
- `r1_penalty(real_logits, real_inputs)`

### `trainer.py`
训练引擎：
- 每个 batch 同时跑两路：
  - Reference-guided (`s_ref = E(x_ref, y_trg)`)
  - Latent-guided (`s_lat = F(z, y_trg)`)
- 判别器与生成器分别更新。
- 判别器与生成器分别更新；判别器更新包含 R1 梯度惩罚（`loss.w_r1`）。
- 使用 `torch.amp.autocast`（支持 `bf16` / `fp16`）。
- `RuntimeError` 中包含 OOM 时，自动 `empty_cache()` 并跳过当前 batch。
- 日志全部 `.detach().cpu().item()` 后输出，防止图引用导致显存增长。
- 按 `save_interval` 保存 checkpoint。
- 训练输入 latent 会先乘 `training.latent_scale`（默认 0.18215）进入网络；可视化解码时会除回原始 latent 尺度。
- `evaluate(epoch)` 会解码并保存网格图到 `visualization.save_dir`（若未配置则默认保存到 checkpoint 目录下的 `vis/`）。

### `run.py`
实验入口：
- `argparse` 读取 YAML/JSON 配置。
- 全局设置：`PYTORCH_ALLOC_CONF=expandable_segments:True`。
- 构建 dataset/dataloader、models、trainer，并循环调用 `train_epoch()`。

### `configs/stargan_v2_latent.yaml`
包含可调参数：模型维度、损失权重（含 `w_r1`）、训练超参（含 `lr_mapping`、`latent_scale`、`eval_interval`）、域列表、预加载开关、checkpoint 路径、VAE 可视化配置（含 `visualization.save_dir`）等。

## 数据组织

默认配置下，目录结构应为：

```text
../../latent-256/
  photo/
    *.pt or *.npy
  Hayao/
    *.pt or *.npy
  monet/
    *.pt or *.npy
  cezanne/
    *.pt or *.npy
  vangogh/
    *.pt or *.npy
```

每个文件应可解析为 `[4, 32, 32]`（或 `[1,4,32,32]`，会自动 squeeze）。

## 运行方式

在 `latent_cyclegan` 目录下：

```bash
python run.py --config configs/stargan_v2_latent.yaml
```

如果使用 JSON 配置：

```bash
python run.py --config path/to/your_config.json
```

## 备注

- 数据按 **unscaled latent** 读取；为训练稳定性，进入网络前会按 `training.latent_scale` 做缩放。
- 若数据量过大且显存不足，`dataset.py` 会自动回退到 CPU 存储，避免初始化阶段 OOM。
