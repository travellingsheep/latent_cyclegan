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
- R1 使用 lazy regularization，默认每 `loss.r1_interval`（默认16）步计算一次。
- 使用 `torch.amp.autocast`（支持 `bf16` / `fp16`）。
- 终端训练显示支持三种模式：`progress_style=bar|percent|off`；`percent` 模式为单行百分比+loss 刷新，避免刷屏换行。
- `RuntimeError` 中包含 OOM 时，自动 `empty_cache()` 并跳过当前 batch。
- 日志全部 `.detach().cpu().item()` 后输出，防止图引用导致显存增长。
- 按 `save_interval` 保存 checkpoint。
- 训练输入 latent 会先乘 `training.latent_scale`（默认 0.18215）进入网络；可视化解码时会除回原始 latent 尺度。
- `evaluate(epoch)` 会解码并保存网格图到 `visualization.save_dir`（若未配置则默认保存到 checkpoint 目录下的 `vis/`），触发频率由 `visualization.every_epochs` 控制。
- 可视化网格会使用 `data.domains` 作为行/列标签（需要环境中有 Pillow；若缺失则仅保存无标签网格，不影响训练）。

### `run.py`
实验入口：
- `argparse` 读取 YAML/JSON 配置。
- 全局设置：`PYTORCH_ALLOC_CONF=expandable_segments:True`。
- 构建 dataset/dataloader、models、trainer，并循环调用 `train_epoch()`。

### `configs/stargan_v2_latent.yaml`
包含可调参数：模型维度、损失权重（含 `w_r1` 与 `r1_interval`）、训练超参（含 `lr_mapping`、`latent_scale`、`num_workers`、`pin_memory`、`use_compile`、`compile_mode`、`progress_style`、`display_interval`、`log_json`）、域列表、预加载开关、checkpoint 路径、VAE 可视化配置（含 `visualization.every_epochs`、`visualization.save_dir`、`suppress_hf_warnings`）等。

## 训练显示与间隔参数的语义

下面这些参数都在 `training:` 段里（以及少量历史兼容项），它们控制“终端输出频率 / 保存频率 / 可视化频率”：

- `display_interval`
  - **作用**：控制“实时刷新显示”（进度百分比与当前 loss）的刷新步频。
  - **实现逻辑**：在每个 epoch 内，训练循环里当 `step % display_interval == 0` 时，刷新一次显示。
  - **注意**：它只影响终端显示刷新，不影响训练、保存或可视化。
  - **补充：什么是 step？**
    - 这里的 `step` 指的是 **一个 batch 的迭代**（也就是 dataloader 里 `for step, batch in ...` 的那个 step）。
    - `display_interval: 100` 的意思是：每 100 个 batch，在终端刷新一次“当前进度 + loss”。
  - **补充：为什么你关了 `progress_bar` 但还想要 percent？**
    - `progress_bar` 只控制是否启用 tqdm 进度条（`progress_style: bar`）。
    - 如果你用的是 `progress_style: percent`，即使 `progress_bar: false`，也会按 `display_interval` 刷新单行百分比显示（不画进度条）。

- `log_json`
  - **作用**：是否打印结构化的 JSON 日志行（便于重定向到文件或后处理）。
  - **实现逻辑**：仅当 `log_json: true` 时，才会按 `log_interval` 打印一条 JSON。
  - **会打印什么内容？**
    - 当前实现打印的是一行 JSON（单行），包含本次 step 的关键标量（都已 `.detach().cpu().item()`）：
      - `epoch` / `step`
      - `d_loss` / `g_loss` / `g_adv`
      - `sty`（style reconstruction）/ `ds`（diversity-sensitive）/ `cyc`（cycle consistency）
      - `id`（identity loss，若启用 `loss.w_id`）
      - `r1`（lazy R1 的原始值；只有触发 R1 的 step 才非 0）
      - `lambda_ds`（当前 epoch 下 diversity weight 的衰减后系数）
    - 例子（字段可能会随你打开/关闭 loss 而略有变化）：
      - `{"d_loss": 0.82341, "g_loss": 1.23018, "g_adv": 0.91234, "sty": 0.10203, "ds": 0.33122, "cyc": 0.28811, "id": 0.0, "r1": 0.0, "epoch": 3, "step": 150, "lambda_ds": 0.85}`
    - 另外会附带 `time` 字段（本地时间戳，格式 `YYYY-mm-dd HH:MM:SS`），方便你回放日志定位。

- `log_interval`
  - **作用**：JSON 日志输出的步频。
  - **实现逻辑**：当 `log_json: true` 且 `step % log_interval == 0` 时，打印一行 JSON（包含 `d_loss/g_loss/.../r1` 等）。
  - **注意**：如果 `log_json: false`，这个参数不会触发任何输出。

### 每个 epoch 有多少 step？如何计算？

在本工程里：
- `step/total_steps` 的 `total_steps` 就是 `len(dataloader)`。
- `dataloader` 是由 `torch.utils.data.DataLoader(dataset, batch_size=..., drop_last=...)` 构成。

因此每个 epoch 的 step 数（batch 数）由下面这些决定：
- `len(dataset)`：这里等于 `data.epoch_size`（见 `dataset.py` 的 `LatentMultiDomainDataset.__len__`）。
  - 如果 `data.epoch_size: null`，会被自动设为 `max_count * num_domains`，其中 `max_count` 是所有域目录中样本数的最大值。
- `training.batch_size`
- `training.drop_last`

直观公式：
- 当 `drop_last: true`：$\text{steps\_per\_epoch} = \left\lfloor \frac{\text{epoch\_size}}{\text{batch\_size}} \right\rfloor$
- 当 `drop_last: false`：$\text{steps\_per\_epoch} = \left\lceil \frac{\text{epoch\_size}}{\text{batch\_size}} \right\rceil$

训练总 step 数大致是 $\text{num\_epochs} \times \text{steps\_per\_epoch}$（如果从 checkpoint resume，会从 `start_epoch` 接着跑）。

### 把运行期打印内容保存到文件

如果你希望把训练过程中终端打印的内容（包括 `[Info]/[Warn]`、epoch summary、`log_json` 行、tqdm/percent 进度输出等）单独保存到一个文件里，可以在配置里加：

```yaml
training:
  console_log_path: "outputs/stargan_v2_baseline/console.log"
```

说明：
- 该文件会以追加模式写入（append）。如果想重新开始一份干净日志，手动删除旧文件即可。
- 如果你使用 `progress_style: percent` 或 `progress_style: bar`，进度输出包含回车刷新（`\r`），写到文件里可能看起来不够“整齐”；这时更建议同时开启 `log_json: true`，用 JSON 行做离线分析。

## Loss-epoch 折线图

训练会在 `run.py` 里记录每个 epoch 的 loss 均值（就是 `[Epoch Summary]` 打印的那些 key），并用 matplotlib 画出 **loss vs epoch** 的多条折线（不同颜色代表不同 loss）。

配置项（都在 `training:` 里）：

```yaml
training:
  loss_plot_interval: 1
  loss_plot_path: "{outputs_dir}/loss_curve.png"
```

说明：
- `loss_plot_interval: 1` 表示每个 epoch 保存一次图。
- 默认会保存到 `{outputs_dir}/loss_curve.png`；你可以改成任意路径。

## 全局实验名称（自动生成输出路径）

很多时候你会把同一个字符串当作“实验名”，并希望它自动影响这些路径：
- `training.console_log_path`
- `visualization.save_dir`
- `checkpoint.save_dir`

本工程支持在配置顶部声明：

```yaml
experiment:
  name: "stargan_v2_baseline_id2"
  outputs_dir: "outputs/{exp_name}"   # 可选，默认就是 outputs/{exp_name}
```

然后在任意路径字符串里使用占位符：
- `{exp_name}`：实验名
- `{outputs_dir}`：输出根目录（默认 `outputs/{exp_name}`）

例子：

```yaml
training:
  console_log_path: "{outputs_dir}/console.log"

visualization:
  save_dir: "{outputs_dir}/vis"

checkpoint:
  save_dir: "{outputs_dir}/ckpt"
```

这样你只需要改 `experiment.name`，所有相关输出路径会在启动时自动展开。

- `save_interval`
  - **作用**：checkpoint 保存频率（以 epoch 为单位）。
  - **实现逻辑**：当 `epoch % save_interval == 0` 时保存一次（同时写 `epoch_xxxx.pt` 和 `latest.pt`）。

- `eval_interval`
  - **作用**：**历史兼容项**，用于在未配置 `visualization.every_epochs` 时，作为可视化频率的 fallback。
  - **实现逻辑**：实际可视化频率优先使用 `visualization.every_epochs`；如果 YAML 里没写 `visualization.every_epochs`，才会退回读取 `training.eval_interval`。
  - **建议**：新实验直接用 `visualization.every_epochs` 控制可视化频率，避免歧义。

## torch.compile 参数用法

这些参数都在 `training:` 段里，用于控制 PyTorch 2.x 的 `torch.compile` 行为。一般来说：**先保证稳定跑通，再逐步打开 compile**。

- `use_compile`
  - **作用**：总开关。`true` 时会尝试对模型做 `torch.compile`；失败会自动回退 eager 并打印 warning。

- `compile_mode`
  - **作用**：`torch.compile(..., mode=...)` 的模式。
  - **常用值**：`default` / `reduce-overhead` / `max-autotune`。
  - **建议**：训练优先用 `reduce-overhead`，更偏向降低 Python 开销。

- `compile_dynamic`
  - **作用**：`torch.compile(..., dynamic=...)`。
  - **说明**：`true` 时更容忍动态形状/最后一个 batch 不满等情况，但可能更慢。

- `compile_fullgraph`
  - **作用**：`torch.compile(..., fullgraph=...)`。
  - **说明**：`true` 更“严格”，遇到图 break 更容易失败；通常保持 `false` 更稳。

- `compile_cudagraphs`
  - **作用**：控制 Inductor 是否使用 CUDAGraph 相关优化路径。
  - **说明**：开启后可能更快，但在某些 PyTorch/驱动组合下训练阶段可能触发 Inductor/CUDAGraph 内部错误。
  - **建议**：默认 `false`；如你确认环境稳定再尝试 `true`。

- `compile_generator / compile_mapping_network / compile_style_encoder / compile_discriminator`
  - **作用**：按模块选择是否编译（分别对应 G/F/E/D）。
  - **典型用法**：
    - 先 `compile_generator=true`，其他 false，确认稳定
    - 再逐个打开 F/E
    - 如果遇到 Inductor 报错，优先把 `compile_discriminator=false`（判别器常是最容易触发图 break/内部问题的部分）

排障建议（遇到 Inductor/CUDAGraph 报错时）：
- 先设 `compile_cudagraphs: false`
- 再设 `compile_discriminator: false`
- 若仍不稳，临时 `use_compile: false`，确认训练逻辑本身 OK 后再逐步打开

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

可视化与进度条控制示例：

```yaml
training:
  progress_bar: true
  progress_style: percent   # bar | percent | off
  display_interval: 100     # 每多少 step 刷新一次显示
  log_json: false           # 是否打印 JSON 日志行
  num_workers: 4
  pin_memory: true
  use_compile: false
  compile_mode: default     # default | reduce-overhead | max-autotune
  compile_dynamic: false
  compile_fullgraph: false
  drop_last: true

说明：
当 `data.preload_to_gpu=true` 时，样本已是 CUDA Tensor，`pin_memory` 会被自动禁用以避免运行时报错。
若希望利用 `compile_dynamic` 处理最后一个不满 batch，建议设置 `training.drop_last=false`。

loss:
  r1_interval: 16

visualization:
  every_epochs: 5
  save_dir: "outputs/stargan_v2_baseline/vis"
  suppress_hf_warnings: true
```

## 备注

- 数据按 **unscaled latent** 读取；为训练稳定性，进入网络前会按 `training.latent_scale` 做缩放。
- 若数据量过大且显存不足，`dataset.py` 会自动回退到 CPU 存储，避免初始化阶段 OOM。
