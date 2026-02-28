# utils 说明

本目录包含一些数据处理/训练辅助脚本。

## make_pt_dataset_sd15_vae.py

**用途**：将图片目录中的 RGB 图片编码为 Stable Diffusion 1.5 的 VAE latent，并以 `.pt`（单张图一个文件）形式保存，方便后续训练/推理直接读取 latent。

- 输入：图片文件（支持后缀：`.png` `.jpg` `.jpeg` `.webp` `.bmp`）
- 输出：与输入同相对路径的 `.pt` 文件（每个 `.pt` 是一个 `torch.Tensor`，形状通常为 `4 x H/8 x W/8`）
- 注意：脚本里明确写了 **输出是未做 0.18215 缩放的 latents**（unscaled latents），即直接来自 `vae.encode(imgs).latent_dist.sample()`。

### 两种运行模式

#### 1) 默认数据集模式（推荐用于 CycleGAN 风格数据结构）
默认会处理 `--dataset_dir` 下的若干个 split 子目录（默认 `trainA trainB`），并写到 `--out_dir` 下同名子目录。

例子：

```bash
python utils/make_pt_dataset_sd15_vae.py \
  --dataset_dir dataset \
  --out_dir pt_dataset \
  --splits trainA trainB \
  --device cuda \
  --batch_size 8 \
  --num_workers 2 \
  --amp
```

目录示意：

- 输入：`dataset/trainA/**.jpg`、`dataset/trainB/**.jpg`
- 输出：`pt_dataset/trainA/**.pt`、`pt_dataset/trainB/**.pt`

#### 2) 自定义输入目录模式（`--in_dirs`）
当提供 `--in_dirs` 时，会忽略 `--dataset_dir/--splits`，改为编码任意输入目录列表。每个输入目录会写到 `--out_dir/<name>/...`：

- `<name>` 默认是输入目录 basename
- 或者你可以用 `--out_names` 为每个输入目录指定输出子目录名

例子（不指定 out_names，输出子目录名 = 输入目录名）：

```bash
python utils/make_pt_dataset_sd15_vae.py \
  --in_dirs data/A data/B \
  --out_dir pt_dataset_custom \
  --device cuda
```

例子（指定 out_names）：

```bash
python utils/make_pt_dataset_sd15_vae.py \
  --in_dirs data/domainA data/domainB \
  --out_names trainA trainB \
  --out_dir pt_dataset \
  --device cuda
```

### 参数说明（逐项）

#### 输入/输出与模式控制

- `--in_dirs DIR [DIR ...]`
  - 作用：启用“自定义输入目录模式”。提供一个或多个待编码的图片根目录。
  - 行为：会覆盖 `--dataset_dir`/`--splits`。

- `--out_names NAME [NAME ...]`
  - 作用：给 `--in_dirs` 中每个输入目录指定对应的输出子目录名。
  - 约束：长度必须与 `--in_dirs` 完全一致。
  - 例子：`--in_dirs data/A data/B --out_names trainA trainB`

- `--dataset_dir PATH`（默认：`dataset`）
  - 作用：默认模式下的数据集根目录。
  - 期望结构：通常包含 `trainA/ trainB/`（或你在 `--splits` 中指定的子目录）。

- `--out_dir PATH`（默认：`pt_dataset`）
  - 作用：输出根目录。
  - 默认模式：输出到 `out_dir/<split>/...`
  - `--in_dirs` 模式：输出到 `out_dir/<name>/...`

- `--splits NAME [NAME ...]`（默认：`trainA trainB`）
  - 作用：默认模式下，要处理的子目录列表。
  - 例子：`--splits trainA trainB testA testB`

#### VAE 模型加载

- `--vae_model_name_or_path STR`（默认：`runwayml/stable-diffusion-v1-5`）
  - 作用：diffusers 的模型名或本地路径。
  - 常见用法：`runwayml/stable-diffusion-v1-5`

- `--vae_subfolder STR`（默认：`vae`）
  - 作用：当 `--vae_model_name_or_path` 是一个包含多个子目录的仓库/目录时，指定 VAE 权重所在子目录。
  - 说明：脚本会把空字符串视为未设置。

#### 运行设备与性能

- `--device STR`（默认：`cuda`）
  - 作用：选择 `cuda` 或 `cpu`。
  - 行为：如果你指定 `cuda` 但当前环境 `torch.cuda.is_available()` 为 false，会自动打印 warning 并回退到 `cpu`。

- `--batch_size INT`（默认：`8`）
  - 作用：每个 batch 编码的图片数量。
  - 建议：显存不够时减小；CPU 编码时也可以适当减小避免内存压力。

- `--num_workers INT`（默认：`2`）
  - 作用：PyTorch DataLoader 的 worker 数量。
  - 建议：Linux 下可适当增大以提升吞吐，但过大可能导致 IO 争用。

- `--amp`
  - 作用：在 CUDA 上启用自动混合精度（AMP）。
  - 说明：仅当 `--device cuda` 时才会生效；CPU 下会自动禁用。

#### 分辨率/缩放策略

该脚本要求输入给 VAE 的图片尺寸是 8 的倍数（因为 latent 空间下采样因子为 8）。你有两种方式确保这一点：

- `--resolution INT`（默认：不设置）
  - 作用：将所有图片 resize 为 `resolution x resolution`。
  - 约束：必须是 8 的倍数（例如 256、512、768）。
  - 适用：你希望所有 latent 尺寸完全一致（便于后续 batch 训练）。

- `--no_auto_resize_to_multiple_of_8`
  - 作用：当未设置 `--resolution` 时，默认会“自动把图片缩小到最接近的 8 的倍数尺寸”。加上此开关会关闭该行为。
  - 默认行为（不加该参数）：
    - 若图片尺寸不是 8 的倍数，会将 `W` 和 `H` 分别变为 `W - (W % 8)`、`H - (H % 8)`（只会变小，不会变大）。
    - 若缩到 0 或负数会报错（图片太小）。

#### 进度条显示

- `--tqdm_ncols INT`（默认：`120`）
  - 作用：tqdm 进度条宽度。

- `--tqdm_bar_len INT`（默认：`30`）
  - 作用：进度条中 bar 的长度（脚本用自定义 `bar_format`）。

### 依赖说明

脚本运行会用到：

- `torch`
- `diffusers`（用于 `AutoencoderKL`）
- `Pillow`（PIL 读取图片）
- `torchvision`（transforms）
- `tqdm`

如果缺少依赖，脚本会抛出带安装提示的错误信息。
