# StarGAN-Latent 自动化评测流水线开发需求说明书 (PRD)

## 1. 项目背景与工程全局规范
本项目用于评测 StarGAN v2 潜空间的生成质量、多样性及语义方向。
为了保证实验的可复现性与底层逻辑透明度，**严禁在 FID 计算中使用 `clean-fid` 或 `pytorch-fid` 等第三方库**，必须从零实现底层的特征提取与距离计算。

**全局约束：**
1. **配置文件驱动**：必须通过 `argparse` 读取 `configs/*.yaml`（如 `data.eval_data_root`, `checkpoint.save_dir` 等）。
2. **统一工作区**：根据 `checkpoint.save_dir` 定位到 `latest.pt` 的父目录，在该目录下建立 `metrics/` 工作区。所有脚本的输出必须存入此目录：
   - 特征缓存：`metrics/cache/`
   - 生成图片：`metrics/images/`
   - 最终报表：`metrics/summary.json`
3. **耗时与进度记录**：每个独立脚本必须使用 `time` 记录起始时间与总耗时。所有的遍历操作必须包上 `tqdm` 进度条。

---

## 2. 核心算法与数学对齐规范（Agent 必读）

### 2.1 定制化 FID 计算对齐（严禁调用第三方库）
必须自己编写 `_InceptionFeatRunner` 类与 `_frechet_distance` 函数，严格按照以下逻辑实现：
* **图像预处理**：使用 `torchvision.transforms` 严格执行：
  `Resize((299, 299))` -> `ToTensor()` -> `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`。
* **模型加载**：使用 `torchvision.models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)`，并将最后的全连接层替换为 `torch.nn.Identity()`。提取特征后需转换为双精度浮点 `double().numpy()`。
* **协方差计算**：提取特征后，计算均值和协方差。**必须使用** NumPy 默认的无偏估计（即 `ddof=1`）：
  `cov_matrix = np.cov(features, rowvar=False)`
* **Fréchet 距离容错补丁 (极其重要)**：
  在计算 $\Sigma_1 \cdot \Sigma_2$ 的矩阵平方根 `scipy.linalg.sqrtm` 时，若出现 `inf` 或 `nan`，需给矩阵对角线加上 $\epsilon = 10^{-6}$ 的极小值偏移（`offset = np.eye(...) * 1e-6`）重新计算。且最后取值必须丢弃虚部：`covmean = covmean.real`。

### 2.2 CLIP 特征与 Directional Similarity
* **模型**：使用 `openai/clip-vit-base-patch32`。
* **原型计算**：提取图像的 CLIP Embeddings，进行 L2 归一化。Target Domain 原型 $P_Y$ 为该域真实图像特征之和的再归一化：
  $P_Y = \sum f_{clip}(x_i) / \|\sum f_{clip}(x_i)\|_2$
* **方向相似度**：$\Delta I = f_{clip}(\hat{x}_{gen}) - f_{clip}(x_{src})$，$\Delta S = P_Y - f_{clip}(x_{src})$。计算 $\Delta I$ 与 $\Delta S$ 的余弦相似度。

### 2.3 多样性 LPIPS (Pairwise)
* 针对 5 张同源不同噪声的生成图 $\hat{x}_{1\dots5}$，计算并平均所有配对（10对）的 LPIPS 距离。

---

## 3. 模块执行需求分解（共 4 个脚本）全部放在utils子文件夹下

### 脚本 0：`main_eval.py` (总控流水线)
* 负责解析 `yaml` 配置文件，获取并构建基础路径。
* 依次 `import` 并调用下游 3 个脚本的执行主函数。若检测到 `metrics/cache/` 目录已存在所需缓存且未开启 `--force`，则打印跳过缓存构建的日志。

### 脚本 1：`01_build_cache.py` (基线预计算)
* 遍历真实的测试集 RGB 图像。
* **任务 A**：调用定制的 Inception Runner，提取各 Domain 真实图片的特征，计算出 $\mu$ 和 $\Sigma$，序列化存至 `metrics/cache/fid_stats_{domain}.pkl`。
* **任务 B**：调用 CLIP 提取特征，计算各 Domain 的 $P_Y$，存为字典至 `metrics/cache/clip_prototypes.pt`。

### 脚本 2：`02_generate_images.py` (离线推理落盘)
* 加载 `latest.pt`。从 `.pt` 潜变量测试集中为每个 Source Domain 提取 30 张作为 Content ($x_{src}$)，解码为原图并落盘（留作比对）。
* **模式 A (Reference)**：
  * 遍历 Target Domain。为 30 张 Source **随机一一匹配** 30 张 Target 潜变量作为 $x_{ref}$。
  * 调用 `StyleEncoder` 与 `Generator` 进行生成。
  * 双重落盘：纯净版存入 `images/ref/pure/{src}_to_{tgt}/`；并用 PIL 将 $x_{ref}$ 与生成图**上下拼接**，存入 `images/ref/vis_concat/{src}_to_{tgt}/`。
* **模式 B (Standard Noise)**：
  * 初始化 **1 个固定种子** 的全局噪声 $z$。
  * 调用 `MappingNetwork` 获取风格并生成。
  * 纯净版存入 `images/noise_std/pure/{src}_to_{tgt}/`。
* **模式 C (Diversity Noise)**：
  * 对每个 Content，采样 **5 个独立噪声**，生成 5 张风格变体。
  * 将这 5 张单图存入 `images/noise_div/{src}_to_{tgt}/`。
  * 使用 PIL 将这 5 张图**竖直拼接**为一张预览图（`_div_concat.png`）单独保存。

### 脚本 3：`03_evaluate_metrics.py` (独立计算与报表)
* 完全脱离生成网络，读取 `metrics/images/` 与 `metrics/cache/` 执行计算。
* **计算项**：
  1. 读取纯净图，计算 LPIPS Content 与 CLIP Content。
  2. 读取缓存，计算生成的 target 图文件夹的 FID，结合 baseline 计算 **Delta FID** = $FID_{base} - FID_{gen}$。
  3. 计算 **Art-FID** = $(1 + FID_{gen}) \times (1 + LPIPS_{content})$。
  4. 利用缓存的 $P_Y$，计算 **CLIP-Dir**。
  5. 针对 `noise_div` 目录的同源 5 张图，计算 Pairwise LPIPS (Diversity)。
* **输出**：计算各项指标均值，整合各模块的运行耗时，最终写入 `metrics/summary.json`。