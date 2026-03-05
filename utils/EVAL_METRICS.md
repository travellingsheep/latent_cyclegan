# StarGANv2 Latent Eval Metrics（utils/eval.py）

本文档说明 `utils/eval.py` 里实现的评估指标定义、计算口径与实现细节。

> 评估分两种生成模式：
> - **ref**：用 `StyleEncoder E(x_ref, y_tgt)` 从目标域参考 latent 提取风格。
> - **random_noise**：用 `MappingNetwork F(z, y_tgt)` 从固定种子噪声生成目标域风格。
>
> 注意：ref 模式的“参考”在**生成**阶段仍然需要目标域的 latent（因为 `E` 的输入是 latent）。本文档里提到的“参考图/真实图集合”主要指**指标**（CLIP prototype、FID 等）使用的真实图像集合。

---

## 0. 数据与集合

### Latent 数据（必需）
- `data_root`：按 domain 子文件夹组织的 `.pt/.npy` latent 数据根目录。
- 生成时始终从该 latent 集合读取：
  - source latent：作为 content 输入 `G`。
  - ref 模式还会读取目标域某个 latent 作为 `E` 的输入（默认取该 domain 下排序后的第 1 个）。

### 真实图像集合（必需）
- `real_images_root`（CLI 参数 `--real_images_root`）：按 domain 子文件夹组织的**原始 RGB 图像**根目录。
- 也可直接写进 config：`config.data.image_root`。
- 用途：
  1) 计算每个 domain 的 CLIP prototype（用于 `clip_dir`）
  2) 作为 FID 的真实参考分布（`fid` 与 `fid_baseline`）
  3) 作为 source 图像（用于 `clip_content` / `lpips_content`）的首选来源

如果不提供 `real_images_root` / `config.data.image_root`：
- 脚本会直接报错，不再做 latent→图像回退解码。

### latent → 原图映射
- 对某个 latent 文件 `data_root/<dom>/<rel>.pt`：
  - 尝试在 `real_images_root/<dom>/<rel>.(png/jpg/jpeg/webp/bmp)` 中寻找同相对路径同文件名（仅扩展名不同）的原图。
  - 若找不到会直接报错并中断评估（避免口径混用）。

---

## 1. clip_content（内容一致性）

**定义**（逐样本）：

- 令 $e(\cdot)$ 为 CLIP 图像编码后得到的单位向量（L2 normalize）。
- $x$ 为源图（source image），$\hat{y}$ 为生成图。

$$
\text{clip\_content}(x,\hat{y}) = \cos\big(e(\hat{y}), e(x)\big)
$$

**实现**：
- CLIP 模型：默认 `openai/clip-vit-base-patch32`（可通过 `--clip_model_name` 改）。
- 用 `CLIPProcessor` 做预处理，`CLIPModel.get_image_features` 得到 embedding。
- embedding 做 L2 归一化后，用 `torch.nn.functional.cosine_similarity` 计算相似度。
- 输出写入 `metrics.csv` 的 `clip_content` 列。

---

## 2. lpips_content（感知距离，内容保持）

**定义**（逐样本）：

- 使用 LPIPS（VGG backbone）。
- 输入要求像素在 $[-1,1]$。

$$
\text{lpips\_content}(x,\hat{y}) = \text{LPIPS}(\hat{y}, x)
$$

**实现**：
- 使用 `lpips.LPIPS(net="vgg")`。
- 图像张量范围：脚本内部图像统一为 $[0,1]$，喂给 LPIPS 前会变换成 $[-1,1]$。
- 为避免显存峰值：支持 chunk 计算（`--lpips_chunk_size`），遇到 CUDA OOM 会自动减小 chunk；必要时可 CPU fallback（默认开启，可用 `--lpips_no_cpu_fallback` 关闭）。
- 输出写入 `metrics.csv` 的 `lpips_content` 列。

---

## 3. clip_dir（CLIP direction alignment）

**直觉**：
- 希望生成图相对源图的“语义变化方向”，与“目标域原型相对源图”的方向一致。

**定义**（逐样本）：

- $x$：源图
- $\hat{y}$：生成图
- $p_t$：目标域真实图像集合的 CLIP prototype（domain prototype）

先定义方向向量并归一化：

$$
\Delta_{gen} = \frac{e(\hat{y}) - e(x)}{\lVert e(\hat{y}) - e(x)\rVert_2 + \epsilon},\quad
\Delta_{tgt} = \frac{p_t - e(x)}{\lVert p_t - e(x)\rVert_2 + \epsilon}
$$

然后：

$$
\text{clip\_dir}(x,\hat{y},t) = \cos(\Delta_{gen}, \Delta_{tgt})
$$

**prototype 的计算**：
- 对每个 domain，取最多 64 张真实图（脚本里固定上限），计算归一化 CLIP embedding 后求均值，再做一次 L2 归一化。

**实现注意**：
- `clip_dir` 的计算依赖原始真实图（`real_images_root` / `config.data.image_root`）。

---

## 4. fid（生成分布 vs 目标真实分布）

**定义**（按 (src_domain, tgt_domain) 这一对聚合计算）：

- 用 InceptionV3 提取 2048 维特征。
- 令生成图特征的均值/协方差为 $(\mu_g, \Sigma_g)$，目标域真实图特征为 $(\mu_r, \Sigma_r)$。

$$
\text{fid} = \lVert \mu_g - \mu_r \rVert_2^2 + \mathrm{Tr}\big(\Sigma_g + \Sigma_r - 2(\Sigma_g\Sigma_r)^{1/2}\big)
$$

**实现**：
- Inception：`torchvision.models.inception_v3(weights=DEFAULT)`，取 `fc` 前特征（把 `fc` 替换成 `Identity`）。
- 真实集合：目标域 `tgt_domain` 下的真实图（来自 `real_images_root` / `config.data.image_root`）。
- 生成集合：当前 (src,tgt) 下生成并保存到 `out_dir/images/` 的图片。
- 子采样：
  - 生成侧最多取 `--art_fid_max_gen` 张
  - 真实侧最多取 `--art_fid_max_ref` 张

---

## 5. fid_baseline（真实 src vs 真实 tgt 的基线差异）

**定义**（按 (src_domain, tgt_domain) 聚合计算）：
- 与 `fid` 同公式，但两侧都是**真实图**：
  - src 侧：`src_domain` 的真实图
  - tgt 侧：`tgt_domain` 的真实图

用途：
- 作为域间“天然差异”的基线，用于计算 `delta_fid`。

---

## 6. delta_fid（相对提升）

**定义**（按 (src_domain, tgt_domain) 聚合计算）：

$$
\text{delta\_fid} = \text{fid\_baseline} - \text{fid}
$$

解释：
- 若生成让分布更接近目标域真实分布，则 `fid` 小于 `fid_baseline`，从而 `delta_fid` 为正。

---

## 7. artfid（组合指标）

**定义**（按 (src_domain, tgt_domain) 聚合计算）：

脚本当前口径：
- 先对该 (src,tgt) 的所有样本求 `lpips_content` 的均值 $\overline{\text{lpips\_content}}$。
- 再与该对的 `fid` 组合：

$$
\text{artfid} = (1 + \text{fid}) (1 + \overline{\text{lpips\_content}})
$$

实现说明：
- 当 `fid` 或 `lpips_content` 变大，`artfid` 会变大（越小越好）。

---

## 8. 聚合与输出

### metrics.csv
逐生成样本输出：
- `src_style`, `tgt_style`
- `src_image`：source 对应原图文件名
- `gen_image`：生成图相对路径（在 `out_dir/images/` 下）
- `lpips_content`, `clip_content`, `clip_dir`

### summary.json
按 (src,tgt) 聚合输出 `matrix_breakdown`，并给出两类池化均值：
- identity：`src == tgt`
- transfer：`src != tgt`

其中每个 cell 会包含：
- `clip_content`, `lpips_content`, `clip_dir`
- `fid_baseline`, `fid`, `delta_fid`, `artfid`

---

## 9. 实践建议

- 必须提供原图根目录（`--real_images_root` 或 `config.data.image_root`）：
  - 真实参考集合直接来自原始数据，避免“VAE 解码真值”带来的分布漂移。
  - `clip_dir` 的 prototype 与 FID 口径保持统一。
- 若你的 latent 数据是从某个图像数据集转换来的，尽量保证转换时保留了相对路径/文件名，这样 latent→原图映射命中率最高。

---

## 10. 用 config 直接运行

本仓库的 `configs/stargan_v2_latent*.yaml` 已加入 `eval:` 段，`utils/eval.py` 支持从 config 读取评估参数。

常用方式：
- `python -m utils.eval --config configs/stargan_v2_latent.yaml`

它会读取：
- `eval.checkpoint_path` / `eval.out_dir` / `eval.mode` 以及其他 eval 超参
- `data.eval_data_root` / `data.data_root` 作为 latent 数据来源
- `data.image_root` 作为原始图像来源（未设置会报错）

CLI 参数仍可覆盖 config（例如临时换 checkpoint 或 out_dir）。
