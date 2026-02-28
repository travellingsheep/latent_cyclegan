# Loss 曲线指标说明（latent_cyclegan）

这份文档解释你在训练日志/曲线图里看到的这些标量：

- `d_loss`, `g_loss`
- `g_adv`, `sty`, `ds`, `cyc`, `id`, `r1`

它们来自训练循环：`StarGANv2Trainer.train_epoch()`（trainer.py），基础 loss 实现在 loss.py。

> 重要提醒（关于 latent 缩放）：
>
> - 数据集里通常存的是 **unscaled** 的 SD VAE latent（例如你用 `utils/make_pt_dataset_sd15_vae.py` 生成的 `.pt`）。
> - 在训练时，代码会做：`content = content * latent_scale`、`x_ref = x_ref * latent_scale`。
> - `latent_scale` 默认来自配置 `training.latent_scale`，常见为 `0.18215`。
>
> 所以曲线里的所有 loss（无论是对抗、cycle、identity、diversity）都是在“乘过 latent_scale 后”的 latent 空间上计算的。

---

## 这些数是怎么统计成“按 epoch 的一条曲线”的？

在每个 step（一个 batch）上会计算一次各个标量，然后：

- `running[key] += logs[key]`
- epoch 结束时再除以 step 数得到 epoch 平均

因此你画的 **epoch 曲线** 基本上是“每个 epoch 内所有 step 的平均值”。

### 关于 `r1` 的特殊性
`r1` 不是每一步都算：只有当 `step % r1_interval == 0` 时才计算，否则直接记为 0。
所以：

- 曲线上的 `r1` 往往很小（因为很多 step 是 0 被平均进去了）
- 但 `d_loss` 里加入 R1 的方式做了“间隔补偿”（见下文），使得 **R1 对 `d_loss` 的平均贡献不随 interval 改变**（近似成立）。

---

## 总览：判别器与生成器的优化目标

训练里对 D 和 G 分开优化：

- 判别器（D）：hinge loss + R1 正则（间隔计算）
- 生成器/风格相关模块（G/E/F）：对抗 + 风格重建 + cycle + identity（可选） - diversity（带权重衰减）

其中：

- `G`：Generator
- `E`：Style Encoder（从参考图/latent 提取 style 向量）
- `F`：Mapping Network（从随机噪声 z 生成 style 向量）
- `D`：Discriminator（带 domain label 条件）

训练会同时走两条“目标风格来源”的路径：

1) **ref 路径**：用参考样本 `x_ref` 经 `E` 得到 `s_ref`
2) **latent 路径**：用随机噪声 `z_trg` 经 `F` 得到 `s_lat`

所以你会看到很多子 loss 都是 `ref` + `lat` 两项相加。

---

## 每条曲线分别是什么？

下面按你图例中的名字逐个说明。

### 1) `d_loss`（判别器总 loss）

在每个 step：

- 先算两份 hinge loss：
  - `d_loss_ref = d_hinge_loss(D(x_ref, y_trg), D(G(content, s_ref).detach(), y_trg))`
  - `d_loss_lat = d_hinge_loss(D(x_ref, y_trg), D(G(content, s_lat).detach(), y_trg))`
- 再取平均：
  - `d_loss = 0.5 * (d_loss_ref + d_loss_lat)`

其中 hinge 的定义（loss.py）：

- `d_hinge_loss(real, fake) = mean(relu(1 - real)) + mean(relu(1 + fake))`

#### 为什么 `d_loss` 常常接近 2？
当 `D` 的输出 logits 大约都在 0 附近时：

- `relu(1 - 0) = 1`
- `relu(1 + 0) = 1`

所以 hinge loss 大约是 2。

如果你的曲线长期贴近 2，通常表示：

- D 的判别没有明显变强（logits 没怎么跑出 margin），或者
- 训练设置让 D/G 处于某种平衡（也可能是学习率/正则导致 D 不愿意过拟合）

#### `d_loss` 里还可能包含 R1（见 r1）
当触发 R1 step 时，会额外加：

- `d_loss += (w_r1 * r1_interval) * r1_loss`

注意：`r1_loss` 是 **原始 r1 penalty**；乘上 `r1_interval` 是一种常见做法，用于抵消“不是每一步都算”的稀疏性。

相关配置：

- `loss.w_r1`
- `loss.r1_interval`

---

### 2) `r1`（R1 gradient penalty，原始值）

只在 `step % r1_interval == 0` 时计算：

- 令 `x_ref_r1 = x_ref.detach().requires_grad_(True)`
- `real_logits_r1 = D(x_ref_r1.float(), y_trg)`（强制关闭 autocast，确保梯度更稳定）
- `r1_loss = r1_penalty(real_logits_r1, x_ref_r1)`

R1 的定义（loss.py）：

- 先求梯度：`grad = ∂(sum logits)/∂x`
- 展平后做二范数平方：`mean(sum(grad^2))`

> 曲线上 `r1` 是 `r1_loss` 本身，并没有乘 `w_r1` 或 `r1_interval`。

---

### 3) `g_loss`（生成器总 loss，实际反向传播用的目标）

代码里：

```
G_loss = w_adv * g_adv
       + w_sty * sty
       + w_cyc * cyc
       + w_id  * id
       - lambda_ds(epoch) * ds
```

其中权重来自配置 `loss.*`，而 `lambda_ds(epoch)` 是一个按 epoch 衰减的系数（见下文 `ds`）。

**关键点**：

- `ds` 在总 loss 里是“减号”：训练是在 **最大化（鼓励）** fake 的差异。
- 你图里画的 `ds` 通常是 **正数**（L1 距离），但它对 `g_loss` 的贡献是负的。

---

### 4) `g_adv`（生成器对抗项，ref + latent 两条路径相加）

每个 step：

- `adv_ref = g_adversarial_loss(D(fake_ref, y_trg))`
- `adv_lat = g_adversarial_loss(D(fake_lat, y_trg))`
- `g_adv = adv_ref + adv_lat`

生成器对抗 loss 定义（loss.py）：

- `g_adversarial_loss(fake_logits) = -mean(fake_logits)`

直觉解释：

- 生成器希望 `D(fake)` 的 logits 越大越好
- 因为最小化 `-mean(logits)` 等价于最大化 `mean(logits)`

相关配置：

- `loss.w_adv`

---

### 5) `sty`（style reconstruction loss，风格重建）

目标：生成结果应该能被 style encoder 还原回原本的 style 向量。

每个 step：

- `s_ref = E(x_ref, y_trg)`
- `fake_ref = G(content, s_ref)`
- `s_ref_hat = E(fake_ref, y_trg)`
- `sty_ref = L1(s_ref_hat, s_ref)`

latent 路径同理：

- `s_lat = F(z_trg, y_trg)`
- `fake_lat = G(content, s_lat)`
- `s_lat_hat = E(fake_lat, y_trg)`
- `sty_lat = L1(s_lat_hat, s_lat)`

最后：

- `sty = sty_ref + sty_lat`

style 重建损失（loss.py）：

- `style_reconstruction_loss = L1(pred_style, target_style)`

相关配置：

- `loss.w_sty`

---

### 6) `ds`（diversity sensitive loss，鼓励多样性）

这里的 `ds` 记录的是 **两次随机采样 style 之后生成结果的 L1 距离**：

- `s_1 = F(z_1, y_trg)`, `s_2 = F(z_2, y_trg)`
- `fake_1 = G(content, s_1)`, `fake_2 = G(content, s_2)`
- `ds = L1(fake_1, fake_2)`

它的定义（loss.py）：

- `dist = L1(fake_1, fake_2)`
- 如果设置了 `ds_margin`：返回 `clamp(dist, max=ds_margin)`

#### 这项在总 loss 里是“减号”
`g_loss` 里是：

- `... - lambda_ds(epoch) * ds`

因此：

- `ds` 越大，`g_loss` 越小（更“好”），训练会推动 `ds` 变大
- 但因为它会被衰减到 0，后期对训练几乎不再起作用

#### diversity 权重是不是“20 epoch 内衰减到 0”？
取决于你的配置 `loss.ds_decay_epochs`。

代码中的衰减函数（trainer.py）是：

- `total_decay = ds_decay_epochs`
- `progress = clamp((epoch - 1) / total_decay, 0, 1)`
- `lambda_ds(epoch) = w_ds * (1 - progress)`

这意味着：

- epoch=1：`lambda_ds = w_ds`
- epoch=ds_decay_epochs+1：`lambda_ds = 0`

所以当你配置 `ds_decay_epochs: 20` 时：

- **第 1 个 epoch 权重最大**
- **到第 21 个 epoch 开始权重变为 0**

直觉上可以理解为“在前 20 个 epoch 线性衰减到 0（到第 21 个 epoch 为 0）”。

相关配置：

- `loss.w_ds`
- `loss.ds_decay_epochs`
- `loss.ds_margin`

---

### 7) `cyc`（cycle consistency loss）

目标：把 content 翻译到目标域后，再用源域风格翻译回来，应当回到原 content（latent）。

每个 step：

- 先得到源域 style：`s_src = E(content, y_src)`
- 对 ref 路径：
  - `rec_ref = G(fake_ref, s_src)`
  - `cyc_ref = L1(rec_ref, content)`
- 对 latent 路径：
  - `rec_lat = G(fake_lat, s_src)`
  - `cyc_lat = L1(rec_lat, content)`

最后：

- `cyc = cyc_ref + cyc_lat`

cycle loss（loss.py）：

- `cycle_consistency_loss = L1(reconstructed, original)`

相关配置：

- `loss.w_cyc`

---

### 8) `id`（identity loss，可选）

目标：当你用“源域 style”去生成时，输出不应改变输入（保持 identity）。

在代码里只有当 `w_id > 0` 时启用：

- `s_src = E(content, y_src)`
- `id_out = G(content, s_src)`
- `id = L1(id_out, content)`

如果 `w_id == 0`：

- `id` 直接记为 0

相关配置：

- `loss.w_id`

---

## 结合你的曲线，如何读这些走势？（常见现象）

1) `ds` 往往前期下降/趋稳，而 `lambda_ds` 在衰减
- 你画的是 `ds` 的原始距离；即使它不变，真正影响 `g_loss` 的是 `lambda_ds * ds`
- 由于 `lambda_ds` 会线性下降，**后期 diversity 项对训练的“推动力”会越来越弱**，这很常见

2) `d_loss` 贴近 2
- hinge loss 在 logits≈0 时就是 2
- 这不必然代表坏，但通常说明 D 没有强到把 real 推到 >1、fake 推到 <-1 很远

3) `g_loss` 持续下降
- `g_loss` 是带权重加和的总目标，其中还包含 `-lambda_ds*ds` 这种“越大越好”的项
- 所以它下降并不等价于“生成质量单调变好”，需要结合可视化结果一起看

---

## 我想确认你当前实验的衰减是否真的是 20 epoch

看你当前配置（例如 `configs/stargan_v2_latent.yaml`）里：

- `loss.ds_decay_epochs: 20`

因此按实现，`lambda_ds` 确实会从 epoch=1 开始线性衰减，并在 epoch=21 变为 0。

如果你把曲线图里 `ds` 的 label 理解成“diversity 权重”，那就要注意：

- 图里的 `ds` 是 **原始 ds_loss**
- 真正衰减的是 `lambda_ds(epoch)`（训练日志 JSON 里会打印 `lambda_ds`）

---

## 相关配置项速查

位于你的 config（YAML/JSON）里的 `loss:`：

- `w_adv`：对抗项权重（乘 `g_adv`）
- `w_sty`：风格重建权重（乘 `sty`）
- `w_cyc`：cycle 权重（乘 `cyc`）
- `w_id`：identity 权重（乘 `id`）
- `w_ds`：diversity 初始权重（乘 `ds`，但注意是减号）
- `ds_decay_epochs`：diversity 权重线性衰减长度（按 epoch）
- `ds_margin`：diversity 距离的截断上限（可选）
- `w_r1`：R1 权重
- `r1_interval`：每隔多少 step 计算一次 R1

以及 `training:`：

- `latent_scale`：训练时对输入 latent 乘的缩放系数（常见 0.18215）

