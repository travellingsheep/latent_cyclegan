# Latent CycleGAN

这个仓库用于在 SD1.5 VAE latent 上训练、可视化和评估 CycleGAN，并支持 few-shot 消融实验。

## 主要脚本

- [train_latent_cyclegan.py](train_latent_cyclegan.py)
  主训练脚本。当前使用 step/kimg 训练循环，支持断点续训、few-shot 截断、JSONL 日志、checkpoint 和可视化输出。

- [visualize_logs.py](visualize_logs.py)
  训练日志看板。使用 Streamlit + Plotly 读取 JSONL 日志，支持自动刷新，并按标签页查看损失与学习率曲线。

- [run_sweep.py](run_sweep.py)
  few-shot 消融扫描脚本。它会读取 [configs/example.yaml](configs/example.yaml#L4) 中的 `exp_root`，为不同的 `data.max_samples_b` 生成隔离实验目录、配置快照、日志和汇总表。

- [utils/eval_latent_cyclegan.py](utils/eval_latent_cyclegan.py)
  latent-space CycleGAN 评估脚本，用于加载 checkpoint 并输出评估结果。

## 关键配置

主要配置文件是 [configs/example.yaml](configs/example.yaml)。常用字段如下：

- `exp_root`：实验输出根目录。单次训练/评估默认会写到 `exp_root/model`、`exp_root/logs`、`exp_root/vis`、`exp_root/eval`；`run_sweep.py` 还会在它下面继续创建各实验子目录。
- `style_a` / `style_b`：全局定义实验使用的两个风格名，训练与评估默认都会继承它们。
- `data.path`：训练 latent 数据根目录；脚本会自动读取 `{path}/{style_a}` 与 `{path}/{style_b}`。
- `data.max_samples_a` / `data.max_samples_b`：few-shot 截断开关，`-1` 表示使用全部样本。
- `train.total_kimg`：总训练量，单位为千张图。
- `logging.use_tqdm`：是否显示长进度条；默认关闭，单跑日志会和 sweep 对齐。
- `logging.log_file`：训练 JSONL 日志文件名，目录默认是 `exp_root/logs`。

## 数据格式

- 输入数据是 latent `.pt` 文件。
- 每个文件可以是 `[4,H,W]`、`[1,4,H,W]` 的 `torch.Tensor`，或包含 `latent` 键的字典。
- 训练时 A/B 两域独立 shuffle，并通过 infinite dataloader 持续供数。

## 安装依赖

```bash
/home/winchester/miniconda3/envs/latent_cyclegan/bin/python -m pip install -r requirements.txt
```

## 运行方式

训练：

```bash
/home/winchester/miniconda3/envs/latent_cyclegan/bin/python train_latent_cyclegan.py --config configs/example.yaml
```

训练 dry-run：

```bash
/home/winchester/miniconda3/envs/latent_cyclegan/bin/python train_latent_cyclegan.py --config configs/example.yaml --dry-run
```

训练日志看板：

```bash
/home/winchester/miniconda3/envs/latent_cyclegan/bin/python -m streamlit run visualize_logs.py
```

消融扫描 dry-run：

```bash
/home/winchester/miniconda3/envs/latent_cyclegan/bin/python run_sweep.py --dry-run
```

消融扫描正式运行：

```bash
/home/winchester/miniconda3/envs/latent_cyclegan/bin/python run_sweep.py
```

评估：

```bash
/home/winchester/miniconda3/envs/latent_cyclegan/bin/python utils/eval_latent_cyclegan.py --config configs/example.yaml
```

评估 dry-run：

```bash
/home/winchester/miniconda3/envs/latent_cyclegan/bin/python eval/evaluate_latent.py --config configs/example.yaml --dry-run
```

## 输出位置

- 单次训练默认输出到 `exp_root/model`、`exp_root/logs`、`exp_root/vis`。
- 单次评估默认输出到 `exp_root/eval`，同时会在该目录下生成 `metrics.csv` 与 `metrics_report.json`。
- sweep 输出会写入 `exp_root/<run_name>` 子目录。
- sweep 汇总表会保存为 `exp_root/sweep_summary.csv`。
