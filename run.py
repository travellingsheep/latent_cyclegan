from __future__ import annotations

import argparse
import json
import os
import random
import sys
import atexit
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import create_dataset_from_config
from model import build_models
from trainer import StarGANv2Trainer


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        if cfg_path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        if cfg_path.suffix.lower() == ".json":
            return json.load(f)
    raise ValueError("Config file must be .yaml/.yml/.json")


def _config_contains_placeholder(cfg: Any, needle: str) -> bool:
    if isinstance(cfg, dict):
        return any(_config_contains_placeholder(v, needle) for v in cfg.values())
    if isinstance(cfg, list):
        return any(_config_contains_placeholder(v, needle) for v in cfg)
    if isinstance(cfg, str):
        return needle in cfg
    return False


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def expand_config_templates(config: dict) -> dict:
    """Expand string templates like {exp_name} and {outputs_dir} recursively.

    - exp_name is read from config['experiment']['name'] (preferred) or config['exp_name'] (fallback).
    - outputs_dir defaults to f"outputs/{exp_name}" unless overridden by config['experiment']['outputs_dir'].
    """

    exp_cfg = config.get("experiment", {}) if isinstance(config.get("experiment", {}), dict) else {}
    exp_name = str(exp_cfg.get("name", config.get("exp_name", ""))).strip()

    has_exp_placeholder = _config_contains_placeholder(config, "{exp_name}") or _config_contains_placeholder(
        config, "${exp_name}"
    )
    has_outputs_placeholder = _config_contains_placeholder(config, "{outputs_dir}") or _config_contains_placeholder(
        config, "${outputs_dir}"
    )
    if (has_exp_placeholder or has_outputs_placeholder) and not exp_name:
        raise ValueError(
            "Config uses {exp_name}/{outputs_dir} placeholders but experiment.name is missing. "
            "Please add:\nexperiment:\n  name: <your_experiment_name>"
        )

    outputs_dir = str(exp_cfg.get("outputs_dir", f"outputs/{exp_name}" if exp_name else "outputs")).strip()

    fmt = _SafeFormatDict(exp_name=exp_name, outputs_dir=outputs_dir)

    def rec(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: rec(v) for k, v in node.items()}
        if isinstance(node, list):
            return [rec(v) for v in node]
        if isinstance(node, str):
            s = node.replace("${exp_name}", "{exp_name}").replace("${outputs_dir}", "{outputs_dir}")
            try:
                return s.format_map(fmt)
            except Exception:
                return node
        return node

    return rec(config)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent-space StarGAN v2")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stargan_v2_latent.yaml",
        help="Path to config file (.yaml/.yml/.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = expand_config_templates(config)
    training_cfg = config["training"]

    console_log_path = str(training_cfg.get("console_log_path", "")).strip()
    if console_log_path:
        log_path = Path(console_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("a", encoding="utf-8")

        class _TeeStream:
            def __init__(self, primary, secondary):
                self.primary = primary
                self.secondary = secondary
                self.encoding = getattr(primary, "encoding", "utf-8")

            def write(self, data):
                self.primary.write(data)
                self.secondary.write(data)
                return len(data)

            def flush(self):
                self.primary.flush()
                self.secondary.flush()

            def isatty(self):
                return bool(getattr(self.primary, "isatty", lambda: False)())

        sys.stdout = _TeeStream(sys.stdout, log_f)
        sys.stderr = _TeeStream(sys.stderr, log_f)
        atexit.register(log_f.close)
        print(f"[Info] console logs are being saved to: {log_path}")

    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] using device: {device}")

    dataset = create_dataset_from_config(config, device=str(device))

    requested_pin_memory = bool(training_cfg.get("pin_memory", device.type == "cuda"))
    storage_device = getattr(dataset, "storage_device", torch.device("cpu"))
    data_on_gpu = isinstance(storage_device, torch.device) and storage_device.type == "cuda"
    effective_pin_memory = requested_pin_memory and not data_on_gpu
    if requested_pin_memory and data_on_gpu:
        print("[Warn] pin_memory is disabled because dataset is preloaded on CUDA tensors.")

    compile_dynamic = bool(training_cfg.get("compile_dynamic", False))
    drop_last = bool(training_cfg.get("drop_last", not compile_dynamic))

    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=effective_pin_memory,
        drop_last=drop_last,
    )

    models = build_models(config)
    generator = models.generator.to(device)
    mapping_network = models.mapping_network.to(device)
    style_encoder = models.style_encoder.to(device)
    discriminator = models.discriminator.to(device)

    use_compile = bool(training_cfg.get("use_compile", False))
    if use_compile:
        if hasattr(torch, "compile"):
            compile_mode = str(training_cfg.get("compile_mode", "default"))
            compile_fullgraph = bool(training_cfg.get("compile_fullgraph", False))
            compile_cudagraphs = bool(training_cfg.get("compile_cudagraphs", False))
            compile_g = bool(training_cfg.get("compile_generator", True))
            compile_f = bool(training_cfg.get("compile_mapping_network", True))
            compile_e = bool(training_cfg.get("compile_style_encoder", True))
            compile_d = bool(training_cfg.get("compile_discriminator", True))

            try:
                import torch._inductor.config as inductor_config  # type: ignore

                if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                    inductor_config.triton.cudagraphs = compile_cudagraphs
                if hasattr(inductor_config, "cudagraphs"):
                    inductor_config.cudagraphs = compile_cudagraphs
                print(f"[Info] torch.compile cudagraphs={compile_cudagraphs}")
            except Exception:
                pass
            try:
                if compile_g:
                    generator = torch.compile(
                        generator,
                        mode=compile_mode,
                        dynamic=compile_dynamic,
                        fullgraph=compile_fullgraph,
                    )
                if compile_f:
                    mapping_network = torch.compile(
                        mapping_network,
                        mode=compile_mode,
                        dynamic=compile_dynamic,
                        fullgraph=compile_fullgraph,
                    )
                if compile_e:
                    style_encoder = torch.compile(
                        style_encoder,
                        mode=compile_mode,
                        dynamic=compile_dynamic,
                        fullgraph=compile_fullgraph,
                    )
                if compile_d:
                    discriminator = torch.compile(
                        discriminator,
                        mode=compile_mode,
                        dynamic=compile_dynamic,
                        fullgraph=compile_fullgraph,
                    )
                print(
                    f"[Info] torch.compile enabled (mode={compile_mode}, dynamic={compile_dynamic}, fullgraph={compile_fullgraph})"
                )
            except Exception as compile_exc:
                print(f"[Warn] torch.compile failed, fallback to eager mode: {compile_exc}")
        else:
            print("[Warn] torch.compile is unavailable in current PyTorch version.")

    trainer = StarGANv2Trainer(
        generator=generator,
        mapping_network=mapping_network,
        style_encoder=style_encoder,
        discriminator=discriminator,
        train_loader=loader,
        config=config,
        device=device,
    )

    num_epochs = int(training_cfg["num_epochs"])

    outputs_root = Path(
        config.get("experiment", {}).get("outputs_dir", Path(config["checkpoint"]["save_dir"]).parent)
    )
    loss_plot_path = Path(training_cfg.get("loss_plot_path", str(outputs_root / "loss_curve.png")))
    loss_plot_interval = int(training_cfg.get("loss_plot_interval", 1))
    loss_plot_path.parent.mkdir(parents=True, exist_ok=True)

    def _now_str() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    history: dict[str, list[float]] = {}
    epochs_seen: list[int] = []

    def _update_history(epoch: int, logs: dict) -> None:
        epochs_seen.append(epoch)
        for k, v in logs.items():
            if k == "epoch":
                continue
            history.setdefault(k, []).append(float(v))

    def _save_loss_plot() -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[Warn] matplotlib is unavailable, skip loss plot: {exc}")
            return

        if not epochs_seen or not history:
            return

        plt.figure(figsize=(10, 6))
        for k, series in history.items():
            if len(series) != len(epochs_seen):
                continue
            plt.plot(epochs_seen, series, label=k)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss vs Epoch")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        plt.close()

    for epoch in range(trainer.start_epoch, num_epochs + 1):
        dataset.set_epoch(epoch)
        logs = trainer.train_epoch(epoch)
        printable = {k: round(v, 5) for k, v in logs.items()}
        printable["epoch"] = epoch
        print(f"[{_now_str()}] [Epoch Summary] {json.dumps(printable, ensure_ascii=False)}")

        _update_history(epoch, printable)
        if loss_plot_interval > 0 and (epoch % loss_plot_interval == 0):
            _save_loss_plot()


if __name__ == "__main__":
    main()
