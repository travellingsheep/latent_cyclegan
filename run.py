from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

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
    training_cfg = config["training"]

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
    for epoch in range(trainer.start_epoch, num_epochs + 1):
        dataset.set_epoch(epoch)
        logs = trainer.train_epoch(epoch)
        printable = {k: round(v, 5) for k, v in logs.items()}
        printable["epoch"] = epoch
        print(f"[Epoch Summary] {json.dumps(printable, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
