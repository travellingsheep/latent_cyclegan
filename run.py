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

    seed = int(config["training"].get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] using device: {device}")

    dataset = create_dataset_from_config(config, device=str(device))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=False,
        drop_last=True,
    )

    models = build_models(config)
    generator = models.generator.to(device)
    mapping_network = models.mapping_network.to(device)
    style_encoder = models.style_encoder.to(device)
    discriminator = models.discriminator.to(device)

    trainer = StarGANv2Trainer(
        generator=generator,
        mapping_network=mapping_network,
        style_encoder=style_encoder,
        discriminator=discriminator,
        train_loader=loader,
        config=config,
        device=device,
    )

    num_epochs = int(config["training"]["num_epochs"])
    for epoch in range(trainer.start_epoch, num_epochs + 1):
        dataset.set_epoch(epoch)
        logs = trainer.train_epoch(epoch)
        printable = {k: round(v, 5) for k, v in logs.items()}
        printable["epoch"] = epoch
        print(f"[Epoch Summary] {json.dumps(printable, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
