from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_latent_tensor(path: Path) -> torch.Tensor:
    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            if "latent" in data:
                data = data["latent"]
            elif "x" in data:
                data = data["x"]
            else:
                first_value = next(iter(data.values()))
                data = first_value
        tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    elif path.suffix == ".npy":
        array = np.load(path)
        tensor = torch.from_numpy(array)
    else:
        raise ValueError(f"Unsupported latent format: {path}")

    tensor = tensor.float()
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 3:
        raise ValueError(f"Expected latent shape [C,H,W], got {tuple(tensor.shape)} from {path}")
    return tensor.contiguous()


class LatentMultiDomainDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        domains: List[str],
        preload_to_gpu: bool = True,
        device: str = "cuda",
        epoch_size: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(data_root)
        self.domains = domains
        self.num_domains = len(domains)
        self.preload_to_gpu = preload_to_gpu
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

        self.domain_files: List[List[Path]] = []
        for domain in domains:
            domain_dir = self.root / domain
            if not domain_dir.exists():
                raise FileNotFoundError(f"Domain directory not found: {domain_dir}")
            files = sorted([*domain_dir.glob("*.pt"), *domain_dir.glob("*.npy")])
            if not files:
                raise RuntimeError(f"No .pt/.npy files found in {domain_dir}")
            self.domain_files.append(files)

        if epoch_size is None:
            max_count = max(len(files) for files in self.domain_files)
            self.epoch_size = max_count * self.num_domains
        else:
            self.epoch_size = int(epoch_size)

        self.storage_device = self._decide_storage_device()
        self.domain_tensors: List[torch.Tensor] = self._preload_all()

        self.src_domain_idx = torch.zeros(self.epoch_size, dtype=torch.long)
        self.tgt_domain_idx = torch.zeros(self.epoch_size, dtype=torch.long)
        self.src_file_idx = torch.zeros(self.epoch_size, dtype=torch.long)
        self.tgt_file_idx = torch.zeros(self.epoch_size, dtype=torch.long)
        self.set_epoch(0)

    def _estimate_total_bytes(self) -> int:
        total = 0
        for files in self.domain_files:
            for path in files:
                total += path.stat().st_size
        return total

    def _decide_storage_device(self) -> torch.device:
        if not self.preload_to_gpu or self.device.type != "cuda":
            return torch.device("cpu")
        free_mem, _ = torch.cuda.mem_get_info(self.device)
        required = self._estimate_total_bytes() * 2
        if required < int(free_mem * 0.9):
            return self.device
        return torch.device("cpu")

    def _preload_all(self) -> List[torch.Tensor]:
        domain_tensors: List[torch.Tensor] = []
        for files in self.domain_files:
            tensors = [_load_latent_tensor(path) for path in files]
            stacked = torch.stack(tensors, dim=0)
            stacked = stacked.to(self.storage_device, non_blocking=True)
            domain_tensors.append(stacked)
        return domain_tensors

    def set_epoch(self, epoch: int) -> None:
        self.generator.manual_seed(int(epoch) + 12345)

        self.src_domain_idx = torch.randint(
            low=0,
            high=self.num_domains,
            size=(self.epoch_size,),
            generator=self.generator,
            dtype=torch.long,
        )

        tgt_offset = torch.randint(
            low=1,
            high=self.num_domains,
            size=(self.epoch_size,),
            generator=self.generator,
            dtype=torch.long,
        )
        self.tgt_domain_idx = (self.src_domain_idx + tgt_offset) % self.num_domains

        src_sizes = torch.tensor([len(files) for files in self.domain_files], dtype=torch.long)
        src_max = int(src_sizes.max().item())
        rand_src = torch.randint(
            low=0,
            high=src_max,
            size=(self.epoch_size,),
            generator=self.generator,
            dtype=torch.long,
        )
        self.src_file_idx = rand_src % src_sizes[self.src_domain_idx]

        tgt_sizes = src_sizes
        tgt_max = int(tgt_sizes.max().item())
        rand_tgt = torch.randint(
            low=0,
            high=tgt_max,
            size=(self.epoch_size,),
            generator=self.generator,
            dtype=torch.long,
        )
        self.tgt_file_idx = rand_tgt % tgt_sizes[self.tgt_domain_idx]

    def __len__(self) -> int:
        return self.epoch_size

    def _get_tensor(self, domain_id: int, file_id: int) -> torch.Tensor:
        return self.domain_tensors[domain_id][file_id]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        src_domain = int(self.src_domain_idx[index].item())
        tgt_domain = int(self.tgt_domain_idx[index].item())
        src_file = int(self.src_file_idx[index].item())
        tgt_file = int(self.tgt_file_idx[index].item())

        content = self._get_tensor(src_domain, src_file)
        target_style = self._get_tensor(tgt_domain, tgt_file)

        return {
            "content": content,
            "target_style": target_style,
            "target_style_id": torch.tensor(tgt_domain, dtype=torch.long, device=content.device),
            "source_style_id": torch.tensor(src_domain, dtype=torch.long, device=content.device),
        }


def create_dataset_from_config(config: dict, device: str) -> LatentMultiDomainDataset:
    data_cfg = config["data"]
    train_cfg = config["training"]
    return LatentMultiDomainDataset(
        data_root=data_cfg["data_root"],
        domains=data_cfg["domains"],
        preload_to_gpu=data_cfg.get("preload_to_gpu", True),
        device=device,
        epoch_size=data_cfg.get("epoch_size"),
        seed=train_cfg.get("seed", 42),
    )
