from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    def __init__(self, num_features: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False, eps=1e-5)
        self.fc = nn.Linear(style_dim, num_features * 2)
        torch.nn.init.constant_(self.fc.weight, 0.0)
        torch.nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        style_params = self.fc(style)
        gamma, beta = torch.chunk(style_params, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        normalized = self.norm(x)
        return (1.0 + gamma) * normalized + beta


class AdaINResBlock(nn.Module):
    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.adain1 = AdaIN(channels, style_dim)
        self.adain2 = AdaIN(channels, style_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.adain1(x, style)
        out = self.act(out)
        out = self.conv1(out)
        out = self.adain2(out, style)
        out = self.act(out)
        out = self.conv2(out)
        return residual + out


class Generator(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        style_dim: int = 64,
        base_dim: int = 256,
        n_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(base_dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(base_dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res_blocks = nn.ModuleList(
            [AdaINResBlock(base_dim, style_dim) for _ in range(n_res_blocks)]
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(base_dim, base_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_dim // 2, latent_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for block in self.res_blocks:
            h = block(h, style)
        return self.decoder(h)


class MappingNetwork(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        style_dim: int,
        num_domains: int,
        hidden_dim: int = 512,
        n_shared_layers: int = 3,
    ) -> None:
        super().__init__()
        shared_layers = []
        in_dim = latent_dim
        for _ in range(n_shared_layers):
            shared_layers.append(nn.Linear(in_dim, hidden_dim))
            shared_layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.shared = nn.Sequential(*shared_layers)
        self.unshared = nn.ModuleList(
            [nn.Linear(hidden_dim, style_dim) for _ in range(num_domains)]
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.shared(z)
        outputs = torch.stack([head(h) for head in self.unshared], dim=1)
        y_expanded = y.view(-1, 1, 1).expand(-1, 1, outputs.size(-1))
        style = torch.gather(outputs, dim=1, index=y_expanded).squeeze(1)
        return style


class StyleEncoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        style_dim: int,
        num_domains: int,
        base_dim: int = 256,
    ) -> None:
        super().__init__()
        mid_dim = max(base_dim // 2, 64)
        self.shared = nn.Sequential(
            nn.Conv2d(latent_channels, mid_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_dim, base_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.unshared = nn.ModuleList(
            [nn.Linear(base_dim, style_dim) for _ in range(num_domains)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.shared(x).flatten(1)
        outputs = torch.stack([head(h) for head in self.unshared], dim=1)
        y_expanded = y.view(-1, 1, 1).expand(-1, 1, outputs.size(-1))
        style = torch.gather(outputs, dim=1, index=y_expanded).squeeze(1)
        return style


class Discriminator(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        num_domains: int,
        base_dim: int = 256,
        d_layers: int = 3,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(latent_channels, base_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_ch = base_dim // 4
        for i in range(max(1, d_layers - 1)):
            out_ch = min(base_dim, in_ch * 2)
            stride = 2 if i < 2 else 1
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, num_domains, kernel_size=1, stride=1, padding=0))
        self.main = nn.Sequential(*layers)

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.forward_all(x)
        y_index = y.view(-1, 1, 1, 1).expand(-1, 1, logits.size(2), logits.size(3))
        return torch.gather(logits, dim=1, index=y_index)


@dataclass
class StarGANv2Models:
    generator: Generator
    mapping_network: MappingNetwork
    style_encoder: StyleEncoder
    discriminator: Discriminator


def build_models(config: dict) -> StarGANv2Models:
    model_cfg = config["model"]
    generator = Generator(
        latent_channels=model_cfg["latent_channels"],
        style_dim=model_cfg["style_dim"],
        base_dim=model_cfg["base_dim"],
        n_res_blocks=model_cfg["n_res_blocks"],
    )
    mapping_network = MappingNetwork(
        latent_dim=model_cfg.get("latent_dim", 16),
        style_dim=model_cfg["style_dim"],
        num_domains=model_cfg["num_domains"],
        hidden_dim=model_cfg.get("mapping_hidden_dim", 512),
        n_shared_layers=model_cfg.get("mapping_shared_layers", 3),
    )
    style_encoder = StyleEncoder(
        latent_channels=model_cfg["latent_channels"],
        style_dim=model_cfg["style_dim"],
        num_domains=model_cfg["num_domains"],
        base_dim=model_cfg["base_dim"],
    )
    discriminator = Discriminator(
        latent_channels=model_cfg["latent_channels"],
        num_domains=model_cfg["num_domains"],
        base_dim=model_cfg["base_dim"],
        d_layers=model_cfg["d_layers"],
    )
    return StarGANv2Models(
        generator=generator,
        mapping_network=mapping_network,
        style_encoder=style_encoder,
        discriminator=discriminator,
    )
