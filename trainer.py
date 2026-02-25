from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

try:
    from diffusers import AutoencoderKL
except ImportError:
    AutoencoderKL = None

from loss import (
    cycle_consistency_loss,
    d_hinge_loss,
    diversity_sensitive_loss,
    g_adversarial_loss,
    r1_penalty,
    style_reconstruction_loss,
)


class StarGANv2Trainer:
    def __init__(
        self,
        generator: torch.nn.Module,
        mapping_network: torch.nn.Module,
        style_encoder: torch.nn.Module,
        discriminator: torch.nn.Module,
        train_loader: DataLoader,
        config: dict,
        device: torch.device,
    ) -> None:
        self.G = generator
        self.F = mapping_network
        self.E = style_encoder
        self.D = discriminator
        self.loader = train_loader
        self.cfg = config
        self.device = device

        training_cfg = config["training"]
        self.use_amp = bool(training_cfg.get("use_amp", True) and device.type == "cuda")
        self.amp_dtype = training_cfg.get("amp_dtype", "bf16").lower()
        self.autocast_dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16

        self.scaler_d = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == "fp16")
        self.scaler_g = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == "fp16")

        self.z_dim = int(config["model"].get("latent_dim", 16))
        self.num_domains = int(config["model"]["num_domains"])
        self.latent_scale = float(config.get("training", {}).get("latent_scale", 0.18215))
        self.w_r1 = float(config.get("loss", {}).get("w_r1", 10.0))

        self.opt_d = torch.optim.AdamW(
            self.D.parameters(),
            lr=float(training_cfg["lr_d"]),
            weight_decay=float(training_cfg.get("weight_decay", 0.0)),
            betas=(0.0, 0.99),
        )

        lr_ge = float(training_cfg.get("lr_g", 1.0e-4))
        lr_mapping = float(training_cfg.get("lr_mapping", 1.0e-6))
        g_param_groups = [
            {"params": self.G.parameters(), "lr": lr_ge},
            {"params": self.E.parameters(), "lr": lr_ge},
            {"params": self.F.parameters(), "lr": lr_mapping},
        ]
        self.opt_g = torch.optim.AdamW(
            g_param_groups,
            lr=lr_ge,
            weight_decay=float(training_cfg.get("weight_decay", 0.0)),
            betas=(0.0, 0.99),
        )

        self.save_dir = Path(config["checkpoint"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = int(training_cfg.get("save_interval", 10))
        self.eval_interval = int(training_cfg.get("eval_interval", self.save_interval))
        self.log_interval = int(training_cfg.get("log_interval", 50))

        if AutoencoderKL is None:
            raise ImportError("diffusers is required for VAE visualization. Please install diffusers.")

        vae_cfg = config.get("visualization", {})
        vis_save_dir = vae_cfg.get("save_dir", "")
        self.vis_dir = (Path(vis_save_dir) if vis_save_dir else (self.save_dir / "vis"))
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        vae_model = vae_cfg.get("vae_model_name_or_path", "runwayml/stable-diffusion-v1-5")
        vae_subfolder = vae_cfg.get("vae_subfolder", "vae")
        self.vae_dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
        if self.device.type != "cuda":
            self.vae_dtype = torch.float32

        if vae_subfolder:
            self.vae = AutoencoderKL.from_pretrained(
                vae_model,
                subfolder=vae_subfolder,
                torch_dtype=self.vae_dtype,
            )
        else:
            self.vae = AutoencoderKL.from_pretrained(
                vae_model,
                torch_dtype=self.vae_dtype,
            )
        self.vae.to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad_(False)
        self.eval_num_samples = int(vae_cfg.get("num_samples", 4))
        self._cached_eval_batch = None

        self.start_epoch = 1
        resume_path = config["checkpoint"].get("resume_path", "").strip()
        if resume_path:
            self._load_checkpoint(resume_path)

    def _lambda_ds(self, epoch: int) -> float:
        loss_cfg = self.cfg["loss"]
        total_decay = max(1, int(loss_cfg.get("ds_decay_epochs", 1)))
        init_weight = float(loss_cfg.get("w_ds", 1.0))
        progress = min(max((epoch - 1) / total_decay, 0.0), 1.0)
        return init_weight * (1.0 - progress)

    def _save_checkpoint(self, epoch: int) -> None:
        payload = {
            "epoch": epoch,
            "G": self.G.state_dict(),
            "F": self.F.state_dict(),
            "E": self.E.state_dict(),
            "D": self.D.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "config": self.cfg,
        }
        ckpt_path = self.save_dir / f"epoch_{epoch:04d}.pt"
        torch.save(payload, ckpt_path)
        torch.save(payload, self.save_dir / "latest.pt")

    def _load_checkpoint(self, resume_path: str) -> None:
        ckpt = torch.load(resume_path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.F.load_state_dict(ckpt["F"])
        self.E.load_state_dict(ckpt["E"])
        self.D.load_state_dict(ckpt["D"])
        if "opt_g" in ckpt:
            self.opt_g.load_state_dict(ckpt["opt_g"])
        if "opt_d" in ckpt:
            self.opt_d.load_state_dict(ckpt["opt_d"])
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1

    def _sample_latent(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.z_dim, device=self.device)

    def _detach_log(self, logs: Dict[str, torch.Tensor | float]) -> Dict[str, float]:
        detached = {}
        for k, v in logs.items():
            if isinstance(v, torch.Tensor):
                detached[k] = v.detach().cpu().item()
            else:
                detached[k] = float(v)
        return detached

    def _get_eval_batch(self) -> dict:
        if self._cached_eval_batch is None:
            self._cached_eval_batch = next(iter(self.loader))
        return self._cached_eval_batch

    def _decode_latent_to_image(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.to(self.device, dtype=self.vae_dtype)
        with torch.no_grad():
            decoded = self.vae.decode(latent).sample
        decoded = decoded.float().clamp(-1.0, 1.0)
        return (decoded + 1.0) / 2.0

    def evaluate(self, epoch: int) -> None:
        self.G.eval()
        self.E.eval()

        batch = self._get_eval_batch()
        content_raw = batch["content"].to(self.device, non_blocking=True)
        style_ref_raw = batch["target_style"].to(self.device, non_blocking=True)
        y_trg = batch["target_style_id"].to(self.device, non_blocking=True).long().view(-1)

        content_scaled = content_raw * self.latent_scale
        style_ref_scaled = style_ref_raw * self.latent_scale

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
                dtype=self.autocast_dtype,
            ):
                s_ref = self.E(style_ref_scaled, y_trg)
                fake_latent_scaled = self.G(content_scaled, s_ref)

        content_for_decode = content_scaled / self.latent_scale
        style_for_decode = style_ref_scaled / self.latent_scale
        fake_for_decode = fake_latent_scaled / self.latent_scale

        n_vis = min(self.eval_num_samples, content_for_decode.size(0))
        content_img = self._decode_latent_to_image(content_for_decode[:n_vis])
        style_img = self._decode_latent_to_image(style_for_decode[:n_vis])
        fake_img = self._decode_latent_to_image(fake_for_decode[:n_vis])

        rows = []
        for i in range(n_vis):
            rows.extend([content_img[i], style_img[i], fake_img[i]])
        grid = make_grid(torch.stack(rows, dim=0), nrow=3)
        out_path = self.vis_dir / f"epoch_{epoch:04d}.png"
        save_image(grid, out_path)

        self.G.train()
        self.E.train()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.G.train()
        self.F.train()
        self.E.train()
        self.D.train()

        running = {
            "d_loss": 0.0,
            "g_loss": 0.0,
            "g_adv": 0.0,
            "sty": 0.0,
            "ds": 0.0,
            "cyc": 0.0,
            "r1": 0.0,
        }
        num_steps = 0

        ds_weight = self._lambda_ds(epoch)
        loss_cfg = self.cfg["loss"]
        w_adv = float(loss_cfg.get("w_adv", 1.0))
        w_sty = float(loss_cfg.get("w_sty", 1.0))
        w_cyc = float(loss_cfg.get("w_cyc", 1.0))

        for step, batch in enumerate(self.loader, start=1):
            try:
                content = batch["content"].to(self.device, non_blocking=True)
                x_ref = batch["target_style"].to(self.device, non_blocking=True)
                y_trg = batch["target_style_id"].to(self.device, non_blocking=True).long().view(-1)
                y_src = batch["source_style_id"].to(self.device, non_blocking=True).long().view(-1)

                content = content * self.latent_scale
                x_ref = x_ref * self.latent_scale

                batch_size = content.size(0)
                z_trg = self._sample_latent(batch_size)
                z_1 = self._sample_latent(batch_size)
                z_2 = self._sample_latent(batch_size)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.autocast_dtype,
                ):
                    x_ref_r1 = x_ref.detach().requires_grad_(True)
                    s_ref = self.E(x_ref, y_trg)
                    s_lat = self.F(z_trg, y_trg)

                    fake_ref = self.G(content, s_ref)
                    fake_lat = self.G(content, s_lat)

                    real_logits = self.D(x_ref_r1, y_trg)
                    fake_ref_logits_d = self.D(fake_ref.detach(), y_trg)
                    fake_lat_logits_d = self.D(fake_lat.detach(), y_trg)

                    d_loss_ref = d_hinge_loss(real_logits, fake_ref_logits_d)
                    d_loss_lat = d_hinge_loss(real_logits, fake_lat_logits_d)
                    r1_loss = r1_penalty(real_logits, x_ref_r1)
                    d_loss = 0.5 * (d_loss_ref + d_loss_lat) + self.w_r1 * r1_loss

                self.opt_d.zero_grad(set_to_none=True)
                if self.scaler_d.is_enabled():
                    self.scaler_d.scale(d_loss).backward()
                    self.scaler_d.step(self.opt_d)
                    self.scaler_d.update()
                else:
                    d_loss.backward()
                    self.opt_d.step()

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.autocast_dtype,
                ):
                    s_ref = self.E(x_ref, y_trg)
                    s_lat = self.F(z_trg, y_trg)

                    fake_ref = self.G(content, s_ref)
                    fake_lat = self.G(content, s_lat)

                    adv_ref = g_adversarial_loss(self.D(fake_ref, y_trg))
                    adv_lat = g_adversarial_loss(self.D(fake_lat, y_trg))
                    g_adv = adv_ref + adv_lat

                    s_ref_hat = self.E(fake_ref, y_trg)
                    s_lat_hat = self.E(fake_lat, y_trg)
                    sty_ref = style_reconstruction_loss(s_ref_hat, s_ref)
                    sty_lat = style_reconstruction_loss(s_lat_hat, s_lat)
                    sty_loss = sty_ref + sty_lat

                    s_1 = self.F(z_1, y_trg)
                    s_2 = self.F(z_2, y_trg)
                    fake_1 = self.G(content, s_1)
                    fake_2 = self.G(content, s_2)
                    ds_loss = diversity_sensitive_loss(fake_1, fake_2)

                    s_src = self.E(content, y_src)
                    rec_ref = self.G(fake_ref, s_src)
                    rec_lat = self.G(fake_lat, s_src)
                    cyc_ref = cycle_consistency_loss(rec_ref, content)
                    cyc_lat = cycle_consistency_loss(rec_lat, content)
                    cyc_loss = cyc_ref + cyc_lat

                    g_loss = w_adv * g_adv + w_sty * sty_loss + w_cyc * cyc_loss - ds_weight * ds_loss

                self.opt_g.zero_grad(set_to_none=True)
                if self.scaler_g.is_enabled():
                    self.scaler_g.scale(g_loss).backward()
                    self.scaler_g.step(self.opt_g)
                    self.scaler_g.update()
                else:
                    g_loss.backward()
                    self.opt_g.step()

                logs = self._detach_log(
                    {
                        "d_loss": d_loss,
                        "g_loss": g_loss,
                        "g_adv": g_adv,
                        "sty": sty_loss,
                        "ds": ds_loss,
                        "cyc": cyc_loss,
                        "r1": r1_loss,
                    }
                )

                for key in running:
                    running[key] += logs[key]

                num_steps += 1

                if step % self.log_interval == 0:
                    printable = {k: round(v, 5) for k, v in logs.items()}
                    printable["epoch"] = epoch
                    printable["step"] = step
                    printable["lambda_ds"] = round(ds_weight, 6)
                    print(json.dumps(printable, ensure_ascii=False))

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"[OOM] epoch={epoch}, step={step}. batch skipped.")
                    self.opt_d.zero_grad(set_to_none=True)
                    self.opt_g.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

        if num_steps == 0:
            mean_logs = {k: float("nan") for k in running}
        else:
            mean_logs = {k: v / num_steps for k, v in running.items()}

        if epoch % self.save_interval == 0:
            self._save_checkpoint(epoch)
        if epoch % self.eval_interval == 0:
            self.evaluate(epoch)

        return mean_logs
