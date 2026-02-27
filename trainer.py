from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict
from datetime import datetime
import time
import warnings

import torch
import torch.nn.functional as nnF
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.utils import make_grid, save_image
import math

try:
    from diffusers import AutoencoderKL
except ImportError:
    AutoencoderKL = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

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

        scaler_enabled = self.use_amp and self.amp_dtype == "fp16"
        if self.device.type == "cuda":
            self.scaler_d = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
            self.scaler_g = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            self.scaler_d = torch.amp.GradScaler("cpu", enabled=False)
            self.scaler_g = torch.amp.GradScaler("cpu", enabled=False)

        self.z_dim = int(config["model"].get("latent_dim", 16))
        self.num_domains = int(config["model"]["num_domains"])
        self.latent_scale = float(config.get("training", {}).get("latent_scale", 0.18215))
        self.w_r1 = float(config.get("loss", {}).get("w_r1", 10.0))
        self.r1_interval = max(1, int(config.get("loss", {}).get("r1_interval", 16)))
        self.ds_margin = config.get("loss", {}).get("ds_margin", None)

        optim_fused = bool(training_cfg.get("optim_fused", True))
        fused_ok = self.device.type == "cuda" and optim_fused

        def _adamw_kwargs() -> dict:
            kwargs = {
                "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
                "betas": (0.0, 0.99),
            }
            try:
                if fused_ok:
                    kwargs["fused"] = True
            except Exception:
                pass
            return kwargs

        self.opt_d = torch.optim.AdamW(
            self.D.parameters(),
            lr=float(training_cfg["lr_d"]),
            **_adamw_kwargs(),
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
            **_adamw_kwargs(),
        )

        self.save_dir = Path(config["checkpoint"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = int(training_cfg.get("save_interval", 10))
        self.log_interval = int(training_cfg.get("log_interval", 50))
        # progress_bar controls tqdm bar only; percent printing is controlled by progress_style.
        self.show_progress = bool(training_cfg.get("progress_bar", True))
        self.progress_style = str(training_cfg.get("progress_style", "percent")).lower()
        if self.progress_style not in {"bar", "percent", "off"}:
            self.progress_style = "percent"
        self.display_interval = int(training_cfg.get("display_interval", 100))
        self.log_json = bool(training_cfg.get("log_json", False))
        self.num_epochs = int(training_cfg.get("num_epochs", 1))
        self.domain_names = list(config.get("data", {}).get("domains", []))

        # Step-based LR schedule (optional)
        self.global_step = 0
        self.steps_per_epoch = int(len(self.loader))
        self.total_steps = int(self.steps_per_epoch * self.num_epochs)
        sched = training_cfg.get("lr_schedule", "none")
        self.lr_schedule = str(sched).lower().strip() if sched is not None else "none"
        self.warmup_steps = int(training_cfg.get("warmup_steps", 0))
        self.min_lr_ratio = float(training_cfg.get("min_lr_ratio", 0.0))

        self.sched_g = None
        self.sched_d = None
        if self.lr_schedule not in {"none", "off", ""}:
            def lr_factor(step: int) -> float:
                total = max(1, int(self.total_steps))
                warm = max(0, int(self.warmup_steps))
                min_r = float(self.min_lr_ratio)
                if warm > 0 and step < warm:
                    return float(step + 1) / float(warm)
                if self.lr_schedule == "cosine":
                    denom = max(1, total - warm)
                    progress = min(max((step - warm) / denom, 0.0), 1.0)
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return min_r + (1.0 - min_r) * cosine
                if self.lr_schedule == "linear":
                    denom = max(1, total - warm)
                    progress = min(max((step - warm) / denom, 0.0), 1.0)
                    return min_r + (1.0 - min_r) * (1.0 - progress)
                # Fallback: constant
                return 1.0

            self.sched_d = torch.optim.lr_scheduler.LambdaLR(self.opt_d, lr_lambda=lambda _: lr_factor(self.global_step))
            self.sched_g = torch.optim.lr_scheduler.LambdaLR(self.opt_g, lr_lambda=lambda _: lr_factor(self.global_step))

        if AutoencoderKL is None:
            raise ImportError("diffusers is required for VAE visualization. Please install diffusers.")

        vae_cfg = config.get("visualization", {})
        self.visualize_every = int(
            vae_cfg.get("every_epochs", training_cfg.get("eval_interval", self.save_interval))
        )
        vis_save_dir = vae_cfg.get("save_dir", "")
        self.vis_dir = (Path(vis_save_dir) if vis_save_dir else (self.save_dir / "vis"))
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        if bool(vae_cfg.get("suppress_hf_warnings", True)):
            warnings.filterwarnings(
                "ignore",
                message="The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`.*",
                category=UserWarning,
            )
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

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
            "global_step": int(self.global_step),
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
        self.global_step = int(ckpt.get("global_step", 0))
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

    def _get_eval_domain_latents(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (content_raw, style_ref_raw) each with shape [N,4,H,W], one sample per domain.

        Prefer pulling directly from dataset.domain_tensors to guarantee full domain coverage.
        Fallback to a loader batch if dataset interface is unknown.
        """
        dataset = getattr(self.loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "domain_tensors") and hasattr(dataset, "num_domains"):
            num_domains = int(getattr(dataset, "num_domains"))
            latents = []
            for domain_id in range(num_domains):
                domain_bank = dataset.domain_tensors[domain_id]
                latents.append(domain_bank[0])
            content_raw = torch.stack(latents, dim=0)
            style_ref_raw = torch.stack(latents, dim=0)
            return content_raw, style_ref_raw

        batch = self._get_eval_batch()
        content_raw = batch["content"]
        style_ref_raw = batch["target_style"]
        return content_raw, style_ref_raw

    def _decode_latent_to_image(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.to(self.device, dtype=self.vae_dtype)
        with torch.no_grad():
            decoded = self.vae.decode(latent).sample
        decoded = decoded.float().clamp(-1.0, 1.0)
        return (decoded + 1.0) / 2.0

    def _save_grid_with_labels(
        self,
        grid_tensor: torch.Tensor,
        domain_names: list[str],
        out_path: Path,
        n_vis: int,
        cell_hw: tuple[int, int],
        padding: int,
    ) -> None:
        if Image is None or ImageDraw is None or ImageFont is None:
            return

        if n_vis <= 0:
            return

        # grid_tensor: [3, H, W] in [0,1]
        grid_u8 = (grid_tensor.detach().cpu().clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        grid_u8 = grid_u8.permute(1, 2, 0).contiguous().numpy()
        grid_img = Image.fromarray(grid_u8)

        font_size = 18
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        left_margin = max(80, int(font_size * 4.5))
        top_margin = max(26, int(font_size * 1.6))
        canvas = Image.new("RGB", (grid_img.width + left_margin, grid_img.height + top_margin), (0, 0, 0))
        canvas.paste(grid_img, (left_margin, top_margin))
        draw = ImageDraw.Draw(canvas)

        cell_h, cell_w = cell_hw

        def cell_center(col: int, row: int) -> tuple[int, int]:
            # col,row are indices in the (n_vis+1)x(n_vis+1) grid.
            x0 = left_margin + padding + col * (cell_w + padding)
            y0 = top_margin + padding + row * (cell_h + padding)
            return (int(x0 + cell_w / 2), int(y0 + cell_h / 2))

        # Column labels (targets) on top header row.
        for j in range(n_vis):
            name = domain_names[j] if j < len(domain_names) else str(j)
            x, _ = cell_center(col=j + 1, row=0)
            y = int(top_margin / 2)
            draw.text((x, y), name, fill=(255, 255, 255), font=font, anchor="mm")

        # Row labels (sources) on left header col.
        for i in range(n_vis):
            name = domain_names[i] if i < len(domain_names) else str(i)
            x = int(left_margin / 2)
            _, y = cell_center(col=0, row=i + 1)
            draw.text((x, y), name, fill=(255, 255, 255), font=font, anchor="mm")

        canvas.save(out_path)

    def evaluate(self, epoch: int) -> None:
        self.G.eval()
        self.E.eval()

        content_raw, style_ref_raw = self._get_eval_domain_latents()
        content_raw = content_raw.to(self.device, non_blocking=True)
        style_ref_raw = style_ref_raw.to(self.device, non_blocking=True)

        num_domains = int(self.cfg["model"]["num_domains"])
        n_vis = min(num_domains, int(self.eval_num_samples))
        content_raw = content_raw[:n_vis]
        style_ref_raw = style_ref_raw[:n_vis]

        content_scaled = content_raw * self.latent_scale
        style_ref_scaled = style_ref_raw * self.latent_scale

        content_for_decode = content_scaled / self.latent_scale
        style_for_decode = style_ref_scaled / self.latent_scale

        content_img = self._decode_latent_to_image(content_for_decode)
        style_img = self._decode_latent_to_image(style_for_decode)

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
                dtype=self.autocast_dtype,
            ):
                # Precompute one style vector per target domain.
                target_styles = []
                for j in range(n_vis):
                    yj = torch.full((1,), j, device=self.device, dtype=torch.long)
                    sj = self.E(style_ref_scaled[j : j + 1], yj)
                    target_styles.append(sj)

                # Generate translations: for each target j, translate all sources i.
                translated_latents = []
                for j in range(n_vis):
                    sj = target_styles[j].expand(n_vis, -1)
                    fake_scaled = self.G(content_scaled, sj)
                    translated_latents.append(fake_scaled)

        # translated_latents[j] has shape [n_vis,4,H,W] for target j
        # Convert to images and form matrix (including same-domain i->i generation)
        translated_imgs = []
        for j in range(n_vis):
            fake_for_decode = translated_latents[j] / self.latent_scale
            translated_imgs.append(self._decode_latent_to_image(fake_for_decode))

        blank = torch.zeros_like(content_img[0])
        grid_cells = [blank]

        # Header row: target domain reference images
        grid_cells.extend(style_img)

        # Rows: source content header + translations
        for i in range(n_vis):
            grid_cells.append(content_img[i])
            for j in range(n_vis):
                # if i == j:
                #     grid_cells.append(content_img[i])
                # else:
                grid_cells.append(translated_imgs[j][i])

        padding = 4
        grid = make_grid(torch.stack(grid_cells, dim=0), nrow=n_vis + 1, padding=padding)
        out_path = self.vis_dir / f"epoch_{epoch:04d}.png"
        save_image(grid, out_path)

        # Overwrite with labeled version if Pillow is available.
        names = self.domain_names[:n_vis] if self.domain_names else [str(i) for i in range(n_vis)]
        cell_hw = (int(content_img.shape[-2]), int(content_img.shape[-1]))
        self._save_grid_with_labels(grid, names, out_path, n_vis=n_vis, cell_hw=cell_hw, padding=padding)

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
            "id": 0.0,
            "r1": 0.0,
        }
        num_steps = 0

        ds_weight = self._lambda_ds(epoch)
        loss_cfg = self.cfg["loss"]
        w_adv = float(loss_cfg.get("w_adv", 1.0))
        w_sty = float(loss_cfg.get("w_sty", 1.0))
        w_cyc = float(loss_cfg.get("w_cyc", 1.0))
        w_id = float(loss_cfg.get("w_id", 0.0))

        total_steps = len(self.loader)
        use_tqdm = self.show_progress and self.progress_style == "bar"
        epoch_start_t = time.time()

        def _fmt_hms(seconds: float) -> str:
            seconds = max(0.0, float(seconds))
            m, s = divmod(int(seconds + 0.5), 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f"{h:d}:{m:02d}:{s:02d}"
            return f"{m:02d}:{s:02d}"

        def _now_str() -> str:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if use_tqdm:
            epoch_iter = tqdm(
                self.loader,
                desc=f"Epoch {epoch}/{self.num_epochs}",
                total=total_steps,
                dynamic_ncols=True,
                leave=True,
            )
            iterator = enumerate(epoch_iter, start=1)
        else:
            epoch_iter = None
            iterator = enumerate(self.loader, start=1)

        for step, batch in iterator:
            try:
                if self.device.type == "cuda" and hasattr(torch, "compiler"):
                    mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
                    if callable(mark_step_begin):
                        mark_step_begin()

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
                    s_ref = self.E(x_ref, y_trg)
                    s_lat = self.F(z_trg, y_trg)

                    fake_ref = self.G(content, s_ref)
                    fake_lat = self.G(content, s_lat)

                    real_logits = self.D(x_ref, y_trg)
                    fake_ref_logits_d = self.D(fake_ref.detach(), y_trg)
                    fake_lat_logits_d = self.D(fake_lat.detach(), y_trg)

                    d_loss_ref = d_hinge_loss(real_logits, fake_ref_logits_d)
                    d_loss_lat = d_hinge_loss(real_logits, fake_lat_logits_d)
                    d_loss = 0.5 * (d_loss_ref + d_loss_lat)

                do_r1 = (step % self.r1_interval == 0)
                if do_r1:
                    x_ref_r1 = x_ref.detach().requires_grad_(True)
                    with torch.amp.autocast(device_type=self.device.type, enabled=False):
                        real_logits_r1 = self.D(x_ref_r1.float(), y_trg)
                        r1_loss = r1_penalty(real_logits_r1, x_ref_r1)
                    d_loss = d_loss + (self.w_r1 * self.r1_interval) * r1_loss
                else:
                    r1_loss = torch.zeros((), device=self.device)

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
                    ds_loss = diversity_sensitive_loss(fake_1, fake_2, margin=self.ds_margin)

                    s_src = self.E(content, y_src)
                    rec_ref = self.G(fake_ref, s_src)
                    rec_lat = self.G(fake_lat, s_src)
                    cyc_ref = cycle_consistency_loss(rec_ref, content)
                    cyc_lat = cycle_consistency_loss(rec_lat, content)
                    cyc_loss = cyc_ref + cyc_lat

                    if w_id > 0.0:
                        id_out = self.G(content, s_src)
                        id_loss = nnF.l1_loss(id_out, content)
                    else:
                        id_loss = torch.zeros((), device=self.device)

                    g_loss = (
                        w_adv * g_adv
                        + w_sty * sty_loss
                        + w_cyc * cyc_loss
                        + w_id * id_loss
                        - ds_weight * ds_loss
                    )

                self.opt_g.zero_grad(set_to_none=True)
                if self.scaler_g.is_enabled():
                    self.scaler_g.scale(g_loss).backward()
                    self.scaler_g.step(self.opt_g)
                    self.scaler_g.update()
                else:
                    g_loss.backward()
                    self.opt_g.step()

                # Step-based LR schedule (one step per batch)
                self.global_step += 1
                if self.sched_d is not None:
                    self.sched_d.step()
                if self.sched_g is not None:
                    self.sched_g.step()

                logs = self._detach_log(
                    {
                        "d_loss": d_loss,
                        "g_loss": g_loss,
                        "g_adv": g_adv,
                        "sty": sty_loss,
                        "ds": ds_loss,
                        "cyc": cyc_loss,
                        "id": id_loss,
                        "r1": r1_loss,
                    }
                )

                # Add LR to logs for debugging/plotting (current optimizer lr after stepping)
                logs["lr_d"] = float(self.opt_d.param_groups[0]["lr"])
                logs["lr_g"] = float(self.opt_g.param_groups[0]["lr"])

                for key in running:
                    running[key] += logs[key]

                num_steps += 1

                if step % self.log_interval == 0 and self.log_json:
                    printable = {k: round(v, 5) for k, v in logs.items()}
                    printable["epoch"] = epoch
                    printable["step"] = step
                    printable["lambda_ds"] = round(ds_weight, 6)
                    printable["time"] = _now_str()
                    print(json.dumps(printable, ensure_ascii=False))

                if step % self.display_interval == 0:
                    progress = 100.0 * step / max(1, total_steps)
                    elapsed = time.time() - epoch_start_t
                    step_avg = elapsed / max(1, step)
                    eta = step_avg * max(0, total_steps - step)
                    t_pair = f"{_fmt_hms(elapsed)}/{_fmt_hms(eta)}"
                    ts = _now_str()
                    if use_tqdm:
                        epoch_iter.set_postfix(
                            pct=f"{progress:.1f}%",
                            t=t_pair,
                            d=round(logs["d_loss"], 4),
                            g=round(logs["g_loss"], 4),
                            r1=round(logs["r1"], 4),
                        )
                    elif self.progress_style == "percent":
                        msg = (
                            f"\r{ts} | Epoch {epoch}/{self.num_epochs} | "
                            f"{progress:6.2f}% | t={t_pair} | "
                            f"d={logs['d_loss']:.4f} g={logs['g_loss']:.4f} id={logs['id']:.4f} r1={logs['r1']:.4f}"
                        )
                        print(msg, end="", flush=True)

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if use_tqdm:
                        epoch_iter.write(f"[OOM] epoch={epoch}, step={step}. batch skipped.")
                    else:
                        print(f"\n[OOM] epoch={epoch}, step={step}. batch skipped.")
                    self.opt_d.zero_grad(set_to_none=True)
                    self.opt_g.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

        if use_tqdm:
            epoch_iter.close()
        elif self.progress_style == "percent":
            print()

        if num_steps == 0:
            mean_logs = {k: float("nan") for k in running}
        else:
            mean_logs = {k: v / num_steps for k, v in running.items()}

        if epoch % self.save_interval == 0:
            self._save_checkpoint(epoch)
        if epoch % self.visualize_every == 0:
            self.evaluate(epoch)

        return mean_logs
