import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from train_latent_cyclegan import (
    ResnetGenerator,
    _load_vae,
    decode_latents_to_images,
    load_latent_tensor,
    load_yaml_config,
    resolve_vae_source,
)


@dataclass
class ExportConfig:
    shared_cfg: Dict[str, Any]
    model_cfg: Dict[str, Any]
    data_cfg: Dict[str, Any]
    vis_cfg: Dict[str, Any]
    export_cfg: Dict[str, Any]
    eval_cfg: Dict[str, Any]
    domain_A_name: str
    domain_B_name: str
    base_latent_dir: Path
    base_orig_rgb_dir: Path
    base_vae_recon_dir: Path
    model_path: Path
    output_dir: Path
    eval_image_size: int
    batch_size: int
    run_device: torch.device
    use_bf16_autocast: bool


@dataclass
class ExportDirection:
    name: str
    src_domain: str
    tgt_domain: str
    src_dir: Path
    orig_rgb_dir: Path
    vae_recon_dir: Path
    out_dir: Path


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _torch_load_trusted(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def _is_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and len(obj) > 0 and all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items()
    )


def _pick_state_dict(payload: Dict[str, Any], keys: Sequence[str]) -> Optional[Dict[str, torch.Tensor]]:
    for key in keys:
        value = payload.get(key)
        if _is_state_dict(value):
            return value
    return None


def _resolve_domain_dir(base_dir: Path, domain_name: str, key_name: str) -> Path:
    path = base_dir / domain_name
    _require(path.exists(), f"{key_name} not found for domain '{domain_name}': {path}")
    return path


def _list_latents(root: Path) -> List[Path]:
    _require(root.exists(), f"latent directory not found: {root}")
    files = sorted([p for p in root.rglob("*.pt") if p.is_file()])
    _require(len(files) > 0, f"no latent .pt files found in: {root}")
    return files


def _build_stem_index(root: Path) -> Dict[str, Path]:
    files: List[Path] = []
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    index: Dict[str, Path] = {}
    for path in sorted(p for p in files if p.is_file()):
        index[path.stem] = path
    return index


def _pil_rgb(path: Path, image_size: Optional[int] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return img


def _make_3panel(orig: Image.Image, recon: Image.Image, transfer: Image.Image, stem: str) -> Image.Image:
    w = orig.width
    h = orig.height
    label_h = 28
    canvas = Image.new("RGB", (w * 3, h + label_h), (245, 245, 245))
    canvas.paste(orig, (0, label_h))
    canvas.paste(recon, (w, label_h))
    canvas.paste(transfer, (w * 2, label_h))

    draw = ImageDraw.Draw(canvas)
    for i, txt in enumerate(["Orig_Img", "Recon_Img", "Transferred_Img"]):
        draw.text((10 + i * w, 6), txt, fill=(20, 20, 20))
    draw.text((10, h + 8), stem, fill=(20, 20, 20))
    return canvas


def _make_generator(model_cfg: Dict[str, Any]) -> ResnetGenerator:
    return ResnetGenerator(
        in_ch=int(model_cfg.get("in_channels", 4)),
        out_ch=int(model_cfg.get("out_channels", 4)),
        ngf=int(model_cfg.get("ngf", 256)),
        n_res_blocks=int(model_cfg.get("n_res_blocks", 9)),
        out_activation=str(model_cfg.get("out_activation", "none")),
    )


def _load_generators(model_path: Path, model_cfg: Dict[str, Any], device: torch.device) -> Dict[str, torch.nn.Module]:
    _require(model_path.exists(), f"model checkpoint not found: {model_path}")
    payload = _torch_load_trusted(model_path, map_location="cpu")

    sd_g: Optional[Dict[str, torch.Tensor]] = None
    sd_f: Optional[Dict[str, torch.Tensor]] = None

    if _is_state_dict(payload):
        sd_g = payload
    elif isinstance(payload, dict):
        sd_g = _pick_state_dict(payload, ("G", "netG_A", "generator", "state_dict", "model"))
        sd_f = _pick_state_dict(payload, ("F", "netG_B"))
    else:
        raise ValueError(f"unsupported checkpoint payload type: {type(payload)}")

    _require(sd_g is not None, f"cannot locate A2B generator weights in: {model_path}")
    _require(sd_f is not None, f"cannot locate B2A generator weights (F/netG_B) in: {model_path}")

    g_ab = _make_generator(model_cfg)
    g_ba = _make_generator(model_cfg)

    missing_g, unexpected_g = g_ab.load_state_dict(sd_g, strict=False)
    missing_f, unexpected_f = g_ba.load_state_dict(sd_f, strict=False)
    if missing_g or unexpected_g:
        _log(
            f"[warn] A2B load_state_dict strict=False: "
            f"missing={len(missing_g)} unexpected={len(unexpected_g)}"
        )
    if missing_f or unexpected_f:
        _log(
            f"[warn] B2A load_state_dict strict=False: "
            f"missing={len(missing_f)} unexpected={len(unexpected_f)}"
        )

    g_ab.to(device).eval()
    g_ba.to(device).eval()
    return {"A2B": g_ab, "B2A": g_ba}


def _phase0_parse_config(args: argparse.Namespace) -> ExportConfig:
    cfg = load_yaml_config(args.config)

    shared_cfg = cfg.get("shared", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    vis_cfg = cfg.get("visualization", {})
    export_cfg = cfg.get("image_export", {})
    eval_cfg = cfg.get("cyclegan_eval", {})

    _require(isinstance(shared_cfg, dict), "config.shared must be a dict")
    _require(isinstance(model_cfg, dict), "config.model must be a dict")
    _require(isinstance(data_cfg, dict), "config.data must be a dict")
    _require(isinstance(vis_cfg, dict), "config.visualization must be a dict")
    _require(isinstance(export_cfg, dict), "config.image_export must be a dict")
    _require(isinstance(eval_cfg, dict), "config.cyclegan_eval must be a dict")

    domain_A_name = str(export_cfg.get("domain_A_name", eval_cfg.get("domain_A_name", "")) or "").strip()
    domain_B_name = str(export_cfg.get("domain_B_name", eval_cfg.get("domain_B_name", "")) or "").strip()
    base_latent_dir = Path(args.base_latent_dir or export_cfg.get("base_latent_dir", eval_cfg.get("base_latent_dir", "")) or "")
    base_orig_rgb_dir = Path(export_cfg.get("base_orig_rgb_dir", eval_cfg.get("base_orig_rgb_dir", "")) or "")
    base_vae_recon_dir = Path(export_cfg.get("base_vae_recon_dir", eval_cfg.get("base_vae_recon_dir", "")) or "")
    model_path = Path(args.model_path or export_cfg.get("model_path", "") or "")
    output_dir = Path(args.output_dir or export_cfg.get("output_dir", "outputs/exported_images"))
    batch_size = int(args.batch_size or export_cfg.get("batch_size", 8))
    eval_image_size = int(export_cfg.get("eval_image_size", eval_cfg.get("eval_image_size", 256)))

    device_str = str(args.run_device or export_cfg.get("run_device", "cuda") or "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    run_device = torch.device(device_str)
    use_bf16_autocast = bool(
        args.use_bf16_autocast if args.use_bf16_autocast is not None else export_cfg.get("use_bf16_autocast", True)
    )

    _require(domain_A_name, "image_export.domain_A_name is required")
    _require(domain_B_name, "image_export.domain_B_name is required")
    _require(str(base_latent_dir), "image_export.base_latent_dir is required")
    _require(str(base_orig_rgb_dir), "image_export.base_orig_rgb_dir or cyclegan_eval.base_orig_rgb_dir is required")
    _require(str(base_vae_recon_dir), "image_export.base_vae_recon_dir or cyclegan_eval.base_vae_recon_dir is required")
    _require(str(model_path), "image_export.model_path is required")
    _require(batch_size > 0, "image_export.batch_size must be > 0")
    _require(eval_image_size > 0, "image_export.eval_image_size must be > 0")
    _require(base_latent_dir.exists(), f"base_latent_dir not found: {base_latent_dir}")
    _require(base_orig_rgb_dir.exists(), f"base_orig_rgb_dir not found: {base_orig_rgb_dir}")
    _require(base_vae_recon_dir.exists(), f"base_vae_recon_dir not found: {base_vae_recon_dir}")

    _resolve_domain_dir(base_latent_dir, domain_A_name, "base_latent_dir")
    _resolve_domain_dir(base_latent_dir, domain_B_name, "base_latent_dir")
    _resolve_domain_dir(base_orig_rgb_dir, domain_A_name, "base_orig_rgb_dir")
    _resolve_domain_dir(base_orig_rgb_dir, domain_B_name, "base_orig_rgb_dir")
    _resolve_domain_dir(base_vae_recon_dir, domain_A_name, "base_vae_recon_dir")
    _resolve_domain_dir(base_vae_recon_dir, domain_B_name, "base_vae_recon_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    return ExportConfig(
        shared_cfg=shared_cfg,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        vis_cfg=vis_cfg,
        export_cfg=export_cfg,
        eval_cfg=eval_cfg,
        domain_A_name=domain_A_name,
        domain_B_name=domain_B_name,
        base_latent_dir=base_latent_dir,
        base_orig_rgb_dir=base_orig_rgb_dir,
        base_vae_recon_dir=base_vae_recon_dir,
        model_path=model_path,
        output_dir=output_dir,
        eval_image_size=eval_image_size,
        batch_size=batch_size,
        run_device=run_device,
        use_bf16_autocast=use_bf16_autocast,
    )


def _build_directions(ecfg: ExportConfig) -> List[ExportDirection]:
    def _mk(name: str, src_domain: str, tgt_domain: str) -> ExportDirection:
        src_dir = _resolve_domain_dir(ecfg.base_latent_dir, src_domain, "base_latent_dir")
        orig_rgb_dir = _resolve_domain_dir(ecfg.base_orig_rgb_dir, src_domain, "base_orig_rgb_dir")
        vae_recon_dir = _resolve_domain_dir(ecfg.base_vae_recon_dir, src_domain, "base_vae_recon_dir")
        out_dir = ecfg.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        return ExportDirection(
            name=name,
            src_domain=src_domain,
            tgt_domain=tgt_domain,
            src_dir=src_dir,
            orig_rgb_dir=orig_rgb_dir,
            vae_recon_dir=vae_recon_dir,
            out_dir=out_dir,
        )

    return [
        _mk("A2B", ecfg.domain_A_name, ecfg.domain_B_name),
        _mk("B2A", ecfg.domain_B_name, ecfg.domain_A_name),
    ]


def _save_tensor_as_png(img_tensor: torch.Tensor, out_path: Path) -> None:
    img = img_tensor.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
    img = img.permute(1, 2, 0).contiguous().numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)


def _get_module_param_dtype(module: torch.nn.Module) -> torch.dtype:
    for param in module.parameters():
        return param.dtype
    return torch.float32


def _infer_direction(
    ecfg: ExportConfig,
    direction: ExportDirection,
    generator: torch.nn.Module,
    vae: torch.nn.Module,
) -> None:
    latent_paths = _list_latents(direction.src_dir)
    orig_index = _build_stem_index(direction.orig_rgb_dir)
    recon_index = _build_stem_index(direction.vae_recon_dir)
    latents_scaled = bool(ecfg.data_cfg.get("latents_scaled", False))
    latent_divisor = float(ecfg.data_cfg.get("latent_divisor", 1.0))
    vae_scaling_factor = float(ecfg.vis_cfg.get("vae_scaling_factor", 0.18215))
    _require(latent_divisor > 0, "config.data.latent_divisor must be > 0")

    _log(
        f"[{direction.name}] src={direction.src_domain} tgt={direction.tgt_domain} "
        f"latents={len(latent_paths)} out_dir={direction.out_dir}"
    )

    for start in tqdm(
        range(0, len(latent_paths), ecfg.batch_size),
        desc=f"{direction.name} export",
        unit="batch",
    ):
        batch_paths = latent_paths[start : start + ecfg.batch_size]
        latent_batch = torch.stack([load_latent_tensor(path) for path in batch_paths], dim=0)
        latent_in = latent_batch / latent_divisor if not math.isclose(latent_divisor, 1.0) else latent_batch
        latent_in = latent_in.to(ecfg.run_device)

        with torch.no_grad():
            if ecfg.run_device.type == "cuda":
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=bool(ecfg.use_bf16_autocast),
                    dtype=torch.bfloat16,
                ):
                    fake_latent = generator(latent_in)
            else:
                fake_latent = generator(latent_in)

            # Keep VAE decode input aligned with VAE weights to avoid bf16/float32 mismatches.
            fake_latent = fake_latent.to(
                device=ecfg.run_device,
                dtype=_get_module_param_dtype(vae),
            )

            fake_images = decode_latents_to_images(
                vae=vae,
                latents=fake_latent,
                latents_scaled=latents_scaled,
                scaling_factor=vae_scaling_factor,
                latent_divisor=latent_divisor,
            )

        for src_path, fake_img in zip(batch_paths, fake_images, strict=False):
            relative_path = src_path.relative_to(direction.src_dir).with_suffix(".png")
            out_path = direction.out_dir / relative_path
            stem = src_path.stem
            orig_path = orig_index.get(stem)
            recon_path = recon_index.get(stem)
            if orig_path is None or recon_path is None:
                _log(f"[warn] [{direction.name}] missing orig/recon for stem={stem}, skip")
                continue

            transfer_img = Image.fromarray(
                fake_img.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().permute(1, 2, 0).contiguous().numpy()
            ).resize((ecfg.eval_image_size, ecfg.eval_image_size), Image.Resampling.BICUBIC)
            orig_img = _pil_rgb(orig_path, image_size=ecfg.eval_image_size)
            recon_img = _pil_rgb(recon_path, image_size=ecfg.eval_image_size)
            panel = _make_3panel(orig_img, recon_img, transfer_img, stem)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(out_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export A2B/B2A translated RGB images from latent .pt files")
    parser.add_argument("--config", type=str, default="configs/example.yaml")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--base_latent_dir", type=str, default="")
    parser.add_argument("--run_device", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--use_bf16_autocast", action="store_true", default=None)
    parser.add_argument("--no_use_bf16_autocast", dest="use_bf16_autocast", action="store_false")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    _log("[Phase0] Parsing config...")
    ecfg = _phase0_parse_config(args)

    _log("[Setup] Loading generators...")
    generators = _load_generators(ecfg.model_path, ecfg.model_cfg, ecfg.run_device)

    _log("[Setup] Loading VAE...")
    vae_name, vae_subfolder = resolve_vae_source({"shared": ecfg.shared_cfg}, ecfg.vis_cfg)
    vae = _load_vae(
        model_name_or_path=vae_name or "runwayml/stable-diffusion-v1-5",
        subfolder=vae_subfolder,
        device=ecfg.run_device,
    )

    for direction in _build_directions(ecfg):
        generator = generators[direction.name]
        _infer_direction(ecfg, direction, generator, vae)

    _log(f"[Done] Exported translated images to: {ecfg.output_dir}")


if __name__ == "__main__":
    sys.exit(main())
