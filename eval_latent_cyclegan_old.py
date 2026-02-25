import argparse
import csv
import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'pyyaml'. Install: pip install pyyaml") from e

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping/dict at the top level")
    return cfg


def _list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    paths: List[Path] = []
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return sorted(paths)


def _load_image_tensor(path: Path, size: int) -> torch.Tensor:
    from PIL import Image
    import torchvision.transforms as T

    img = Image.open(path).convert("RGB").resize((size, size))
    x = T.ToTensor()(img)  # [0,1]
    # match utils/make_pt_dataset_sd15_vae.py: normalize to [-1,1]
    x = x * 2.0 - 1.0
    return x


@torch.no_grad()
def _load_vae(model_name_or_path: str, subfolder: Optional[str], device: torch.device) -> nn.Module:
    try:
        from diffusers import AutoencoderKL  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'diffusers'. Install: pip install diffusers transformers accelerate safetensors"
        ) from e

    kwargs: Dict[str, Any] = {}
    if subfolder:
        kwargs["subfolder"] = subfolder

    vae = AutoencoderKL.from_pretrained(model_name_or_path, **kwargs)
    vae.to(device)
    vae.eval()
    return vae


@torch.no_grad()
def encode_images_to_latents(
    vae: nn.Module,
    images: torch.Tensor,
    latents_scaled: bool,
    scaling_factor: float,
) -> torch.Tensor:
    """images: [-1,1] in shape [B,3,H,W]. Returns latents [B,4,h,w].

    diffusers VAE returns *unscaled* latents; if training expects scaled latents,
    multiply by scaling_factor to match.
    """
    lat = vae.encode(images).latent_dist.sample()  # type: ignore[attr-defined]
    if latents_scaled:
        lat = lat * float(scaling_factor)
    return lat


@torch.no_grad()
def decode_latents_to_images(
    vae: nn.Module,
    latents: torch.Tensor,
    latents_scaled: bool,
    scaling_factor: float,
) -> torch.Tensor:
    """latents: [B,4,H,W] in training's latent scale. Returns images [B,3,h,w] in [0,1]."""
    x = latents
    if latents_scaled:
        x = x / float(scaling_factor)
    decoded = vae.decode(x).sample  # type: ignore[attr-defined]
    images = (decoded / 2 + 0.5).clamp(0, 1)
    return images


class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        in_ch: int = 4,
        out_ch: int = 4,
        ngf: int = 32,
        n_res_blocks: int = 6,
        out_activation: str = "none",
    ):
        super().__init__()
        layers: List[nn.Module] = []

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ngf, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        ]

        mult = 1
        for _ in range(2):
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
            ]
            mult *= 2

        for _ in range(n_res_blocks):
            layers += [ResnetBlock(ngf * mult)]

        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    ngf * mult // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(ngf * mult // 2, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
            ]
            mult //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, kernel_size=7, stride=1, padding=0, bias=True),
        ]

        if out_activation.lower() == "tanh":
            layers += [nn.Tanh()]
        elif out_activation.lower() in ("none", "identity", "linear", ""):
            pass
        else:
            raise ValueError(f"Unsupported out_activation: {out_activation}")

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_ckpt(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint: {path}")
    # train_latent_cyclegan.py stores config under 'cfg'
    if "cfg" not in payload and "config" in payload:
        payload["cfg"] = payload.get("config")
    return payload


def _extract_clip_embeddings(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if isinstance(output, dict):
        if "image_embeds" in output:
            return output["image_embeds"]
        if "pooler_output" in output:
            return output["pooler_output"]
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError(f"Could not extract CLIP embeddings from output type={type(output)}")


@dataclass
class EvalConfig:
    image_size: int = 256
    batch_size: int = 8
    max_src_samples: int = 0
    max_ref_cache: int = 256
    max_ref_compare: int = 50
    cache_dir: str = "outputs/eval_cache"
    force_regen_cache: bool = False
    disable_lpips: bool = False
    disable_clip: bool = False
    clip_model_id: str = "openai/clip-vit-base-patch32"
    clip_cache_dir: str = ""
    clip_local_files_only: bool = False
    clip_use_safetensors: bool = True
    compact_paths: bool = False
    amp_bf16: bool = True
    device: str = "cuda"


def _hash_dir(path: Path) -> str:
    return hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:10]


@torch.no_grad()
def _compute_ref_clip_matrix(
    ref_paths: Sequence[Path],
    clip_model: Any,
    clip_processor: Any,
    device: torch.device,
    image_size: int,
    max_ref_cache: int,
) -> Optional[torch.Tensor]:
    from PIL import Image

    if max_ref_cache > 0:
        ref_paths = ref_paths[: min(len(ref_paths), int(max_ref_cache))]
    if not ref_paths:
        return None

    embs: List[torch.Tensor] = []
    bs = 32
    for i in range(0, len(ref_paths), bs):
        batch_paths = ref_paths[i : i + bs]
        batch_pils = [Image.open(p).convert("RGB").resize((image_size, image_size)) for p in batch_paths]
        inputs = clip_processor(images=batch_pils, return_tensors="pt").to(device)
        out = clip_model.get_image_features(**inputs)
        e = _extract_clip_embeddings(out).to(device, dtype=torch.float32)
        if e.ndim == 1:
            e = e.unsqueeze(0)
        e = e / (e.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        embs.append(e.detach().cpu())

    mat = torch.cat(embs, dim=0)
    return mat


def _to_lpips_input(img01: torch.Tensor) -> torch.Tensor:
    # lpips expects [-1,1]
    return img01 * 2.0 - 1.0


def _load_lpips(device: torch.device):
    try:
        import lpips  # type: ignore

        return lpips.LPIPS(net="vgg", verbose=False).to(device)
    except Exception:
        return None


def _load_clip(device: torch.device):
    model, processor, _ = _load_clip_with_cfg(device=device)
    return model, processor


def _load_clip_with_cfg(
    *,
    device: torch.device,
    model_id: str = "openai/clip-vit-base-patch32",
    cache_dir: str = "",
    local_files_only: bool = False,
    use_safetensors: bool = True,
) -> Tuple[Any, Any, Optional[str]]:
    try:
        from transformers import CLIPModel, CLIPProcessor  # type: ignore
    except ImportError as e:
        return None, None, f"transformers not installed ({e})"
    except Exception as e:
        return None, None, f"failed to import transformers ({e})"

    kwargs: Dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if local_files_only:
        kwargs["local_files_only"] = True
    # Prefer safetensors to avoid torch.load restrictions in newer transformers.
    kwargs["use_safetensors"] = bool(use_safetensors)

    try:
        try:
            model = CLIPModel.from_pretrained(model_id, **kwargs).to(device)
        except TypeError:
            # Older transformers: retry without use_safetensors.
            kwargs.pop("use_safetensors", None)
            model = CLIPModel.from_pretrained(model_id, **kwargs).to(device)
        processor = CLIPProcessor.from_pretrained(model_id, **kwargs)
        model.eval()
        return model, processor, None
    except Exception as e:
        msg = str(e)
        if "serious vulnerability issue" in msg and "upgrade torch" in msg:
            return (
                None,
                None,
                "torch is too old for loading .bin weights in this transformers version; "
                "either upgrade torch>=2.6 OR enable safetensors (pip install safetensors and set eval.clip_use_safetensors=true)"
                f" (original error: {e})",
            )
        return None, None, f"failed to load CLIP model '{model_id}' ({e})"


def _common_root(paths: Sequence[Path]) -> Optional[Path]:
    if not paths:
        return None
    try:
        root = Path(os.path.commonpath([str(p.resolve()) for p in paths]))
        if root.suffix:
            root = root.parent
        return root
    except Exception:
        return None


def _relpath_or_str(path: Path, root: Optional[Path]) -> str:
    if root is None:
        return str(path)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def _fmt_metric(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "NA"


def _save_side_by_side_with_metrics(
    *,
    out_path: Path,
    src01: torch.Tensor,
    gen01: torch.Tensor,
    direction: str,
    index: int,
    content_lpips: Optional[float],
    style_lpips: Optional[float],
    content_clip: Optional[float],
    style_clip: Optional[float],
) -> None:
    """Save a single PNG: left=source, right=generated, with header + 2x2 metric table."""

    from PIL import Image, ImageDraw, ImageFont
    import torchvision.transforms as T

    src_pil = T.ToPILImage()(src01.detach().float().cpu())
    gen_pil = T.ToPILImage()(gen01.detach().float().cpu())

    w, h = src_pil.size
    header_h = max(28, int(h * 0.10))
    table_h = max(88, int(h * 0.28))

    canvas = Image.new("RGB", (w * 2, header_h + h + table_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title = f"{direction}  #{index}"
    font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textlength(title, font=font), 12
    draw.text(((canvas.size[0] - tw) / 2, (header_h - th) / 2), title, fill=(0, 0, 0), font=font)

    canvas.paste(src_pil, (0, header_h))
    canvas.paste(gen_pil, (w, header_h))

    top = header_h + h
    draw.rectangle([0, top, w * 2 - 1, top + table_h - 1], outline=(0, 0, 0), width=1)
    draw.line([w, top, w, top + table_h], fill=(0, 0, 0), width=1)
    draw.line([0, top + table_h / 2, w * 2, top + table_h / 2], fill=(0, 0, 0), width=1)

    cells = [
        (0, 0, "LPIPS content", _fmt_metric(content_lpips)),
        (1, 0, "LPIPS style", _fmt_metric(style_lpips)),
        (0, 1, "CLIP content", _fmt_metric(content_clip)),
        (1, 1, "CLIP style", _fmt_metric(style_clip)),
    ]
    pad = 6
    cw = w
    ch = table_h / 2
    for cx, cy, label, val in cells:
        x0 = cx * cw
        y0 = top + cy * ch
        text = f"{label}\n{val}"
        draw.multiline_text((x0 + pad, y0 + pad), text, fill=(0, 0, 0), font=font, spacing=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def avg(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    return {
        "count": len(rows),
        "content_lpips": avg("content_lpips"),
        "style_lpips": avg("style_lpips"),
        "content_clip": avg("content_clip"),
        "style_clip": avg("style_clip"),
    }


def evaluate_direction(
    direction: str,
    src_paths: Sequence[Path],
    ref_paths: Sequence[Path],
    gen: nn.Module,
    vae: nn.Module,
    device: torch.device,
    latents_scaled: bool,
    vae_scaling_factor: float,
    latent_divisor: float,
    out_writer: csv.DictWriter,
    out_dir: Path,
    cfg: EvalConfig,
    save_images: bool,
    src_root: Optional[Path] = None,
    gen_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    _require(direction in ("A2B", "B2A"), "direction must be A2B or B2A")

    if cfg.max_src_samples > 0:
        src_paths = src_paths[: min(len(src_paths), int(cfg.max_src_samples))]

    images_dir = out_dir / "images" / direction
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    lpips_fn = None if cfg.disable_lpips else _load_lpips(device)
    if (not cfg.disable_lpips) and lpips_fn is None:
        print("[warn] LPIPS not available. Install: pip install lpips")

    clip_model, clip_processor = (None, None)
    if not cfg.disable_clip:
        clip_model, clip_processor, clip_err = _load_clip_with_cfg(
            device=device,
            model_id=str(cfg.clip_model_id or "openai/clip-vit-base-patch32"),
            cache_dir=str(cfg.clip_cache_dir or ""),
            local_files_only=bool(cfg.clip_local_files_only),
            use_safetensors=bool(cfg.clip_use_safetensors),
        )
        if clip_model is None:
            if clip_err and "not installed" in clip_err:
                print("[warn] CLIP not available: transformers is missing. Install: pip install transformers")
            else:
                print(
                    "[warn] CLIP disabled: model could not be loaded. "
                    + (clip_err or "")
                    + " (If offline, pre-download/cached model or set --clip_local_files_only)"
                )

    cache_root = Path(cfg.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = f"{direction}_{_hash_dir(Path(ref_paths[0]).parents[1] if ref_paths else out_dir)}_{len(ref_paths)}_{cfg.image_size}_{cfg.max_ref_cache}"
    cache_file = cache_root / f"ref_clip_{hashlib.md5(cache_key.encode('utf-8')).hexdigest()[:10]}.pt"

    ref_clip_cpu: Optional[torch.Tensor] = None
    if (not cfg.disable_clip) and clip_model is not None and clip_processor is not None and ref_paths:
        if cache_file.exists() and not cfg.force_regen_cache:
            try:
                ref_clip_cpu = torch.load(cache_file, map_location="cpu")
            except Exception:
                ref_clip_cpu = None
        if ref_clip_cpu is None:
            ref_clip_cpu = _compute_ref_clip_matrix(
                ref_paths,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device,
                image_size=cfg.image_size,
                max_ref_cache=cfg.max_ref_cache,
            )
            if ref_clip_cpu is not None:
                torch.save(ref_clip_cpu, cache_file)

    ref_clip_gpu = ref_clip_cpu.to(device, dtype=torch.float32) if ref_clip_cpu is not None else None

    rows_out: List[Dict[str, Any]] = []
    bs = max(1, int(cfg.batch_size))

    amp_ctx = nullcontext()
    if device.type == "cuda" and cfg.amp_bf16:
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)

    from PIL import Image
    import torchvision.transforms as T

    # Pre-load a small list of reference tensors for LPIPS style (on-the-fly per image)
    ref_for_lpips: List[Path] = list(ref_paths)
    if cfg.max_ref_compare > 0:
        ref_for_lpips = ref_for_lpips[: min(len(ref_for_lpips), int(cfg.max_ref_compare))]

    for start in range(0, len(src_paths), bs):
        batch_paths = src_paths[start : start + bs]
        src_batch = torch.stack([_load_image_tensor(p, cfg.image_size) for p in batch_paths], dim=0).to(device)

        with amp_ctx:
            lat_raw = encode_images_to_latents(
                vae,
                src_batch,
                latents_scaled=latents_scaled,
                scaling_factor=vae_scaling_factor,
            )
            lat_in = lat_raw
            if latent_divisor != 1.0:
                lat_in = lat_in / float(latent_divisor)

            lat_gen = gen(lat_in)

            lat_out = lat_gen
            if latent_divisor != 1.0:
                lat_out = lat_out * float(latent_divisor)

            img_gen = decode_latents_to_images(
                vae,
                lat_out,
                latents_scaled=latents_scaled,
                scaling_factor=vae_scaling_factor,
            )

        # src in [0,1] for metrics
        src01 = (src_batch / 2 + 0.5).clamp(0, 1)

        # content LPIPS + clip_content
        content_lpips_vals: List[Optional[float]] = [None] * len(batch_paths)
        content_clip_vals: List[Optional[float]] = [None] * len(batch_paths)

        if lpips_fn is not None:
            d = lpips_fn(_to_lpips_input(img_gen.float()), _to_lpips_input(src01.float()))
            content_lpips_vals = d.view(-1).detach().cpu().float().tolist()

        gen_clip = None
        src_clip = None
        if clip_model is not None and clip_processor is not None:
            # AMP bf16 can produce bfloat16 tensors; torchvision PIL conversion requires float32/uint8.
            pil_gen = [T.ToPILImage()(img_gen[i].detach().float().cpu()) for i in range(img_gen.shape[0])]
            pil_src = [T.ToPILImage()(src01[i].detach().float().cpu()) for i in range(src01.shape[0])]
            with torch.no_grad():
                inputs_gen = clip_processor(images=pil_gen, return_tensors="pt").to(device)
                out_gen = clip_model.get_image_features(**inputs_gen)
                gen_clip = _extract_clip_embeddings(out_gen).to(device, dtype=torch.float32)
                if gen_clip.ndim == 1:
                    gen_clip = gen_clip.unsqueeze(0)
                gen_clip = gen_clip / (gen_clip.norm(p=2, dim=-1, keepdim=True) + 1e-8)

                inputs_src = clip_processor(images=pil_src, return_tensors="pt").to(device)
                out_src = clip_model.get_image_features(**inputs_src)
                src_clip = _extract_clip_embeddings(out_src).to(device, dtype=torch.float32)
                if src_clip.ndim == 1:
                    src_clip = src_clip.unsqueeze(0)
                src_clip = src_clip / (src_clip.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            content_clip_vals = F.cosine_similarity(gen_clip, src_clip).detach().cpu().float().tolist()

        # per-image style metrics
        for i, p in enumerate(batch_paths):
            gen_rel = f"{p.stem}_{direction}.png"
            gen_path = images_dir / gen_rel

            style_lpips_val: Optional[float] = None
            style_clip_val: Optional[float] = None

            if lpips_fn is not None and ref_for_lpips:
                # average LPIPS to a subset of target refs
                # (smaller is better: closer to target style manifold)
                lpips_chunk = 8
                all_dists: List[torch.Tensor] = []
                for cs in range(0, len(ref_for_lpips), lpips_chunk):
                    ce = min(cs + lpips_chunk, len(ref_for_lpips))
                    chunk = ref_for_lpips[cs:ce]
                    ref_imgs = []
                    for rp in chunk:
                        img = Image.open(rp).convert("RGB").resize((cfg.image_size, cfg.image_size))
                        ref_imgs.append(T.ToTensor()(img))
                    ref_batch01 = torch.stack(ref_imgs, dim=0).to(device)
                    gen01 = img_gen[i : i + 1].float().expand(ref_batch01.shape[0], -1, -1, -1)
                    d = lpips_fn(_to_lpips_input(gen01), _to_lpips_input(ref_batch01))
                    all_dists.append(d.detach().view(-1))
                if all_dists:
                    style_lpips_val = float(torch.cat(all_dists).mean().detach().cpu().item())

            if ref_clip_gpu is not None and gen_clip is not None:
                # mean cosine similarity to target ref set (bigger is better)
                g = gen_clip[i : i + 1]
                sims = torch.matmul(g, ref_clip_gpu.t())
                style_clip_val = float(sims.mean().detach().cpu().item())

            if save_images:
                try:
                    _save_side_by_side_with_metrics(
                        out_path=gen_path,
                        src01=src01[i],
                        gen01=img_gen[i],
                        direction=direction,
                        index=int(start + i + 1),
                        content_lpips=(content_lpips_vals[i] if content_lpips_vals else None),
                        style_lpips=style_lpips_val,
                        content_clip=(content_clip_vals[i] if content_clip_vals else None),
                        style_clip=style_clip_val,
                    )
                except Exception:
                    pass

            row = {
                "direction": direction,
                "src_path": _relpath_or_str(p, src_root) if cfg.compact_paths else str(p),
                "src_image": str(p.name),
                "gen_path": (
                    _relpath_or_str(gen_path, gen_root) if (cfg.compact_paths and save_images) else (str(gen_path) if save_images else "")
                ),
                "gen_image": str(gen_path.name) if save_images else "",
                "content_lpips": content_lpips_vals[i] if content_lpips_vals else None,
                "style_lpips": style_lpips_val,
                "content_clip": content_clip_vals[i] if content_clip_vals else None,
                "style_clip": style_clip_val,
            }
            out_writer.writerow(row)
            rows_out.append(row)

    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate latent CycleGAN checkpoints with LPIPS/CLIP metrics")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/example.yaml",
        help=(
            "YAML config path. Defaults to configs/example.yaml if present. "
            "Reads eval.testA_dir/eval.testB_dir/eval.checkpoint_path etc."
        ),
    )

    parser.add_argument("--checkpoint", type=str, default="", help="Path to outputs/model/epoch_XXXX.pt (overrides config)")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory (overrides config)")
    parser.add_argument("--testA", type=str, default="", help="Directory of domain A test images (overrides config)")
    parser.add_argument("--testB", type=str, default="", help="Directory of domain B test images (overrides config)")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_src_samples", type=int, default=0)
    parser.add_argument("--max_ref_cache", type=int, default=256)
    parser.add_argument("--max_ref_compare", type=int, default=50)
    parser.add_argument("--cache_dir", type=str, default="outputs/eval_cache")
    parser.add_argument("--force_regen_cache", action="store_true")

    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--disable_clip", action="store_true")

    parser.add_argument(
        "--clip_model_id",
        type=str,
        default="",
        help="HuggingFace model id or local path for CLIP (default: openai/clip-vit-base-patch32)",
    )
    parser.add_argument(
        "--clip_cache_dir",
        type=str,
        default="",
        help="Optional transformers cache_dir for CLIP weights",
    )
    parser.add_argument(
        "--clip_local_files_only",
        action="store_true",
        help="Do not try to download CLIP weights; use local cache only",
    )
    parser.add_argument(
        "--compact_paths",
        action="store_true",
        help="Write src_path/gen_path as relative paths and save roots to metrics_paths.json",
    )

    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--no_amp_bf16", action="store_true")

    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save generated images under {out_dir}/{checkpoint_stem}/images/{A2B|B2A}",
    )

    # Optional VAE override (otherwise read from checkpoint cfg.visualization)
    parser.add_argument("--vae_model", type=str, default="")
    parser.add_argument("--vae_subfolder", type=str, default="")
    parser.add_argument("--vae_scaling_factor", type=float, default=0.18215)

    args = parser.parse_args()

    # ----------------- load YAML config (default: configs/example.yaml) -----------------
    cfg_file: Dict[str, Any] = {}
    default_cfg = "configs/example.yaml"
    cfg_arg = str(args.config) if isinstance(args.config, str) else ""
    cfg_path = Path(cfg_arg) if cfg_arg.strip() else Path(default_cfg)
    if cfg_path.exists():
        cfg_file = load_yaml_config(str(cfg_path))
    else:
        # If user explicitly provided a config path and it doesn't exist, fail fast.
        # If the default config is missing, keep previous behavior (run with defaults).
        if cfg_arg.strip() and cfg_arg.strip() != default_cfg:
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        if cfg_arg.strip() == default_cfg:
            print(f"[warn] Default config not found: {cfg_path}. Running with built-in defaults.")

    eval_cfg = cfg_file.get("eval", {}) if isinstance(cfg_file.get("eval", {}), dict) else {}
    train_cfg = cfg_file.get("train", {}) if isinstance(cfg_file.get("train", {}), dict) else {}
    vis_cfg_from_file = cfg_file.get("visualization", {}) if isinstance(cfg_file.get("visualization", {}), dict) else {}

    # Resolve paths with precedence: CLI (non-empty) > config.eval > defaults
    ckpt_raw = str(args.checkpoint).strip() or str(eval_cfg.get("checkpoint_path", "")).strip()
    if not ckpt_raw:
        ckpt_dir = str(train_cfg.get("checkpoint_dir", "outputs/model")).strip() or "outputs/model"
        ckpt_raw = str(Path(ckpt_dir) / "last.pt")

    out_dir_raw = str(args.out_dir).strip() or str(eval_cfg.get("out_dir", "outputs/eval")).strip() or "outputs/eval"
    testA_raw = str(args.testA).strip() or str(eval_cfg.get("testA_dir", "dataset/testA")).strip() or "dataset/testA"
    testB_raw = str(args.testB).strip() or str(eval_cfg.get("testB_dir", "dataset/testB")).strip() or "dataset/testB"

    # Resolve device: CLI (if set) > eval.device > default cuda
    device_str = str(args.device).strip() or str(eval_cfg.get("device", "cuda")).strip() or "cuda"

    ckpt_path = Path(ckpt_raw)
    _require(ckpt_path.exists(), f"checkpoint not found: {ckpt_path}")

    payload = _load_ckpt(ckpt_path)
    cfg_saved = payload.get("cfg", {})

    # Pull model hyperparams from saved cfg when possible
    model_cfg = (cfg_saved.get("model", {}) if isinstance(cfg_saved, dict) else {})
    data_cfg = (cfg_saved.get("data", {}) if isinstance(cfg_saved, dict) else {})
    vis_cfg = (cfg_saved.get("visualization", {}) if isinstance(cfg_saved, dict) else {})

    in_ch = int(model_cfg.get("in_channels", 4))
    out_ch = int(model_cfg.get("out_channels", 4))
    ngf = int(model_cfg.get("ngf", 32))
    n_res_blocks = int(model_cfg.get("n_res_blocks", 6))
    out_activation = str(model_cfg.get("out_activation", "none"))

    latents_scaled = bool(data_cfg.get("latents_scaled", False))
    latent_divisor = float(data_cfg.get("latent_divisor", 1.0))
    _require(latent_divisor > 0, "latent_divisor must be > 0")

    # If requested cuda but unavailable, fall back to cpu.
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    G = ResnetGenerator(
        in_ch=in_ch,
        out_ch=out_ch,
        ngf=ngf,
        n_res_blocks=n_res_blocks,
        out_activation=out_activation,
    ).to(device)
    Fnet = ResnetGenerator(
        in_ch=in_ch,
        out_ch=out_ch,
        ngf=ngf,
        n_res_blocks=n_res_blocks,
        out_activation=out_activation,
    ).to(device)

    _require("G" in payload and "F" in payload, "Checkpoint missing keys: expected 'G' and 'F'")
    G.load_state_dict(payload["G"], strict=True)
    Fnet.load_state_dict(payload["F"], strict=True)
    G.eval()
    Fnet.eval()

    # Resolve VAE: CLI override > YAML visualization > checkpoint visualization
    vae_model = str(args.vae_model).strip() or str(vis_cfg_from_file.get("vae_model_name_or_path", "")).strip() or str(
        vis_cfg.get("vae_model_name_or_path", "")
    ).strip()
    _require(vae_model != "", "VAE model is not set. Provide --vae_model or set visualization.vae_model_name_or_path in training config")

    vae_sub = str(args.vae_subfolder).strip() or str(vis_cfg_from_file.get("vae_subfolder", "")).strip() or str(
        vis_cfg.get("vae_subfolder", "")
    ).strip()
    vae_sub = vae_sub if vae_sub else None

    vae_scaling_factor = float(args.vae_scaling_factor)
    if isinstance(vis_cfg.get("vae_scaling_factor"), (float, int)) and not str(args.vae_model).strip():
        # use checkpoint's vis scaling factor when not overriding vae_model explicitly
        vae_scaling_factor = float(vis_cfg.get("vae_scaling_factor"))

    vae = _load_vae(vae_model, vae_sub, device)

    testA = Path(testA_raw)
    testB = Path(testB_raw)
    srcA = _list_images(testA)
    srcB = _list_images(testB)
    _require(len(srcA) > 0, f"No images found under testA: {testA}")
    _require(len(srcB) > 0, f"No images found under testB: {testB}")

    out_root = Path(out_dir_raw)
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / ckpt_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = out_dir / "metrics.csv"
    # Scalar options: CLI flags/values override YAML when explicitly passed.
    image_size = int(eval_cfg.get("image_size", args.image_size))
    batch_size = int(eval_cfg.get("batch_size", args.batch_size))
    max_src_samples = int(eval_cfg.get("max_src_samples", args.max_src_samples))
    max_ref_cache = int(eval_cfg.get("max_ref_cache", args.max_ref_cache))
    max_ref_compare = int(eval_cfg.get("max_ref_compare", args.max_ref_compare))

    cache_dir = str(eval_cfg.get("cache_dir", args.cache_dir)).strip() or str(args.cache_dir)
    force_regen_cache = bool(eval_cfg.get("force_regen_cache", False)) or bool(args.force_regen_cache)
    save_images = bool(eval_cfg.get("save_images", False)) or bool(args.save_images)

    disable_lpips = bool(eval_cfg.get("disable_lpips", False)) or bool(args.disable_lpips)
    disable_clip = bool(eval_cfg.get("disable_clip", False)) or bool(args.disable_clip)

    clip_model_id = str(args.clip_model_id).strip() or str(eval_cfg.get("clip_model_id", "openai/clip-vit-base-patch32")).strip()
    clip_cache_dir = str(args.clip_cache_dir).strip() or str(eval_cfg.get("clip_cache_dir", "")).strip()
    clip_local_files_only = bool(eval_cfg.get("clip_local_files_only", False)) or bool(args.clip_local_files_only)
    clip_use_safetensors = bool(eval_cfg.get("clip_use_safetensors", True))
    compact_paths = bool(eval_cfg.get("compact_paths", False)) or bool(args.compact_paths)

    amp_bf16 = bool(eval_cfg.get("amp_bf16", True)) and (not bool(args.no_amp_bf16))

    cfg_eval = EvalConfig(
        image_size=image_size,
        batch_size=batch_size,
        max_src_samples=max_src_samples,
        max_ref_cache=max_ref_cache,
        max_ref_compare=max_ref_compare,
        cache_dir=cache_dir,
        force_regen_cache=force_regen_cache,
        disable_lpips=disable_lpips,
        disable_clip=disable_clip,
        clip_model_id=clip_model_id or "openai/clip-vit-base-patch32",
        clip_cache_dir=clip_cache_dir,
        clip_local_files_only=clip_local_files_only,
        clip_use_safetensors=clip_use_safetensors,
        compact_paths=compact_paths,
        amp_bf16=amp_bf16,
        device=device_str,
    )

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        columns = [
            "direction",
            "src_path",
            "src_image",
            "gen_path",
            "gen_image",
            "content_lpips",
            "style_lpips",
            "content_clip",
            "style_clip",
        ]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        rows_a2b = evaluate_direction(
            direction="A2B",
            src_paths=srcA,
            ref_paths=srcB,
            gen=G,
            vae=vae,
            device=device,
            latents_scaled=latents_scaled,
            vae_scaling_factor=vae_scaling_factor,
            latent_divisor=latent_divisor,
            out_writer=writer,
            out_dir=out_dir,
            cfg=cfg_eval,
            save_images=save_images,
            src_root=_common_root([testA, testB]) if cfg_eval.compact_paths else None,
            gen_root=(out_dir / "images") if cfg_eval.compact_paths else None,
        )

        rows_b2a = evaluate_direction(
            direction="B2A",
            src_paths=srcB,
            ref_paths=srcA,
            gen=Fnet,
            vae=vae,
            device=device,
            latents_scaled=latents_scaled,
            vae_scaling_factor=vae_scaling_factor,
            latent_divisor=latent_divisor,
            out_writer=writer,
            out_dir=out_dir,
            cfg=cfg_eval,
            save_images=save_images,
            src_root=_common_root([testA, testB]) if cfg_eval.compact_paths else None,
            gen_root=(out_dir / "images") if cfg_eval.compact_paths else None,
        )

    summary = {
        "checkpoint": str(ckpt_path),
        "out_dir": str(out_dir),
        "testA": str(testA),
        "testB": str(testB),
        "compact_paths": bool(cfg_eval.compact_paths),
        "src_root": str(_common_root([testA, testB])) if cfg_eval.compact_paths else "",
        "gen_root": str(out_dir / "images") if cfg_eval.compact_paths else "",
        "latents_scaled": latents_scaled,
        "latent_divisor": latent_divisor,
        "vae_model": vae_model,
        "vae_subfolder": vae_sub,
        "vae_scaling_factor": vae_scaling_factor,
        "A2B": _summarize_rows(rows_a2b),
        "B2A": _summarize_rows(rows_b2a),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if cfg_eval.compact_paths:
        with open(out_dir / "metrics_paths.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "src_root": str(_common_root([testA, testB]) or ""),
                    "gen_root": str(out_dir / "images"),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    print(f"Saved: {metrics_csv}")
    print(f"Saved: {out_dir / 'summary.json'}")
    if cfg_eval.compact_paths:
        print(f"Saved: {out_dir / 'metrics_paths.json'}")


if __name__ == "__main__":
    main()
