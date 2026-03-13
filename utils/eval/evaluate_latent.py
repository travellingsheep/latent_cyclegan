import argparse
import csv
import hashlib
import json
import math
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm

# Ensure repository root is importable when launching from utils/eval.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_latent_cyclegan import (  # noqa: E402
    ResnetGenerator,
    _load_vae,
    decode_latents_to_images,
    load_yaml_config,
)


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass
class EvalConfig:
    model_cfg: Dict[str, Any]
    data_cfg: Dict[str, Any]
    vis_cfg: Dict[str, Any]
    eval_cfg: Dict[str, Any]
    latent_src_dir: Path
    orig_rgb_dir: Path
    vae_recon_dir: Path
    testB_dir: Path
    out_dir: Path
    cache_dir: Path
    checkpoint_dir: Path
    checkpoint_path: Path
    image_size: int
    batch_size: int
    device: torch.device
    amp_bf16: bool


def _warn(msg: str) -> None:
    warnings.warn(msg, stacklevel=2)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _safe_stem_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _pil_rgb(path: Path, image_size: Optional[int] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return img


def _list_images(root: Path) -> List[Path]:
    _require(root.exists(), f"Image directory not found: {root}")
    files: List[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    files = sorted([p for p in files if p.is_file()])
    _require(len(files) > 0, f"No RGB images found in: {root}")
    return files


def _list_latents(root: Path) -> List[Path]:
    _require(root.exists(), f"Latent directory not found: {root}")
    files = sorted([p for p in root.rglob("*.pt") if p.is_file()])
    _require(len(files) > 0, f"No latent .pt files found in: {root}")
    return files


def _build_stem_index(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in _list_images(root):
        index[p.stem] = p
    return index


def _load_latent(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict) and isinstance(obj.get("latent"), torch.Tensor):
        t = obj["latent"]
    else:
        raise ValueError(f"Unsupported latent format: {path}")
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    _require(t.ndim == 3, f"Latent must be [C,H,W] or [1,C,H,W], got {tuple(t.shape)} for {path}")
    return t.detach().float()


def _get_cache_name(testB_dir: Path, image_size: int, clip_model_id: str) -> str:
    key = f"{str(testB_dir.resolve())}|{image_size}|inception_v3_pool3|{clip_model_id}"
    md5 = hashlib.md5(key.encode("utf-8")).hexdigest()
    clip_tag = clip_model_id.replace("/", "_").replace("-", "_")
    return f"ref_cache_{md5}_res{image_size}_inception_v3_pool3_{clip_tag}.pt"


def _batch_iter(seq: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def _mean_cov(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _kid_mean_std_from_features(
    feats_ref: np.ndarray,
    feats_gen: np.ndarray,
    num_subsets: int = 100,
    max_subset_size: int = 1000,
    seed: int = 7,
) -> Tuple[float, float]:
    _require(feats_ref.ndim == 2 and feats_gen.ndim == 2, "KID features must be 2D arrays")
    _require(feats_ref.shape[1] == feats_gen.shape[1], "KID features dim mismatch")

    n_dim = int(feats_ref.shape[1])
    m = min(int(feats_ref.shape[0]), int(feats_gen.shape[0]), int(max_subset_size))
    _require(m >= 2, "KID requires at least 2 samples in each set")

    rng = np.random.default_rng(seed)
    subset_vals: List[float] = []
    for _ in range(max(1, int(num_subsets))):
        x = feats_gen[rng.choice(feats_gen.shape[0], m, replace=False)]
        y = feats_ref[rng.choice(feats_ref.shape[0], m, replace=False)]
        a = (x @ x.T / n_dim + 1.0) ** 3 + (y @ y.T / n_dim + 1.0) ** 3
        b = (x @ y.T / n_dim + 1.0) ** 3
        kid_subset = (a.sum() - np.diag(a).sum()) / (m - 1) - 2.0 * b.sum() / m
        kid_subset = kid_subset / m
        subset_vals.append(float(kid_subset))

    arr = np.asarray(subset_vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    try:
        from scipy import linalg as scipy_linalg
    except Exception as exc:
        raise RuntimeError("scipy is required for FID computation from cached statistics") from exc

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = scipy_linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = scipy_linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(fid)


def _load_clip_model(eval_cfg: Dict[str, Any], device: torch.device):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:
        raise RuntimeError("Missing transformers CLIP deps. Install transformers.") from exc

    clip_model_id = str(eval_cfg.get("clip_model_id", "openai/clip-vit-base-patch32"))
    clip_cache_dir = str(eval_cfg.get("clip_cache_dir", "") or "")
    clip_local_files_only = bool(eval_cfg.get("clip_local_files_only", False))
    clip_use_safetensors = bool(eval_cfg.get("clip_use_safetensors", False))

    kwargs: Dict[str, Any] = {
        "local_files_only": clip_local_files_only,
    }
    if clip_cache_dir:
        kwargs["cache_dir"] = clip_cache_dir
    if clip_use_safetensors:
        kwargs["use_safetensors"] = True

    model = CLIPModel.from_pretrained(clip_model_id, **kwargs).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_id, **kwargs)
    model.eval()
    return clip_model_id, model, processor


@torch.no_grad()
def _clip_image_features(
    image_paths: Sequence[Path],
    clip_model,
    clip_processor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    for batch_paths in tqdm(list(_batch_iter(list(image_paths), batch_size)), desc="CLIP image feats", ascii=True):
        imgs = [_pil_rgb(p) for p in batch_paths]
        inputs = clip_processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = clip_model.get_image_features(**inputs)
        outputs = F.normalize(outputs.float(), dim=-1)
        feats.append(outputs.cpu())
        del imgs, inputs, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(feats, dim=0)


@torch.no_grad()
def _clip_text_feature(
    prompt: str,
    clip_model,
    clip_processor,
    device: torch.device,
) -> torch.Tensor:
    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    text_feat = clip_model.get_text_features(**inputs)
    text_feat = F.normalize(text_feat.float(), dim=-1)
    return text_feat[0].cpu()


def _phase0_parse_config(args: argparse.Namespace) -> EvalConfig:
    cfg = load_yaml_config(args.config)

    model_cfg = cfg.get("model", {})
    vis_cfg = cfg.get("visualization", {})
    eval_cfg = cfg.get("cyclegan_eval", {})
    data_cfg = cfg.get("data", {})

    _require(isinstance(model_cfg, dict), "config.model must be a dict")
    _require(isinstance(vis_cfg, dict), "config.visualization must be a dict")
    _require(isinstance(eval_cfg, dict), "config.cyclegan_eval must be a dict")
    _require(isinstance(data_cfg, dict), "config.data must be a dict")

    latent_src_dir = Path(args.latent_src_dir or eval_cfg.get("latent_src_dir", "") or data_cfg.get("a_dir", ""))
    orig_rgb_dir = Path(args.orig_rgb_dir or eval_cfg.get("orig_rgb_dir", "") or eval_cfg.get("testA_dir", ""))
    vae_recon_dir = Path(args.vae_recon_dir or eval_cfg.get("vae_recon_dir", ""))
    testB_dir = Path(args.testB_dir or eval_cfg.get("testB_dir", ""))

    out_dir = Path(args.out_dir or eval_cfg.get("out_dir", "outputs/eval_latent"))
    cache_dir = Path(args.cache_dir or eval_cfg.get("cache_dir", "outputs/eval_cache_latent"))
    checkpoint_dir = Path(args.checkpoint_dir or cfg.get("train", {}).get("checkpoint_dir", "outputs/model"))
    checkpoint_path = Path(args.checkpoint_path or eval_cfg.get("checkpoint_path", "") or checkpoint_dir / "last.pt")

    image_size = int(args.image_size or eval_cfg.get("image_size", 256))
    batch_size = int(args.batch_size or eval_cfg.get("batch_size", 8))

    device_str = str(args.device or eval_cfg.get("device", "cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    amp_bf16 = bool(args.amp_bf16 if args.amp_bf16 is not None else eval_cfg.get("amp_bf16", True))

    _require(str(latent_src_dir), "latent_src_dir is required")
    _require(str(orig_rgb_dir), "orig_rgb_dir is required")
    _require(str(vae_recon_dir), "vae_recon_dir is required")
    _require(str(testB_dir), "testB_dir is required")

    _require(latent_src_dir.exists(), f"latent_src_dir not found: {latent_src_dir}")
    _require(orig_rgb_dir.exists(), f"orig_rgb_dir not found: {orig_rgb_dir}")
    _require(vae_recon_dir.exists(), f"vae_recon_dir not found: {vae_recon_dir}")
    _require(testB_dir.exists(), f"testB_dir not found: {testB_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return EvalConfig(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        vis_cfg=vis_cfg,
        eval_cfg=eval_cfg,
        latent_src_dir=latent_src_dir,
        orig_rgb_dir=orig_rgb_dir,
        vae_recon_dir=vae_recon_dir,
        testB_dir=testB_dir,
        out_dir=out_dir,
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        amp_bf16=amp_bf16,
    )


def _phase1_build_or_load_ref_cache(ecfg: EvalConfig, force_regen_cache: bool = False) -> Dict[str, Any]:
    from cleanfid import fid

    clip_model_id, clip_model, clip_processor = _load_clip_model(ecfg.eval_cfg, ecfg.device)
    cache_name = _get_cache_name(ecfg.testB_dir, ecfg.image_size, clip_model_id)
    cache_path = ecfg.cache_dir / cache_name

    if cache_path.exists() and not force_regen_cache:
        _log(f"[Phase1] Use existing strict cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        cache["_clip_runtime"] = {
            "clip_model_id": clip_model_id,
            "clip_model": clip_model,
            "clip_processor": clip_processor,
        }
        cache["cache_path"] = str(cache_path)
        return cache

    _log("[Phase1] Building strict reference cache...")
    ref_paths = _list_images(ecfg.testB_dir)
    max_ref_cache = int(ecfg.eval_cfg.get("max_ref_cache", 0))
    if max_ref_cache > 0 and len(ref_paths) > max_ref_cache:
        random.seed(42)
        ref_paths = random.sample(ref_paths, max_ref_cache)

    cleanfid_features = fid.get_folder_features(
        fdir=str(ecfg.testB_dir),
        model="inception_v3",
        mode="clean",
        num_workers=4,
        batch_size=max(8, ecfg.batch_size),
        device=ecfg.device,
        verbose=True,
    )
    cleanfid_features = np.asarray(cleanfid_features)
    mu, sigma = _mean_cov(cleanfid_features)

    clip_feats = _clip_image_features(
        image_paths=ref_paths,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=ecfg.device,
        batch_size=ecfg.batch_size,
    )
    clip_proto_img = F.normalize(clip_feats.mean(dim=0, keepdim=False), dim=0).cpu()

    clip_text_prompt = str(ecfg.eval_cfg.get("clip_text_prompt", "") or "")
    clip_proto_text = None
    if clip_text_prompt:
        clip_proto_text = _clip_text_feature(
            prompt=clip_text_prompt,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=ecfg.device,
        )

    payload: Dict[str, Any] = {
        "meta": {
            "testB_dir": str(ecfg.testB_dir.resolve()),
            "image_size": ecfg.image_size,
            "extractors": ["inception_v3_pool3", clip_model_id],
            "num_ref_images_cleanfid": int(cleanfid_features.shape[0]),
            "num_ref_images_clip": int(clip_feats.shape[0]),
            "clip_text_prompt": clip_text_prompt,
        },
        "fid_kid": {
            "mu": mu.astype(np.float64),
            "sigma": sigma.astype(np.float64),
        },
        "clip": {
            "image_prototype": clip_proto_img,
            "text_prototype": clip_proto_text,
        },
        "cache_path": str(cache_path),
    }
    torch.save(payload, cache_path)
    _log(f"[Phase1] Reference cache saved: {cache_path}")

    payload["_clip_runtime"] = {
        "clip_model_id": clip_model_id,
        "clip_model": clip_model,
        "clip_processor": clip_processor,
    }
    return payload


def _load_generator(ecfg: EvalConfig) -> torch.nn.Module:
    _require(ecfg.checkpoint_path.exists(), f"checkpoint not found: {ecfg.checkpoint_path}")

    G = ResnetGenerator(
        in_ch=int(ecfg.model_cfg.get("in_channels", 4)),
        out_ch=int(ecfg.model_cfg.get("out_channels", 4)),
        ngf=int(ecfg.model_cfg.get("ngf", 256)),
        n_res_blocks=int(ecfg.model_cfg.get("n_res_blocks", 9)),
        out_activation=str(ecfg.model_cfg.get("out_activation", "none")),
    )

    payload = torch.load(ecfg.checkpoint_path, map_location="cpu")
    sd = None
    if isinstance(payload, dict):
        for k in ("G", "netG_A", "generator", "state_dict"):
            if isinstance(payload.get(k), dict):
                sd = payload[k]
                break
        if sd is None and all(isinstance(v, torch.Tensor) for v in payload.values()):
            sd = payload
    if sd is None:
        raise ValueError(f"Unsupported generator checkpoint structure: {ecfg.checkpoint_path}")

    missing, unexpected = G.load_state_dict(sd, strict=False)
    if missing:
        _warn(f"Generator state_dict missing keys: {len(missing)}")
    if unexpected:
        _warn(f"Generator state_dict unexpected keys: {len(unexpected)}")

    G.to(ecfg.device)
    G.eval()
    return G


def _save_tensor_image(img: torch.Tensor, out_path: Path) -> None:
    img = img.detach().cpu().clamp(0, 1)
    pil = transforms.ToPILImage()(img)
    pil.save(out_path)


def _make_3panel(orig: Image.Image, recon: Image.Image, transfer: Image.Image, stem: str) -> Image.Image:
    w = orig.width
    h = orig.height
    label_h = 28
    canvas = Image.new("RGB", (w * 3, h + label_h), (245, 245, 245))
    canvas.paste(orig, (0, label_h))
    canvas.paste(recon, (w, label_h))
    canvas.paste(transfer, (w * 2, label_h))

    draw = ImageDraw.Draw(canvas)
    labels = ["Orig_Img", "Recon_Img", "Transferred_Img"]
    for i, txt in enumerate(labels):
        draw.text((10 + i * w, 6), txt, fill=(20, 20, 20))
    draw.text((10, h + 8), stem, fill=(20, 20, 20))
    return canvas


@torch.no_grad()
def _phase2_infer_and_visualize(ecfg: EvalConfig) -> Dict[str, Any]:
    image_out_dir = ecfg.out_dir / "images"
    vis3_out_dir = ecfg.out_dir / "vis_3panel"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    vis3_out_dir.mkdir(parents=True, exist_ok=True)

    latent_paths = _list_latents(ecfg.latent_src_dir)
    orig_index = _build_stem_index(ecfg.orig_rgb_dir)
    recon_index = _build_stem_index(ecfg.vae_recon_dir)

    G = _load_generator(ecfg)
    vae = _load_vae(
        model_name_or_path=str(ecfg.vis_cfg.get("vae_model_name_or_path", "runwayml/stable-diffusion-v1-5")),
        subfolder=str(ecfg.vis_cfg.get("vae_subfolder", "vae") or "vae"),
        device=ecfg.device,
    )

    latents_scaled = bool(ecfg.data_cfg.get("latents_scaled", False))
    latent_divisor = float(ecfg.data_cfg.get("latent_divisor", 1.0))
    vae_scaling_factor = float(ecfg.vis_cfg.get("vae_scaling_factor", 0.18215))

    transferred_paths: List[Path] = []
    paired_stems: List[str] = []

    pbar = tqdm(list(_batch_iter(latent_paths, ecfg.batch_size)), desc="Phase2 inference", ascii=True)
    for latent_batch_paths in pbar:
        stems = [_safe_stem_name(p) for p in latent_batch_paths]
        latent_tensors = [_load_latent(p) for p in latent_batch_paths]

        try:
            latent_batch = torch.stack(latent_tensors, dim=0).to(ecfg.device, non_blocking=True)
            latent_in = latent_batch / latent_divisor if not math.isclose(latent_divisor, 1.0) else latent_batch
            use_amp = ecfg.amp_bf16 and ecfg.device.type == "cuda"
            amp_dtype = torch.bfloat16 if use_amp else torch.float32
            with torch.autocast(device_type=ecfg.device.type, enabled=use_amp, dtype=amp_dtype):
                fake_latent = G(latent_in)
                fake_img = decode_latents_to_images(
                    vae=vae,
                    latents=fake_latent,
                    latents_scaled=latents_scaled,
                    scaling_factor=vae_scaling_factor,
                    latent_divisor=latent_divisor,
                )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            _warn(f"OOM on batch size={len(latent_batch_paths)}; fallback to per-sample for this batch")
            if ecfg.device.type == "cuda":
                torch.cuda.empty_cache()
            fake_list: List[torch.Tensor] = []
            for latent_tensor in latent_tensors:
                one = latent_tensor.unsqueeze(0).to(ecfg.device, non_blocking=True)
                one_in = one / latent_divisor if not math.isclose(latent_divisor, 1.0) else one
                one_fake_latent = G(one_in)
                one_fake_img = decode_latents_to_images(
                    vae=vae,
                    latents=one_fake_latent,
                    latents_scaled=latents_scaled,
                    scaling_factor=vae_scaling_factor,
                    latent_divisor=latent_divisor,
                )
                fake_list.append(one_fake_img[0].detach().cpu())
                del one, one_in, one_fake_latent, one_fake_img
            fake_img = torch.stack(fake_list, dim=0)

        for i, stem in enumerate(stems):
            out_path = image_out_dir / f"{stem}.png"
            _save_tensor_image(fake_img[i], out_path)
            transferred_paths.append(out_path)

            orig_path = orig_index.get(stem)
            recon_path = recon_index.get(stem)
            if orig_path is None or recon_path is None:
                _warn(f"Missing stem={stem} in orig/recon dir, skip 3-panel visualization")
                continue

            orig_img = _pil_rgb(orig_path, image_size=ecfg.image_size)
            recon_img = _pil_rgb(recon_path, image_size=ecfg.image_size)
            transfer_img = _pil_rgb(out_path, image_size=ecfg.image_size)
            panel = _make_3panel(orig_img, recon_img, transfer_img, stem)
            panel.save(vis3_out_dir / f"{stem}.png")
            paired_stems.append(stem)

        del latent_tensors
        if "latent_batch" in locals():
            del latent_batch
        if "latent_in" in locals():
            del latent_in
        if "fake_latent" in locals():
            del fake_latent
        if "fake_img" in locals():
            del fake_img
        if ecfg.device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "transferred_paths": transferred_paths,
        "transferred_dir": image_out_dir,
        "paired_stems": paired_stems,
    }

def _phase3_metrics(
    ecfg: EvalConfig,
    ref_cache: Dict[str, Any],
    phase2_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    from cleanfid import fid

    disable_lpips = bool(ecfg.eval_cfg.get("disable_lpips", False))
    disable_clip = bool(ecfg.eval_cfg.get("disable_clip", False))

    transferred_dir = Path(phase2_outputs["transferred_dir"])
    transferred_paths = _list_images(transferred_dir)

    # Distribution metrics.
    gen_feats = fid.get_folder_features(
        fdir=str(transferred_dir),
        model="inception_v3",
        mode="clean",
        num_workers=4,
        batch_size=max(8, ecfg.batch_size),
        device=ecfg.device,
        verbose=True,
    )
    gen_feats = np.asarray(gen_feats)
    mu_g, sigma_g = _mean_cov(gen_feats)
    mu_r = np.asarray(ref_cache["fid_kid"]["mu"])
    sigma_r = np.asarray(ref_cache["fid_kid"]["sigma"])
    fid_value = _frechet_distance(mu_g, sigma_g, mu_r, sigma_r)

    max_ref_compare = int(ecfg.eval_cfg.get("max_ref_compare", 0))
    ref_feats_kid = fid.get_folder_features(
        fdir=str(ecfg.testB_dir),
        model="inception_v3",
        mode="clean",
        num_workers=4,
        num=(max_ref_compare if max_ref_compare > 0 else None),
        shuffle=(max_ref_compare > 0),
        seed=7,
        batch_size=max(8, ecfg.batch_size),
        device=ecfg.device,
        verbose=True,
    )
    ref_feats_kid = np.asarray(ref_feats_kid)

    kid_subsets = int(ecfg.eval_cfg.get("kid_subsets", 100))
    kid_subset_size = int(ecfg.eval_cfg.get("kid_subset_size", 1000))
    kid_value, kid_std_value = _kid_mean_std_from_features(
        feats_ref=ref_feats_kid,
        feats_gen=gen_feats,
        num_subsets=kid_subsets,
        max_subset_size=kid_subset_size,
        seed=7,
    )

    # Pairwise metrics.
    orig_index = _build_stem_index(ecfg.orig_rgb_dir)
    gen_index = _build_stem_index(transferred_dir)
    pair_stems = sorted(set(orig_index.keys()).intersection(set(gen_index.keys())))

    lpips_vals: List[float] = []
    clip_dir_vals: List[float] = []
    clip_style_vals: List[float] = []
    rows: List[Dict[str, Any]] = []

    lpips_model = None
    if not disable_lpips:
        try:
            import lpips

            lpips_model = lpips.LPIPS(net="vgg").to(ecfg.device)
            lpips_model.eval()
        except Exception as exc:
            _warn(f"LPIPS disabled due to import/init error: {exc}")

    clip_model = None
    clip_processor = None
    clip_target_proto = None
    if not disable_clip:
        runtime = ref_cache.get("_clip_runtime", {})
        clip_model = runtime.get("clip_model")
        clip_processor = runtime.get("clip_processor")
        clip_target_proto = ref_cache.get("clip", {}).get("image_prototype")
        if clip_model is None or clip_processor is None or clip_target_proto is None:
            _warn("CLIP runtime/prototype missing in cache, CLIP metrics skipped")
            disable_clip = True

    src_clip_feats: List[torch.Tensor] = []
    gen_clip_feats: List[torch.Tensor] = []

    for batch_stems in tqdm(list(_batch_iter(pair_stems, ecfg.batch_size)), desc="Phase3 pairwise", ascii=True):
        orig_tensors: List[torch.Tensor] = []
        gen_tensors: List[torch.Tensor] = []
        orig_imgs: List[Image.Image] = []
        gen_imgs: List[Image.Image] = []

        for stem in batch_stems:
            o = _pil_rgb(orig_index[stem], ecfg.image_size)
            g = _pil_rgb(gen_index[stem], ecfg.image_size)
            orig_imgs.append(o)
            gen_imgs.append(g)
            orig_tensors.append(transforms.ToTensor()(o))
            gen_tensors.append(transforms.ToTensor()(g))

        orig_batch = torch.stack(orig_tensors, dim=0).to(ecfg.device)
        gen_batch = torch.stack(gen_tensors, dim=0).to(ecfg.device)

        if lpips_model is not None:
            # LPIPS expects [-1, 1]
            o_lp = orig_batch * 2 - 1
            g_lp = gen_batch * 2 - 1
            lp = lpips_model(o_lp, g_lp).view(-1)
            lp_cpu = lp.detach().cpu().tolist()
        else:
            lp_cpu = [float("nan")] * len(batch_stems)

        if not disable_clip and clip_model is not None and clip_processor is not None:
            inp_src = clip_processor(images=orig_imgs, return_tensors="pt")
            inp_gen = clip_processor(images=gen_imgs, return_tensors="pt")
            inp_src = {k: v.to(ecfg.device) for k, v in inp_src.items()}
            inp_gen = {k: v.to(ecfg.device) for k, v in inp_gen.items()}
            feat_src = clip_model.get_image_features(**inp_src)
            feat_gen = clip_model.get_image_features(**inp_gen)
            feat_src = F.normalize(feat_src.float(), dim=-1)
            feat_gen = F.normalize(feat_gen.float(), dim=-1)
            src_clip_feats.append(feat_src.detach().cpu())
            gen_clip_feats.append(feat_gen.detach().cpu())

            tgt = F.normalize(clip_target_proto.to(feat_gen.device).unsqueeze(0), dim=-1)
            style_vals = F.cosine_similarity(feat_gen, tgt.expand_as(feat_gen), dim=-1)
            style_cpu = style_vals.detach().cpu().tolist()
        else:
            style_cpu = [float("nan")] * len(batch_stems)

        for i, stem in enumerate(batch_stems):
            row = {
                "stem": stem,
                "orig_path": str(orig_index[stem]),
                "transferred_path": str(gen_index[stem]),
                "lpips": float(lp_cpu[i]) if not math.isnan(lp_cpu[i]) else None,
                "clip_dir": None,
                "clip_style": float(style_cpu[i]) if not math.isnan(style_cpu[i]) else None,
            }
            rows.append(row)
            if not math.isnan(lp_cpu[i]):
                lpips_vals.append(float(lp_cpu[i]))
            if not math.isnan(style_cpu[i]):
                clip_style_vals.append(float(style_cpu[i]))

        del orig_batch, gen_batch
        if ecfg.device.type == "cuda":
            torch.cuda.empty_cache()

    if (not disable_clip) and src_clip_feats and gen_clip_feats and clip_target_proto is not None:
        src_all = torch.cat(src_clip_feats, dim=0)
        gen_all = torch.cat(gen_clip_feats, dim=0)
        src_proto = F.normalize(src_all.mean(dim=0), dim=0)
        tgt_proto = F.normalize(clip_target_proto.float(), dim=0)
        style_vec = F.normalize(tgt_proto - src_proto, dim=0)

        edit_vec = F.normalize(gen_all - src_all, dim=-1)
        dir_vals = F.cosine_similarity(edit_vec, style_vec.unsqueeze(0).expand_as(edit_vec), dim=-1)
        dir_vals_list = dir_vals.cpu().tolist()

        for i, v in enumerate(dir_vals_list):
            rows[i]["clip_dir"] = float(v)
            clip_dir_vals.append(float(v))

    report = {
        "fid": float(fid_value),
        "kid_mean": kid_value,
        "kid_std": kid_std_value,
        "lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else None,
        "clip_dir_mean": float(np.mean(clip_dir_vals)) if clip_dir_vals else None,
        "clip_style_mean": float(np.mean(clip_style_vals)) if clip_style_vals else None,
        "num_generated": len(transferred_paths),
        "num_pairs": len(rows),
        "reference_cache": ref_cache.get("cache_path", ""),
    }
    return {
        "summary": report,
        "rows": rows,
    }


def _write_reports(ecfg: EvalConfig, metric_payload: Dict[str, Any]) -> None:
    report_json = ecfg.out_dir / "metrics_report.json"
    report_csv = ecfg.out_dir / "metrics.csv"

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(metric_payload["summary"], f, indent=2, ensure_ascii=False)

    rows = metric_payload["rows"]
    fields = ["stem", "orig_path", "transferred_path", "lpips", "clip_dir", "clip_style"]
    with open(report_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})

    _log("[Phase3] Metrics summary:")
    for k, v in metric_payload["summary"].items():
        _log(f"  - {k}: {v}")
    _log(f"[Phase3] Saved JSON report to: {report_json}")
    _log(f"[Phase3] Saved CSV report to: {report_csv}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Latent CycleGAN evaluation in RGB space")
    p.add_argument("--config", type=str, default="configs/example.yaml")

    # Phase 0 override with highest priority.
    p.add_argument("--orig_rgb_dir", type=str, default="", help="source-domain original RGB directory")
    p.add_argument("--vae_recon_dir", type=str, default="", help="source-domain VAE reconstruction RGB directory")
    p.add_argument("--latent_src_dir", type=str, default="", help="source-domain latent .pt directory")

    p.add_argument("--testB_dir", type=str, default="", help="target-domain RGB directory for reference cache")
    p.add_argument("--checkpoint_dir", type=str, default="")
    p.add_argument("--checkpoint_path", type=str, default="")
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--cache_dir", type=str, default="")
    p.add_argument("--image_size", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=0)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--amp_bf16", action="store_true", default=None)
    p.add_argument("--no_amp_bf16", dest="amp_bf16", action="store_false")
    p.add_argument("--force_regen_cache", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    _log("[Phase0] Parsing YAML and applying CLI overrides...")
    ecfg = _phase0_parse_config(args)

    _log("[Phase1] Strict cache pre-computation...")
    ref_cache = _phase1_build_or_load_ref_cache(ecfg, force_regen_cache=args.force_regen_cache)

    _log("[Phase2] Latent inference and 3-panel visualization...")
    phase2_outputs = _phase2_infer_and_visualize(ecfg)

    _log("[Phase3] Metric computation and aggregation...")
    metric_payload = _phase3_metrics(ecfg, ref_cache, phase2_outputs)
    _write_reports(ecfg, metric_payload)


if __name__ == "__main__":
    main()
