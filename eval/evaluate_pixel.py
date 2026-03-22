import argparse
import csv
import functools
import hashlib
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Ensure repository root is importable when launching from eval/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_latent_cyclegan import load_yaml_config  # noqa: E402


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _get_norm_layer(norm_type: str = "instance"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        def norm_layer(_: int) -> nn.Module:
            return _Identity()

        return norm_layer
    raise NotImplementedError(f"normalization layer [{norm_type}] is not found")


class OfficialResnetBlock(nn.Module):
    def __init__(self, dim: int, padding_type: str, norm_layer, use_dropout: bool, use_bias: bool):
        super().__init__()
        self.conv_block = self._build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def _build_conv_block(self, dim: int, padding_type: str, norm_layer, use_dropout: bool, use_bias: bool) -> nn.Sequential:
        conv_block: List[nn.Module] = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class OfficialResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        norm: str = "instance",
        use_dropout: bool = False,
        n_blocks: int = 9,
        padding_type: str = "reflect",
    ):
        super().__init__()
        _require(n_blocks >= 0, "n_blocks must be >= 0")

        norm_layer = _get_norm_layer(norm_type=norm)
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model: List[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [
                OfficialResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class EvalConfig:
    eval_cfg: Dict[str, Any]
    domain_A_name: str
    domain_B_name: str
    direction_mode: str
    single_direction: str
    base_test_image_dir: Path
    base_ref_image_dir: Path
    eval_output_dir: Path
    reference_cache_dir: Path
    generator_checkpoint_path: Path
    max_eval_samples: int
    eval_seed: int
    eval_image_size: int
    eval_batch_size: int
    run_device: torch.device
    use_bf16_autocast: bool


@dataclass
class EvalDirection:
    name: str
    src_domain: str
    tgt_domain: str
    test_image_dir: Path
    tgt_ref_dir: Path
    out_dir: Path


def _warn(msg: str) -> None:
    warnings.warn(msg, stacklevel=2)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _resolve_repo_path(path_value: Any) -> Path:
    path_str = str(path_value or "").strip()
    if not path_str:
        return Path("")
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _torch_load_trusted(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _patch_instance_norm_state_dict(state_dict: Dict[str, Any], module: torch.nn.Module, keys: List[str], i: int = 0) -> None:
    key = keys[i]
    if i + 1 == len(keys):
        if module.__class__.__name__.startswith("InstanceNorm") and key in ("running_mean", "running_var"):
            if getattr(module, key, None) is None:
                state_dict.pop(".".join(keys), None)
        if module.__class__.__name__.startswith("InstanceNorm") and key == "num_batches_tracked":
            state_dict.pop(".".join(keys), None)
    else:
        if hasattr(module, key):
            _patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


def _normalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        normalized[new_key] = value
    return normalized


def _safe_stem_name(path: Path) -> str:
    return path.stem


def _pil_rgb(path: Path, image_size: Optional[int] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return img


def _build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _load_image_tensor(path: Path, image_size: int, transform: transforms.Compose) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transform(img)


def _list_images(root: Path) -> List[Path]:
    _require(root.exists(), f"Image directory not found: {root}")
    files: List[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    files = sorted([p for p in files if p.is_file()])
    _require(len(files) > 0, f"No RGB images found in: {root}")
    return files


def _build_stem_index(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in _list_images(root):
        index[p.stem] = p
    return index


def _resolve_domain_dir(base_dir: Path, domain_name: str, key_name: str) -> Path:
    path = base_dir / domain_name
    _require(path.exists(), f"{key_name} not found for domain '{domain_name}': {path}")
    return path


def _select_image_paths(
    image_paths: Sequence[Path],
    max_eval_samples: int,
    eval_seed: int,
) -> List[Path]:
    selected = list(image_paths)
    if max_eval_samples > 0 and len(selected) > max_eval_samples:
        rng = random.Random(eval_seed)
        selected = rng.sample(selected, max_eval_samples)
    return sorted(selected, key=lambda p: p.stem)


def _get_cache_name(target_reference_rgb_dir: Path, eval_image_size: int, clip_backbone_id: str) -> str:
    key = f"{str(target_reference_rgb_dir.resolve())}|{eval_image_size}|inception_v3_pool3|{clip_backbone_id}"
    md5 = hashlib.md5(key.encode("utf-8")).hexdigest()
    clip_tag = clip_backbone_id.replace("/", "_").replace("-", "_")
    return f"eval_ref_cache_{md5}_res{eval_image_size}_inception_v3_pool3_{clip_tag}.pt"


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
        from transformers.utils import logging as transformers_logging
    except Exception as exc:
        raise RuntimeError("Missing transformers CLIP deps. Install transformers.") from exc

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        transformers_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass

    clip_backbone_id = str(eval_cfg.get("clip_backbone_id", "openai/clip-vit-base-patch32"))
    clip_model_cache_dir = str(eval_cfg.get("clip_model_cache_dir", "") or "")
    clip_local_only = bool(eval_cfg.get("clip_local_only", False))
    clip_use_safetensors = bool(eval_cfg.get("clip_use_safetensors", False))

    kwargs: Dict[str, Any] = {
        "local_files_only": clip_local_only,
    }
    if clip_model_cache_dir:
        kwargs["cache_dir"] = clip_model_cache_dir
    if clip_use_safetensors:
        kwargs["use_safetensors"] = True

    model = CLIPModel.from_pretrained(clip_backbone_id, **kwargs).to(device)
    processor = CLIPProcessor.from_pretrained(clip_backbone_id, **kwargs)
    model.eval()
    return clip_backbone_id, model, processor


def _to_feature_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if hasattr(output, "pooler_output") and torch.is_tensor(output.pooler_output):
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state[:, 0]
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported CLIP output type: {type(output)}")


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
        outputs = _to_feature_tensor(clip_model.get_image_features(**inputs))
        outputs = F.normalize(outputs.float(), dim=-1)
        feats.append(outputs.cpu())
        del imgs, inputs, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(feats, dim=0)


def _phase0_parse_config(args: argparse.Namespace) -> EvalConfig:
    config_path = _resolve_repo_path(args.config)
    cfg = load_yaml_config(str(config_path))
    eval_cfg = cfg.get("pixel_cyclegan_eval", {})
    _require(isinstance(eval_cfg, dict), "config.pixel_cyclegan_eval must be a dict")

    domain_A_name = str(eval_cfg.get("domain_A_name", "") or "").strip()
    domain_B_name = str(eval_cfg.get("domain_B_name", "") or "").strip()
    direction_mode = str(eval_cfg.get("direction_mode", "both") or "both").strip().lower()
    single_direction = str(eval_cfg.get("single_direction", "A2B") or "A2B").strip().upper()

    base_test_image_dir = _resolve_repo_path(args.base_test_image_dir or eval_cfg.get("base_test_image_dir", "") or "")
    base_ref_image_dir = _resolve_repo_path(args.base_ref_image_dir or eval_cfg.get("base_ref_image_dir", "") or "")
    eval_output_dir = _resolve_repo_path(args.eval_output_dir or eval_cfg.get("eval_output_dir", "outputs/eval_pixel_cyclegan"))
    reference_cache_dir = _resolve_repo_path(
        args.reference_cache_dir or eval_cfg.get("reference_cache_dir", "outputs/eval_cache_pixel_cyclegan")
    )
    generator_checkpoint_path = _resolve_repo_path(
        args.generator_checkpoint_path or eval_cfg.get("generator_checkpoint_path", "") or ""
    )

    eval_image_size = int(args.eval_image_size or eval_cfg.get("eval_image_size", 256))
    eval_batch_size = int(args.eval_batch_size or eval_cfg.get("eval_batch_size", 8))
    device_str = str(args.run_device or eval_cfg.get("run_device", "cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    run_device = torch.device(device_str)
    use_bf16_autocast = bool(
        args.use_bf16_autocast if args.use_bf16_autocast is not None else eval_cfg.get("use_bf16_autocast", True)
    )
    max_eval_samples = int(eval_cfg.get("max_eval_samples", -1))
    eval_seed = int(eval_cfg.get("eval_seed", 42))

    _require(domain_A_name, "pixel_cyclegan_eval.domain_A_name is required")
    _require(domain_B_name, "pixel_cyclegan_eval.domain_B_name is required")
    _require(direction_mode in {"single", "both"}, "pixel_cyclegan_eval.direction_mode must be 'single' or 'both'")
    _require(single_direction in {"A2B", "B2A"}, "pixel_cyclegan_eval.single_direction must be 'A2B' or 'B2A'")
    _require(str(base_test_image_dir), "pixel_cyclegan_eval.base_test_image_dir is required")
    _require(str(base_ref_image_dir), "pixel_cyclegan_eval.base_ref_image_dir is required")
    _require(str(generator_checkpoint_path), "pixel_cyclegan_eval.generator_checkpoint_path is required")

    _require(base_test_image_dir.exists(), f"base_test_image_dir not found: {base_test_image_dir}")
    _require(base_ref_image_dir.exists(), f"base_ref_image_dir not found: {base_ref_image_dir}")
    _resolve_domain_dir(base_test_image_dir, domain_A_name, "base_test_image_dir")
    _resolve_domain_dir(base_test_image_dir, domain_B_name, "base_test_image_dir")
    _resolve_domain_dir(base_ref_image_dir, domain_A_name, "base_ref_image_dir")
    _resolve_domain_dir(base_ref_image_dir, domain_B_name, "base_ref_image_dir")

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    reference_cache_dir.mkdir(parents=True, exist_ok=True)

    return EvalConfig(
        eval_cfg=eval_cfg,
        domain_A_name=domain_A_name,
        domain_B_name=domain_B_name,
        direction_mode=direction_mode,
        single_direction=single_direction,
        base_test_image_dir=base_test_image_dir,
        base_ref_image_dir=base_ref_image_dir,
        eval_output_dir=eval_output_dir,
        reference_cache_dir=reference_cache_dir,
        generator_checkpoint_path=generator_checkpoint_path,
        max_eval_samples=max_eval_samples,
        eval_seed=eval_seed,
        eval_image_size=eval_image_size,
        eval_batch_size=eval_batch_size,
        run_device=run_device,
        use_bf16_autocast=use_bf16_autocast,
    )


def _phase1_build_or_load_ref_cache(
    ecfg: EvalConfig,
    direction: EvalDirection,
    clip_runtime: Optional[Dict[str, Any]] = None,
    force_regen_cache: bool = False,
) -> Dict[str, Any]:
    from cleanfid import fid

    cleanfid_model = fid.build_feature_extractor(mode="clean", device=ecfg.run_device, use_dataparallel=False)
    disable_clip = bool(ecfg.eval_cfg.get("skip_clip", False))

    clip_model_id = "clip_skipped"
    clip_model = None
    clip_processor = None
    if not disable_clip:
        if clip_runtime is None:
            clip_model_id, clip_model, clip_processor = _load_clip_model(ecfg.eval_cfg, ecfg.run_device)
        else:
            clip_model_id = str(clip_runtime["clip_model_id"])
            clip_model = clip_runtime["clip_model"]
            clip_processor = clip_runtime["clip_processor"]

    def _build_or_load_domain_cache(reference_rgb_dir: Path, cache_tag: str) -> Dict[str, Any]:
        cache_name = f"{cache_tag}_" + _get_cache_name(reference_rgb_dir, ecfg.eval_image_size, clip_model_id)
        cache_path = ecfg.reference_cache_dir / cache_name

        if cache_path.exists() and not force_regen_cache:
            _log(f"[Phase1] Use existing {cache_tag} cache: {cache_path}")
            payload = _torch_load_trusted(cache_path, map_location="cpu")
            payload["cache_path"] = str(cache_path)
            return payload

        _log(f"[Phase1] Building {cache_tag} cache: {reference_rgb_dir}")
        ref_paths = _list_images(reference_rgb_dir)
        ref_cache_max_images = int(ecfg.eval_cfg.get("ref_cache_max_images", 0))

        num_ref_images = None
        shuffle_ref = False
        if ref_cache_max_images > 0:
            num_ref_images = ref_cache_max_images
            shuffle_ref = True
            if len(ref_paths) > ref_cache_max_images:
                random.seed(42)
                ref_paths = random.sample(ref_paths, ref_cache_max_images)

        cleanfid_features = fid.get_folder_features(
            fdir=str(reference_rgb_dir),
            model=cleanfid_model,
            mode="clean",
            num_workers=4,
            num=num_ref_images,
            shuffle=shuffle_ref,
            seed=42,
            batch_size=max(8, ecfg.eval_batch_size),
            device=ecfg.run_device,
            verbose=True,
        )
        cleanfid_features = np.asarray(cleanfid_features)
        mu, sigma = _mean_cov(cleanfid_features)

        clip_payload: Dict[str, Any] = {}
        if not disable_clip and clip_model is not None and clip_processor is not None:
            clip_feats = _clip_image_features(
                image_paths=ref_paths,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=ecfg.run_device,
                batch_size=ecfg.eval_batch_size,
            )
            clip_payload["image_prototype"] = F.normalize(clip_feats.mean(dim=0, keepdim=False), dim=0).cpu()
            num_ref_images_clip = int(clip_feats.shape[0])
        else:
            num_ref_images_clip = 0

        payload: Dict[str, Any] = {
            "meta": {
                "cache_tag": cache_tag,
                "reference_rgb_dir": str(reference_rgb_dir.resolve()),
                "eval_image_size": ecfg.eval_image_size,
                "extractors": ["inception_v3_pool3"] + ([] if disable_clip else [clip_model_id]),
                "num_ref_images_cleanfid": int(cleanfid_features.shape[0]),
                "num_ref_images_clip": num_ref_images_clip,
            },
            "fid_kid": {
                "mu": mu.astype(np.float64),
                "sigma": sigma.astype(np.float64),
            },
            "clip": clip_payload,
            "cache_path": str(cache_path),
        }
        torch.save(payload, cache_path)
        _log(f"[Phase1] {cache_tag} cache saved: {cache_path}")
        return payload

    source_cache = _build_or_load_domain_cache(direction.test_image_dir, "source")
    target_cache = _build_or_load_domain_cache(direction.tgt_ref_dir, "target")

    merged_cache: Dict[str, Any] = {
        "meta": {
            "direction": direction.name,
            "source_domain": direction.src_domain,
            "target_domain": direction.tgt_domain,
        },
        "fid_kid": target_cache["fid_kid"],
        "clip": {
            "source_image_prototype": source_cache.get("clip", {}).get("image_prototype"),
            "target_image_prototype": target_cache.get("clip", {}).get("image_prototype"),
        },
        "cache_path": {
            "source": source_cache.get("cache_path", ""),
            "target": target_cache.get("cache_path", ""),
        },
        "_clip_runtime": {
            "clip_model_id": clip_model_id,
            "clip_model": clip_model,
            "clip_processor": clip_processor,
        },
    }
    return merged_cache


def _extract_tensor_state_dict(payload: Dict[str, Any], keys: Sequence[str]) -> Optional[Dict[str, torch.Tensor]]:
    for k in keys:
        sd_obj = payload.get(k)
        if isinstance(sd_obj, dict) and all(isinstance(v, torch.Tensor) for v in sd_obj.values()):
            return _normalize_state_dict_keys(sd_obj)
    return None


def _load_generators(ecfg: EvalConfig, directions: Sequence[EvalDirection]) -> Dict[str, torch.nn.Module]:
    _require(ecfg.generator_checkpoint_path.exists(), f"checkpoint not found: {ecfg.generator_checkpoint_path}")

    payload = _torch_load_trusted(ecfg.generator_checkpoint_path, map_location="cpu")
    if isinstance(payload, dict):
        checkpoint_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)}")

    direct_state_dict = None
    if all(isinstance(v, torch.Tensor) for v in checkpoint_dict.values()):
        direct_state_dict = _normalize_state_dict_keys(checkpoint_dict)

    state_dict_map = {
        "A2B": _extract_tensor_state_dict(checkpoint_dict, ("G", "netG_A", "generator", "state_dict")),
        "B2A": _extract_tensor_state_dict(checkpoint_dict, ("F", "netG_B")),
    }
    required_names = [direction.name for direction in directions]
    if direct_state_dict is not None:
        for name in required_names:
            if state_dict_map[name] is None:
                state_dict_map[name] = direct_state_dict
                break

    generators: Dict[str, torch.nn.Module] = {}
    for name in required_names:
        state_dict = state_dict_map.get(name)
        if state_dict is None:
            _warn(f"[{name}] Missing generator weights in checkpoint: {ecfg.generator_checkpoint_path}")
            continue

        generator = OfficialResnetGenerator(
            input_nc=3,
            output_nc=3,
            ngf=64,
            norm="instance",
            use_dropout=False,
            n_blocks=9,
            padding_type="reflect",
        )
        for key in list(state_dict.keys()):
            _patch_instance_norm_state_dict(state_dict, generator, key.split("."))
        missing, unexpected = generator.load_state_dict(state_dict, strict=False)
        if missing:
            _warn(f"[{name}] state_dict missing keys: {len(missing)}")
        if unexpected:
            _warn(f"[{name}] state_dict unexpected keys: {len(unexpected)}")
        generator.to(ecfg.run_device).eval()
        generators[name] = generator

    missing_required = [name for name in required_names if name not in generators]
    _require(len(generators) > 0, f"Cannot load any generator weights from: {ecfg.generator_checkpoint_path}")
    if missing_required:
        _warn(f"Some requested directions have no weights and will be skipped: {missing_required}")
    return generators


def _save_tensor_image(img: torch.Tensor, out_path: Path) -> None:
    img = img.detach().to(dtype=torch.float32).cpu().clamp(0, 1)
    pil = transforms.ToPILImage()(img)
    pil.save(out_path)


@torch.no_grad()
def _phase2_infer_and_visualize(
    ecfg: EvalConfig,
    direction: EvalDirection,
    generator: torch.nn.Module,
) -> Dict[str, Any]:
    image_out_dir = direction.out_dir / "eval_images"
    image_out_dir.mkdir(parents=True, exist_ok=True)

    image_paths_all = _list_images(direction.test_image_dir)
    selected_image_paths = _select_image_paths(
        image_paths=image_paths_all,
        max_eval_samples=ecfg.max_eval_samples,
        eval_seed=ecfg.eval_seed,
    )
    selected_stems = [_safe_stem_name(p) for p in selected_image_paths]
    preprocess = _build_eval_transform(ecfg.eval_image_size)

    transferred_paths: List[Path] = []
    pbar = tqdm(list(_batch_iter(selected_image_paths, ecfg.eval_batch_size)), desc=f"[{direction.name}] Phase2 inference", ascii=True)
    for image_batch_paths in pbar:
        stems = [_safe_stem_name(p) for p in image_batch_paths]
        image_tensors = [_load_image_tensor(p, ecfg.eval_image_size, preprocess) for p in image_batch_paths]

        try:
            image_batch = torch.stack(image_tensors, dim=0).to(ecfg.run_device, non_blocking=True)
            use_amp = ecfg.use_bf16_autocast and ecfg.run_device.type == "cuda"
            amp_dtype = torch.bfloat16 if use_amp else torch.float32
            with torch.autocast(device_type=ecfg.run_device.type, enabled=use_amp, dtype=amp_dtype):
                fake_img = generator(image_batch)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            _warn(f"OOM on batch size={len(image_batch_paths)}; fallback to per-sample for this batch")
            if ecfg.run_device.type == "cuda":
                torch.cuda.empty_cache()
            fake_list: List[torch.Tensor] = []
            for image_tensor in image_tensors:
                one = image_tensor.unsqueeze(0).to(ecfg.run_device, non_blocking=True)
                use_amp = ecfg.use_bf16_autocast and ecfg.run_device.type == "cuda"
                amp_dtype = torch.bfloat16 if use_amp else torch.float32
                with torch.autocast(device_type=ecfg.run_device.type, enabled=use_amp, dtype=amp_dtype):
                    one_fake_img = generator(one)
                fake_list.append(one_fake_img[0].detach().cpu())
                del one, one_fake_img
            fake_img = torch.stack(fake_list, dim=0)

        fake_img = (fake_img * 0.5 + 0.5).clamp(0, 1)
        for i, stem in enumerate(stems):
            out_path = image_out_dir / f"{stem}.png"
            _save_tensor_image(fake_img[i], out_path)
            transferred_paths.append(out_path)

        del image_tensors
        if "image_batch" in locals():
            del image_batch
        if "fake_img" in locals():
            del fake_img
        if ecfg.run_device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "transferred_paths": transferred_paths,
        "transferred_dir": image_out_dir,
        "selected_stems": selected_stems,
    }


def _phase3_metrics(
    ecfg: EvalConfig,
    direction: EvalDirection,
    ref_cache: Dict[str, Any],
    phase2_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    from cleanfid import fid

    cleanfid_model = fid.build_feature_extractor(mode="clean", device=ecfg.run_device, use_dataparallel=False)

    disable_lpips = bool(ecfg.eval_cfg.get("skip_lpips", False))
    disable_clip = bool(ecfg.eval_cfg.get("skip_clip", False))

    transferred_dir = Path(phase2_outputs["transferred_dir"])
    transferred_paths = _list_images(transferred_dir)

    gen_feats = fid.get_folder_features(
        fdir=str(transferred_dir),
        model=cleanfid_model,
        mode="clean",
        num_workers=4,
        batch_size=max(8, ecfg.eval_batch_size),
        device=ecfg.run_device,
        verbose=True,
    )
    gen_feats = np.asarray(gen_feats)
    mu_g, sigma_g = _mean_cov(gen_feats)
    mu_r = np.asarray(ref_cache["fid_kid"]["mu"])
    sigma_r = np.asarray(ref_cache["fid_kid"]["sigma"])
    fid_value = _frechet_distance(mu_g, sigma_g, mu_r, sigma_r)

    max_ref_compare = int(ecfg.eval_cfg.get("kid_ref_max_images", 0))
    ref_feats_kid = fid.get_folder_features(
        fdir=str(direction.tgt_ref_dir),
        model=cleanfid_model,
        mode="clean",
        num_workers=4,
        num=(max_ref_compare if max_ref_compare > 0 else None),
        shuffle=(max_ref_compare > 0),
        seed=7,
        batch_size=max(8, ecfg.eval_batch_size),
        device=ecfg.run_device,
        verbose=True,
    )
    ref_feats_kid = np.asarray(ref_feats_kid)

    kid_subsets = int(ecfg.eval_cfg.get("kid_num_subsets", 100))
    kid_subset_size = int(ecfg.eval_cfg.get("kid_subset_size", 1000))
    kid_value, kid_std_value = _kid_mean_std_from_features(
        feats_ref=ref_feats_kid,
        feats_gen=gen_feats,
        num_subsets=kid_subsets,
        max_subset_size=kid_subset_size,
        seed=7,
    )

    rows: List[Dict[str, Any]] = []
    lpips_vals: List[float] = []
    clip_dir_vals: List[float] = []
    clip_style_vals: List[float] = []

    if not disable_lpips or not disable_clip:
        orig_index = _build_stem_index(direction.test_image_dir)
        gen_index = _build_stem_index(transferred_dir)
        selected_stems = [str(s) for s in phase2_outputs.get("selected_stems", [])]
        pair_stems = [s for s in selected_stems if s in orig_index and s in gen_index]
    else:
        orig_index = {}
        gen_index = {}
        pair_stems = []

    lpips_model = None
    if not disable_lpips:
        try:
            import lpips

            lpips_model = lpips.LPIPS(net="vgg").to(ecfg.run_device)
            lpips_model.eval()
        except Exception as exc:
            _warn(f"LPIPS disabled due to import/init error: {exc}")

    clip_model = None
    clip_processor = None
    clip_source_proto = None
    clip_target_proto = None
    if not disable_clip:
        runtime = ref_cache.get("_clip_runtime", {})
        clip_model = runtime.get("clip_model")
        clip_processor = runtime.get("clip_processor")
        clip_source_proto = ref_cache.get("clip", {}).get("source_image_prototype")
        clip_target_proto = ref_cache.get("clip", {}).get("target_image_prototype")
        if (
            clip_model is None
            or clip_processor is None
            or clip_source_proto is None
            or clip_target_proto is None
        ):
            _warn("CLIP runtime/prototype missing in cache, CLIP metrics skipped")
            disable_clip = True

    style_vec = None
    if not disable_clip and clip_source_proto is not None and clip_target_proto is not None:
        src_proto = F.normalize(clip_source_proto.float().to(ecfg.run_device), dim=0)
        tgt_proto = F.normalize(clip_target_proto.float().to(ecfg.run_device), dim=0)
        style_vec = F.normalize(tgt_proto - src_proto, dim=0)

    for batch_stems in tqdm(list(_batch_iter(pair_stems, ecfg.eval_batch_size)), desc=f"[{direction.name}] Phase3 pairwise", ascii=True):
        orig_tensors: List[torch.Tensor] = []
        gen_tensors: List[torch.Tensor] = []
        orig_imgs: List[Image.Image] = []
        gen_imgs: List[Image.Image] = []

        for stem in batch_stems:
            o = _pil_rgb(orig_index[stem], ecfg.eval_image_size)
            g = _pil_rgb(gen_index[stem], ecfg.eval_image_size)
            orig_imgs.append(o)
            gen_imgs.append(g)
            orig_tensors.append(transforms.ToTensor()(o))
            gen_tensors.append(transforms.ToTensor()(g))

        orig_batch = torch.stack(orig_tensors, dim=0).to(ecfg.run_device)
        gen_batch = torch.stack(gen_tensors, dim=0).to(ecfg.run_device)

        if lpips_model is not None:
            o_lp = orig_batch * 2 - 1
            g_lp = gen_batch * 2 - 1
            lp = lpips_model(o_lp, g_lp).view(-1)
            lp_cpu = lp.detach().cpu().tolist()
        else:
            lp_cpu = [float("nan")] * len(batch_stems)

        if not disable_clip and clip_model is not None and clip_processor is not None:
            inp_src = clip_processor(images=orig_imgs, return_tensors="pt")
            inp_gen = clip_processor(images=gen_imgs, return_tensors="pt")
            inp_src = {k: v.to(ecfg.run_device) for k, v in inp_src.items()}
            inp_gen = {k: v.to(ecfg.run_device) for k, v in inp_gen.items()}
            feat_src = _to_feature_tensor(clip_model.get_image_features(**inp_src))
            feat_gen = _to_feature_tensor(clip_model.get_image_features(**inp_gen))
            feat_src = F.normalize(feat_src.float(), dim=-1)
            feat_gen = F.normalize(feat_gen.float(), dim=-1)

            tgt = F.normalize(clip_target_proto.to(feat_gen.device).unsqueeze(0), dim=-1)
            style_vals = F.cosine_similarity(feat_gen, tgt.expand_as(feat_gen), dim=-1)
            style_cpu = style_vals.detach().cpu().tolist()

            if style_vec is not None:
                edit_vec = F.normalize(feat_gen - feat_src, dim=-1)
                dir_vals = F.cosine_similarity(edit_vec, style_vec.unsqueeze(0).expand_as(edit_vec), dim=-1)
                dir_cpu = dir_vals.detach().cpu().tolist()
            else:
                dir_cpu = [float("nan")] * len(batch_stems)
        else:
            style_cpu = [float("nan")] * len(batch_stems)
            dir_cpu = [float("nan")] * len(batch_stems)

        for i, stem in enumerate(batch_stems):
            row = {
                "stem": stem,
                "orig_path": str(orig_index[stem]),
                "transferred_path": str(gen_index[stem]),
                "lpips": float(lp_cpu[i]) if not math.isnan(lp_cpu[i]) else None,
                "clip_dir": float(dir_cpu[i]) if not math.isnan(dir_cpu[i]) else None,
                "clip_style": float(style_cpu[i]) if not math.isnan(style_cpu[i]) else None,
            }
            rows.append(row)
            if not math.isnan(lp_cpu[i]):
                lpips_vals.append(float(lp_cpu[i]))
            if not math.isnan(dir_cpu[i]):
                clip_dir_vals.append(float(dir_cpu[i]))
            if not math.isnan(style_cpu[i]):
                clip_style_vals.append(float(style_cpu[i]))

        del orig_batch, gen_batch
        if ecfg.run_device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "direction": direction.name,
        "src_domain": direction.src_domain,
        "tgt_domain": direction.tgt_domain,
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


def _write_reports(direction: EvalDirection, metric_payload: Dict[str, Any]) -> None:
    report_json = direction.out_dir / f"eval_metrics_{direction.name}.json"
    report_csv = direction.out_dir / f"eval_metrics_{direction.name}.csv"

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(metric_payload["summary"], f, indent=2, ensure_ascii=False)

    rows = metric_payload["rows"]
    fields = ["stem", "orig_path", "transferred_path", "lpips", "clip_dir", "clip_style"]
    with open(report_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})

    _log(f"[{direction.name}][Phase3] Metrics summary:")
    for k, v in metric_payload["summary"].items():
        _log(f"  - {k}: {v}")
    _log(f"[{direction.name}][Phase3] Saved JSON report to: {report_json}")
    _log(f"[{direction.name}][Phase3] Saved CSV report to: {report_csv}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standard pixel-space CycleGAN evaluation")
    p.add_argument("--config", type=str, default="configs/example.yaml")
    p.add_argument("--base_test_image_dir", type=str, default="")
    p.add_argument("--base_ref_image_dir", type=str, default="")
    p.add_argument("--generator_checkpoint_path", type=str, default="")
    p.add_argument("--eval_output_dir", type=str, default="")
    p.add_argument("--reference_cache_dir", type=str, default="")
    p.add_argument("--eval_image_size", type=int, default=0)
    p.add_argument("--eval_batch_size", type=int, default=0)
    p.add_argument("--run_device", type=str, default="")
    p.add_argument("--use_bf16_autocast", action="store_true", default=None)
    p.add_argument("--no_use_bf16_autocast", dest="use_bf16_autocast", action="store_false")
    p.add_argument("--force_regen_cache", action="store_true")
    return p


def _build_directions(ecfg: EvalConfig) -> List[EvalDirection]:
    def _mk(name: str, src_domain: str, tgt_domain: str) -> EvalDirection:
        test_image_dir = _resolve_domain_dir(ecfg.base_test_image_dir, src_domain, "base_test_image_dir")
        tgt_ref_dir = _resolve_domain_dir(ecfg.base_ref_image_dir, tgt_domain, "base_ref_image_dir")
        out_dir = ecfg.eval_output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        return EvalDirection(
            name=name,
            src_domain=src_domain,
            tgt_domain=tgt_domain,
            test_image_dir=test_image_dir,
            tgt_ref_dir=tgt_ref_dir,
            out_dir=out_dir,
        )

    all_directions = {
        "A2B": _mk("A2B", ecfg.domain_A_name, ecfg.domain_B_name),
        "B2A": _mk("B2A", ecfg.domain_B_name, ecfg.domain_A_name),
    }
    if ecfg.direction_mode == "single":
        return [all_directions[ecfg.single_direction]]
    return [all_directions["A2B"], all_directions["B2A"]]


def main() -> None:
    args = build_argparser().parse_args()

    _log("[Phase0] Parsing YAML and applying CLI overrides...")
    ecfg = _phase0_parse_config(args)
    directions = _build_directions(ecfg)

    _log("[Setup] Loading requested generator(s) from checkpoint...")
    generators = _load_generators(ecfg, directions)

    clip_runtime = None
    if not bool(ecfg.eval_cfg.get("skip_clip", False)):
        _log("[Setup] Loading CLIP once for all directions...")
        clip_model_id, clip_model, clip_processor = _load_clip_model(ecfg.eval_cfg, ecfg.run_device)
        clip_runtime = {
            "clip_model_id": clip_model_id,
            "clip_model": clip_model,
            "clip_processor": clip_processor,
        }

    for direction in directions:
        generator = generators.get(direction.name)
        if generator is None:
            _warn(f"[{direction.name}] No generator loaded, skip this direction")
            continue

        _log(
            f"[{direction.name}] src={direction.src_domain} tgt={direction.tgt_domain} "
            f"max_eval_samples={ecfg.max_eval_samples} eval_seed={ecfg.eval_seed}"
        )

        _log(f"[{direction.name}][Phase1] Strict cache pre-computation...")
        ref_cache = _phase1_build_or_load_ref_cache(
            ecfg=ecfg,
            direction=direction,
            clip_runtime=clip_runtime,
            force_regen_cache=args.force_regen_cache,
        )

        _log(f"[{direction.name}][Phase2] Pixel-space inference and image export...")
        phase2_outputs = _phase2_infer_and_visualize(
            ecfg=ecfg,
            direction=direction,
            generator=generator,
        )

        _log(f"[{direction.name}][Phase3] Metric computation and aggregation...")
        metric_payload = _phase3_metrics(
            ecfg=ecfg,
            direction=direction,
            ref_cache=ref_cache,
            phase2_outputs=phase2_outputs,
        )
        _write_reports(direction=direction, metric_payload=metric_payload)


if __name__ == "__main__":
    main()
