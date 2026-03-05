from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("Missing dependency 'pyyaml'. Install: pip install pyyaml") from exc

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a dict")
    return cfg


def _config_contains_placeholder(cfg: Any, needle: str) -> bool:
    if isinstance(cfg, dict):
        return any(_config_contains_placeholder(v, needle) for v in cfg.values())
    if isinstance(cfg, list):
        return any(_config_contains_placeholder(v, needle) for v in cfg)
    if isinstance(cfg, str):
        return needle in cfg
    return False


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def expand_config_templates(config: dict[str, Any]) -> dict[str, Any]:
    exp_cfg = config.get("experiment", {}) if isinstance(config.get("experiment", {}), dict) else {}
    exp_name = str(exp_cfg.get("name", config.get("exp_name", ""))).strip()

    has_exp = _config_contains_placeholder(config, "{exp_name}") or _config_contains_placeholder(config, "${exp_name}")
    has_out = _config_contains_placeholder(config, "{outputs_dir}") or _config_contains_placeholder(config, "${outputs_dir}")
    if (has_exp or has_out) and (not exp_name):
        raise ValueError("Config uses {exp_name}/{outputs_dir} but experiment.name is missing")

    outputs_dir = str(exp_cfg.get("outputs_dir", f"outputs/{exp_name}" if exp_name else "outputs")).strip()
    fmt = _SafeFormatDict(exp_name=exp_name, outputs_dir=outputs_dir)

    def rec(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: rec(v) for k, v in node.items()}
        if isinstance(node, list):
            return [rec(v) for v in node]
        if isinstance(node, str):
            s = node.replace("${exp_name}", "{exp_name}").replace("${outputs_dir}", "{outputs_dir}")
            try:
                return s.format_map(fmt)
            except Exception:
                return node
        return node

    return rec(config)


@dataclass
class EvalWorkspace:
    config_path: Path
    config: dict[str, Any]
    checkpoint_path: Path
    run_dir: Path
    metrics_dir: Path
    images_dir: Path
    domains: list[str]
    device: torch.device


def build_workspace(*, config_path: str | Path, checkpoint: str | None = None, device: str | None = None) -> EvalWorkspace:
    cfg_path = Path(config_path).expanduser().resolve()
    cfg = expand_config_templates(load_yaml_config(cfg_path))

    ckpt_cfg = cfg.get("checkpoint", {}) if isinstance(cfg.get("checkpoint", {}), dict) else {}
    eval_cfg = cfg.get("eval", {}) if isinstance(cfg.get("eval", {}), dict) else {}
    run_dir = Path(str(ckpt_cfg.get("save_dir", "")).strip() or "outputs/ckpt").expanduser().resolve()

    ckpt_arg = str(checkpoint or "").strip()
    ckpt_from_eval = str(eval_cfg.get("checkpoint_path", "")).strip()
    ckpt_raw = ckpt_arg or ckpt_from_eval
    ckpt_path = Path(ckpt_raw).expanduser().resolve() if ckpt_raw else (run_dir / "latest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    exp_dir = run_dir.parent  
    # 获取类似 "latest" 或 "epoch_0080" 的名字
    ckpt_name = ckpt_path.stem  
    
    # 建立平级的 metrics 文件夹，并按 epoch 隔离
    metrics_dir = exp_dir / "metrics" / ckpt_name
    
    images_dir = metrics_dir / "images"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    domains = [str(x) for x in data_cfg.get("domains", [])] if isinstance(data_cfg.get("domains", []), list) else []
    if not domains:
        raise ValueError("Config must provide data.domains")

    dev_raw = str(device or "").strip() or str(eval_cfg.get("device", "")).strip()
    if not dev_raw:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(dev_raw)
        if dev.type == "cuda" and (not torch.cuda.is_available()):
            dev = torch.device("cpu")

    return EvalWorkspace(
        config_path=cfg_path,
        config=cfg,
        checkpoint_path=ckpt_path,
        run_dir=run_dir,
        metrics_dir=metrics_dir,
        images_dir=images_dir,
        domains=domains,
        device=dev,
    )


def list_image_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def list_latent_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pt", ".npy"}:
            files.append(p)
    return sorted(files)


def load_latent_tensor(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".pt":
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            if "latent" in data:
                data = data["latent"]
            elif "x" in data:
                data = data["x"]
            else:
                data = next(iter(data.values()))
        tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    elif path.suffix.lower() == ".npy":
        arr = np.load(path)
        tensor = torch.from_numpy(arr)
    else:
        raise ValueError(f"Unsupported latent file: {path}")

    tensor = tensor.float()
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(tensor.shape)} from {path}")
    return tensor.contiguous()


def load_image_tensor_01(path: Path, size: int | None = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None and int(size) > 0:
        img = img.resize((int(size), int(size)), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_tensor_image_01(x: torch.Tensor, path: Path) -> None:
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(x.detach().cpu().clamp(0.0, 1.0), str(path))


def normalize_state_dict_keys(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in state.items():
        nk = str(k)
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod.") :]
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        out[nk] = v
    return out


def load_checkpoint_payload(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return payload


def load_stargan_modules(payload: dict[str, Any], config: dict[str, Any], device: torch.device):
    from model import build_models

    models = build_models(config)
    G = models.generator.to(device).eval()
    E = models.style_encoder.to(device).eval()
    M = models.mapping_network.to(device).eval()

    if "G" not in payload or "E" not in payload or "F" not in payload:
        raise KeyError("Checkpoint must contain keys: G, E, F")

    G.load_state_dict(normalize_state_dict_keys(payload["G"]), strict=True)
    E.load_state_dict(normalize_state_dict_keys(payload["E"]), strict=True)
    M.load_state_dict(normalize_state_dict_keys(payload["F"]), strict=True)
    return G, E, M


def load_vae_from_config(config: dict[str, Any], device: torch.device):
    try:
        from diffusers import AutoencoderKL
    except Exception as exc:
        raise RuntimeError("Missing dependency 'diffusers'. Install: pip install diffusers") from exc

    vis_cfg = config.get("visualization", {}) if isinstance(config.get("visualization", {}), dict) else {}
    training_cfg = config.get("training", {}) if isinstance(config.get("training", {}), dict) else {}
    model_name = str(vis_cfg.get("vae_model_name_or_path", "runwayml/stable-diffusion-v1-5")).strip()
    subfolder = str(vis_cfg.get("vae_subfolder", "vae")).strip() or None
    amp_dtype = str(training_cfg.get("amp_dtype", "bf16")).lower()
    vae_dtype = torch.bfloat16 if (device.type == "cuda" and amp_dtype == "bf16") else torch.float16
    if device.type != "cuda":
        vae_dtype = torch.float32

    kwargs: dict[str, Any] = {"torch_dtype": vae_dtype}
    if subfolder:
        kwargs["subfolder"] = subfolder
    vae = AutoencoderKL.from_pretrained(model_name, **kwargs)
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, vae_dtype


@torch.no_grad()
def decode_latents_to_images_01(vae, latents_unscaled: torch.Tensor, device: torch.device, vae_dtype: torch.dtype) -> torch.Tensor:
    lat = latents_unscaled.to(device=device, dtype=vae_dtype)
    decoded = vae.decode(lat).sample
    decoded = decoded.float().clamp(-1.0, 1.0)
    return (decoded + 1.0) / 2.0


def resolve_real_eval_root(config: dict[str, Any]) -> Path:
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    domains = [str(d) for d in data_cfg.get("domains", [])] if isinstance(data_cfg.get("domains", []), list) else []

    candidates: list[str] = []
    for k in ("eval_data_root", "image_root", "data_root"):
        v = str(data_cfg.get(k, "")).strip()
        if v:
            candidates.append(v)

    for raw in candidates:
        root = Path(raw).expanduser().resolve()
        if not root.exists():
            continue
        if not domains:
            return root
        ok = True
        for dom in domains:
            dom_dir = root / dom
            if not list_image_files(dom_dir):
                ok = False
                break
        if ok:
            return root

    raise FileNotFoundError(
        "Could not find a valid RGB eval root from data.eval_data_root / data.image_root / data.data_root"
    )


def resolve_domain_cache_paths(real_root: Path, domain: str) -> tuple[Path, Path]:
    domain_dir = real_root / domain
    return domain_dir / "fid_stats.pkl", domain_dir / "clip_prototype.pt"


def find_real_image_by_stem(domain_dir: Path, stem: str) -> Path | None:
    for ext in sorted(IMG_EXTS):
        cand = domain_dir / f"{stem}{ext}"
        if cand.exists() and cand.is_file():
            return cand
    for p in sorted(domain_dir.glob(f"{stem}.*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return p
    return None


class InceptionFeatRunner:
    def __init__(self, *, device: torch.device, batch_size: int):
        import torchvision.models as models
        import torchvision.transforms as T

        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            transform_input=False,
        )
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)
        self.tfm = T.Compose(
            [
                T.Resize((299, 299)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def extract(self, image_paths: Iterable[Path], show_progress: bool = True) -> np.ndarray:
        paths = list(image_paths)
        if not paths:
            return np.empty((0, 2048), dtype=np.float64)

        feats: list[np.ndarray] = []
        iterator = range(0, len(paths), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Inception特征提取", dynamic_ncols=True)
        for s in iterator:
            batch = paths[s : s + self.batch_size]
            imgs = [self.tfm(Image.open(p).convert("RGB")) for p in batch]
            x = torch.stack(imgs, dim=0).to(self.device)
            y = self.model(x)
            if y.ndim > 2:
                y = torch.flatten(y, 1)
            feats.append(y.detach().cpu().double().numpy())
        if not feats:
            return np.empty((0, 2048), dtype=np.float64)
        return np.concatenate(feats, axis=0)


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if (not np.isfinite(covmean).all()) or np.isnan(covmean).any():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("Need at least 2 samples to compute covariance")
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def save_fid_stats(path: Path, *, mu: np.ndarray, sigma: np.ndarray, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"mu": mu, "sigma": sigma, "n": int(n)}, f)


def load_fid_stats(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or ("mu" not in data) or ("sigma" not in data):
        raise ValueError(f"Invalid FID stats file: {path}")
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_clip(device: torch.device, model_name: str = "openai/clip-vit-base-patch32"):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:
        raise RuntimeError("Missing dependency 'transformers'. Install: pip install transformers") from exc

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def _extract_clip_embeddings(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "image_embeds") and getattr(output, "image_embeds") is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and getattr(output, "pooler_output") is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and getattr(output, "last_hidden_state") is not None:
        hidden = output.last_hidden_state
        if isinstance(hidden, torch.Tensor):
            return hidden[:, 0, :]
    if isinstance(output, dict):
        if "image_embeds" in output and isinstance(output["image_embeds"], torch.Tensor):
            return output["image_embeds"]
        if "pooler_output" in output and isinstance(output["pooler_output"], torch.Tensor):
            return output["pooler_output"]
        if "last_hidden_state" in output and isinstance(output["last_hidden_state"], torch.Tensor):
            return output["last_hidden_state"][:, 0, :]
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        first = output[0]
        if first.ndim == 3:
            return first[:, 0, :]
        return first
    raise RuntimeError(f"Could not extract CLIP embeddings from output type={type(output)}")


@torch.no_grad()
def clip_embed_paths(paths: list[Path], clip_model, clip_processor, device: torch.device, batch_size: int = 32, show_progress: bool = True) -> torch.Tensor:
    feats: list[torch.Tensor] = []
    iterator = range(0, len(paths), max(1, int(batch_size)))
    if show_progress:
        iterator = tqdm(iterator, desc="CLIP特征提取", dynamic_ncols=True)
    for s in iterator:
        batch = paths[s : s + max(1, int(batch_size))]
        pil = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=pil, return_tensors="pt").to(device)
        output = clip_model.get_image_features(**inputs)
        emb = _extract_clip_embeddings(output).to(device=device, dtype=torch.float32)
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        emb = F.normalize(emb, p=2, dim=-1)
        feats.append(emb.detach().cpu())
    if not feats:
        return torch.empty((0, 512), dtype=torch.float32)
    return torch.cat(feats, dim=0)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def script_timer_start(name: str) -> tuple[float, str]:
    t0 = time.time()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{name}] start: {ts}")
    return t0, ts


def script_timer_end(name: str, t0: float) -> float:
    dt = float(time.time() - t0)
    print(f"[{name}] done in {dt:.2f}s")
    return dt