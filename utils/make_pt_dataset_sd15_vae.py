import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def resize_to_multiple_of_8_pil(img):
    w, h = img.size
    new_w = w - (w % 8)
    new_h = h - (h % 8)
    if new_w <= 0 or new_h <= 0:
        raise ValueError(f"Image too small to make /8: {(w, h)}")
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h))
    return img


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    paths: List[Path] = []
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return sorted(paths)


@torch.no_grad()
def _load_vae(model_name_or_path: str, subfolder: Optional[str], device: torch.device) -> torch.nn.Module:
    try:
        from diffusers import AutoencoderKL  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'diffusers'. Install: pip install diffusers transformers accelerate safetensors"
        ) from e

    kwargs = {}
    if subfolder:
        kwargs["subfolder"] = subfolder

    vae = AutoencoderKL.from_pretrained(model_name_or_path, **kwargs)
    vae.to(device)
    vae.eval()
    return vae


def _make_transform(resolution: Optional[int], auto_resize_to_multiple_of_8: bool):
    try:
        from PIL import Image  # noqa: F401
        from torchvision import transforms
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'pillow'/'torchvision'. Install: pip install pillow torchvision") from e

    ops = []
    if resolution is not None:
        _require(resolution % 8 == 0, "--resolution must be a multiple of 8")
        ops.append(transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC))
    elif auto_resize_to_multiple_of_8:
        ops.append(transforms.Lambda(resize_to_multiple_of_8_pil))

    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # to [-1,1]
    ]
    return transforms.Compose(ops)


def _load_image(path: Path):
    from PIL import Image

    img = Image.open(path)
    img = img.convert("RGB")
    return img


class ImageEncodeDataset(torch.utils.data.Dataset):
    def __init__(self, in_dir: Path, paths: Sequence[Path], transform):
        self.in_dir = in_dir
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        p = self.paths[idx]
        rel = p.relative_to(self.in_dir)
        img = self.transform(_load_image(p))
        return str(rel.as_posix()), img


def collate_relpath_images(batch: Sequence[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    rel_paths = [x[0] for x in batch]
    imgs = torch.stack([x[1] for x in batch], dim=0)
    return rel_paths, imgs


def encode_dir_to_pt(
    in_dir: Path,
    image_paths: Sequence[Path],
    vae: torch.nn.Module,
    device: torch.device,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    resolution: Optional[int],
    auto_resize_to_multiple_of_8: bool,
    amp: bool,
    tqdm_ncols: int,
    tqdm_bar_len: int,
) -> None:
    from torch.utils.data import DataLoader
    from tqdm import tqdm  # type: ignore

    transform = _make_transform(resolution=resolution, auto_resize_to_multiple_of_8=auto_resize_to_multiple_of_8)

    ds = ImageEncodeDataset(in_dir=in_dir, paths=image_paths, transform=transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_relpath_images,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(
        total=len(dl),
        desc="encode",
        dynamic_ncols=False,
        ncols=tqdm_ncols,
        ascii=True,
        bar_format=f"{{l_bar}}{{bar:{tqdm_bar_len}}}{{r_bar}}",
    )

    from torch.cuda.amp import autocast

    for rel_paths, imgs in dl:
        imgs = imgs.to(device, non_blocking=True)

        with torch.no_grad():
            with autocast(enabled=amp and device.type == "cuda"):
                # NOTE: output is *unscaled* latents (no 0.18215 scaling)
                lat = vae.encode(imgs).latent_dist.sample()  # type: ignore[attr-defined]

        lat = lat.detach().to(dtype=torch.float32).cpu()

        for i, rel in enumerate(rel_paths):
            rel_path = Path(rel)
            out_path = (out_dir / rel_path).with_suffix(".pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(lat[i], out_path)

        pbar.update(1)

    pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Encode dataset/trainA and dataset/trainB images independently into SD1.5 VAE latents (.pt). "
            "Output filenames follow input (same relative path, suffix .pt)."
        )
    )
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Root dataset dir containing trainA/trainB")
    parser.add_argument("--out_dir", type=str, default="pt_dataset", help="Output root dir containing trainA/trainB")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["trainA", "trainB"],
        help="Which subfolders under dataset_dir to process (default: trainA trainB)",
    )

    parser.add_argument(
        "--vae_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Diffusers model name/path. Typically runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--vae_subfolder", type=str, default="vae", help="Subfolder for VAE weights (optional)")

    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")

    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Optional square resize (must be multiple of 8). If unset, auto-resize to multiple of 8.",
    )
    parser.add_argument(
        "--no_auto_resize_to_multiple_of_8",
        action="store_true",
        help="Disable auto-resize to multiple-of-8 when --resolution is not set",
    )

    parser.add_argument("--tqdm_ncols", type=int, default=120)
    parser.add_argument("--tqdm_bar_len", type=int, default=30)

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_root = Path(args.out_dir)

    splits: List[str] = list(args.splits)
    _require(len(splits) > 0, "--splits must be non-empty")

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available; fallback to cpu")
        device_str = "cpu"
    device = torch.device(device_str)

    vae_subfolder = args.vae_subfolder.strip() if isinstance(args.vae_subfolder, str) else ""
    vae_subfolder = vae_subfolder if vae_subfolder else None

    print("Loading VAE...")
    vae = _load_vae(args.vae_model_name_or_path, vae_subfolder, device)

    for split in splits:
        in_dir = dataset_dir / split
        out_dir = out_root / split
        image_paths = _list_images(in_dir)
        if not image_paths:
            print(f"[warn] No images found under: {in_dir}")
            continue

        print(f"Split '{split}': {len(image_paths)} images")
        encode_dir_to_pt(
            in_dir=in_dir,
            image_paths=image_paths,
            vae=vae,
            device=device,
            out_dir=out_dir,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            resolution=args.resolution,
            auto_resize_to_multiple_of_8=not bool(args.no_auto_resize_to_multiple_of_8),
            amp=bool(args.amp),
            tqdm_ncols=int(args.tqdm_ncols),
            tqdm_bar_len=int(args.tqdm_bar_len),
        )

    print(f"Done. Saved to: {out_root}")


if __name__ == "__main__":
    main()
