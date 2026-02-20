import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_STATS_NAME = "fid_stats.npz"


def list_images_recursive(dir_path: Path) -> List[Path]:
    paths: List[Path] = []
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return sorted(paths)


def _ensure_cleanfid():
    try:
        from cleanfid import fid as _  # type: ignore

        return _
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'clean-fid'. Install it with: pip install clean-fid"
        ) from e


def _try_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)
    except Exception:
        # Fallback: copy (slower but works even without symlink permission)
        shutil.copy2(src, dst)


def extract_features_for_paths(
    image_paths: Sequence[Path],
    *,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
    mode: str = "clean",
) -> Tuple[List[Path], np.ndarray]:
    """Extract clean-fid features for an explicit list of image paths.

    clean-fid APIs are folder-based (often non-recursive). To make behavior predictable,
    this function symlinks/copies images into a temp flat folder with deterministic
    sortable names, then calls clean-fid feature extraction.

    Returns (paths_in_same_order, features[N, D]).
    """

    cfid = _ensure_cleanfid()

    img_list = [Path(p) for p in image_paths]
    if not img_list:
        return [], np.zeros((0, 0), dtype=np.float32)

    # Create a temp flat folder.
    tmp_dir = Path(tempfile.mkdtemp(prefix="cleanfid_flat_"))
    try:
        link_names: List[str] = []
        for i, p in enumerate(img_list):
            name = f"{i:06d}__{p.name}"
            link_names.append(name)
            _try_symlink(p, tmp_dir / name)

        # clean-fid reads from folder; it typically sorts filenames.
        # We enforce the same by our 000000__ prefix.
        feats = cfid.get_folder_features(
            str(tmp_dir),
            mode=str(mode),
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            device=str(device),
            verbose=False,
        )

        # Ensure numpy array
        feats_np = np.asarray(feats)
        if feats_np.ndim != 2:
            raise RuntimeError(f"Unexpected feature shape from clean-fid: {feats_np.shape}")

        if feats_np.shape[0] != len(img_list):
            raise RuntimeError(
                f"Feature count mismatch: got {feats_np.shape[0]} features for {len(img_list)} images"
            )

        return img_list, feats_np
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def compute_mu_sigma(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError(f"features must be [N,D] with N>=1, got {features.shape}")

    x = features.astype(np.float64, copy=False)
    mu = np.mean(x, axis=0)
    sigma = np.cov(x, rowvar=False)
    return mu, sigma


def save_vectors_and_stats(
    *,
    dataset_dir: Path,
    image_paths: Sequence[Path],
    features: np.ndarray,
    mode: str = "clean",
    stats_name: str = DEFAULT_STATS_NAME,
) -> Path:
    """Save per-image vectors under dataset_dir/vectors and dataset-level mu/sigma.

    - vectors path mirrors relative path under dataset_dir.
    - stats saved to dataset_dir/{stats_name} as npz with keys: mu, sigma, mode, count, dim.
    """

    dataset_dir = Path(dataset_dir)
    vectors_root = dataset_dir / "vectors"

    _ = vectors_root.mkdir(parents=True, exist_ok=True)

    if len(image_paths) != int(features.shape[0]):
        raise ValueError("image_paths/features length mismatch")

    # Save vectors
    for p, v in zip(image_paths, features, strict=True):
        try:
            rel = p.resolve().relative_to(dataset_dir.resolve())
            out_path = vectors_root / rel
        except Exception:
            # If images are not strictly under dataset_dir, fall back to basename.
            out_path = vectors_root / p.name

        out_path = out_path.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, np.asarray(v, dtype=np.float32))

    mu, sigma = compute_mu_sigma(features)
    stats_path = dataset_dir / stats_name
    np.savez(
        stats_path,
        mu=np.asarray(mu, dtype=np.float64),
        sigma=np.asarray(sigma, dtype=np.float64),
        mode=str(mode),
        count=int(features.shape[0]),
        dim=int(features.shape[1]),
    )

    return stats_path


def load_stats(stats_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(stats_path)
    mu = np.asarray(d["mu"], dtype=np.float64)
    sigma = np.asarray(d["sigma"], dtype=np.float64)
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Compute FID using clean-fid's implementation when available."""
    try:
        from cleanfid.fid import frechet_distance as _fd  # type: ignore

        return float(_fd(mu1, sigma1, mu2, sigma2))
    except Exception:
        # Fallback: ask cleanfid's public API by creating temporary stats names is messy.
        # If this fails, it's better to error out than to silently use a different impl.
        raise RuntimeError(
            "Could not import cleanfid.fid.frechet_distance. Please upgrade clean-fid to a newer version."
        )


def compute_folder_mu_sigma(
    folder: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
    mode: str = "clean",
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Compute mu/sigma for all images under folder (recursive listing)."""
    folder = Path(folder)
    paths = list_images_recursive(folder)
    if not paths:
        raise FileNotFoundError(f"No images found under: {folder}")

    _, feats = extract_features_for_paths(
        paths,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        mode=mode,
    )
    mu, sigma = compute_mu_sigma(feats)
    return mu, sigma, int(feats.shape[0]), int(feats.shape[1])
