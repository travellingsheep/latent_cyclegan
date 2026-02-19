import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def _extract_tensor(obj: Any, path: Path) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        if "latent" in obj and isinstance(obj["latent"], torch.Tensor):
            return obj["latent"]
        raise ValueError(f"Unsupported dict format in {path}. Expected key 'latent' -> Tensor")
    raise ValueError(f"Unsupported content type in {path}: {type(obj)}")


def _summarize_tensor(t: torch.Tensor) -> str:
    shape = tuple(t.shape)
    dtype = str(t.dtype).replace("torch.", "")
    req = bool(t.requires_grad)
    return f"shape={shape} dtype={dtype} requires_grad={req}"


def _list_pt_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.pt") if p.is_file()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect .pt latent tensor shapes")
    parser.add_argument("path", type=str, help="A .pt file or a directory containing .pt files")
    parser.add_argument("--limit", type=int, default=50, help="Max files to print when path is a directory")
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")

    if p.is_file():
        obj = torch.load(p, map_location="cpu")
        t = _extract_tensor(obj, p)
        print(f"{p}: {_summarize_tensor(t)}")
        return

    files = _list_pt_files(p)
    if not files:
        raise FileNotFoundError(f"No .pt files found under: {p}")

    limit = max(1, int(args.limit))
    for i, fp in enumerate(files[:limit], start=1):
        obj = torch.load(fp, map_location="cpu")
        t = _extract_tensor(obj, fp)
        print(f"[{i:04d}] {fp}: {_summarize_tensor(t)}")

    if len(files) > limit:
        print(f"... ({len(files) - limit} more files)")


if __name__ == "__main__":
    main()
