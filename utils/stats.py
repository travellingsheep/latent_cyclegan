import argparse
from pathlib import Path
from typing import Any

import torch


def _extract_tensor(obj: Any, path: Path) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        if "latent" in obj and isinstance(obj["latent"], torch.Tensor):
            return obj["latent"]
        raise ValueError(f"Unsupported dict format in {path}. Expected key 'latent' -> Tensor")
    raise ValueError(f"Unsupported content type in {path}: {type(obj)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a .pt tensor: stats + histogram (not saved)")
    parser.add_argument("pt_path", type=str, help="Path to a .pt file")
    parser.add_argument("--bins", type=int, default=100, help="Histogram bins (default: 100)")
    args = parser.parse_args()

    pt_path = Path(args.pt_path)
    if not pt_path.exists() or not pt_path.is_file():
        raise FileNotFoundError(f"Not found: {pt_path}")

    obj = torch.load(pt_path, map_location="cpu")
    t = _extract_tensor(obj, pt_path).detach().to(dtype=torch.float32)

    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]

    flat = t.reshape(-1)
    # Use double for stable moments
    flat64 = flat.to(dtype=torch.float64)

    min_v = float(flat64.min().item())
    max_v = float(flat64.max().item())
    mean_v = float(flat64.mean().item())
    var_v = float(flat64.var(unbiased=False).item())

    print(f"File: {pt_path}")
    print(f"Shape: {tuple(t.shape)}  dtype: {t.dtype}  requires_grad: {bool(t.requires_grad)}")
    print(f"Min:  {min_v}")
    print(f"Max:  {max_v}")
    print(f"Mean: {mean_v}")
    print(f"Var:  {var_v}")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'matplotlib'. Install: pip install matplotlib") from e

    data = flat64.cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=int(args.bins))
    plt.title(f"Histogram: {pt_path.name}")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
