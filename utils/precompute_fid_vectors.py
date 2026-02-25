import argparse
from pathlib import Path

from fid_utils import (
    DEFAULT_STATS_NAME,
    extract_features_for_paths,
    list_images_recursive,
    save_vectors_and_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute clean-fid feature vectors for a dataset folder and save: "
            "(1) per-image vectors under {dataset}/vectors, (2) dataset mu/sigma to {dataset}/fid_stats.npz."
        )
    )
    parser.add_argument("data_dir", type=str, help="Dataset image directory (will be scanned recursively)")

    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mode", type=str, default="clean", help="clean-fid mode (usually 'clean')")
    parser.add_argument(
        "--stats_name",
        type=str,
        default=DEFAULT_STATS_NAME,
        help="Stats filename saved under data_dir (default: fid_stats.npz)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    paths = list_images_recursive(data_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under: {data_dir}")

    print(f"Found {len(paths)} images under {data_dir}")

    paths_out, feats = extract_features_for_paths(
        paths,
        device=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        mode=str(args.mode),
    )

    stats_path = save_vectors_and_stats(
        dataset_dir=data_dir,
        image_paths=paths_out,
        features=feats,
        mode=str(args.mode),
        stats_name=str(args.stats_name),
    )

    print(f"Saved vectors to: {data_dir / 'vectors'}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
