from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from eval_common import (
    InceptionFeatRunner,
    build_workspace,
    clip_embed_paths,
    compute_stats,
    load_clip,
    list_image_files,
    resolve_domain_cache_paths,
    resolve_real_eval_root,
    save_fid_stats,
    script_timer_end,
    script_timer_start,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evaluation caches: FID stats + CLIP prototypes")
    parser.add_argument("--config", type=str, default="configs/stargan_v2_latent.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path override (default from eval.checkpoint_path)")
    parser.add_argument("--batch_size", type=int, default=None, help="Feature extraction batch size (default from eval.batch_size)")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu (default from eval.device)")
    return parser.parse_args()


def run(config: str, checkpoint: str = "", batch_size: int | None = None, device: str = "") -> dict[str, float | int]:
    t0, _ = script_timer_start("01_build_cache")
    ws = build_workspace(config_path=config, checkpoint=(checkpoint or None), device=(device or None))
    eval_cfg = ws.config.get("eval", {}) if isinstance(ws.config.get("eval", {}), dict) else {}
    if batch_size is None:
        batch_size = int(eval_cfg.get("batch_size", 32))

    real_root = resolve_real_eval_root(ws.config)
    print(f"[01_build_cache] RGB eval root: {real_root}")

    domain_to_paths: dict[str, list[Path]] = {}
    for domain in tqdm(ws.domains, desc="扫描真实图", dynamic_ncols=True):
        files = list_image_files(real_root / domain)
        if len(files) < 2:
            raise RuntimeError(f"Domain {domain} has <2 images under {real_root / domain}")
        domain_to_paths[domain] = files
        print(f"[01_build_cache] {domain}: {len(files)} images")

    clip_model, clip_processor = load_clip(ws.device, model_name="openai/clip-vit-base-patch32")
    fid_runner = InceptionFeatRunner(device=ws.device, batch_size=batch_size)
    for domain in tqdm(ws.domains, desc="构建domain缓存", dynamic_ncols=True):
        fid_path, clip_path = resolve_domain_cache_paths(real_root, domain)
        if fid_path.exists() and clip_path.exists():
            print(f"[01_build_cache] skip {domain}: cache exists")
            continue

        feats = fid_runner.extract(domain_to_paths[domain], show_progress=False)
        mu, sigma = compute_stats(feats)
        save_fid_stats(fid_path, mu=mu, sigma=sigma, n=int(feats.shape[0]))

        emb = clip_embed_paths(
            domain_to_paths[domain],
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=ws.device,
            batch_size=batch_size,
            show_progress=False,
        )
        if emb.shape[0] < 1:
            raise RuntimeError(f"No CLIP embeddings for domain {domain}")
        proto = emb.sum(dim=0)
        proto = proto / (proto.norm(p=2) + 1e-8)
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(proto.cpu(), clip_path)

        print(f"[01_build_cache] saved {fid_path}")
        print(f"[01_build_cache] saved {clip_path}")

    elapsed = script_timer_end("01_build_cache", t0)
    return {
        "elapsed_sec": elapsed,
        "num_domains": len(ws.domains),
    }


def main() -> None:
    args = parse_args()
    run(
        config=args.config,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()