from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm.auto import tqdm

from eval_common import (
    build_workspace,
    decode_latents_to_images_01,
    find_real_image_by_stem,
    list_latent_files,
    load_checkpoint_payload,
    load_latent_tensor,
    load_stargan_modules,
    load_vae_from_config,
    resolve_real_eval_root,
    script_timer_end,
    script_timer_start,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline images for evaluation")
    parser.add_argument("--config", type=str, default="configs/stargan_v2_latent.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path override (default from eval.checkpoint_path)")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu (default from eval.device)")
    parser.add_argument("--max_src", type=int, default=None, help="Number of source samples per source domain (<=0 means use all; default from eval.max_src_samples)")
    return parser.parse_args()


def _resolve_eval_latent_root(config: dict[str, Any]) -> Path:
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    for key in ("eval_data_root", "data_root"):
        raw = str(data_cfg.get(key, "")).strip()
        if not raw:
            continue
        root = Path(raw).expanduser().resolve()
        if root.exists():
            return root
    raise FileNotFoundError("Cannot resolve latent eval root from data.eval_data_root / data.data_root")


def _resolve_eval_latent_root_with_fallback(primary_cfg: dict[str, Any], fallback_cfg: dict[str, Any]) -> Path:
    try:
        return _resolve_eval_latent_root(primary_cfg)
    except Exception:
        return _resolve_eval_latent_root(fallback_cfg)


def _resolve_real_root_with_fallback(primary_cfg: dict[str, Any], fallback_cfg: dict[str, Any]) -> Path:
    try:
        return resolve_real_eval_root(primary_cfg)
    except Exception:
        return resolve_real_eval_root(fallback_cfg)


def _pick_refs(files: list[Path], count: int, rng: random.Random) -> list[Path]:
    if len(files) >= count:
        picked = files.copy()
        rng.shuffle(picked)
        return picked[:count]
    return [files[rng.randrange(0, len(files))] for _ in range(count)]


def _to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr)


def _resize_keep(pil_img: Image.Image, width: int, height: int) -> Image.Image:
    if pil_img.size == (width, height):
        return pil_img
    return pil_img.resize((width, height), resample=Image.BILINEAR)


@torch.no_grad()
def run(config: str, checkpoint: str = "", device: str = "", max_src: int | None = None) -> dict[str, float | int]:
    t0, _ = script_timer_start("02_generate_images")
    ws = build_workspace(config_path=config, checkpoint=(checkpoint or None), device=(device or None))

    payload = load_checkpoint_payload(ws.checkpoint_path, ws.device)
    cfg_ckpt = payload.get("config", ws.config) if isinstance(payload.get("config", {}), dict) else ws.config
    cfg_runtime = ws.config

    eval_cfg = cfg_runtime.get("eval", {}) if isinstance(cfg_runtime.get("eval", {}), dict) else {}
    if max_src is None:
        max_src = int(eval_cfg.get("max_src_samples", 30))

    latent_root = _resolve_eval_latent_root_with_fallback(cfg_runtime, cfg_ckpt)
    real_root = _resolve_real_root_with_fallback(cfg_runtime, cfg_ckpt)

    cfg = cfg_ckpt

    G, E, M = load_stargan_modules(payload, cfg, ws.device)
    vae, vae_dtype = load_vae_from_config(cfg, ws.device)

    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    latent_scale = float(training_cfg.get("latent_scale", 0.18215))
    z_dim = int(model_cfg.get("latent_dim", 16))
    base_seed = int(training_cfg.get("seed", 42))
    set_seed(base_seed)
    rng = random.Random(base_seed)

    src_files_by_domain: dict[str, list[Path]] = {}
    tgt_files_by_domain: dict[str, list[Path]] = {}
    for dom in ws.domains:
        files = list_latent_files(latent_root / dom)
        if not files:
            raise RuntimeError(f"No latent files in {latent_root / dom}")
        if int(max_src) <= 0:
            src_used = files
        else:
            src_used = files[: min(len(files), int(max_src))]
        src_files_by_domain[dom] = src_used
        tgt_files_by_domain[dom] = files
        print(f"[02_generate_images] domain={dom} src_used={len(src_used)} (total={len(files)}, max_src={max_src})")

    source_dir = ws.images_dir / "source"
    ref_pure_dir = ws.images_dir / "ref" / "pure"
    ref_vis_dir = ws.images_dir / "ref" / "vis_concat"
    std_pure_dir = ws.images_dir / "noise_std" / "pure"
    div_dir = ws.images_dir / "noise_div"
    for p in [source_dir, ref_pure_dir, ref_vis_dir, std_pure_dir, div_dir]:
        p.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "checkpoint": str(ws.checkpoint_path),
        "latent_root": str(latent_root),
        "domains": ws.domains,
        "source_images": [],
        "pairs": [],
        "diversity_groups": [],
    }

    src_tensor_cache: dict[tuple[str, int], torch.Tensor] = {}
    for src_dom in tqdm(ws.domains, desc="解码source", dynamic_ncols=True):
        src_files = src_files_by_domain[src_dom]
        for i, src_path in enumerate(tqdm(src_files, desc=f"source:{src_dom}", dynamic_ncols=True, leave=False)):
            raw_lat = load_latent_tensor(src_path).unsqueeze(0)
            src_tensor_cache[(src_dom, i)] = raw_lat
            img = decode_latents_to_images_01(vae, raw_lat, device=ws.device, vae_dtype=vae_dtype)[0]
            src_name = src_path.stem
            out_path = source_dir / src_dom / f"{src_name}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _to_pil(img).save(out_path)
            manifest["source_images"].append(
                {
                    "src_domain": src_dom,
                    "index": i,
                    "latent_path": str(src_path),
                    "src_name": src_name,
                    "image_path": str(out_path.relative_to(ws.metrics_dir)),
                }
            )

    z_global = torch.randn(1, z_dim, device=ws.device)

    pair_list = [(s, t) for s in ws.domains for t in ws.domains]
    for src_dom, tgt_dom in tqdm(pair_list, desc="生成(ref/std/div)", dynamic_ncols=True):
        src_files = src_files_by_domain[src_dom]
        ref_files = _pick_refs(tgt_files_by_domain[tgt_dom], len(src_files), rng=rng)

        y_tgt = torch.tensor([ws.domains.index(tgt_dom)], device=ws.device, dtype=torch.long)

        for i in tqdm(range(len(src_files)), desc=f"{src_dom}->{tgt_dom}", dynamic_ncols=True, leave=False):
            src_path = src_files[i]
            src_name = src_path.stem
            src_raw = src_tensor_cache[(src_dom, i)].to(ws.device)
            src_scaled = src_raw * latent_scale

            ref_path = ref_files[i]
            ref_name = ref_path.stem
            ref_raw = load_latent_tensor(ref_path).unsqueeze(0).to(ws.device)
            ref_scaled = ref_raw * latent_scale
            # IMPORTANT: style encoder input keeps latent tensor form (ref_raw/ref_scaled),
            # not RGB image. The middle visualization column below is exactly VAE-decoded
            # reconstruction from this latent, i.e. what the model actually sees for style.
            style_ref = E(ref_scaled, y_tgt)
            gen_ref_scaled = G(src_scaled, style_ref)
            gen_ref = decode_latents_to_images_01(vae, gen_ref_scaled / latent_scale, device=ws.device, vae_dtype=vae_dtype)[0]
            ref_img = decode_latents_to_images_01(vae, ref_raw, device=ws.device, vae_dtype=vae_dtype)[0]

            real_ref_path = find_real_image_by_stem(real_root / tgt_dom, ref_name)
            if real_ref_path is None:
                raise FileNotFoundError(f"Cannot find real RGB ref image for {tgt_dom}/{ref_name} under {real_root / tgt_dom}")
            real_ref_img = Image.open(real_ref_path).convert("RGB")

            out_ref = ref_pure_dir / f"{src_dom}_to_{tgt_dom}" / f"{src_name}_to_{ref_name}.png"
            out_ref.parent.mkdir(parents=True, exist_ok=True)
            _to_pil(gen_ref).save(out_ref)

            recon_ref_pil = _to_pil(ref_img)
            gen_ref_pil = _to_pil(gen_ref)
            col_w, col_h = gen_ref_pil.size
            real_ref_pil = _resize_keep(real_ref_img, col_w, col_h)
            recon_ref_pil = _resize_keep(recon_ref_pil, col_w, col_h)
            gen_ref_pil = _resize_keep(gen_ref_pil, col_w, col_h)

            concat_ref = Image.new("RGB", (col_w * 3, col_h))
            concat_ref.paste(real_ref_pil, (0, 0))
            concat_ref.paste(recon_ref_pil, (col_w, 0))
            concat_ref.paste(gen_ref_pil, (col_w * 2, 0))
            out_ref_vis = ref_vis_dir / f"{src_dom}_to_{tgt_dom}" / f"{src_name}_to_{ref_name}.png"
            out_ref_vis.parent.mkdir(parents=True, exist_ok=True)
            concat_ref.save(out_ref_vis)

            style_std = M(z_global, y_tgt)
            gen_std_scaled = G(src_scaled, style_std)
            gen_std = decode_latents_to_images_01(vae, gen_std_scaled / latent_scale, device=ws.device, vae_dtype=vae_dtype)[0]
            out_std = std_pure_dir / f"{src_dom}_to_{tgt_dom}" / f"{i:03d}.png"
            out_std.parent.mkdir(parents=True, exist_ok=True)
            _to_pil(gen_std).save(out_std)

            manifest["pairs"].append(
                {
                    "mode": "ref",
                    "src_domain": src_dom,
                    "tgt_domain": tgt_dom,
                    "index": i,
                    "src_image": str((source_dir / src_dom / f"{src_name}.png").relative_to(ws.metrics_dir)),
                    "gen_image": str(out_ref.relative_to(ws.metrics_dir)),
                    "ref_image": str(ref_path),
                    "real_ref_image": str(real_ref_path),
                }
            )
            manifest["pairs"].append(
                {
                    "mode": "noise_std",
                    "src_domain": src_dom,
                    "tgt_domain": tgt_dom,
                    "index": i,
                    "src_image": str((source_dir / src_dom / f"{src_name}.png").relative_to(ws.metrics_dir)),
                    "gen_image": str(out_std.relative_to(ws.metrics_dir)),
                }
            )

            div_paths: list[str] = []
            div_imgs: list[Image.Image] = []
            for k in range(5):
                z = torch.randn(1, z_dim, device=ws.device)
                style = M(z, y_tgt)
                gen_div_scaled = G(src_scaled, style)
                gen_div = decode_latents_to_images_01(vae, gen_div_scaled / latent_scale, device=ws.device, vae_dtype=vae_dtype)[0]
                out_div = div_dir / f"{src_dom}_to_{tgt_dom}" / f"{i:03d}_k{k}.png"
                out_div.parent.mkdir(parents=True, exist_ok=True)
                img_pil = _to_pil(gen_div)
                img_pil.save(out_div)
                div_imgs.append(img_pil)
                div_paths.append(str(out_div.relative_to(ws.metrics_dir)))

            w, h = div_imgs[0].size
            concat_div = Image.new("RGB", (w, h * 5))
            for k, im in enumerate(div_imgs):
                concat_div.paste(im, (0, k * h))
            out_div_concat = div_dir / f"{src_dom}_to_{tgt_dom}" / f"{i:03d}_div_concat.png"
            concat_div.save(out_div_concat)

            manifest["diversity_groups"].append(
                {
                    "src_domain": src_dom,
                    "tgt_domain": tgt_dom,
                    "index": i,
                    "src_image": str((source_dir / src_dom / f"{src_name}.png").relative_to(ws.metrics_dir)),
                    "images": div_paths,
                    "concat": str(out_div_concat.relative_to(ws.metrics_dir)),
                }
            )

    manifest_path = ws.images_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"[02_generate_images] saved manifest: {manifest_path}")

    elapsed = script_timer_end("02_generate_images", t0)
    return {
        "elapsed_sec": elapsed,
        "num_pairs": len(manifest["pairs"]),
        "num_div_groups": len(manifest["diversity_groups"]),
    }


def main() -> None:
    args = parse_args()
    run(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        max_src=args.max_src,
    )


if __name__ == "__main__":
    main()