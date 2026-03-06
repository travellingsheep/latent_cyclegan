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


def _resolve_source_cache_dir(cfg_runtime: dict[str, Any], run_dir: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    data_cfg = cfg_runtime.get("data", {}) if isinstance(cfg_runtime.get("data", {}), dict) else {}
    decoded_raw = str(data_cfg.get("decoded_source_dir", "")).strip()
    if decoded_raw:
        p = Path(decoded_raw).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.resolve()
        return p

    exp_dir = run_dir.parent
    return (exp_dir / "metrics" / "common_source_images").resolve()


@torch.no_grad()
def _decode_in_micro_batches(
    vae,
    latents_unscaled: torch.Tensor,
    *,
    device: torch.device,
    vae_dtype: torch.dtype,
    micro_batch: int,
) -> torch.Tensor:
    mb = max(1, int(micro_batch))
    outs: list[torch.Tensor] = []
    for s in range(0, int(latents_unscaled.shape[0]), mb):
        e = min(s + mb, int(latents_unscaled.shape[0]))
        outs.append(
            decode_latents_to_images_01(
                vae,
                latents_unscaled[s:e],
                device=device,
                vae_dtype=vae_dtype,
            )
        )
    return torch.cat(outs, dim=0)


@torch.no_grad()
def run(config: str, checkpoint: str = "", device: str = "", max_src: int | None = None) -> dict[str, float | int]:
    t0, _ = script_timer_start("02_generate_images")
    ws = build_workspace(config_path=config, checkpoint=(checkpoint or None), device=(device or None))

    payload = load_checkpoint_payload(ws.checkpoint_path, ws.device)
    cfg_ckpt = payload.get("config", ws.config) if isinstance(payload.get("config", {}), dict) else ws.config
    cfg_runtime = ws.config

    eval_cfg = cfg_runtime.get("eval", {}) if isinstance(cfg_runtime.get("eval", {}), dict) else {}
    batch_size = int(eval_cfg.get("batch_size", 8))
    micro_batch = int(eval_cfg.get("micro_batch", 4))
    if max_src is None:
        max_src = int(eval_cfg.get("max_src_samples", 30))

    latent_root = _resolve_eval_latent_root_with_fallback(cfg_runtime, cfg_ckpt)
    real_root = _resolve_real_root_with_fallback(cfg_runtime, cfg_ckpt)
    source_dir = _resolve_source_cache_dir(cfg_runtime, ws.run_dir)
    print(f"[02_generate_images] source cache dir: {source_dir}")
    print(f"[02_generate_images] batch_size={batch_size}, micro_batch={micro_batch}")

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

    ref_pure_dir = ws.images_dir / "ref" / "pure"
    ref_vis_dir = ws.images_dir / "ref" / "vis_concat"
    std_pure_dir = ws.images_dir / "noise_std" / "pure"
    div_dir = ws.images_dir / "noise_div"
    for p in [source_dir, ref_pure_dir, ref_vis_dir, std_pure_dir, div_dir]:
        p.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "checkpoint": str(ws.checkpoint_path),
        "latent_root": str(latent_root),
        "source_dir": str(source_dir),
        "domains": ws.domains,
        "source_images": [],
        "pairs": [],
        "diversity_groups": [],
    }

    src_tensor_cache: dict[tuple[str, int], torch.Tensor] = {}
    source_cache_hit = 0
    source_decode_new = 0
    source_cache_stats: dict[str, dict[str, int]] = {}
    for src_dom in tqdm(ws.domains, desc="解码source", dynamic_ncols=True):
        src_files = src_files_by_domain[src_dom]
        dom_hit = 0
        dom_new = 0
        for i, src_path in enumerate(tqdm(src_files, desc=f"source:{src_dom}", dynamic_ncols=True, leave=False)):
            # MUST keep this memory cache for downstream generator pass.
            raw_lat = load_latent_tensor(src_path).unsqueeze(0)
            src_tensor_cache[(src_dom, i)] = raw_lat
            src_name = src_path.stem
            out_path = source_dir / src_dom / f"{src_name}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                img = decode_latents_to_images_01(vae, raw_lat, device=ws.device, vae_dtype=vae_dtype)[0]
                _to_pil(img).save(out_path)
                source_decode_new += 1
                dom_new += 1
            else:
                source_cache_hit += 1
                dom_hit += 1
            manifest["source_images"].append(
                {
                    "src_domain": src_dom,
                    "index": i,
                    "latent_path": str(src_path),
                    "src_name": src_name,
                    "image_path": str(out_path),
                }
            )
        source_cache_stats[src_dom] = {
            "cache_hit": dom_hit,
            "decode_new": dom_new,
            "total": len(src_files),
        }
        print(f"[02_generate_images] source cache stats [{src_dom}] hit={dom_hit}, new={dom_new}, total={len(src_files)}")

    print(
        "[02_generate_images] source cache summary "
        f"hit={source_cache_hit}, new={source_decode_new}, total={source_cache_hit + source_decode_new}"
    )

    z_global = torch.randn(1, z_dim, device=ws.device)

    pair_list = [(s, t) for s in ws.domains for t in ws.domains]
    for src_dom, tgt_dom in tqdm(pair_list, desc="生成(ref/std/div)", dynamic_ncols=True):
        src_files = src_files_by_domain[src_dom]
        ref_files = _pick_refs(tgt_files_by_domain[tgt_dom], len(src_files), rng=rng)

        tgt_id = ws.domains.index(tgt_dom)
        pair_iter = range(0, len(src_files), max(1, int(batch_size)))
        for s in tqdm(pair_iter, desc=f"{src_dom}->{tgt_dom}", dynamic_ncols=True, leave=False):
            e = min(s + max(1, int(batch_size)), len(src_files))
            batch_indices = list(range(s, e))
            bsz = len(batch_indices)

            src_raw_batch = torch.stack(
                [src_tensor_cache[(src_dom, i)].squeeze(0) for i in batch_indices],
                dim=0,
            ).to(ws.device)
            ref_raw_batch = torch.stack(
                [load_latent_tensor(ref_files[i]) for i in batch_indices],
                dim=0,
            ).to(ws.device)

            src_scaled = src_raw_batch * latent_scale
            ref_scaled = ref_raw_batch * latent_scale
            y_tgt_batch = torch.full((bsz,), tgt_id, device=ws.device, dtype=torch.long)

            # IMPORTANT: style encoder input keeps latent tensor form (ref_raw/ref_scaled),
            # not RGB image. The reconstructed ref image is decoded from this latent source.
            style_ref_batch = E(ref_scaled, y_tgt_batch)
            gen_ref_scaled_batch = G(src_scaled, style_ref_batch)

            gen_ref_batch = _decode_in_micro_batches(
                vae,
                gen_ref_scaled_batch / latent_scale,
                device=ws.device,
                vae_dtype=vae_dtype,
                micro_batch=micro_batch,
            ).cpu()
            ref_img_batch = _decode_in_micro_batches(
                vae,
                ref_raw_batch,
                device=ws.device,
                vae_dtype=vae_dtype,
                micro_batch=micro_batch,
            ).cpu()

            z_std_batch = z_global.expand(bsz, -1)
            style_std_batch = M(z_std_batch, y_tgt_batch)
            gen_std_scaled_batch = G(src_scaled, style_std_batch)
            gen_std_batch = _decode_in_micro_batches(
                vae,
                gen_std_scaled_batch / latent_scale,
                device=ws.device,
                vae_dtype=vae_dtype,
                micro_batch=micro_batch,
            ).cpu()

            div_decoded_batches: list[torch.Tensor] = []
            for _k in range(5):
                z_div_batch = torch.randn(bsz, z_dim, device=ws.device)
                style_div_batch = M(z_div_batch, y_tgt_batch)
                gen_div_scaled_batch = G(src_scaled, style_div_batch)
                div_decoded_batches.append(
                    _decode_in_micro_batches(
                        vae,
                        gen_div_scaled_batch / latent_scale,
                        device=ws.device,
                        vae_dtype=vae_dtype,
                        micro_batch=micro_batch,
                    ).cpu()
                )

            for j, i in enumerate(batch_indices):
                src_path = src_files[i]
                src_name = src_path.stem
                ref_path = ref_files[i]
                ref_name = ref_path.stem

                gen_ref = gen_ref_batch[j]
                ref_img = ref_img_batch[j]
                gen_std = gen_std_batch[j]

                real_ref_path = find_real_image_by_stem(real_root / tgt_dom, ref_name)
                if real_ref_path is None:
                    raise FileNotFoundError(
                        f"Cannot find real RGB ref image for {tgt_dom}/{ref_name} under {real_root / tgt_dom}"
                    )
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

                out_std = std_pure_dir / f"{src_dom}_to_{tgt_dom}" / f"{i:03d}.png"
                out_std.parent.mkdir(parents=True, exist_ok=True)
                _to_pil(gen_std).save(out_std)

                manifest["pairs"].append(
                    {
                        "mode": "ref",
                        "src_domain": src_dom,
                        "tgt_domain": tgt_dom,
                        "index": i,
                        "src_image": str(source_dir / src_dom / f"{src_name}.png"),
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
                        "src_image": str(source_dir / src_dom / f"{src_name}.png"),
                        "gen_image": str(out_std.relative_to(ws.metrics_dir)),
                    }
                )

                div_paths: list[str] = []
                div_imgs: list[Image.Image] = []
                for k in range(5):
                    gen_div = div_decoded_batches[k][j]
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
                        "src_image": str(source_dir / src_dom / f"{src_name}.png"),
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
        "source_cache_hit": source_cache_hit,
        "source_decode_new": source_decode_new,
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