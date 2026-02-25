import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

import eval_latent_cyclegan as ev
import fid_utils


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _load_cfg_maybe(path: Path) -> Dict[str, Any]:
    if path.exists():
        return ev.load_yaml_config(str(path))
    return {}


def _resolve_ckpt_path(args: argparse.Namespace, cfg_file: Dict[str, Any]) -> Path:
    eval_cfg = cfg_file.get("eval", {}) if isinstance(cfg_file.get("eval"), dict) else {}
    train_cfg = cfg_file.get("train", {}) if isinstance(cfg_file.get("train"), dict) else {}

    ckpt_raw = str(getattr(args, "checkpoint", "") or "").strip() or str(eval_cfg.get("checkpoint_path", "")).strip()
    if not ckpt_raw:
        ckpt_dir = str(train_cfg.get("checkpoint_dir", "outputs/model")).strip() or "outputs/model"
        ckpt_raw = str(Path(ckpt_dir) / "last.pt")

    ckpt_path = Path(ckpt_raw)
    _require(ckpt_path.exists(), f"checkpoint not found: {ckpt_path}")
    return ckpt_path


def _resolve_device(args: argparse.Namespace, cfg_file: Dict[str, Any]) -> torch.device:
    eval_cfg = cfg_file.get("eval", {}) if isinstance(cfg_file.get("eval"), dict) else {}
    device_str = str(getattr(args, "device", "") or "").strip() or str(eval_cfg.get("device", "cuda")).strip() or "cuda"
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    return torch.device(device_str)


def _resolve_out_dir(args: argparse.Namespace, ckpt_path: Path, image_tag: str) -> Path:
    out_root = str(getattr(args, "out_dir", "") or "").strip() or "outputs/eval_single"
    out_root_p = Path(out_root)
    # Keep runs separated by (ckpt_stem / image_tag)
    out_dir = out_root_p / ckpt_path.stem / image_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _resolve_single_images(args: argparse.Namespace) -> Tuple[Optional[Path], Optional[Path], str]:
    image = str(getattr(args, "image", "") or "").strip()
    imageA = str(getattr(args, "imageA", "") or "").strip()
    imageB = str(getattr(args, "imageB", "") or "").strip()

    pA: Optional[Path] = Path(imageA) if imageA else None
    pB: Optional[Path] = Path(imageB) if imageB else None

    if image and (pA is None and pB is None):
        # If user gives only one image, use it for both directions (matches the user's request).
        pA = Path(image)
        pB = Path(image)

    _require(pA is not None or pB is not None, "Provide --image OR --imageA/--imageB")

    if pA is not None:
        _require(pA.exists() and pA.is_file(), f"imageA not found: {pA}")
    if pB is not None:
        _require(pB.exists() and pB.is_file(), f"imageB not found: {pB}")

    tag_a = pA.stem if pA is not None else ""
    tag_b = pB.stem if pB is not None else ""

    if tag_a and tag_b:
        if pA is not None and pB is not None and pA.resolve() == pB.resolve():
            tag = tag_a
        else:
            tag = f"A_{tag_a}__B_{tag_b}"
    else:
        tag = tag_a or tag_b

    return pA, pB, tag


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-image inference + metrics for latent CycleGAN. "
            "Loads the latest checkpoint by default and computes the same metrics as eval_latent_cyclegan.py."
        )
    )
    parser.add_argument("--config", type=str, default="configs/example.yaml", help="YAML config path")

    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path (default: {train.checkpoint_dir}/last.pt)")
    parser.add_argument("--out_dir", type=str, default="outputs/eval_single", help="Output root directory")

    parser.add_argument("--image", type=str, default="", help="A single image path (will be used for BOTH A2B and B2A)")
    parser.add_argument("--imageA", type=str, default="", help="Domain A image path (for A2B)")
    parser.add_argument("--imageB", type=str, default="", help="Domain B image path (for B2A)")

    parser.add_argument("--testA", type=str, default="", help="Domain A reference set folder (for default FID stats path)")
    parser.add_argument("--testB", type=str, default="", help="Domain B reference set folder (for default FID stats path)")

    parser.add_argument(
        "--fid_statsA",
        type=str,
        default="",
        help="Path to fid_stats.npz for domain A reference set (default: {testA}/fid_stats.npz)",
    )
    parser.add_argument(
        "--fid_statsB",
        type=str,
        default="",
        help="Path to fid_stats.npz for domain B reference set (default: {testB}/fid_stats.npz)",
    )

    # Metrics settings (subset; defaults mirror eval_latent_cyclegan.py)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    # style_clip settings: compare the generated image to a cached reference set.
    # (A2B uses --testB, B2A uses --testA as reference set roots.)
    parser.add_argument("--max_ref_cache", type=int, default=256, help="Reference cache size for style_clip (0 disables)")
    parser.add_argument("--max_ref_compare", type=int, default=50, help="How many cached refs to compare per image for style_clip (<=0 means all)")
    parser.add_argument("--cache_dir", type=str, default="outputs/eval_cache", help="Cache directory for style_clip reference embeddings")
    parser.add_argument("--force_regen_cache", action="store_true", help="Recompute style_clip reference cache even if exists")

    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--disable_clip", action="store_true")

    parser.add_argument("--clip_model_id", type=str, default="", help="CLIP model id/path (default: openai/clip-vit-base-patch32)")
    parser.add_argument("--clip_cache_dir", type=str, default="")
    parser.add_argument("--clip_local_files_only", action="store_true")
    parser.add_argument("--compact_paths", action="store_true")

    parser.add_argument("--device", type=str, default="", help="cuda or cpu")
    parser.add_argument("--no_amp_bf16", action="store_true")

    parser.add_argument("--no_save_images", action="store_true", help="Do not save side-by-side images")

    # Optional VAE override (otherwise read from checkpoint cfg.visualization / YAML)
    parser.add_argument("--vae_model", type=str, default="")
    parser.add_argument("--vae_subfolder", type=str, default="")
    parser.add_argument("--vae_scaling_factor", type=float, default=0.18215)

    args = parser.parse_args()

    if str(args.image).strip() and (not str(args.imageA).strip()) and (not str(args.imageB).strip()):
        print("[info] --image is provided; will run BOTH A2B and B2A using the same source image.")

    cfg_path = Path(str(args.config).strip() or "configs/example.yaml")
    cfg_file = _load_cfg_maybe(cfg_path)

    ckpt_path = _resolve_ckpt_path(args, cfg_file)
    device = _resolve_device(args, cfg_file)

    imageA_path, imageB_path, image_tag = _resolve_single_images(args)
    out_dir = _resolve_out_dir(args, ckpt_path, image_tag)

    payload = ev._load_ckpt(ckpt_path)
    _require(
        "cfg" in payload and isinstance(payload.get("cfg"), dict),
        "Checkpoint missing key: 'cfg' (expected current training format)",
    )
    cfg_saved = payload.get("cfg", {})

    model_cfg = cfg_saved.get("model", {}) if isinstance(cfg_saved, dict) else {}
    data_cfg = cfg_saved.get("data", {}) if isinstance(cfg_saved, dict) else {}
    vis_cfg_from_ckpt = cfg_saved.get("visualization", {}) if isinstance(cfg_saved, dict) else {}

    in_ch = int(model_cfg.get("in_channels", 4))
    out_ch = int(model_cfg.get("out_channels", 4))
    ngf = int(model_cfg.get("ngf", 32))
    n_res_blocks = int(model_cfg.get("n_res_blocks", 6))
    out_activation = str(model_cfg.get("out_activation", "none"))

    latents_scaled = bool(data_cfg.get("latents_scaled", False))
    latent_divisor = float(data_cfg.get("latent_divisor", 1.0))
    _require(latent_divisor > 0, "latent_divisor must be > 0")

    _require("G" in payload and "F" in payload, "Checkpoint missing keys: expected 'G' and 'F'")

    G = ev.ResnetGenerator(
        in_ch=in_ch,
        out_ch=out_ch,
        ngf=ngf,
        n_res_blocks=n_res_blocks,
        out_activation=out_activation,
    ).to(device)
    Fnet = ev.ResnetGenerator(
        in_ch=in_ch,
        out_ch=out_ch,
        ngf=ngf,
        n_res_blocks=n_res_blocks,
        out_activation=out_activation,
    ).to(device)

    G.load_state_dict(payload["G"], strict=True)
    Fnet.load_state_dict(payload["F"], strict=True)
    G.eval()
    Fnet.eval()

    # Resolve VAE: CLI override > YAML visualization > checkpoint visualization
    vis_cfg_from_file = cfg_file.get("visualization", {}) if isinstance(cfg_file.get("visualization"), dict) else {}

    vae_model = (
        str(args.vae_model).strip()
        or str(vis_cfg_from_file.get("vae_model_name_or_path", "")).strip()
        or str(vis_cfg_from_ckpt.get("vae_model_name_or_path", "")).strip()
    )
    _require(
        vae_model != "",
        "VAE model is not set. Provide --vae_model or set visualization.vae_model_name_or_path in config",
    )

    vae_sub = (
        str(args.vae_subfolder).strip()
        or str(vis_cfg_from_file.get("vae_subfolder", "")).strip()
        or str(vis_cfg_from_ckpt.get("vae_subfolder", "")).strip()
    )
    vae_sub = vae_sub if vae_sub else None

    vae_scaling_factor = float(args.vae_scaling_factor)
    if isinstance(vis_cfg_from_ckpt.get("vae_scaling_factor"), (float, int)) and (not str(args.vae_model).strip()):
        vae_scaling_factor = float(vis_cfg_from_ckpt.get("vae_scaling_factor"))

    vae = ev._load_vae(vae_model, vae_sub, device)

    # Build EvalConfig (merge YAML eval defaults with CLI)
    eval_cfg = cfg_file.get("eval", {}) if isinstance(cfg_file.get("eval"), dict) else {}

    clip_model_id = str(args.clip_model_id).strip() or str(eval_cfg.get("clip_model_id", "openai/clip-vit-base-patch32")).strip()
    clip_cache_dir = str(args.clip_cache_dir).strip() or str(eval_cfg.get("clip_cache_dir", "")).strip()
    clip_local_files_only = bool(eval_cfg.get("clip_local_files_only", False)) or bool(args.clip_local_files_only)
    clip_use_safetensors = bool(eval_cfg.get("clip_use_safetensors", True))

    image_size = int(eval_cfg.get("image_size", args.image_size))
    batch_size = int(eval_cfg.get("batch_size", args.batch_size))
    max_ref_cache = int(eval_cfg.get("max_ref_cache", args.max_ref_cache))
    max_ref_compare = int(eval_cfg.get("max_ref_compare", args.max_ref_compare))
    cache_dir = str(eval_cfg.get("cache_dir", args.cache_dir)).strip() or str(args.cache_dir)
    force_regen_cache = bool(eval_cfg.get("force_regen_cache", False)) or bool(args.force_regen_cache)

    disable_lpips = bool(eval_cfg.get("disable_lpips", False)) or bool(args.disable_lpips)
    disable_clip = bool(eval_cfg.get("disable_clip", False)) or bool(args.disable_clip)

    amp_bf16 = bool(eval_cfg.get("amp_bf16", True)) and (not bool(args.no_amp_bf16))

    cfg_eval = ev.EvalConfig(
        image_size=image_size,
        batch_size=batch_size,
        max_src_samples=0,
        max_ref_cache=max_ref_cache,
        max_ref_compare=max_ref_compare,
        cache_dir=cache_dir or "outputs/eval_cache",
        force_regen_cache=force_regen_cache,
        disable_lpips=disable_lpips,
        disable_clip=disable_clip,
        clip_model_id=clip_model_id or "openai/clip-vit-base-patch32",
        clip_cache_dir=clip_cache_dir,
        clip_local_files_only=clip_local_files_only,
        clip_use_safetensors=clip_use_safetensors,
        compact_paths=bool(args.compact_paths),
        amp_bf16=amp_bf16,
        device=str(device),
    )

    # Pre-load LPIPS/CLIP once to avoid duplicate model loads/prints for A2B and B2A.
    lpips_fn = None if cfg_eval.disable_lpips else ev._load_lpips(device)
    clip_model, clip_processor = (None, None)
    if not cfg_eval.disable_clip:
        clip_model, clip_processor, clip_err = ev._load_clip_with_cfg(
            device=device,
            model_id=str(cfg_eval.clip_model_id or "openai/clip-vit-base-patch32"),
            cache_dir=str(cfg_eval.clip_cache_dir or ""),
            local_files_only=bool(cfg_eval.clip_local_files_only),
            use_safetensors=bool(cfg_eval.clip_use_safetensors),
        )
        if clip_model is None:
            if clip_err and "not installed" in clip_err:
                print("[warn] CLIP not available: transformers is missing. Install: pip install transformers")
            else:
                print(
                    "[warn] CLIP disabled: model could not be loaded. "
                    + (clip_err or "")
                    + " (If offline, pre-download/cached model or set --clip_local_files_only)"
                )
            clip_model, clip_processor = None, None

    # Run directions
    metrics_csv = out_dir / "metrics.csv"
    summary_path = out_dir / "summary.json"

    save_images = not bool(args.no_save_images)

    columns = [
        "direction",
        "src_path",
        "src_image",
        "gen_path",
        "gen_image",
        "content_lpips",
        "content_clip",
        "style_clip",
        "fid",
    ]

    rows_a2b = []
    rows_b2a = []

    src_root = None
    gen_root = (out_dir / "gen") if cfg_eval.compact_paths else None

    # style reference embeddings (optional): use --testA/--testB as reference sets.
    style_ref_a = None
    style_ref_b = None
    if (clip_model is not None) and (clip_processor is not None) and (not cfg_eval.disable_clip):
        if testA is not None and testA.exists():
            try:
                style_ref_a = ev._load_or_build_clip_ref_cache(
                    ref_dir=testA,
                    ref_paths=ev._list_images(testA),
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    device=device,
                    cfg=cfg_eval,
                )
            except Exception:
                style_ref_a = None
        if testB is not None and testB.exists():
            try:
                style_ref_b = ev._load_or_build_clip_ref_cache(
                    ref_dir=testB,
                    ref_paths=ev._list_images(testB),
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    device=device,
                    cfg=cfg_eval,
                )
            except Exception:
                style_ref_b = None

    if imageA_path is not None:
        rows_a2b = ev.evaluate_direction(
            direction="A2B",
            src_paths=[imageA_path],
            gen=G,
            vae=vae,
            device=device,
            latents_scaled=latents_scaled,
            vae_scaling_factor=vae_scaling_factor,
            latent_divisor=latent_divisor,
            out_dir=out_dir,
            cfg=cfg_eval,
            save_images=save_images,
            src_root=src_root,
            gen_root=gen_root,
            lpips_fn=lpips_fn,
            clip_model=clip_model,
            clip_processor=clip_processor,
            style_ref_clip=style_ref_b,
        )

    if imageB_path is not None:
        rows_b2a = ev.evaluate_direction(
            direction="B2A",
            src_paths=[imageB_path],
            gen=Fnet,
            vae=vae,
            device=device,
            latents_scaled=latents_scaled,
            vae_scaling_factor=vae_scaling_factor,
            latent_divisor=latent_divisor,
            out_dir=out_dir,
            cfg=cfg_eval,
            save_images=save_images,
            src_root=src_root,
            gen_root=gen_root,
            lpips_fn=lpips_fn,
            clip_model=clip_model,
            clip_processor=clip_processor,
            style_ref_clip=style_ref_a,
        )

    # FID: A2B uses domain B stats; B2A uses domain A stats.
    testA = Path(str(args.testA).strip()) if str(args.testA).strip() else None
    testB = Path(str(args.testB).strip()) if str(args.testB).strip() else None
    statsA_path = Path(str(args.fid_statsA).strip()) if str(args.fid_statsA).strip() else ((testA / fid_utils.DEFAULT_STATS_NAME) if testA else None)
    statsB_path = Path(str(args.fid_statsB).strip()) if str(args.fid_statsB).strip() else ((testB / fid_utils.DEFAULT_STATS_NAME) if testB else None)

    fid_dir = out_dir / "fid"
    fid_dir.mkdir(parents=True, exist_ok=True)
    fid_a2b: Optional[float] = None
    fid_b2a: Optional[float] = None

    def _compute_fid(gen_folder: Path, ref_stats: Optional[Path], tag: str) -> Optional[float]:
        if ref_stats is None or (not ref_stats.exists()):
            return None
        mu_ref, sigma_ref = fid_utils.load_stats(ref_stats)
        mu_gen, sigma_gen, count, dim = fid_utils.compute_folder_mu_sigma(
            gen_folder,
            device=str(device.type),
            batch_size=max(1, int(cfg_eval.batch_size)),
            num_workers=0,
            mode="clean",
        )
        fid_utils.np.savez(
            fid_dir / f"{tag}_gen_stats.npz",
            mu=fid_utils.np.asarray(mu_gen, dtype=fid_utils.np.float64),
            sigma=fid_utils.np.asarray(sigma_gen, dtype=fid_utils.np.float64),
            count=int(count),
            dim=int(dim),
            gen_dir=str(gen_folder),
            ref_stats=str(ref_stats),
        )
        return float(fid_utils.frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref))

    if rows_a2b:
        fid_a2b = _compute_fid(out_dir / "gen" / "A2B", statsB_path, "A2B")
        for r in rows_a2b:
            r["fid"] = fid_a2b

    if rows_b2a:
        fid_b2a = _compute_fid(out_dir / "gen" / "B2A", statsA_path, "B2A")
        for r in rows_b2a:
            r["fid"] = fid_b2a

    # Write metrics.csv after FID is known.
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows_a2b + rows_b2a:
            writer.writerow(r)

    # Render composites after FID is known.
    if save_images:
        img_root = out_dir / "images"
        for direction, rows, fid_val in (
            ("A2B", rows_a2b, fid_a2b),
            ("B2A", rows_b2a, fid_b2a),
        ):
            if not rows:
                continue
            out_img_dir = img_root / direction
            out_img_dir.mkdir(parents=True, exist_ok=True)
            for idx, r in enumerate(rows, start=1):
                try:
                    src_abs = Path(str(r.get("src_path_abs", "")).strip() or str(r.get("src_path", "")).strip())
                    gen_abs = out_dir / "gen" / direction / str(r.get("gen_image", "")).strip()

                    src01 = ev._load_image_tensor01(src_abs, cfg_eval.image_size)
                    gen01 = ev._load_image_tensor01(gen_abs, cfg_eval.image_size)

                    ev._save_side_by_side_with_metrics(
                        out_path=out_img_dir / str(r.get("gen_image", f"{idx:06d}_{direction}.png")),
                        src01=src01,
                        gen01=gen01,
                        direction=direction,
                        index=int(idx),
                        content_lpips=r.get("content_lpips"),
                        content_clip=r.get("content_clip"),
                        fid=fid_val,
                    )
                except Exception:
                    pass

    summary = {
        "checkpoint": str(ckpt_path),
        "out_dir": str(out_dir),
        "imageA": str(imageA_path) if imageA_path is not None else "",
        "imageB": str(imageB_path) if imageB_path is not None else "",
        "compact_paths": bool(cfg_eval.compact_paths),
        "gen_root": str(gen_root) if cfg_eval.compact_paths and gen_root is not None else "",
        "latents_scaled": latents_scaled,
        "latent_divisor": latent_divisor,
        "vae_model": vae_model,
        "vae_subfolder": vae_sub,
        "vae_scaling_factor": vae_scaling_factor,
        "A2B": ev._summarize_rows(rows_a2b) if rows_a2b else {},
        "B2A": ev._summarize_rows(rows_b2a) if rows_b2a else {},
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if cfg_eval.compact_paths:
        with open(out_dir / "metrics_paths.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "gen_root": str(gen_root or ""),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    print(f"Saved: {metrics_csv}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
