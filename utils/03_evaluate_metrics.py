from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from eval_common import (
    InceptionFeatRunner,
    build_workspace,
    clip_embed_paths,
    compute_stats,
    frechet_distance,
    load_clip,
    load_fid_stats,
    resolve_domain_cache_paths,
    resolve_real_eval_root,
    script_timer_end,
    script_timer_start,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated images and write metrics/summary.json")
    parser.add_argument("--config", type=str, default="configs/stargan_v2_latent.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path override (default from eval.checkpoint_path)")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu (default from eval.device)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for FID/CLIP extraction (default from eval.batch_size)")
    return parser.parse_args()


def _to_lpips_input(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def _load_img01(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@torch.no_grad()
def _lpips_dist(loss_fn, a: torch.Tensor, b: torch.Tensor, device: torch.device) -> float:
    x = a.unsqueeze(0).to(device)
    y = b.unsqueeze(0).to(device)
    d = loss_fn(_to_lpips_input(x), _to_lpips_input(y))
    return float(d.view(-1)[0].detach().cpu().item())


def _mean_or_none(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(np.mean(np.asarray(xs, dtype=np.float64)))


def run(config: str, checkpoint: str = "", device: str = "", batch_size: int | None = None) -> dict[str, Any]:
    t0, _ = script_timer_start("03_evaluate_metrics")
    ws = build_workspace(config_path=config, checkpoint=(checkpoint or None), device=(device or None))
    eval_cfg = ws.config.get("eval", {}) if isinstance(ws.config.get("eval", {}), dict) else {}
    if batch_size is None:
        batch_size = int(eval_cfg.get("batch_size", 32))

    manifest_path = ws.images_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing generated manifest: {manifest_path}")
    manifest = torch.load(manifest_path) if manifest_path.suffix == ".pt" else None
    if manifest is None:
        import json

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

    pairs = manifest.get("pairs", []) if isinstance(manifest, dict) else []
    div_groups = manifest.get("diversity_groups", []) if isinstance(manifest, dict) else []
    if not isinstance(pairs, list) or not isinstance(div_groups, list):
        raise ValueError("Invalid manifest.json format")

    real_root = resolve_real_eval_root(ws.config)
    fid_stats: dict[str, dict[str, Any]] = {}
    clip_protos: dict[str, torch.Tensor] = {}
    for dom in ws.domains:
        fid_path, clip_path = resolve_domain_cache_paths(real_root, dom)
        if not fid_path.exists() or not clip_path.exists():
            raise FileNotFoundError(f"Missing domain cache for {dom}: {fid_path} / {clip_path}")
        fid_stats[dom] = load_fid_stats(fid_path)
        clip_protos[dom] = torch.load(clip_path, map_location="cpu").float()

    try:
        import lpips
    except Exception as exc:
        raise RuntimeError("Missing dependency 'lpips'. Install: pip install lpips") from exc
    lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(ws.device).eval()

    clip_model, clip_processor = load_clip(ws.device, model_name="openai/clip-vit-base-patch32")
    inception = InceptionFeatRunner(device=ws.device, batch_size=batch_size)

    # 1) LPIPS content / CLIP content / CLIP-dir / CLIP-style
    pair_metrics: list[dict[str, Any]] = []
    pbar_pairs = tqdm(pairs, desc="逐图指标(LPIPS/CLIP)", dynamic_ncols=True)
    for item in pbar_pairs:
        src = ws.metrics_dir / str(item["src_image"])
        gen = ws.metrics_dir / str(item["gen_image"])
        if (not src.exists()) or (not gen.exists()):
            continue

        src_img = _load_img01(src)
        gen_img = _load_img01(gen)
        lpips_content = _lpips_dist(lpips_fn, src_img, gen_img, ws.device)

        real_ref_raw = str(item.get("real_ref_image", "")).strip()
        real_ref = Path(real_ref_raw).expanduser()
        if not real_ref_raw or not real_ref.exists():
            if str(item.get("mode", "")) == "ref":
                raise FileNotFoundError(f"Missing real_ref_image for ref pair: {item}")
            real_ref = None

        clip_inputs = [src, gen] if real_ref is None else [src, gen, real_ref]
        emb = clip_embed_paths(clip_inputs, clip_model, clip_processor, ws.device, batch_size=3, show_progress=False)
        src_clip = emb[0]
        gen_clip = emb[1]
        ref_clip = emb[2] if emb.shape[0] >= 3 else None
        clip_content = float(torch.sum(src_clip * gen_clip).item())
        clip_style = None if ref_clip is None else float(torch.sum(gen_clip * ref_clip).item())

        tgt = str(item["tgt_domain"])
        if tgt not in clip_protos:
            raise KeyError(f"Missing CLIP prototype for domain: {tgt}")
        proto = F.normalize(clip_protos[tgt].float(), p=2, dim=0)
        delta_i = gen_clip - src_clip
        delta_s = proto - src_clip
        clip_dir = float(F.cosine_similarity(delta_i.unsqueeze(0), delta_s.unsqueeze(0), dim=1).item())

        pair_metrics.append(
            {
                "mode": str(item["mode"]),
                "src_domain": str(item["src_domain"]),
                "tgt_domain": tgt,
                "index": int(item["index"]),
                "lpips_content": lpips_content,
                "clip_content": clip_content,
                "clip_style": clip_style,
                "clip_dir": clip_dir,
                "gen_image": str(item["gen_image"]),
            }
        )

    # 2) FID / Delta FID / Art-FID per (mode, src->tgt)
    grouped_metrics: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for m in pair_metrics:
        grouped_metrics[(m["mode"], m["src_domain"], m["tgt_domain"])].append(m)

    fid_results: list[dict[str, Any]] = []
    for key in tqdm(sorted(grouped_metrics.keys()), desc="分组FID", dynamic_ncols=True):
        mode, src_dom, tgt_dom = key
        group = grouped_metrics[key]
        gen_paths = [ws.metrics_dir / g["gen_image"] for g in group]
        gen_paths = [p for p in gen_paths if p.exists()]
        if len(gen_paths) < 2:
            continue

        gen_feats = inception.extract(gen_paths, show_progress=False)
        mu_g, sigma_g = compute_stats(gen_feats)
        mu_t = np.asarray(fid_stats[tgt_dom]["mu"], dtype=np.float64)
        sg_t = np.asarray(fid_stats[tgt_dom]["sigma"], dtype=np.float64)
        mu_s = np.asarray(fid_stats[src_dom]["mu"], dtype=np.float64)
        sg_s = np.asarray(fid_stats[src_dom]["sigma"], dtype=np.float64)

        fid_gen = float(frechet_distance(mu_g, sigma_g, mu_t, sg_t))
        fid_base = float(frechet_distance(mu_s, sg_s, mu_t, sg_t))
        delta_fid = float(fid_base - fid_gen)

        lpips_mean = _mean_or_none([float(x["lpips_content"]) for x in group])
        art_fid = None if lpips_mean is None else float((1.0 + fid_gen) * (1.0 + lpips_mean))

        fid_results.append(
            {
                "mode": mode,
                "src_domain": src_dom,
                "tgt_domain": tgt_dom,
                "num_images": len(gen_paths),
                "fid_gen": fid_gen,
                "fid_base": fid_base,
                "delta_fid": delta_fid,
                "art_fid": art_fid,
            }
        )

    # 3) Diversity LPIPS (pairwise over 5 variants)
    diversity_scores: list[dict[str, Any]] = []
    for g in tqdm(div_groups, desc="多样性LPIPS", dynamic_ncols=True):
        imgs = [ws.metrics_dir / str(p) for p in g.get("images", [])]
        imgs = [p for p in imgs if p.exists() and ("_div_concat" not in p.name)]
        if len(imgs) != 5:
            continue
        tensors = [_load_img01(p) for p in imgs]
        dvals: list[float] = []
        for i, j in itertools.combinations(range(5), 2):
            dvals.append(_lpips_dist(lpips_fn, tensors[i], tensors[j], ws.device))
        diversity_scores.append(
            {
                "src_domain": str(g["src_domain"]),
                "tgt_domain": str(g["tgt_domain"]),
                "index": int(g["index"]),
                "pairwise_lpips": float(np.mean(np.asarray(dvals, dtype=np.float64))),
            }
        )

    mode_summary: dict[str, dict[str, float | None]] = {}
    for mode in sorted({m["mode"] for m in pair_metrics}):
        ms = [m for m in pair_metrics if m["mode"] == mode]
        fr = [r for r in fid_results if r["mode"] == mode]
        mode_summary[mode] = {
            "lpips_content_mean": _mean_or_none([float(x["lpips_content"]) for x in ms]),
            "clip_content_mean": _mean_or_none([float(x["clip_content"]) for x in ms]),
            "clip_style_mean": _mean_or_none([float(x["clip_style"]) for x in ms if x["clip_style"] is not None]),
            "clip_dir_mean": _mean_or_none([float(x["clip_dir"]) for x in ms]),
            "fid_gen_mean": _mean_or_none([float(x["fid_gen"]) for x in fr]),
            "delta_fid_mean": _mean_or_none([float(x["delta_fid"]) for x in fr]),
            "art_fid_mean": _mean_or_none([float(x["art_fid"]) for x in fr if x["art_fid"] is not None]),
        }

    summary = {
        "config": str(ws.config_path),
        "checkpoint": str(ws.checkpoint_path),
        "metrics_dir": str(ws.metrics_dir),
        "counts": {
            "pairs": len(pair_metrics),
            "fid_groups": len(fid_results),
            "diversity_groups": len(diversity_scores),
        },
        "means": {
            "lpips_content": _mean_or_none([float(x["lpips_content"]) for x in pair_metrics]),
            "clip_content": _mean_or_none([float(x["clip_content"]) for x in pair_metrics]),
            "clip_style": _mean_or_none([float(x["clip_style"]) for x in pair_metrics if x["clip_style"] is not None]),
            "clip_dir": _mean_or_none([float(x["clip_dir"]) for x in pair_metrics]),
            "fid_gen": _mean_or_none([float(x["fid_gen"]) for x in fid_results]),
            "delta_fid": _mean_or_none([float(x["delta_fid"]) for x in fid_results]),
            "art_fid": _mean_or_none([float(x["art_fid"]) for x in fid_results if x["art_fid"] is not None]),
            "diversity_lpips": _mean_or_none([float(x["pairwise_lpips"]) for x in diversity_scores]),
        },
        "by_mode": mode_summary,
        "details": {
            "pairs": pair_metrics,
            "fid": fid_results,
            "diversity": diversity_scores,
        },
    }

    summary_path = ws.metrics_dir / "summary.json"
    write_json(summary_path, summary)
    print(f"[03_evaluate_metrics] saved {summary_path}")

    elapsed = script_timer_end("03_evaluate_metrics", t0)
    summary["elapsed_sec"] = elapsed
    write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    run(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()