from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from eval_common import (
    build_workspace,
    resolve_domain_cache_paths,
    resolve_real_eval_root,
    script_timer_end,
    script_timer_start,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main evaluation pipeline (cache -> generate -> evaluate)")
    parser.add_argument("--config", type=str, default="configs/stargan_v2_latent.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path override (default from eval.checkpoint_path)")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu (default from eval.device)")
    parser.add_argument("--batch_size", type=int, default=None, help="Feature extraction batch size (default from eval.batch_size)")
    parser.add_argument(
        "--max_src",
        type=int,
        default=None,
        help="Number of source samples per source domain (<=0 means use all). If omitted, use eval.max_src_samples from YAML.",
    )
    parser.add_argument("--force", action="store_true", help="Force rebuild cache (or set eval.force_rebuild_cache=true)")
    return parser.parse_args()


def _load_run_fn(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise AttributeError(f"Module {path} has no run(...) function")
    return module.run


def _cache_ready(real_root: Path, domains: list[str]) -> bool:
    for dom in domains:
        fid_path, clip_path = resolve_domain_cache_paths(real_root, dom)
        if (not fid_path.exists()) or (not clip_path.exists()):
            return False
    return True


def run(
    config: str,
    checkpoint: str = "",
    device: str = "",
    batch_size: int | None = None,
    max_src: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    t0, _ = script_timer_start("main_eval")
    ws = build_workspace(config_path=config, checkpoint=(checkpoint or None), device=(device or None))
    real_root = resolve_real_eval_root(ws.config)

    eval_cfg = ws.config.get("eval", {}) if isinstance(ws.config.get("eval", {}), dict) else {}
    if batch_size is None:
        batch_size = int(eval_cfg.get("batch_size", 32))
    if max_src is None:
        max_src = int(eval_cfg.get("max_src_samples", 30))
    force = bool(force) or bool(eval_cfg.get("force_rebuild_cache", False))
    ckpt_eff = checkpoint or str(eval_cfg.get("checkpoint_path", "")).strip()
    device_eff = device or str(eval_cfg.get("device", "")).strip()

    print(f"[main_eval] effective batch_size={batch_size}")
    print(f"[main_eval] effective max_src={max_src}")
    print(f"[main_eval] effective force_rebuild_cache={force}")

    utils_dir = Path(__file__).resolve().parent
    run_cache = _load_run_fn(utils_dir / "01_build_cache.py")
    run_generate = _load_run_fn(utils_dir / "02_generate_images.py")
    run_metrics = _load_run_fn(utils_dir / "03_evaluate_metrics.py")

    stage_info: dict[str, Any] = {}

    if force or (not _cache_ready(real_root, ws.domains)):
        stage_info["01_build_cache"] = run_cache(
            config=config,
            checkpoint=ckpt_eff,
            batch_size=int(batch_size),
            device=device_eff,
        )
    else:
        print("[main_eval] cache already exists, skip 01_build_cache (use --force to rebuild)")
        stage_info["01_build_cache"] = {"skipped": True}

    stage_info["02_generate_images"] = run_generate(
        config=config,
        checkpoint=ckpt_eff,
        device=device_eff,
        max_src=int(max_src),
    )
    summary = run_metrics(
        config=config,
        checkpoint=ckpt_eff,
        device=device_eff,
        batch_size=int(batch_size),
    )
    stage_info["03_evaluate_metrics"] = {"summary_path": str(ws.metrics_dir / "summary.json")}

    total_elapsed = script_timer_end("main_eval", t0)
    pipeline_report = {
        "config": str(ws.config_path),
        "checkpoint": str(ws.checkpoint_path),
        "metrics_dir": str(ws.metrics_dir),
        "stages": stage_info,
        "total_elapsed_sec": total_elapsed,
    }
    write_json(ws.metrics_dir / "pipeline_report.json", pipeline_report)
    return summary


def main() -> None:
    args = parse_args()
    run(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        max_src=args.max_src,
        force=args.force,
    )


if __name__ == "__main__":
    main()