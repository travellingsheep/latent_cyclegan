import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


@dataclass
class RunResult:
    run_name: str
    run_dir: Path
    status: str
    message: str = ""
    return_code: Optional[int] = None
    summary: Optional[Dict[str, Any]] = None

SCRIPT_DIR = Path(__file__).resolve().parent
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep evaluator for latent CycleGAN runs")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("configs") / "example.yaml"),
        help="Base YAML config path (used to resolve default exp_root)",
    )
    parser.add_argument(
        "--exp_root",
        type=str,
        default="",
        help="Experiment root directory containing run_* subfolders; overrides YAML exp_root",
    )
    parser.add_argument(
        "--evaluate_script",
        type=str,
        default=str(SCRIPT_DIR / "evaluate_latent.py"),
        help="Path to evaluate_latent.py",
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python executable used to launch evaluation subprocess",
    )
    parser.add_argument(
        "--force_regen_cache",
        action="store_true",
        help="Forward --force_regen_cache to evaluate_latent.py",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config top-level must be a mapping: {path}")
    return data


def resolve_exp_root(config_path: Path, cfg: Dict[str, Any], exp_root_override: str) -> Path:
    if exp_root_override.strip():
        return Path(exp_root_override).expanduser().resolve()

    exp_root = cfg.get("exp_root")
    if not isinstance(exp_root, str) or not exp_root.strip():
        raise ValueError("exp_root missing in config and --exp_root not provided")

    raw = Path(exp_root)
    if raw.is_absolute():
        return raw
    return (config_path.parent / raw).resolve()


def _resolve_path_value(config_dir: Path, value: str) -> str:
    text = value.strip()
    if not text:
        return value
    if "://" in text:
        return value
    raw = Path(text).expanduser()
    if raw.is_absolute():
        return str(raw)
    return str((config_dir / raw).resolve())


def _set_nested_path_if_present(cfg: Dict[str, Any], keys: Tuple[str, ...], config_dir: Path) -> None:
    cur: Any = cfg
    for key in keys[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return
        cur = cur[key]

    leaf = keys[-1]
    if not isinstance(cur, dict) or leaf not in cur:
        return
    value = cur.get(leaf)
    if isinstance(value, str):
        cur[leaf] = _resolve_path_value(config_dir, value)


def write_resolved_eval_config(base_config_path: Path, exp_root: Path) -> Path:
    cfg = load_yaml(base_config_path)
    config_dir = base_config_path.parent.resolve()

    path_fields: List[Tuple[str, ...]] = [
        ("exp_root",),
        ("data", "a_dir"),
        ("data", "b_dir"),
        ("train", "checkpoint_dir"),
        ("train", "resume_path"),
        ("logging", "log_dir"),
        ("visualization", "out_dir"),
        ("cyclegan_eval", "base_latent_dir"),
        ("cyclegan_eval", "base_orig_rgb_dir"),
        ("cyclegan_eval", "base_vae_recon_dir"),
        ("cyclegan_eval", "eval_output_dir"),
        ("cyclegan_eval", "reference_cache_dir"),
        ("cyclegan_eval", "generator_checkpoint_path"),
        ("cyclegan_eval", "clip_model_cache_dir"),
    ]

    for field_keys in path_fields:
        _set_nested_path_if_present(cfg, field_keys, config_dir)

    cfg["exp_root"] = str(exp_root)

    resolved_path = exp_root / "eval_sweep_resolved_config.yaml"
    with resolved_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return resolved_path


def resolve_reference_cache_dir(config_path: Path, cfg: Dict[str, Any], exp_root: Path) -> Path:
    eval_cfg = cfg.get("cyclegan_eval", {})
    if isinstance(eval_cfg, dict):
        raw = eval_cfg.get("reference_cache_dir", "")
        if isinstance(raw, str) and raw.strip():
            return Path(_resolve_path_value(config_path.parent.resolve(), raw))

    return exp_root / "shared_eval_cache"


def is_valid_existing_run(run_dir: Path) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    summary_path = run_dir / "eval_run_summary.json"
    all_scores_path = run_dir / "eval_run_all_images_scores.csv"
    if not summary_path.exists() or not all_scores_path.exists():
        return False, None, "missing eval_run_summary.json or eval_run_all_images_scores.csv"

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        if not isinstance(summary, dict):
            return False, None, "eval_run_summary.json is not a JSON object"
        if "A2B" not in summary or "B2A" not in summary:
            return False, None, "eval_run_summary.json missing A2B/B2A keys"
        pd.read_csv(all_scores_path)
    except Exception as exc:
        return False, None, f"invalid existing summary files: {exc}"

    return True, summary, "ok"


def _stream_reader(stream, label: str, log_fp, lock: threading.Lock) -> None:
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            out_line = f"[{now_str()}] [{label}] {line.rstrip()}\n"
            with lock:
                print(out_line, end="", flush=True)
                log_fp.write(out_line)
                log_fp.flush()
    finally:
        stream.close()


def run_evaluation_subprocess(
    run_dir: Path,
    resolved_config_path: Path,
    evaluate_script: Path,
    python_exec: str,
    shared_cache_dir: Path,
    repo_root: Path,
    force_regen_cache: bool = False,
) -> int:
    checkpoint_path = run_dir / "model" / "last.pt"
    out_dir = run_dir / "eval_results"
    log_path = run_dir / "eval_eval_stdout_stderr.log"

    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {resolved_config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not evaluate_script.exists():
        raise FileNotFoundError(f"Missing evaluate script: {evaluate_script}")

    out_dir.mkdir(parents=True, exist_ok=True)
    shared_cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exec,
        str(evaluate_script),
        "--config",
        str(resolved_config_path),
        "--generator_checkpoint_path",
        str(checkpoint_path),
        "--eval_output_dir",
        str(out_dir),
        "--reference_cache_dir",
        str(shared_cache_dir),
    ]
    if force_regen_cache:
        cmd.append("--force_regen_cache")

    with log_path.open("a", encoding="utf-8") as log_fp:
        log_fp.write(f"[{now_str()}] {'=' * 80}\n")
        log_fp.write(f"[{now_str()}] Command: {' '.join(cmd)}\n")
        log_fp.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        lock = threading.Lock()
        threads = [
            threading.Thread(target=_stream_reader, args=(proc.stdout, "stdout", log_fp, lock), daemon=True),
            threading.Thread(target=_stream_reader, args=(proc.stderr, "stderr", log_fp, lock), daemon=True),
        ]

        for t in threads:
            t.start()

        return_code = proc.wait()

        for t in threads:
            t.join()

    return return_code


def aggregate_run_outputs(run_dir: Path) -> Dict[str, Any]:
    eval_dir = run_dir / "eval_results"
    a2b_json = eval_dir / "A2B" / "eval_metrics_A2B.json"
    b2a_json = eval_dir / "B2A" / "eval_metrics_B2A.json"
    a2b_csv = eval_dir / "A2B" / "eval_metrics_A2B.csv"
    b2a_csv = eval_dir / "B2A" / "eval_metrics_B2A.csv"

    for path in [a2b_json, b2a_json, a2b_csv, b2a_csv]:
        if not path.exists():
            raise FileNotFoundError(f"Expected eval artifact not found: {path}")

    with a2b_json.open("r", encoding="utf-8") as f:
        a2b_summary = json.load(f)
    with b2a_json.open("r", encoding="utf-8") as f:
        b2a_summary = json.load(f)

    run_summary = {"A2B": a2b_summary, "B2A": b2a_summary}

    summary_path = run_dir / "eval_run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    df_a2b = pd.read_csv(a2b_csv)
    df_a2b["direction"] = "A2B"
    df_b2a = pd.read_csv(b2a_csv)
    df_b2a["direction"] = "B2A"

    df_all = pd.concat([df_a2b, df_b2a], ignore_index=True, sort=False)
    all_scores_path = run_dir / "eval_run_all_images_scores.csv"
    df_all.to_csv(all_scores_path, index=False)

    return run_summary


def flatten_summary(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for direction in ["A2B", "B2A"]:
        metrics = run_summary.get(direction, {})
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            flat_key = f"{direction}_{key}"
            if isinstance(value, (dict, list)):
                row[flat_key] = json.dumps(value, ensure_ascii=False)
            else:
                row[flat_key] = value
    return row


def write_global_summary(exp_root: Path, results: List[RunResult]) -> None:
    rows: List[Dict[str, Any]] = []
    for result in results:
        row: Dict[str, Any] = {
            "run_name": result.run_name,
            "status": result.status,
        }
        if result.summary:
            row.update(flatten_summary(result.summary))
        rows.append(row)

    df = pd.DataFrame(rows)

    fixed = ["run_name", "status"]
    metric_cols = sorted([c for c in df.columns if c not in fixed])
    ordered_cols = [c for c in fixed if c in df.columns] + metric_cols
    df = df.reindex(columns=ordered_cols)

    out_path = exp_root / "eval_sweep_summary.csv"
    df.to_csv(out_path, index=False)
    log(f"[DONE] Global summary saved: {out_path}")


def collect_run_dirs(exp_root: Path) -> List[Path]:
    dirs = [p for p in exp_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    return sorted(dirs, key=lambda p: p.name)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_yaml(config_path)
    exp_root = resolve_exp_root(config_path, cfg, args.exp_root)
    if not exp_root.exists():
        raise FileNotFoundError(f"exp_root not found: {exp_root}")

    resolved_config_path = write_resolved_eval_config(config_path, exp_root)

    evaluate_script = Path(args.evaluate_script).expanduser()
    if not evaluate_script.is_absolute():
        evaluate_script = (repo_root / evaluate_script).resolve()

    shared_cache_dir = resolve_reference_cache_dir(config_path, cfg, exp_root)
    run_dirs = collect_run_dirs(exp_root)
    if not run_dirs:
        raise RuntimeError(f"No run_* directories found under: {exp_root}")

    log(f"Using config: {config_path}")
    log(f"Resolved eval config: {resolved_config_path}")
    log(f"Experiment root: {exp_root}")
    log(f"Evaluate script: {evaluate_script}")
    log(f"Shared cache dir: {shared_cache_dir}")
    log(f"Found {len(run_dirs)} run directories")

    results: List[RunResult] = []
    for run_dir in run_dirs:
        run_name = run_dir.name
        log(f"[RUN] Processing {run_name}")

        valid_existing, existing_summary, msg = is_valid_existing_run(run_dir)
        if valid_existing:
            log(f"[SKIP] {run_name} already evaluated")
            results.append(
                RunResult(
                    run_name=run_name,
                    run_dir=run_dir,
                    status="skipped_existing",
                    message="already evaluated",
                    summary=existing_summary,
                )
            )
            continue

        log(f"[INFO] {run_name} requires evaluation ({msg})")
        try:
            return_code = run_evaluation_subprocess(
                run_dir=run_dir,
                resolved_config_path=resolved_config_path,
                evaluate_script=evaluate_script,
                python_exec=args.python_exec,
                shared_cache_dir=shared_cache_dir,
                repo_root=repo_root,
                force_regen_cache=args.force_regen_cache,
            )
            if return_code != 0:
                err_msg = f"evaluate_latent.py exited with non-zero code: {return_code}"
                log(f"[ERROR] {run_name}: {err_msg}")
                results.append(
                    RunResult(
                        run_name=run_name,
                        run_dir=run_dir,
                        status="failed",
                        message=err_msg,
                        return_code=return_code,
                    )
                )
                continue

            run_summary = aggregate_run_outputs(run_dir)
            log(f"[OK] {run_name} local aggregation completed")
            results.append(
                RunResult(
                    run_name=run_name,
                    run_dir=run_dir,
                    status="success",
                    message="evaluated",
                    return_code=0,
                    summary=run_summary,
                )
            )
        except Exception as exc:
            log(f"[ERROR] {run_name} failed with exception: {exc}")
            results.append(
                RunResult(
                    run_name=run_name,
                    run_dir=run_dir,
                    status="failed",
                    message=str(exc),
                )
            )

    write_global_summary(exp_root, results)


if __name__ == "__main__":
    main()