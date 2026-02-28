from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict at root: {path}")
    return data


def _dump_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def _ensure_dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    node = parent.get(key)
    if node is None:
        node = {}
        parent[key] = node
    if not isinstance(node, dict):
        raise ValueError(f"Expected dict at '{key}', got {type(node).__name__}")
    return node


def _apply_overrides(cfg: Dict[str, Any], run_name: str, epochs: int, overrides: Dict[str, float]) -> Dict[str, Any]:
    # Only change what the user requested:
    # - experiment.name
    # - training.num_epochs
    # - one of loss.w_r1 or loss.w_adv
    out = dict(cfg)  # shallow copy top-level

    exp = _ensure_dict(out, "experiment")
    exp["name"] = str(run_name)

    training = _ensure_dict(out, "training")
    training["num_epochs"] = int(epochs)

    loss = _ensure_dict(out, "loss")
    for k, v in overrides.items():
        if k not in {"w_r1", "w_adv"}:
            raise ValueError(f"Unexpected override key: {k}")
        loss[k] = float(v)

    return out


def _run_one(repo_root: Path, config_path: Path, dry_run: bool) -> int:
    cmd = [sys.executable, "run.py", "--config", str(config_path)]
    print(f"\n[Run] {' '.join(cmd)}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(repo_root))
    return int(proc.returncode)


def _extract_last_epoch_summary(console_log_path: Path) -> Optional[Dict[str, Any]]:
    """Parse outputs/*/console.log and return the last '[Epoch Summary] {json}' payload."""

    if not console_log_path.exists():
        return None

    marker = "[Epoch Summary]"
    last: Optional[Dict[str, Any]] = None
    with console_log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if marker not in line:
                continue
            try:
                payload = line.split(marker, 1)[1].strip()
                obj = json.loads(payload)
                if isinstance(obj, dict):
                    last = obj
            except Exception:
                continue
    return last


def _write_summary_tables(
    outputs_dir: Path,
    experiments: Sequence[Dict[str, Any]],
    epochs: int,
) -> None:
    rows: List[Dict[str, Any]] = []

    for exp in experiments:
        run_name = str(exp.get("name", ""))
        overrides = exp.get("overrides", {})
        if not run_name:
            continue
        if not isinstance(overrides, dict):
            overrides = {}

        run_out_dir = outputs_dir / run_name
        console_log = run_out_dir / "console.log"
        summary = _extract_last_epoch_summary(console_log)

        row: Dict[str, Any] = {
            "run": run_name,
            "epochs": int(epochs),
            "w_r1": overrides.get("w_r1", ""),
            "w_adv": overrides.get("w_adv", ""),
            "epoch": "",
            "d_loss": "",
            "g_loss": "",
            "g_adv": "",
            "sty": "",
            "ds": "",
            "cyc": "",
            "id": "",
            "r1": "",
            "loss_curve": str((run_out_dir / "loss_curve.png").relative_to(outputs_dir.parent)),
            "console_log": str(console_log.relative_to(outputs_dir.parent)),
            "config": str((run_out_dir / "config_sweep.yaml").relative_to(outputs_dir.parent)),
        }

        if isinstance(summary, dict):
            for k in ["epoch", "d_loss", "g_loss", "g_adv", "sty", "ds", "cyc", "id", "r1"]:
                if k in summary:
                    row[k] = summary[k]
        rows.append(row)

    csv_path = outputs_dir / "sweep_summary.csv"
    md_path = outputs_dir / "sweep_summary.md"

    fieldnames = [
        "run",
        "epochs",
        "w_r1",
        "w_adv",
        "epoch",
        "d_loss",
        "g_loss",
        "g_adv",
        "sty",
        "ds",
        "cyc",
        "id",
        "r1",
        "loss_curve",
        "console_log",
        "config",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    def _md_escape(v: Any) -> str:
        s = "" if v is None else str(v)
        return s.replace("|", "\\|")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Sweep Summary\n\n")
        f.write("说明：每行取该实验 `console.log` 中最后一条 `[Epoch Summary]` 的统计值（按 epoch 平均）。\n\n")
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(_md_escape(r.get(k, "")) for k in fieldnames) + " |\n")

    print(f"\n[Summary] wrote: {csv_path}")
    print(f"[Summary] wrote: {md_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequential sweep runner for latent_cyclegan")
    p.add_argument(
        "--base_config",
        type=str,
        default="configs/stargan_v2_latent.yaml",
        help="Base config to load (default: configs/stargan_v2_latent.yaml)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Override training.num_epochs for every run (default: 50)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only write configs and print commands; do not run training",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    base_config_path = (repo_root / str(args.base_config)).resolve()
    base_cfg = _load_yaml(base_config_path)

    experiments: List[Dict[str, Any]] = [
        # {"name": "new_baseline_zero_w_r1", "overrides": {"w_r1": 0.0}},
        # {"name": "new_baseline_micro_w_r1", "overrides": {"w_r1": 0.1}},
        # {"name": "new_baseline_small_w_r1", "overrides": {"w_r1": 0.5}},
        {"name": "new_baseline_medium_w_r1", "overrides": {"w_r1": 1.0}},
        {"name": "new_baseline_bigger_w_adv", "overrides": {"w_adv": 2.0}},
        {"name": "new_baseline_greater_w_adv", "overrides": {"w_adv": 5.0}},
    ]

    outputs_dir = repo_root / "outputs"
    print(f"[Info] repo_root={repo_root}")
    print(f"[Info] base_config={base_config_path}")
    print(f"[Info] epochs={int(args.epochs)}")
    print(f"[Info] outputs_dir={outputs_dir}")

    for i, exp in enumerate(experiments, start=1):
        run_name = str(exp["name"])
        overrides = exp["overrides"]
        if not isinstance(overrides, dict):
            raise ValueError(f"Invalid overrides for {run_name}")

        out_cfg = _apply_overrides(base_cfg, run_name=run_name, epochs=int(args.epochs), overrides=overrides)

        run_out_dir = outputs_dir / run_name
        run_out_dir.mkdir(parents=True, exist_ok=True)

        out_cfg_path = run_out_dir / "config_sweep.yaml"
        _dump_yaml(out_cfg, out_cfg_path)

        print(f"\n========== [{i}/{len(experiments)}] {run_name} ==========")
        print(f"[Info] wrote config: {out_cfg_path}")
        print(f"[Info] overrides: {overrides}")

        rc = _run_one(repo_root=repo_root, config_path=out_cfg_path, dry_run=bool(args.dry_run))
        if rc != 0:
            raise SystemExit(f"[Error] run failed: {run_name} (exit code {rc})")

    _write_summary_tables(outputs_dir=outputs_dir, experiments=experiments, epochs=int(args.epochs))

    print("\n[Done] All sweep runs finished.")


if __name__ == "__main__":
    main()
