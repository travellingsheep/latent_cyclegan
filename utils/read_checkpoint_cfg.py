import json
from pathlib import Path
from typing import Any, Dict

import torch


def _load_script_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'pyyaml'. Install it: pip install pyyaml") from e

    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config YAML format: expected dict, got {type(cfg)}")
    return cfg


def _resolve_path(path_str: str) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _torch_load_checkpoint(path: Path) -> Dict[str, Any]:
    # torch>=2.6 may default to weights_only=True. We need full payload for cfg/rng.
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint format: expected dict, got {type(payload)}")
    return payload


def _dump_cfg_to_file(cfg: Dict[str, Any], out_path: Path, json_indent: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception:
            # Fallback to JSON text if PyYAML is unavailable.
            out_path.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=json_indent),
                encoding="utf-8",
            )
            return

        out_path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return

    out_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=json_indent),
        encoding="utf-8",
    )


def _to_plain_data(obj: Any) -> Any:
    """Convert common config container types to plain JSON/YAML-serializable data."""
    if isinstance(obj, dict):
        return {str(k): _to_plain_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_data(v) for v in obj]
    if hasattr(obj, "items") and callable(getattr(obj, "items")):
        try:
            return {str(k): _to_plain_data(v) for k, v in obj.items()}
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {str(k): _to_plain_data(v) for k, v in vars(obj).items()}
        except Exception:
            pass
    return obj


def _extract_cfg(payload: Dict[str, Any], ckpt_path: Path) -> Dict[str, Any]:
    # Prefer explicit config keys used by different training scripts.
    for key in ("cfg", "config"):
        if key in payload and payload[key] is not None:
            cfg = _to_plain_data(payload[key])
            if isinstance(cfg, dict):
                return cfg
            raise ValueError(
                f"Invalid '{key}' type in checkpoint: expected dict-like, got {type(payload[key])}"
            )

    # Fallback: some tools may save only the config content itself.
    plain_payload = _to_plain_data(payload)
    if isinstance(plain_payload, dict):
        candidate_keys = {"training", "model", "data", "loss", "checkpoint", "visualization", "experiment"}
        if len(candidate_keys.intersection(set(plain_payload.keys()))) >= 2:
            return plain_payload

    raise KeyError(
        f"Neither 'cfg' nor 'config' found in checkpoint: {ckpt_path}. "
        f"Available keys: {list(payload.keys())[:20]}"
    )


def _numel_of_state_dict(state_dict: Dict[str, Any]) -> int:
    total = 0
    for v in state_dict.values():
        if torch.is_tensor(v):
            total += int(v.numel())
    return total


def _print_state_dict_preview(
    name: str,
    state_dict: Dict[str, Any],
    emit,
    limit: int = 1000,
) -> None:
    total_tensors = sum(1 for v in state_dict.values() if torch.is_tensor(v))
    total_params = _numel_of_state_dict(state_dict)
    emit(f"[{name}] tensors={total_tensors}, params={total_params}")

    shown = 0
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        emit(f"  - {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        shown += 1
        if shown >= limit:
            break
    if total_tensors > shown:
        emit(f"  ... ({total_tensors - shown} more tensors)")


def _print_non_config_summary(payload: Dict[str, Any], emit) -> None:
    emit("\ncheckpoint summary (except config/cfg):")
    all_keys = list(payload.keys())
    emit(f"top-level keys: {all_keys}")

    for simple_key in ("epoch", "global_step"):
        if simple_key in payload:
            emit(f"{simple_key}: {payload[simple_key]}")

    for net_key in ("G", "F", "E", "D"):
        obj = payload.get(net_key)
        if isinstance(obj, dict):
            _print_state_dict_preview(net_key, obj, emit=emit)

    for opt_key in ("opt_g", "opt_d"):
        obj = payload.get(opt_key)
        if isinstance(obj, dict):
            state_n = len(obj.get("state", {})) if isinstance(obj.get("state", {}), dict) else 0
            pg_n = len(obj.get("param_groups", [])) if isinstance(obj.get("param_groups", []), list) else 0
            emit(f"[{opt_key}] optimizer_state_entries={state_n}, param_groups={pg_n}")


def main() -> None:
    cfg_yaml_path = Path(__file__).resolve().with_name("read_checkpoint_cfg_config.yaml")
    script_cfg = _load_script_config(cfg_yaml_path)

    ckpt_raw = str(script_cfg.get("checkpoint_path", "")).strip()
    out_raw = str(script_cfg.get("output_cfg_path", "")).strip()
    report_raw = str(script_cfg.get("output_report_path", "")).strip()
    json_indent = int(script_cfg.get("json_indent", 2))

    if not ckpt_raw:
        raise ValueError("Missing 'checkpoint_path' in read_checkpoint_cfg_config.yaml")
    if not out_raw:
        raise ValueError("Missing 'output_cfg_path' in read_checkpoint_cfg_config.yaml")

    ckpt_path = _resolve_path(ckpt_raw)
    out_cfg_path = _resolve_path(out_raw)
    if report_raw:
        out_report_path = _resolve_path(report_raw)
    else:
        out_report_path = out_cfg_path.with_name(f"{out_cfg_path.stem}_terminal_report.txt")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = _torch_load_checkpoint(ckpt_path)
    cfg = _extract_cfg(payload, ckpt_path=ckpt_path)

    _dump_cfg_to_file(cfg, out_cfg_path, json_indent=json_indent)

    report_lines: list[str] = []

    def emit(msg: str) -> None:
        report_lines.append(msg)
        print(msg)

    emit(f"Checkpoint: {ckpt_path}")
    emit(f"Saved cfg to: {out_cfg_path}")
    emit("cfg content:")
    emit(json.dumps(cfg, ensure_ascii=False, indent=json_indent))
    _print_non_config_summary(payload, emit=emit)

    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Saved terminal report to: {out_report_path}")


if __name__ == "__main__":
    main()
