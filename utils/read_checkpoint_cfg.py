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


def main() -> None:
    cfg_yaml_path = Path(__file__).resolve().with_name("read_checkpoint_cfg_config.yaml")
    script_cfg = _load_script_config(cfg_yaml_path)

    ckpt_raw = str(script_cfg.get("checkpoint_path", "")).strip()
    out_raw = str(script_cfg.get("output_cfg_path", "")).strip()
    json_indent = int(script_cfg.get("json_indent", 2))

    if not ckpt_raw:
        raise ValueError("Missing 'checkpoint_path' in read_checkpoint_cfg_config.yaml")
    if not out_raw:
        raise ValueError("Missing 'output_cfg_path' in read_checkpoint_cfg_config.yaml")

    ckpt_path = _resolve_path(ckpt_raw)
    out_cfg_path = _resolve_path(out_raw)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = _torch_load_checkpoint(ckpt_path)
    cfg = payload.get("cfg")

    if cfg is None:
        raise KeyError(f"'cfg' not found in checkpoint: {ckpt_path}")
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid cfg type in checkpoint: expected dict, got {type(cfg)}")

    _dump_cfg_to_file(cfg, out_cfg_path, json_indent=json_indent)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Saved cfg to: {out_cfg_path}")
    print("cfg content:")
    print(json.dumps(cfg, ensure_ascii=False, indent=json_indent))


if __name__ == "__main__":
    main()
