import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).with_name("run_evaluation_config.json")
RUN_EVALUATION_PATH = Path(__file__).with_name("run_evaluation.py")
PATH_KEYS = {
    "checkpoint",
    "output",
    "config",
    "test_dir",
    "cache_dir",
    "image_classifier_path",
    "clip_modelscope_cache_dir",
    "clip_hf_cache_dir",
}


def load_argument_specs(script_path: Path) -> dict[str, dict[str, object]]:
    if not script_path.is_file():
        raise FileNotFoundError(f"run_evaluation script not found: {script_path}")

    source = script_path.read_text(encoding="utf-8")
    marker = "parser.add_argument("
    specs: dict[str, dict[str, object]] = {}
    start = 0

    while True:
        idx = source.find(marker, start)
        if idx == -1:
            break

        call_start = idx + len(marker)
        depth = 1
        pos = call_start
        while pos < len(source) and depth > 0:
            char = source[pos]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            pos += 1

        call_body = source[call_start : pos - 1]
        option = extract_option_string(call_body)
        if option is not None:
            dest = option.lstrip("-").replace("-", "_")
            specs[dest] = {
                "option": option,
                "is_flag": "action='store_true'" in call_body or 'action="store_true"' in call_body,
            }

        start = pos

    if not specs:
        raise RuntimeError(f"No parser.add_argument definitions found in {script_path}")

    return specs


def extract_option_string(call_body: str) -> str | None:
    for quote in ('"', "'"):
        prefix = f"{quote}--"
        idx = call_body.find(prefix)
        if idx == -1:
            continue
        end = call_body.find(quote, idx + 1)
        if end == -1:
            continue
        return call_body[idx + 1 : end]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run run_evaluation.py using parameters loaded from a JSON config file."
    )
    parser.add_argument(
        "--runner-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the JSON config file used to populate run_evaluation arguments.",
    )
    return parser.parse_args()


def load_runner_config(config_path: Path) -> tuple[dict, Path]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Runner config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise TypeError("Runner config must be a JSON object.")

    return data, config_path.resolve().parent


def resolve_path_value(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def build_argv_from_config(config_data: dict, config_dir: Path) -> list[str]:
    arg_specs = load_argument_specs(RUN_EVALUATION_PATH)

    unknown_keys = sorted(set(config_data) - set(arg_specs))
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise KeyError(f"Unknown run_evaluation config keys: {joined}")

    argv: list[str] = []
    for dest, spec in arg_specs.items():
        if dest not in config_data:
            continue

        value = config_data[dest]
        option = str(spec["option"])
        is_flag = bool(spec["is_flag"])

        if is_flag:
            if bool(value):
                argv.append(option)
            continue

        if value is None:
            continue

        if dest in PATH_KEYS and isinstance(value, str) and value.strip():
            value = resolve_path_value(value, config_dir)

        argv.extend([option, str(value)])

    return argv


def main() -> None:
    args = parse_args()
    config_data, config_dir = load_runner_config(args.runner_config)
    argv = build_argv_from_config(config_data, config_dir)
    command = [sys.executable, str(RUN_EVALUATION_PATH), *argv]
    print("Resolved run_evaluation arguments:")
    print(" ".join(argv) if argv else "<none>")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()