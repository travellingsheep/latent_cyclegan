import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONFIG_PATH = Path(__file__).with_name("flatten_rename_images_config.json")


@dataclass(frozen=True)
class DirectionRule:
    subdir_name: str
    direction_tag: str
    source_style: str
    target_style: str


RULES = (
    DirectionRule(
        subdir_name="A2B",
        direction_tag="A2B",
        source_style="monet",
        target_style="photo",
    ),
    DirectionRule(
        subdir_name="B2A",
        direction_tag="B2A",
        source_style="photo",
        target_style="monet",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy generated images into one flat directory with renamed filenames."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a JSON config file containing source_dir and output_dir.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> tuple[Path, Path]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if "source_dir" not in data or "output_dir" not in data:
        raise KeyError("Config file must contain both 'source_dir' and 'output_dir'.")

    base_dir = config_path.resolve().parent
    source_dir = resolve_path(data["source_dir"], base_dir)
    output_dir = resolve_path(data["output_dir"], base_dir)
    return source_dir, output_dir


def resolve_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def validate_source_dir(source_dir: Path) -> None:
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {source_dir}")

    for rule in RULES:
        subdir = source_dir / rule.subdir_name
        if not subdir.is_dir():
            raise NotADirectoryError(f"Required subdirectory not found: {subdir}")


def build_output_name(source_file: Path, rule: DirectionRule) -> str:
    extension = source_file.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension for {source_file.name}: {source_file.suffix}"
        )

    stem = source_file.stem
    split_index = stem.rfind("_")
    if split_index == -1:
        raise ValueError(
            f"Filename does not contain a direction suffix separated by the last underscore: {source_file.name}"
        )

    original_name = stem[:split_index]
    direction_tag = stem[split_index + 1 :]

    if direction_tag != rule.direction_tag:
        raise ValueError(
            f"Filename direction suffix mismatch for {source_file.name}: expected _{rule.direction_tag}"
        )

    if not original_name:
        raise ValueError(f"Original image name is empty in {source_file.name}")

    return f"{rule.source_style}_{original_name}_to_{rule.target_style}{extension}"


def copy_file(source_file: Path, destination: Path) -> None:
    shutil.copy2(source_file, destination)


def copy_images(source_dir: Path, output_dir: Path) -> tuple[int, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    output_names: set[str] = set()
    summaries: list[str] = []

    for rule in RULES:
        input_dir = source_dir / rule.subdir_name
        local_count = 0

        for source_file in sorted(input_dir.iterdir()):
            if not source_file.is_file():
                continue

            output_name = build_output_name(source_file, rule)
            destination = output_dir / output_name

            if output_name in output_names or destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing output file: {destination}"
                )

            copy_file(source_file, destination)
            output_names.add(output_name)
            copied_count += 1
            local_count += 1

        summaries.append(f"{rule.subdir_name}: {local_count} files")

    return copied_count, summaries


def main() -> None:
    args = parse_args()
    source_dir, output_dir = load_config(args.config)
    validate_source_dir(source_dir)
    copied_count, summaries = copy_images(source_dir, output_dir)

    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    for summary in summaries:
        print(summary)
    print(f"Copied {copied_count} files in total.")


if __name__ == "__main__":
    main()