import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageColor, ImageDraw, ImageFont


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_color(value: str) -> Tuple[int, int, int]:
    try:
        rgb = ImageColor.getrgb(value)
    except ValueError as exc:
        raise ValueError(f"Invalid color: {value}") from exc
    if isinstance(rgb, int):
        return (rgb, rgb, rgb)
    if len(rgb) == 4:
        return rgb[:3]
    return rgb


def _load_font(font_path: str, font_size: int) -> ImageFont.ImageFont:
    candidates: List[str] = []
    if font_path.strip():
        candidates.append(font_path.strip())
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
    )

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), font_size)
            except OSError:
                continue

    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    return right - left, bottom - top


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    wrapped_lines: List[str] = []
    for raw_line in stripped.splitlines():
        if not raw_line.strip():
            wrapped_lines.append("")
            continue

        tokens: Sequence[str]
        if " " in raw_line.strip():
            tokens = raw_line.split()
        else:
            tokens = list(raw_line)

        current = tokens[0]
        for token in tokens[1:]:
            candidate = f"{current} {token}" if " " in raw_line.strip() else f"{current}{token}"
            width, _ = _text_size(draw, candidate, font)
            if width <= max_width:
                current = candidate
            else:
                wrapped_lines.append(current)
                current = token
        wrapped_lines.append(current)

    return "\n".join(wrapped_lines)


def _resize_keep_ratio(image: Image.Image, target_height: int) -> Image.Image:
    if target_height <= 0 or image.height == target_height:
        return image
    target_width = max(1, round(image.width * target_height / image.height))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _open_rgb_image(path: Path, target_height: int) -> Image.Image:
    _require(path.exists() and path.is_file(), f"Image not found: {path}")
    image = Image.open(path).convert("RGB")
    return _resize_keep_ratio(image, target_height)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Place two images side by side with a top title and captions below each image."
    )
    parser.add_argument("--left", type=str, required=True, help="Left image path")
    parser.add_argument("--right", type=str, required=True, help="Right image path")
    parser.add_argument("--output", type=str, required=True, help="Output image path")

    parser.add_argument("--title", type=str, default="", help="Title text shown above the images")
    parser.add_argument("--left-caption", type=str, default="", help="Caption below the left image")
    parser.add_argument("--right-caption", type=str, default="", help="Caption below the right image")

    parser.add_argument("--image-height", type=int, default=0, help="Resize both images to this height before composing")
    parser.add_argument("--padding", type=int, default=32, help="Outer padding in pixels")
    parser.add_argument("--gap", type=int, default=24, help="Gap between left and right images in pixels")
    parser.add_argument("--title-gap", type=int, default=18, help="Gap below the title block in pixels")
    parser.add_argument("--caption-gap", type=int, default=16, help="Gap between images and captions in pixels")
    parser.add_argument("--bottom-padding", type=int, default=24, help="Extra padding below captions in pixels")

    parser.add_argument("--title-size", type=int, default=34, help="Title font size")
    parser.add_argument("--caption-size", type=int, default=24, help="Caption font size")
    parser.add_argument("--font", type=str, default="", help="Optional .ttf font path")

    parser.add_argument("--background", type=str, default="white", help="Canvas background color")
    parser.add_argument("--text-color", type=str, default="black", help="Text color")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    padding = max(0, int(args.padding))
    gap = max(0, int(args.gap))
    title_gap = max(0, int(args.title_gap))
    caption_gap = max(0, int(args.caption_gap))
    bottom_padding = max(0, int(args.bottom_padding))
    image_height = max(0, int(args.image_height))

    left_path = Path(args.left)
    right_path = Path(args.right)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    left_image = _open_rgb_image(left_path, image_height)
    right_image = _open_rgb_image(right_path, image_height)

    background = _parse_color(args.background)
    text_color = _parse_color(args.text_color)
    title_font = _load_font(args.font, max(1, int(args.title_size)))
    caption_font = _load_font(args.font, max(1, int(args.caption_size)))

    content_width = left_image.width + gap + right_image.width
    canvas_width = padding * 2 + content_width

    probe = Image.new("RGB", (canvas_width, 1000), background)
    probe_draw = ImageDraw.Draw(probe)

    title_text = _wrap_text(probe_draw, args.title, title_font, max(1, canvas_width - padding * 2))
    left_caption_text = _wrap_text(probe_draw, args.left_caption, caption_font, max(1, left_image.width))
    right_caption_text = _wrap_text(probe_draw, args.right_caption, caption_font, max(1, right_image.width))

    title_height = _text_size(probe_draw, title_text, title_font)[1] if title_text else 0
    left_caption_height = _text_size(probe_draw, left_caption_text, caption_font)[1] if left_caption_text else 0
    right_caption_height = _text_size(probe_draw, right_caption_text, caption_font)[1] if right_caption_text else 0
    caption_height = max(left_caption_height, right_caption_height)

    top_block_height = title_height + (title_gap if title_text else 0)
    caption_block_height = (caption_gap if caption_height else 0) + caption_height
    canvas_height = padding + top_block_height + max(left_image.height, right_image.height) + caption_block_height + bottom_padding

    canvas = Image.new("RGB", (canvas_width, canvas_height), background)
    draw = ImageDraw.Draw(canvas)

    current_y = padding
    if title_text:
        title_width, _ = _text_size(draw, title_text, title_font)
        title_x = (canvas_width - title_width) // 2
        draw.multiline_text((title_x, current_y), title_text, fill=text_color, font=title_font, spacing=4, align="center")
        current_y += title_height + title_gap

    left_x = padding
    right_x = padding + left_image.width + gap
    image_top_y = current_y

    canvas.paste(left_image, (left_x, image_top_y))
    canvas.paste(right_image, (right_x, image_top_y))

    caption_y = image_top_y + max(left_image.height, right_image.height)
    if caption_height:
        caption_y += caption_gap

    if left_caption_text:
        left_caption_width, _ = _text_size(draw, left_caption_text, caption_font)
        left_caption_x = left_x + max(0, (left_image.width - left_caption_width) // 2)
        draw.multiline_text(
            (left_caption_x, caption_y),
            left_caption_text,
            fill=text_color,
            font=caption_font,
            spacing=4,
            align="center",
        )

    if right_caption_text:
        right_caption_width, _ = _text_size(draw, right_caption_text, caption_font)
        right_caption_x = right_x + max(0, (right_image.width - right_caption_width) // 2)
        draw.multiline_text(
            (right_caption_x, caption_y),
            right_caption_text,
            fill=text_color,
            font=caption_font,
            spacing=4,
            align="center",
        )

    canvas.save(output_path)
    print(f"Saved composed image to: {output_path}")


if __name__ == "__main__":
    main()