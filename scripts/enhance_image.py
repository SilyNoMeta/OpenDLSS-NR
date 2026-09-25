#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=2.0",
#   "Pillow>=11.0",
# ]
# ///
"""Run OpenDLSS-NR on ordinary image files."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "build" / ("dlss5vk.exe" if os.name == "nt" else "dlss5vk")
FILTERS = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "box": Image.Resampling.BOX,
    "nearest": Image.Resampling.NEAREST,
}


def bounded(value: str, low: float, high: float) -> float:
    number = float(value)
    if not low <= number <= high:
        raise argparse.ArgumentTypeError(f"must be between {low:g} and {high:g}")
    return number


def integer_bounded(value: str, low: int, high: int) -> int:
    number = int(value)
    if not low <= number <= high:
        raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
    return number


def dimensions(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be WIDTHxHEIGHT") from error
    if width < 33 or height < 33 or width > 16384 or height > 16384:
        raise argparse.ArgumentTypeError("each dimension must be between 33 and 16384 pixels")
    return width, height


def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image.copy()
    width = radius * 2 + 1
    padded = np.pad(image, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    sums = np.cumsum(padded, axis=1, dtype=np.float32)
    sums = np.concatenate((np.zeros_like(sums[:, :1]), sums), axis=1)
    horizontal = (sums[:, width:] - sums[:, :-width]) / width
    padded = np.pad(horizontal, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
    sums = np.cumsum(padded, axis=0, dtype=np.float32)
    sums = np.concatenate((np.zeros_like(sums[:1]), sums), axis=0)
    return (sums[width:] - sums[:-width]) / width


def luminance(rgb: np.ndarray) -> np.ndarray:
    return np.sum(rgb * np.array([0.212639, 0.715169, 0.072192], dtype=np.float32), axis=2, keepdims=True)


def srgb_decode(rgb: np.ndarray) -> np.ndarray:
    bounded = np.clip(rgb, 0.0, 1.0)
    return np.where(bounded <= 0.04045, bounded / 12.92, ((bounded + 0.055) / 1.055) ** 2.4)


def srgb_encode(rgb: np.ndarray) -> np.ndarray:
    bounded = np.clip(rgb, 0.0, 1.0)
    return np.where(bounded <= 0.0031308, bounded * 12.92, 1.055 * bounded ** (1.0 / 2.4) - 0.055)


def compose(source: np.ndarray, generated: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    source_rgb = source[..., :3]
    result = generated[..., :3]
    if args.detail_only:
        residual = result - source_rgb
        result = source_rgb + residual - box_blur(residual, args.detail_radius)

    if args.tone_preservation > 0 or args.color_strength < 1:
        source_linear = srgb_decode(source_rgb)
        result_linear = srgb_decode(result)
        source_y = luminance(source_linear)
        result_y = luminance(result_linear)
    if args.tone_preservation > 0:
        target_y = result_y + (source_y - result_y) * args.tone_preservation
        result_linear = result_linear * np.divide(target_y, np.maximum(result_y, 1e-6))
        result_y = luminance(result_linear)

    if args.color_strength < 1:
        luminance_only = source_linear * np.divide(result_y, np.maximum(source_y, 1e-6))
        result_linear = luminance_only + (result_linear - luminance_only) * args.color_strength

    if args.tone_preservation > 0 or args.color_strength < 1:
        result = srgb_encode(result_linear)

    if args.grain_preservation > 0:
        source_detail = source_rgb - box_blur(source_rgb, 1)
        result += source_detail * args.grain_preservation

    result = np.clip(result, 0.0, 1.0)
    if args.mask:
        mask = Image.open(args.mask)
        mask = ImageOps.exif_transpose(mask).convert("L").resize(
            (source.shape[1], source.shape[0]), FILTERS[args.filter]
        )
        if args.mask_feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(args.mask_feather))
        amount = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
        result = source_rgb + (result - source_rgb) * amount

    return np.concatenate((np.clip(result, 0.0, 1.0), source[..., 3:4]), axis=2).astype("<f4")


def save_image(path: Path, rgba: np.ndarray, quality: int) -> None:
    pixels = np.floor(np.clip(rgba, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    image = Image.fromarray(pixels, "RGBA")
    suffix = path.suffix.lower()
    options: dict[str, object] = {}
    if suffix in {".jpg", ".jpeg"}:
        image = image.convert("RGB")
        options.update(quality=quality, subsampling=0)
    elif suffix in {".webp", ".avif"}:
        options["quality"] = quality
    elif suffix not in {".png", ".tif", ".tiff"}:
        raise ValueError("output extension must be PNG, JPEG, WebP, AVIF, or TIFF")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, **options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance an image with the OpenDLSS-NR model.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=Path(os.environ.get("DLSS5VK_MODEL", ROOT / "models/nr")))
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--size", type=dimensions, help="resize before NR, as WIDTHxHEIGHT")
    size.add_argument("--scale", type=lambda value: bounded(value, 0.25, 4.0), help="resize factor before NR")
    parser.add_argument("--filter", choices=FILTERS, default="lanczos")
    parser.add_argument("--passes", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--style", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--intensity", type=lambda value: bounded(value, 0.0, 2.0), default=1.0)
    parser.add_argument("--tone", type=lambda value: bounded(value, 0.0, 2.0), default=1.0)
    parser.add_argument("--structure", type=lambda value: bounded(value, 0.0, 2.0), default=1.0)
    parser.add_argument("--skin", type=lambda value: bounded(value, -1.0, 2.0), default=-1.0)
    parser.add_argument("--auto-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--detail-only", action="store_true", help="retain only high-frequency NR residuals")
    parser.add_argument("--detail-radius", type=int, choices=range(1, 17), default=3)
    parser.add_argument("--color-strength", type=lambda value: bounded(value, 0.0, 1.0), default=1.0)
    parser.add_argument("--tone-preservation", type=lambda value: bounded(value, 0.0, 1.0), default=0.0)
    parser.add_argument("--grain-preservation", type=lambda value: bounded(value, 0.0, 2.0), default=0.0)
    parser.add_argument("--mask", type=Path, help="grayscale mask; white applies NR")
    parser.add_argument("--mask-feather", type=lambda value: bounded(value, 0.0, 128.0), default=0.0)
    parser.add_argument("--quality", type=lambda value: integer_bounded(value, 1, 100), metavar="1..100", default=95)
    parser.add_argument("--no-verify", action="store_true", help="skip model SHA-256 verification")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(f"{args.binary} not found; run scripts/build.sh")
    if not (args.model / "manifest.json").is_file():
        raise FileNotFoundError(f"{args.model}/manifest.json not found")
    image = Image.open(args.input)
    image.seek(0)
    image = ImageOps.exif_transpose(image).convert("RGBA")
    if args.size:
        image = image.resize(args.size, FILTERS[args.filter])
    elif args.scale:
        image = image.resize(
            (max(33, round(image.width * args.scale)), max(33, round(image.height * args.scale))),
            FILTERS[args.filter],
        )
    if image.width < 33 or image.height < 33 or image.width > 16384 or image.height > 16384:
        raise ValueError("each processed dimension must be between 33 and 16384 pixels")

    source = np.asarray(image, dtype=np.float32) / 255.0
    current = source.astype("<f4")
    with tempfile.TemporaryDirectory(prefix="opendlss-nr-") as temporary:
        directory = Path(temporary)
        for pass_index in range(args.passes):
            input_path = directory / f"pass-{pass_index}-input.f32"
            output_path = directory / f"pass-{pass_index}-output.f32"
            current.tofile(input_path)
            command = [
                str(args.binary),
                "image",
                "--model",
                str(args.model),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--width",
                str(image.width),
                "--height",
                str(image.height),
                "--style",
                str(args.style),
                "--intensity",
                str(args.intensity),
                "--tone",
                str(args.tone),
                "--structure",
                str(args.structure),
                "--skin",
                str(args.skin),
                "--auto-mask",
                "1" if args.auto_mask else "0",
                "--seed",
                str(args.seed + pass_index),
                "--repeat",
                str(args.repeat),
            ]
            if args.no_verify or pass_index > 0:
                command.append("--no-verify")
            print(f"NR pass {pass_index + 1}/{args.passes}: {image.width}x{image.height}", flush=True)
            subprocess.run(command, check=True)
            current = np.fromfile(output_path, dtype="<f4").reshape(image.height, image.width, 4)

    result = compose(source, current, args)
    save_image(args.output, result, args.quality)
    print(f"wrote {args.output} ({image.width}x{image.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
