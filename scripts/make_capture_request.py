"""Write a capture request (request.json + proxy.f32) for ports/browser-webgpu/web/capture.html.

The proxy is the image the network sees: RGBA f32 in [0, 1], display-encoded, tightly packed. The request also
carries the conditioning, the seed and the auto-mask flag, under the names the fixture contract uses.

  python scripts/make_capture_request.py <out dir> --image in.png [--crop x,y,w,h] [--size WxH]
  python scripts/make_capture_request.py <out dir> --pattern bands --size 333x517

Patterns are procedural (no asset needed): "bands" (gradients, a checkerboard, hard edges and text-like strokes)
and "noise" (seeded white noise over a smooth field). Needs NumPy, and Pillow for --image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pattern(name: str, width: int, height: int, seed: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    u, v = x / max(width - 1, 1), y / max(height - 1, 1)
    rgb = np.zeros((height, width, 3), np.float32)
    if name == "bands":
        rgb[..., 0] = u
        rgb[..., 1] = v
        rgb[..., 2] = 0.5 + 0.5 * np.sin(12.0 * u + 7.0 * v)
        checker = ((x // 16 + y // 16) % 2).astype(np.float32)
        quadrant = (u > 0.5) & (v > 0.5)
        rgb[quadrant] = checker[quadrant, None] * 0.9 + 0.05
        strokes = ((x % 23) < 2) & (v < 0.3)
        rgb[strokes] = 0.0
        disc = (u - 0.3) ** 2 + (v - 0.7) ** 2 < 0.02
        rgb[disc] = [0.95, 0.85, 0.7]
    elif name == "noise":
        rng = np.random.default_rng(seed)
        base = 0.5 + 0.35 * np.stack([np.sin(3 * u), np.cos(4 * v), np.sin(5 * (u + v))], axis=-1)
        rgb = np.clip(base + 0.15 * rng.standard_normal((height, width, 3)).astype(np.float32), 0, 1)
    else:
        raise SystemExit(f"unknown pattern {name}")
    return np.clip(rgb, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--pattern", choices=["bands", "noise"])
    ap.add_argument("--crop", help="x,y,w,h in source pixels, before --size")
    ap.add_argument("--size", help="WxH; for --image a Lanczos resize after --crop, for --pattern the size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auto-mask", type=int, choices=[0, 1], default=1)
    ap.add_argument("--tone", type=float, default=1.0)
    ap.add_argument("--structure", type=float, default=1.0)
    ap.add_argument("--skin", type=float, default=-1.0)
    ap.add_argument("--style", type=float, default=0.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    if args.image:
        from PIL import Image, ImageOps
        image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
        if args.crop:
            x, y, w, h = (int(part) for part in args.crop.split(","))
            image = image.crop((x, y, x + w, y + h))
        if args.size:
            w, h = (int(part) for part in args.size.lower().split("x"))
            image = image.resize((w, h), Image.Resampling.LANCZOS)
        rgb = np.asarray(image, np.float32) / 255.0
    else:
        if not args.size:
            raise SystemExit("--pattern needs --size")
        w, h = (int(part) for part in args.size.lower().split("x"))
        rgb = pattern(args.pattern, w, h, args.seed)
    height, width = rgb.shape[:2]
    if width < 33 or height < 33:
        raise SystemExit("each side needs at least 33 pixels")
    rgba = np.concatenate([rgb, np.ones((height, width, 1), np.float32)], axis=-1).astype("<f4")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "proxy.f32").write_bytes(rgba.tobytes())
    request = dict(label=args.label or (args.image.name if args.image else f"pattern {args.pattern}"),
                   proxy=dict(width=width, height=height, file="proxy.f32"),
                   conditioning=dict(localTone=args.tone, localStructure=args.structure,
                                     skinStructure=args.skin, style=args.style),
                   seed=args.seed, autoMask=bool(args.auto_mask))
    (args.out / "request.json").write_text(json.dumps(request, indent=1), encoding="utf-8")
    print(f"{args.out}: {width}x{height} proxy, {request['label']}")


if __name__ == "__main__":
    main()
