"""Compare the composed output of two backends on capture requests, with the network's own seed variation as a scale.

  python scripts/compare_backends.py <request dir> [<request dir> ...] [--model models/nr] [--binary build/dlss5vk]
                                     [--reference compat] [--candidate sm86] [--json report.json]

A request directory holds request.json and proxy.f32 (scripts/make_capture_request.py). For each one, `dlss5vk image`
runs the reference backend, the candidate backend, and the reference again with the seed plus one. The candidate is
compared with the reference; the reseeded reference shows how much the output moves when only the injected noise
changes, which the network does by design. All numbers are on the composed RGB in [0, 1], intensity 1:

  PSNR, the largest and mean absolute difference (in 1/255 units), and the share of channels more than 1/255 and
  4/255 away; the NR effect (mean |output - proxy| of the reference) gives the size of what the network changes.

Byte identity, numerical tolerance and visual quality are different claims: this measures the second, as statistics,
and says nothing about the first (dlss5vk parity) or the third (look at the images).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run_image(binary: Path, model: Path, request: dict, proxy: Path, backend: str, seed: int, output: Path) -> float:
    c = request["conditioning"]
    argv = [str(binary), "image", "--model", str(model), "--input", str(proxy), "--output", str(output),
            "--width", str(request["proxy"]["width"]), "--height", str(request["proxy"]["height"]),
            "--seed", str(seed), "--auto-mask", "1" if request["autoMask"] else "0",
            "--tone", str(c["localTone"]), "--structure", str(c["localStructure"]), "--skin", str(c["skinStructure"]),
            "--style", str(int(c["style"]))]
    env = dict(os.environ, DLSS5VK_BACKEND=backend)
    result = subprocess.run(argv, capture_output=True, text=True, env=env)
    if result.returncode:
        raise SystemExit(f"{backend} failed on {proxy}:\n{result.stdout}\n{result.stderr}")
    for line in result.stdout.splitlines():
        if "GPU" in line and "ms" in line:
            return float(line.split("GPU")[1].split("ms")[0])
    return float("nan")


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    d = candidate[..., :3].astype(np.float64) - reference[..., :3].astype(np.float64)
    mse = float(np.mean(d * d))
    a = np.abs(d)
    return dict(psnr_db=float("inf") if mse == 0 else 10 * np.log10(1.0 / mse), max_abs_255=float(a.max() * 255),
                mean_abs_255=float(a.mean() * 255), over_1_255_pct=float(np.mean(a > 1 / 255) * 100),
                over_4_255_pct=float(np.mean(a > 4 / 255) * 100), identical_pct=float(np.mean(d == 0) * 100))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("requests", nargs="+", type=Path)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "nr")
    ap.add_argument("--binary", type=Path, default=ROOT / "build" / ("dlss5vk.exe" if os.name == "nt" else "dlss5vk"))
    ap.add_argument("--reference", default="compat")
    ap.add_argument("--candidate", default="sm86")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for directory in args.requests:
            request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
            proxy = directory / request["proxy"]["file"]
            w, h = request["proxy"]["width"], request["proxy"]["height"]
            outputs = {}
            for key, backend, seed in (("reference", args.reference, request["seed"]),
                                       ("candidate", args.candidate, request["seed"]),
                                       ("reseeded", args.reference, request["seed"] + 1)):
                path = tmp / f"{key}.f32"
                run_image(args.binary, args.model, request, proxy, backend, seed, path)
                outputs[key] = np.fromfile(path, "<f4").reshape(h, w, 4)
            source = np.fromfile(proxy, "<f4").reshape(h, w, 4)
            row = dict(request=request.get("label", directory.name), size=f"{w}x{h}",
                       nr_effect_mean_abs_255=float(np.abs(outputs["reference"][..., :3] - source[..., :3]).mean() * 255),
                       candidate_vs_reference=metrics(outputs["candidate"], outputs["reference"]),
                       reseeded_reference_vs_reference=metrics(outputs["reseeded"], outputs["reference"]))
            rows.append(row)
            c, s = row["candidate_vs_reference"], row["reseeded_reference_vs_reference"]
            print(f"{row['request']} ({row['size']}): NR effect {row['nr_effect_mean_abs_255']:.2f}/255 | "
                  f"{args.candidate}: PSNR {c['psnr_db']:.2f} dB, max {c['max_abs_255']:.1f}/255, mean {c['mean_abs_255']:.3f}/255, "
                  f">1/255 {c['over_1_255_pct']:.2f}%, >4/255 {c['over_4_255_pct']:.3f}% | "
                  f"seed+1: PSNR {s['psnr_db']:.2f} dB, max {s['max_abs_255']:.1f}/255, mean {s['mean_abs_255']:.3f}/255", flush=True)
    if args.json:
        args.json.write_text(json.dumps(dict(reference=args.reference, candidate=args.candidate, rows=rows), indent=1),
                             encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
