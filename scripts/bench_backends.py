"""Benchmark backends against each other at the same sizes, alternating, with GPU telemetry.

  python scripts/bench_backends.py --sizes 512x512,1920x1080 --backends compat,sm86 [--frames sm86=40,compat=5]
                                   [--rounds 2] [--model models/nr] [--json report.json]

Each round runs `dlss5vk bench` once per backend and size, in alternating order (AB then BA), so a drifting clock
or temperature does not favour one backend. Every run is a separate process: device, model, kernels and graph are
prepared again, and the preparation figures (device, model, kernels, graph, first frame with its compiles) and the
memory in use are recorded beside the recurring frame time (median and minimum of the GPU timestamps around the
whole graph). nvidia-smi is sampled before and after each run.

Same model, same synthetic input, same size, one pass per frame for every backend: nothing is scaled down for one
of them. Do not run two benchmarks on the same GPU at once.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def telemetry() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def bench(binary: Path, model: Path, backend: str, width: int, height: int, frames: int) -> dict:
    env = dict(os.environ, DLSS5VK_BACKEND=backend)
    before = telemetry()
    result = subprocess.run([str(binary), "bench", "--model", str(model), "--width", str(width), "--height", str(height),
                             "--frames", str(frames)], capture_output=True, text=True, env=env)
    after = telemetry()
    out = result.stdout + result.stderr
    if result.returncode:
        return dict(backend=backend, size=f"{width}x{height}", error=out[-2000:])
    row = dict(backend=backend, size=f"{width}x{height}", frames=frames, telemetry_before=before, telemetry_after=after)
    if m := re.search(r"median ([\d.]+) ms, min ([\d.]+) ms over \d+ frames at \S+ \(full (\S+)\)", out):
        row.update(median_ms=float(m[1]), min_ms=float(m[2]), field=m[3])
    if m := re.search(r"preparation: device ([\d.]+) s, model ([\d.]+) s, kernels ([\d.]+) s, graph ([\d.]+) s, "
                      r"first frame \(compiles\) ([\d.]+) s", out):
        row.update(prep_device_s=float(m[1]), prep_model_s=float(m[2]), prep_kernels_s=float(m[3]),
                   prep_graph_s=float(m[4]), prep_first_frame_s=float(m[5]))
    if m := re.search(r"memory: (\d+) MiB device-local in use \(raw tensors (\d+) MiB, re-laid weights (\d+) MiB, "
                      r"activations and scratch (\d+) MiB\), peak (\d+) MiB", out):
        row.update(mem_device_mib=int(m[1]), mem_raw_mib=int(m[2]), mem_weights_mib=int(m[3]), mem_activations_mib=int(m[4]),
                   mem_peak_mib=int(m[5]))
    if m := re.search(r"frame \d+: [\d.]+ ms GPU \((\d+) dispatches\)", out):
        row["dispatches"] = int(m[1])
    if m := re.search(r"backend: (\S+)", out):
        row["backend_reported"] = m[1]
    if m := re.search(r"host frame: median ([\d.]+) ms", out):
        row["host_frame_ms"] = float(m[1])
    if row.get("backend_reported") != backend or "median_ms" not in row:
        row["error"] = "Missing timing or requested backend not used:\n" + out[-2000:]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512x512,1920x1080")
    ap.add_argument("--backends", default="compat,sm86")
    ap.add_argument("--frames", default="sm86=40,native=40,compat=5")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "nr")
    ap.add_argument("--binary", type=Path, default=ROOT / "build" / ("dlss5vk.exe" if os.name == "nt" else "dlss5vk"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    frames = dict(item.split("=") for item in args.frames.split(","))
    backends = args.backends.split(",")
    rows = []
    for size in args.sizes.split(","):
        width, height = (int(v) for v in size.lower().split("x"))
        for round_index in range(args.rounds):
            order = backends if round_index % 2 == 0 else list(reversed(backends))
            for backend in order:
                row = bench(args.binary, args.model, backend, width, height, int(frames.get(backend, 20)))
                row["round"] = round_index
                rows.append(row)
                if "error" in row:
                    print(f"{size} {backend} round {round_index}: FAILED\n{row['error']}", flush=True)
                    continue
                print(f"{size} {backend} round {round_index}: median {row['median_ms']:.2f} ms, min {row['min_ms']:.2f} ms "
                      f"(field {row['field']}, {row.get('dispatches', '?')} dispatches); first frame "
                      f"{row.get('prep_first_frame_s', float('nan')):.2f} s; memory {row.get('mem_device_mib', '?')} MiB "
                      f"(peak {row.get('mem_peak_mib', '?')}); GPU before [{row['telemetry_before']}] after "
                      f"[{row['telemetry_after']}]", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 1 if any("error" in r for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
