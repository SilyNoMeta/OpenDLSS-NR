"""Compare the ways the NR graph can execute for a Direct3D host, on the same frames.

    python scripts/compare_execution.py --model models/nr --fixture fixtures/web/cowboy512/proxy --out build/compare

Per Direct3D API, two modes:
    native   the graph in the host's own API (dlss5vk bench, DLSS5VK_API=d3d12|d3d11)
    bridge   the host hands its frame to a dedicated Vulkan device and takes the head back (dlss5vk bridge)
plus the references that say what a difference means:
    vulkan             Vulkan run directly: no host, no transport
    vulkan-d3dkernels  the same with the kernel set Direct3D runs (native backend only: without Vulkan's
                       cooperative-matrix GLSL), so a native-vs-Vulkan gap can be split into kernels and API
    transport          the bridge with no network: what the sharing and the two fence waits cost alone

Every configuration runs the same fixture (same proxy, seed and conditioning, hence the same features), the same
backend, a graph recorded once and replayed (as a host runs it), one frame in flight, counter chaining off. Each run
is its own process, so its preparation (device, model, kernels, first frame) is measured too and kept apart. Rounds
alternate the order (forward, then reversed), and the first --warmup frames of every run are dropped.

What is reported, per configuration, over the kept frames of all rounds: median, p5, p95, p99, min, max, standard
deviation of
    gpu_ms          the graph on the GPU, by the executing API's own timestamps (one clock domain per figure)
    gpu_submission_ms  everything the executing API submits for the frame: the features generated from the proxy and
                    the graph and, on the bridge, the image <-> buffer copies around them
    record_ms       host CPU to issue the frame's commands (the recurring CPU cost)
    host_frame_ms   wall time from the frame's first command to its usable output (the host CPU clock)
and, for the bridge, the host's copies in and out. Figures of different clock domains are never subtracted from one
another: the bridge's extra cost is read from wall times (one clock) and from the transport-only run.

GPU runs are serial and the GPU must be idle: a busy GPU (another process) stops the comparison instead of
measuring through it. Results go to <out>/results.json and <out>/REPORT.md.
"""
import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def gpu_state():
    fields = "name,driver_version,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,pstate"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        values = [v.strip() for v in out.strip().splitlines()[0].split(",")]
        return dict(zip(fields.split(","), values))
    except Exception as error:  # noqa: BLE001 - diagnostics only
        return {"error": str(error)}


def require_idle(limit, tries=5):
    """The GPU must be idle before a run. Returns its state, or raises when something else keeps it busy."""
    last = None
    for _ in range(tries):
        last = gpu_state()
        try:
            if float(last.get("utilization.gpu", "0")) <= limit:
                return last
        except ValueError:
            return last
        time.sleep(2.0)
    raise SystemExit(f"GPU busy ({last}): another process is using it; the comparison is deferred, not measured through it")


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def summarize(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "samples": len(ordered),
        "median": percentile(ordered, 0.5),
        "p5": percentile(ordered, 0.05),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "min": ordered[0],
        "max": ordered[-1],
        "stdev": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=pathlib.Path, default=ROOT / "build" / "dlss5vk.exe")
    parser.add_argument("--shaders", type=pathlib.Path, default=None, help="default: <exe dir>/shaders")
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--fixture", type=pathlib.Path, required=True, help="a proxy fixture: the frames every mode runs")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="a new directory")
    parser.add_argument("--backend", default="auto", help="native or sm86; auto: what the GPU allows")
    parser.add_argument("--apis", default="d3d12,d3d11")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--idle-limit", type=float, default=15.0, help="GPU utilization (%%) above which a run is not started")
    args = parser.parse_args()
    if args.frames <= args.warmup:
        raise SystemExit("--frames must exceed --warmup")
    args.out.mkdir(parents=True, exist_ok=False)
    shaders = args.shaders or args.exe.parent / "shaders"
    manifest = json.loads((args.fixture / "manifest.json").read_text())
    full_width, full_height = manifest["fullDimensions"]
    apis = [a for a in args.apis.split(",") if a]

    base_env = {k: v for k, v in os.environ.items() if not k.startswith("DLSS5VK_")}
    base_env["DLSS5VK_CHAIN"] = "0"
    if args.backend != "auto":
        base_env["DLSS5VK_BACKEND"] = args.backend

    def bench(api, extra_env=None):
        env = {"DLSS5VK_API": api, **(extra_env or {})}
        return env, ["bench", "--model", args.model, "--fixture", args.fixture, "--shaders", shaders, "--frames", args.frames, "--tape"]

    def bridge(host, extra=()):
        return {}, ["bridge", "--host", host, "--model", args.model, "--fixture", args.fixture, "--shaders", shaders, "--frames", args.frames, *extra]

    def transport(host):
        return {}, ["bridge", "--host", host, "--transport", "--width", full_width, "--height", full_height, "--frames", args.frames]

    # One backend for every configuration. Left to "auto", Vulkan and Direct3D would not agree on an Ampere GPU
    # (Vulkan's auto is native or compat; sm86 is explicit there), so the route the GPU's compute capability allows
    # is resolved once, by a short Direct3D 12 run, and then named to every run.
    probe_json = args.out / "probe.json"
    probe = subprocess.run([str(args.exe), "bench", "--model", str(args.model), "--fixture", str(args.fixture), "--shaders", str(shaders),
                            "--frames", "2", "--tape", "--json", str(probe_json)], env={**base_env, "DLSS5VK_API": "d3d12"},
                           capture_output=True, text=True)
    if probe.returncode:
        raise SystemExit("the probe run failed:\n" + probe.stdout[-2000:] + probe.stderr[-2000:])
    backend = json.loads(probe_json.read_text())["backend"]
    base_env["DLSS5VK_BACKEND"] = backend

    configs = {"vulkan": bench("vulkan")}
    if backend == "native":
        configs["vulkan-d3dkernels"] = bench("vulkan", {"DLSS5VK_SHADER_FP8": "0"})
    for api in apis:
        configs[f"{api}-native"] = bench(api)
        configs[f"{api}-bridge"] = bridge(api)
        if backend == "native":
            configs[f"{api}-bridge-d3dkernels"] = bridge(api, ("--same-kernels",))
        configs[f"{api}-transport"] = transport(api)

    names = list(configs)
    runs = {name: [] for name in names}
    states = []
    for round_index in range(args.rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for name in order:
            before = require_idle(args.idle_limit)
            env, arguments = configs[name]
            output = args.out / f"{name}.round{round_index}.json"
            log = args.out / f"{name}.round{round_index}.log"
            command = [str(args.exe), *map(str, arguments), "--json", str(output)]
            with log.open("w", encoding="utf-8", errors="replace") as stream:
                code = subprocess.run(command, env={**base_env, **env}, stdout=stream, stderr=subprocess.STDOUT, timeout=1800).returncode
            after = gpu_state()
            states.append({"config": name, "round": round_index, "before": before, "after": after, "exit": code})
            if code:
                raise SystemExit(f"{name} round {round_index} failed (exit {code}); see {log}")
            runs[name].append(json.loads(output.read_text()))
            print(f"round {round_index} {name}: ok", flush=True)

    metrics = ("gpu_ms", "gpu_submission_ms", "record_ms", "submit_wait_ms", "host_write_ms", "host_read_ms", "host_frame_ms")
    summary = {}
    for name in names:
        entry = {"runs": len(runs[name])}
        first = runs[name][0]
        for key in ("api", "host", "mode", "backend", "device", "shader_fp8", "taped", "chained", "dispatches", "sm_count", "frame_work",
                    "width", "height", "full_width", "full_height", "transport_width", "transport_height", "memory_bytes"):
            if key in first:
                entry[key] = first[key]
        for metric in metrics:
            samples = [v for run in runs[name] for v in run.get(metric, [])[args.warmup:]]
            if samples:
                entry[metric] = summarize(samples)
        preparation = [run["preparation_s"] for run in runs[name] if "preparation_s" in run]
        if preparation:
            entry["preparation_s"] = {key: statistics.median(p[key] for p in preparation) for key in preparation[0]}
        summary[name] = entry

    results = {
        "fixture": str(args.fixture), "backend": backend, "rounds": args.rounds, "frames": args.frames, "warmup": args.warmup,
        "order": "alternating: forward on even rounds, reversed on odd rounds", "frames_in_flight": 1, "chaining": "off",
        "graph": "recorded once, replayed every frame", "gpu": states[0]["before"] if states else gpu_state(),
        "configs": summary, "runs": states,
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    def cell(entry, metric, key="median"):
        value = entry.get(metric)
        return "-" if not value else f"{value[key]:.3f}"

    lines = [
        "# Execution comparison", "",
        f"Fixture `{args.fixture.name if args.fixture.name != 'proxy' else args.fixture.parent.name}` "
        f"({manifest['sourceDimensions'][0]}x{manifest['sourceDimensions'][1]}, full {full_width}x{full_height}), backend `{backend}`, "
        f"{args.rounds} rounds of {args.frames} frames ({args.warmup} dropped), order alternated, one frame in flight, chaining off, graph replayed from a tape.", "",
        f"GPU: {results['gpu'].get('name')} driver {results['gpu'].get('driver_version')}, "
        f"{results['gpu'].get('temperature.gpu')} C at start.", "",
        "All times in milliseconds. `graph` is the graph alone and `gpu frame` everything the executing API submits for the "
        "frame (features from the proxy, graph, and the bridge's image/buffer copies), both by that API's timestamps; "
        "`record` is host CPU per frame; `frame` is wall time to a usable output (median, p95, p99).", "",
        "| configuration | kernels | graph median | graph p95 | gpu frame median | gpu frame p95 | record median | frame median | frame p95 | frame p99 | frame stdev |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in names:
        entry = summary[name]
        kernels = "-" if entry.get("mode") == "transport" else ("coopmat GLSL + PTX" if entry.get("shader_fp8") else "PTX + exact scalar")
        lines.append(f"| {name} | {kernels} | {cell(entry, 'gpu_ms')} | {cell(entry, 'gpu_ms', 'p95')} | {cell(entry, 'gpu_submission_ms')} | "
                     f"{cell(entry, 'gpu_submission_ms', 'p95')} | {cell(entry, 'record_ms')} | "
                     f"{cell(entry, 'host_frame_ms')} | {cell(entry, 'host_frame_ms', 'p95')} | {cell(entry, 'host_frame_ms', 'p99')} | "
                     f"{cell(entry, 'host_frame_ms', 'stdev')} |")
    lines += ["", "Preparation (seconds, median over rounds; `first_frame` holds the kernel compilation):", "",
              "| configuration | device | model | kernels | graph | first frame | device-local MiB | peak MiB |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in names:
        entry = summary[name]
        p = entry.get("preparation_s", {})
        m = entry.get("memory_bytes", {})
        mib = lambda key: "-" if key not in m else f"{m[key] / 1048576:.0f}"  # noqa: E731
        fmt = lambda key: "-" if key not in p else f"{p[key]:.2f}"  # noqa: E731
        lines.append(f"| {name} | {fmt('device')} | {fmt('model')} | {fmt('kernels')} | {fmt('graph')} | {fmt('first_frame')} | {mib('device_local')} | {mib('peak_device_local')} |")
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
