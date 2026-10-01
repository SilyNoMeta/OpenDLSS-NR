# OpenDLSS-NR on Ampere (sm86)

This fork runs the network on NVIDIA Ampere GPUs, which have tensor cores but no FP8: an **exact** compatibility
backend (`compat`) and an **experimental** Tensor Core backend (`sm86`). Everything below was built and run on one
machine; what was only compiled or only read is said so. Measurements: [RESULTS.md](RESULTS.md). The plan and its
status: [PLAN.md](PLAN.md). What a host integration needs: [TRANSFUSION-CONTRACT.md](TRANSFUSION-CONTRACT.md).

Test machine: RTX 3070 Ti Laptop GPU (GA104, compute capability 8.6, 46 SMs, 8 GB), NVIDIA driver 617.14 (Vulkan
1.4.351), Windows 11 Pro 26200, a hybrid laptop (an Intel Iris Xe is enumerated too), Visual Studio 2026 18.9.3,
the repository's pinned toolchain (glslang 16.6.0, Vulkan-Headers v1.4.363, volk, CMake 3.31, Ninja 1.13), Python
3.11, Node 22 and Chrome for the WebGPU port. The CUDA 13.4 toolkit was used only for side experiments (ptxas checks,
characterisation of the f16 MMA); nothing in the build needs it.

## What the GPU exposes

`VK_KHR_cooperative_matrix` (f16, int8, bf16), `VK_NV_cooperative_matrix2`, `VK_NV_cuda_kernel_launch` and
`VK_NVX_binary_import`; **no `VK_EXT_shader_float8`**: no FP8 type, no FP8 cooperative matrix, no FP8 conversion
instruction usable on this architecture. The native route needs all three FP8 capabilities, so it cannot run here.

## Provenance

| change | origin |
| --- | --- |
| Linux build, the software-E4M3 compatibility backend | [upstream cfa37e2](https://github.com/maanHimself/OpenDLSS-NR/commit/cfa37e2e7126bc8b91a2deecb963737cd10791c2), authorship and co-author preserved (`-x`) |
| `dlss5vk image`, `scripts/enhance_image.py` | [upstream 928dc19](https://github.com/maanHimself/OpenDLSS-NR/commit/928dc19f53895bc99345cb886263e72c2408ed2f), cherry-picked with attribution |
| hybrid-laptop device selection, `--max-fps` | [upstream c856cda](https://github.com/maanHimself/OpenDLSS-NR/commit/c856cda674171e3b4855dc34ab88bdde2e415ea6), attributed and resolved against `cfa37e2` (below) |
| everything else listed here | this fork (`feat/sm86`) |

`cfa37e2` and `c856cda` both started from `9d08f41` and both rewrote the device selection: one added the
compatibility fallback, the other a selector that required the native FP8 features of every device. The resolution
keeps both intents without the global FP8 requirement: requirements per backend, every device checked against each
backend, an explicit `DLSS5VK_DEVICE` / `DLSS5VK_BACKEND`, and a report of what each device lacks for each backend.
For a device borrowed from a renderer, its created queue and backend are checked against the `VkDeviceCreateInfo` it was created with
(what was enabled), not against what the GPU supports.

The sm86 lowering adapts the public PTX generators under this repository's MIT license; the existing license and
credits remain. The Filament patch applies to Google's public Apache-2.0 source and retains its notices. No private
repository history, proprietary kernel payload or derived model data is part of the public source.

## Backends

| | `native` | `compat` | `sm86` |
| --- | --- | --- | --- |
| GEMMs | FP8 cooperative matrices / FP8 PTX | the F13 group arithmetic in integer GLSL | PTX lowered to f16 `mma.sync` |
| fusion, chaining | yes, yes | no, no | fused PTX blocks; chaining off by default |
| exact | upstream fixture claim; hardware confirmation pending here | yes for the 12 WebGPU-derived fixtures here | no: Ampere sums the products differently |
| auto-selected | yes | when native is unavailable | never (`DLSS5VK_BACKEND=sm86`) |

The native route is unchanged by this fork: same SPIR-V (all 13 native files compared after every shader change),
same device requirements, same kernels and routes. It was not run here (no Ada or newer GPU): **its absence of
regression remains to be confirmed on such hardware.**

### The compatibility backend, made exact

Run on this GPU against an independent reference (below), `cfa37e2`'s compatibility route ran end to end, but
block 0 differed on 16% of its bytes, every later boundary on about 99%, and the head was all zeros: an image came
out, without any NR in it. Three causes, all fixed:

1. Its GEMMs and attentions accumulated with sequential f16 FMAs. The tensor cores compute a fixed-point dot
   product over each group of 16 E4M3 products (F13, [numerics.md](../numerics.md)). `shaders/exact_mma.glsl`
   emulates it in integer code, bit for bit the CPU reference, and the `*_compat` kernels follow their native
   twins step for step (residual seeding, K order, the ViT's partitions, which the compat GEMM had ignored).
2. `ops.comp` (pooling, skip merges, post blend) uses the hardware E4M3 conversions. Without `shaderFloat8` the
   driver still builds the pipeline, and the conversions return garbage. `ops.comp` is also built with
   `-DDLSS_SOFTWARE_E4M3` into `ops_compat.spv`.
3. A rare drift (one E4M3 code in millions, then propagated) on four of six images: the native
   `window_normalize.comp` writes its square sums as a `float16_t` fma, which this driver's compiler does not
   evaluate with a single rounding. The compatibility kernels now spell every half operation in f32 published by
   the bit-level `roundF16`, and the norms and softmax reciprocals are the correctly rounded halves in integer
   code. (Native's `rsqrt.approx.ftz` / `rcp.approx.ftz` then `cvt.rn.f16` give the correctly rounded half for
   every half input, checked exhaustively on this GPU.) The native GLSL route keeps that spelling; whether it
   drifts on Ada is to be checked there (the native production route uses the PTX kernels).

### The sm86 backend

The native route's PTX kernels (fused 32-channel blocks, expert FFNs, fused QKV attention, the gemm2 / gemmt /
gemmv GEMMs, global attention) lowered by `scripts/ptx/lower_sm86.py`:

- each `mma.sync.m16n8k32.e4m3` becomes two `m16n8k16.f16` on exactly decoded operands, keeping every product and
  every 16-product group of the native instruction; only the summation inside a group is Ampere's;
- the E4M3 decode and encode instructions become integer code with the instructions' semantics (tested on all finite E4M3 pairs and all half bit patterns in both encode lanes,
  `scripts/ptx/test_lower_sm86.py`; FP8 NaN decode pairs are excluded);
- `.target sm_80`: compiled by the driver for any compute capability 8.x; validated on 8.6 only.

Launched through `VK_NV_cuda_kernel_launch`, with the exact compatibility GLSL for what the native route would run
on FP8 cooperative matrices. The split-512 branch MLP, cooperative-matrix GLSL only, runs as one gemm2 per branch.
Counter chaining reproduced the barrier head on the tested 512x512 fixture but showed no performance benefit here, so it stays off unless `DLSS5VK_CHAIN=1`.
`--accumulate=f32` lowers with an f32 accumulator published to half per group: an optional arithmetic experiment. The qualification below uses the default f16 accumulator;
f32 has no visual-quality certification.

## How the arithmetic was checked

No NVIDIA-native full-network capture was available locally on this Ampere GPU. Expected data was not generated
from the Vulkan output being tested. The
references are independent implementations of the same specification:

- **The WebGPU port** (`ports/browser-webgpu`), an exact WGSL implementation by the upstream author, validated by
  them against native captures. `web/capture.html` (driven by `tools/headless.mjs capture`) runs it on a proxy
  image and writes fixtures with the existing contract: every boundary and the head from recorded features, and
  the head and composed image from the proxy. Its manifests name it as the source.
- **The CPU reference** (`src/reference.cpp`): `dlss5vk verify` checks every block-0 kernel against it.
- To localise a difference: `DLSS5VK_CAPTURE_BLOCK=N` (one block's intermediate tensors) and the same on the
  WebGPU side (`capture.html?block=N`).

## Build and run (Windows; Linux scripts not run here)

```powershell
scripts\fetch_tools.ps1
scripts\build.ps1            # shaders (+ *_compat), build\ptx, build\ptx_sm86, build\dlss5vk.exe
$env:DLSS5VK_BACKEND = 'sm86'                  # opt-in; auto picks compat on this Ampere
build\dlss5vk.exe bench   --model models\nr --width 1920 --height 1080 --frames 40
build\dlss5vk.exe image   --model models\nr --input in.rgba-f32 --output out.rgba-f32 --width W --height H
$env:DLSS5VK_BACKEND = 'compat'                # exact WebGPU-derived fixtures
build\dlss5vk.exe parity  --model models\nr --fixture <fixture dir>
python scripts\ptx\test_lower_sm86.py         # the lowering, on the GPU (NumPy, the driver's CUDA library)
```

Reference fixtures from the WebGPU port (needs Node and Chrome; fixtures stay outside Git):

```powershell
python scripts\make_capture_request.py fixtures\req\img --image photo.png --size 768x432 --seed 5
$taskRoot = (Get-Location).Path
$env:NR_FIXTURES = "$taskRoot\fixtures\req\img"
$env:NR_CAPTURE_OUT = "$taskRoot\fixtures\web\img"
cd ports\browser-webgpu
node tools\headless.mjs capture
cd ..\..
$env:DLSS5VK_BACKEND = 'compat'
build\dlss5vk.exe parity --model models\nr --fixture fixtures\web\img\features
build\dlss5vk.exe parity --model models\nr --fixture fixtures\web\img\proxy
python scripts\compare_backends.py fixtures\req\img          # sm86 against compat, with the seed variation
python scripts\bench_backends.py --sizes 512x512,1920x1080   # alternating, with telemetry
```

The Linux scripts (`scripts/*.sh`) were updated alongside (the compat shader variants, the lowering step) and
checked for shell syntax only: no Linux system was available, so the Linux build is examined, not compiled.

## Data that is not in the repository

- The model directory (`models/nr`, git-ignored), prepared by the user from their own DLSS-NR 310.8.0 runtime;
  the public manifest contract is in the top-level README; model extraction is a user-side preparation step.
- Fixtures (`fixtures/`, git-ignored) and their capture requests.
- Scenes for the demo (`build/scenes`); the Filament checkout brings two that need no conversion.
- Raw captures, logs, derived model tensors and reference captures are local ignored inputs/outputs, never
  published source. No private repository or binary kernel is needed to compile this fork.

## Limitations

- This `sm86` lowering is not bit-exact. RESULTS.md quantifies the measured difference; it does not establish
  perceptual equivalence or full-network equality to NVIDIA. Seed variation is a descriptive comparison only.
- No Ada / Blackwell / Hopper GPU, no A100, no Linux system was available: the native route, compute capability
  8.0 and 8.7, and Linux are unverified here.
- Performance is this engine's alone: no comparison with another NR engine at the same game frames was made.
- sm75 (Turing) is out of scope.
