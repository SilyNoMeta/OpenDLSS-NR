# Ampere (sm86) port: action plan

Living plan for the `feat/sm86` branch of this fork. Status markers: `[x]` done and verified, `[~]` in progress,
`[ ]` not started, `[!]` blocked (the missing input is named).

Test machine: RTX 3070 Ti Laptop GPU (GA104, sm86, 46 SMs, 8 GB), NVIDIA driver 617.14 (Vulkan 1.4.351),
Windows 11 Pro 26200, hybrid laptop (an Intel Iris Xe is enumerated too), Visual Studio 2026 (18.9.3),
CUDA 13.4 toolkit (only for side experiments; the fork does not need it).

## 0. What the device actually exposes

- [x] `VK_KHR_cooperative_matrix` (f16, s8/u8, bf16; 16x16x16, 16x8x16, 16x8x8, int 16x16x32)
- [x] `VK_NV_cooperative_matrix2` with every feature bit the native route enables
- [x] `VK_NV_cuda_kernel_launch` v2 and `VK_NVX_binary_import` v2 (PTX launch from Vulkan is available)
- [x] **no** `VK_EXT_shader_float8`: no FP8 type, no FP8 cooperative matrix
- [x] compute shared memory limit 48 KiB for Vulkan pipelines

## 1. Fork preparation

- [x] `upstream` remote -> maanHimself/OpenDLSS-NR; work on `feat/sm86`, `main` stays the upstream reference
- [x] the three upstream commits fetched by SHA and their graph checked (cfa37e2 -> 928dc19; c856cda; both from 9d08f41)

## 2. Upstream commits

- [x] cherry-pick cfa37e2 (Linux + Ampere software-E4M3 compatibility), authorship preserved (`-x`)
- [x] cherry-pick 928dc19 (offline image workflow)
- [x] cherry-pick c856cda (hybrid laptops), resolved against cfa37e2: per-backend requirements and device scoring,
      `DLSS5VK_DEVICE` / `DLSS5VK_BACKEND`, a per-device, per-backend report, no global native-FP8 requirement
- [x] borrowed device (`vk::BorrowedDevice`): the backend is checked against the `VkDeviceCreateInfo` the caller
      used (enabled extensions and feature bits), not against what the physical device supports

## 3. Backends and non-regression

- [x] `DLSS5VK_BACKEND=auto|native|compat`; an explicit request that cannot be met is an error; `auto` prints the
      backend and the reason, and refuses to leave the native route on a GPU of compute capability >= 8.9
- [x] native SPIR-V byte-identical to before (all 13 files compared after every shader change); the native
      backend loads no `*_compat` module and its device requirements are unchanged
- [!] native numerical/performance non-regression on Ada/Hopper/Blackwell: **hardware unavailable here**

## 4. Compatibility route (reference on sm86)

- [x] qualified cfa37e2's scalar route on this GPU: ran end to end, but block 0 differed on 16% of its bytes, every
      later boundary on ~99%, and the head was all zeros (sequential f16 FMAs instead of the F13 group dot product;
      hardware E4M3 conversions in ops.comp without `shaderFloat8`)
- [x] exact compatibility kernels (`exact_mma.glsl`, `*_compat.comp`, `ops_compat.spv`)
- [x] no half arithmetic left to the GLSL compiler in the compat kernels (the native `window_normalize.comp` fma
      spelling drifted on this driver; native sources untouched)
- [x] model directory prepared locally from the user's DLL (tool and data outside the repository)
- [x] references: WebGPU-port captures (`web/capture.html`), the CPU reference (`dlss5vk verify`)
- [x] 12 fixtures bit-exact (6 images, 33x33 to 1279x721, styles 0-2, auto-mask on/off): 75 boundaries + head,
      head + composed image from the proxy; `verify`: every block-0 kernel equals the CPU reference
- [x] `image` (bit-identical to the reference composition), `bench`, `profile`, `enhance_image.py`
- Final controlled compat benchmark: 643.37-659.63 ms GPU at 512x512, 5051.93-5259.33 ms at 1920x1080.
  These synthetic graph medians differ from the earlier image-tool observations; see RESULTS.md for scope.

## 5. Optimized sm86 route

- [x] Ampere f16 HMMA arithmetic differs from F13; the final random operand/accumulator probe gives 54.3% exact
      results and 100% within its local summation bound. Earlier zero-accumulator experiments are not a full-network claim
- [x] PTX lowering to sm80 ISA for sm86: two f16 MMAs, exact operand conversion; 78 variants assembled
- [x] full engine launches through `VK_NV_cuda_kernel_launch`, including dynamic shared memory above 48 KiB
- [x] barriers default; opt-in chaining compared for repeatability/progress, not enabled automatically
- [x] composed-output comparisons on six fixtures; sm86 differs and remains experimental
- [x] controlled AB/BA performance rerun: setup, GPU graph, host submission/wait, memory and profile reported in RESULTS.md
- [x] backend flags isolated per Kernels instance; concurrent sm86/compat contexts and borrowed enabled-bit checks pass

## 6. Demo and temporal path

- [x] Filament and demo Windows builds; sm86 scripted run (motion/toggle/resize)
- [x] compat demo pipeline eviction fixed with submission fences; 250 frames pass synchronization validation without CPU serialization
- [x] CPU reprojection/history verification on compat and sm86: 160x120 -> 233x137, Fox skinning/camera motion, reset, NR off/on
- [x] sm86 second 250-frame run: all 11 captured histories byte-identical; renderer inputs identical between backends
- [x] sm86 640x360 -> 800x450 demo and no-temporal control; CPU verifier passes
- [x] final 12-fixture rerun, CPU block verify, two-backend profiles and conversion/MMA probe pass
- [x] all 78 PTX variants assemble for sm80 and sm86; Linux script syntax passes (no Linux compiler/runtime qualification)

## 7. Documentation and delivery

- [x] provenance, backends, build/data commands, results, limitations and hardware documented
- [x] integration contract documents enabled queues/features, synchronization, resources, history and failure status
- [~] audited pushes to `origin/feat/sm86` (no force-push, no private data, no binaries)
