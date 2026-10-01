# sm86 qualification, 2026-10-01

Executed on a Windows 11 Pro 26200 hybrid laptop: NVIDIA RTX 3070 Ti Laptop (GA104, sm86, 46 SMs,
8 GB), driver 617.14, Vulkan 1.4.351; Intel Iris Xe also enumerated. Build: MSVC 18.9.3, CMake 3.31.12,
Ninja 1.13.2, glslang 16.6.0, Vulkan-Headers 1.4.363, Filament 1.77.0. Arithmetic probes/assembly:
CUDA toolkit 13.4; the engine itself uses the installed driver and Vulkan, without a toolkit dependency.
Python 3.11/NumPy and Node 22/Chrome were used for verification and independent WebGPU captures.

These are results of the standalone public engine. They do not establish a gain over another NR engine or a game.
The model is the user's locally prepared 310.8.0 network: 71 blocks, 153 tensors, 147,683,778 raw bytes. Model
files, scene assets, reference captures, output pixels and raw logs are ignored local data and are not published.

## Numerical coverage

The independent WebGPU implementation supplies expected full-network values. Its reference lineage is described
in [README.md](README.md#how-the-arithmetic-was-checked). No NVIDIA-native full-network capture was available
on this machine. The CPU implementation supplies independent block-0 values and temporal arithmetic checks.

| check | actual result |
| --- | --- |
| compat, six feature fixtures | 75 boundaries/transitions plus head each, all bit-exact (456 checks) |
| compat, six proxy fixtures | head plus composed f32-half output each, all bit-exact (12 checks) |
| compat block 0, `verify` | every tested kernel equals the CPU reference; validation: zero errors |
| sm86 E4M3 conversion lowering | all 64,516 finite code pairs decoded correctly; all 65,536 half patterns in each encode lane correct |
| sm86 MMA lowering, random operands and accumulators | 25,600 results: 54.3% equal to F13, 100% within the probe's exponent-based summation bound |
| native shaders | all 13 SPIR-V files byte-identical to the pre-port local snapshots |
| backend coexistence | compat does not change an existing sm86 instance's PTX or chain setting |
| borrowed device | disabled required features/extensions and an uncreated queue rejected; valid declaration accepted |
| failure reporting | explicit native on sm86 and Intel selection fail; missing PTX exits nonzero and creates no output |
| image utility | `enhance_image.py`, one pass at 33x33, completes and reports repeatable head |
| PTX assembly | all 78 lowered variants assemble for both sm80 and sm86 with ptxas 13.4 (156 checks) |

The group bound in `scripts/ptx/test_lower_sm86.py` covers the discarded product/accumulator bits at the largest
exponent and publication to half. It is a local arithmetic bound, not a bound for the nonlinear 71-block network.
FP8 NaN decode pairs are excluded; model load reports zero replaced NaN weight codes in these runs.

Requests use a credited public natural image at three sizes, procedural bands and deterministic noise, styles 0-2,
auto-mask on/off, and tone/structure/skin conditioning. Composed RGB differences below are against the independently
qualified compat route, with identical input, model, size, seed and one pass. Values in /255 are differences in
linear arrays of proxy code values, not measurements in HDR luminance.

| request | sm86 PSNR dB | mean absolute /255 | maximum /255 | channels >4/255 |
| --- | ---: | ---: | ---: | ---: |
| natural crop 512x512 | 43.27 | 1.250 | 23.0 | 3.534% |
| bands 333x517, style 1, mask off | 44.93 | 1.137 | 10.0 | 0.840% |
| natural 768x432, tone .5 | 48.51 | .502 | 17.6 | 1.015% |
| noise 1000x562, style 2 | 43.19 | 1.317 | 11.1 | 3.723% |
| natural 1279x721 | 47.17 | .613 | 27.0 | 1.343% |
| bands 33x33 | 37.62 | 2.285 | 14.0 | 17.631% |

All outputs were finite. Compat's mean NR effect relative to the proxy ranged from 6.86 to 19.53/255.
Changing the compat seed by one gives PSNRs 38.74-47.25 dB in this suite, but this is descriptive context:
seed variation is not an acceptance threshold, a proof of perceptual equivalence or a full-network error bound.
Byte identity, numerical deviation and visual quality are separate claims. **sm86 remains opt-in and experimental.**
`parity` on sm86 against the exact fixtures fails, as it should; it must not be relabelled a passing accuracy test.

## Temporal demo

Explicit scene: Filament's skinned Fox, three animations available, first animation running, camera orbit.
The scripted animation advances by 1/60 s per accepted frame, independent of actual rendering time. Each run has
250 accepted frames: static/camera movement, settle, explicit reset at 120, NR off at 160, on at 200,
resize at 210. Eleven snapshots contain scene, velocity, motion, head, input features, previous/next history and
frame metadata. `scripts/verify_demo.py` independently checks the camera matrix, motion unpack, history sampling,
feature history lanes and composition; the observed head is an input to this frame test, not a network reference.

| execution | result |
| --- | --- |
| compat, 160x120 -> 233x137 | all frame checks pass, sync validation reports zero errors |
| sm86, same dimensions/frames | all checks pass, renderer scene/velocity/motion bytes equal compat's |
| second identical sm86 run | all eleven captured histories byte-identical |
| sm86, 640x360 -> 800x450 | all checks pass; largest CPU/history deviation .001464 in [0,1] |
| sm86 with temporal disabled, 160x120 -> 233x137 | all snapshots declare invalid history; checks pass |

CPU comparison bounds are declared in the verifier: history .004 with reprojection, .0011 without; input history
.0006 after the 1/8 centring; unpacked motion within two half ULPs. They cover half publication, shader/CPU
transcendentals and the sampler's permitted subtexel interpolation precision. The recorded camera matrix is checked
within 2e-5 because camera text uses six significant digits. These tolerances validate frame plumbing, not quality.
Reset/re-enable/resize snapshots explicitly require no prior history. Bypass history equals the proxy.

The old compat demo could lose the device at 160x120: cache age advanced during skipped frames and Filament evicted
graphics pipelines still referenced by pending commands. The corrected patch keeps submission fences, resets binding
state after interop, and the NR pass orders reused buffers. Qualification uses asynchronous frames, without an added
CPU wait per compat frame. Khronos validation 1.4.304.1 emits vertex-attribute performance warnings; there were no
validation errors in the qualified runs. Validation is not a complete proof for device-address PTX accesses.

At 800x450 one captured final GPU frame cost 19.208 ms: scene .045, preprocess .124, network 18.953, composite .045,
present/UI .041. This is a frame observation; it is not an average FPS benchmark or an in-game result.

## Performance

`scripts/bench_backends.py --sizes 512x512,1920x1080 --rounds 2 --frames sm86=40,compat=5` runs AB then BA,
one process at a time. Same 71-block model, deterministic synthetic features, padded field, one full pass per frame;
no resolution, pass or cadence reduction. There was no concurrent benchmark. The following range is the two run
medians, after a separate warm-up. GPU clocks were sampled at 1785 MHz, memory 6001-7001 MHz; temperatures at sample
boundaries ranged 57-79 C. Clock/power variation and only five expensive compat samples limit precision.

| valid -> padded field | backend | GPU network median ms | host frame median ms | dispatches |
| --- | --- | ---: | ---: | ---: |
| 512x512 -> 576x512 | compat | 643.37-659.63 | 653.00-667.81 | 513 |
| same | sm86 | 14.16-14.25 | 22.66-27.44 | 481 |
| 1920x1080 -> 1920x1152 | compat | 5051.93-5259.33 | 5060.53-5264.12 | 513 |
| same | sm86 | 75.25-75.42 | 89.69-90.64 | 489 |

GPU spans include the whole graph's conversions and barriers. Host spans include recording, submission and waiting
for the GPU; input upload, output readback, rendering and temporal composition are outside this synthetic graph
benchmark. The demo observation above includes preprocess/composite/render stages. No cross-API synchronization has
been implemented or measured. The secondary-command replay check also passes; its CPU cost is a separate mode.

Preparation per fresh process, separate from the samples: device .46-.97 s, model .38-.66 s, kernel setup .00-.01 s,
graph .01-.05 s; first frame including driver compilation: compat 1.90-2.69 s at 512 and 7.12-8.56 s at 1080;
sm86 1.58-1.79 s and 2.07-3.09 s. Driver shader caches were already populated: these are process setup costs,
not guarantees for a clean driver cache. Resize preserves the model and kernels but rebuilds sized resources.

### Memory

| size | backend | device-local MiB | raw tensors | re-laid matrices | activations/scratch |
| --- | --- | ---: | ---: | ---: | ---: |
| 512x512 | compat | 908 | 141 | 141 | 626 |
| 512x512 | sm86 | 409 | 141 | 146 | 123 |
| 1920x1080 | compat | 4962 | 141 | 141 | 4681 |
| 1920x1080 | sm86 | 1094 | 141 | 146 | 808 |

Values are rounded from the buffer memory allocation sizes returned by Vulkan (`VkMemoryRequirements::size`);
independent rounding may differ by one MiB. Recorded peaks equal these resident totals. This includes buffer
allocation padding but excludes driver/module memory, scene textures,
swapchain/history images and other desktop processes; it is not total system VRAM usage.

Both `profile` commands complete. At 512x512, sums of per-dispatch spans were 632.10 ms compat and 17.29 ms sm86;
400 tiny dispatches plus barriers cost .794/.789 ms. Instrumented per-dispatch spans have different overhead from
the benchmark above. PTX runs on Vulkan buffer device addresses via `VK_NV_cuda_kernel_launch`; neither
`VK_NVX_binary_import` nor a CUDA Driver external-memory bridge is used by the engine. CUDA Driver probes test
arithmetic separately. The 78 generated variants include dynamic shared requests above the 48 KiB SPIR-V limit;
do not infer launch availability from an isolated CUDA test or from extension names alone.

Opt-in counter chaining on the 512x512 proxy reproduced the barrier head across five submissions, with no watchdog
timeout; observed 17.929 ms versus the faster default barrier runs. This proves only that tested schedule's ordering
and progress. It does not authorize default chaining at every shape; chaining stays off on sm86.

## Reproduction and unconfirmed scope

See [build/data commands](README.md#build-and-run-windows-linux-scripts-not-run-here),
the [demo command line](../../demo/README.md#command-line), and [PLAN.md](PLAN.md). Useful additional checks:

```powershell
scripts/test_backend.ps1
$env:DLSS5VK_BACKEND = 'compat'
build/dlss5vk.exe verify --model models/nr --fixture fixtures/web/cowboy512/features
build/dlss5vk.exe bench --model models/nr --width 160 --height 120 --frames 4 --secondary
$env:DLSS5VK_BACKEND = 'sm86'
build/demo/dlss5-demo.exe build/scenes/fox/Fox.glb build/scenes/fox/lighting.hdr --model models/nr --width 640 --height 360 --frames 250 --capture tmp/fox-large --resize-at 210,800x450
python scripts/verify_demo.py tmp/fox-large
```

Ada/Blackwell/Hopper native numerical and performance non-regression remains **unconfirmed**: no such hardware is
available. Native shaders and route defaults are preserved and audited; compilation is not numerical or performance
proof. A100/sm80, sm87 and Linux execution are unconfirmed. Linux shell scripts were syntax-checked only.
No sm75 port, Transfusion integration, cross-API bridge or binary release is part of this delivery.
