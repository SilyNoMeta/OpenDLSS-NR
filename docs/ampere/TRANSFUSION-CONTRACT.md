# Integration contract: OpenDLSS-NR as an experimental engine in DLSSG-Transfusion

What a host (DLSSG-Transfusion, a game hook, any renderer) must provide and may rely on to run this engine. This is
a contract for a **future** integration: nothing here is wired into Transfusion yet. It describes the engine as it
stands on `feat/sm86`; where a point is a recommendation rather than existing code, it says so.

Reference implementation of everything below: `demo/nr_pass.{h,cpp}` (a renderer integration that records into the
renderer's own command buffer) and `src/main.cpp` (`image`, `bench`, `parity`). Background: [frame.md](../frame.md)
(the per-frame pipeline), [numerics.md](../numerics.md) (exactness), [execution.md](../execution.md) (scheduling).

## 1. The model

- A model directory: `manifest.json` plus `model/stages/*.bin` (README, "Model directory"). Nothing in this
  repository produces one; the user prepares it from their own copy of the DLSS-NR 310.8.0 runtime. The engine
  refuses any graph but the 71-block one (`totals.blockCount`). Image/parity and the demo verify stage SHA-256;
  bench/profile skip hash work to measure setup separately. A host should verify once before using a model.
- `block70.layer0.blend_scale` (one f16, 0.7397 in the shipped model) is the cap of the temporal blend; the host
  reads it through `nr::Model::tensor(70, 0, "blend_scale")` when it composes itself.
- Device memory, RTX 3070 Ti Laptop GPU (`dlss5vk bench`): see [RESULTS.md](RESULTS.md#memory). The raw tensors
  stay resident (some kernels read scale vectors from them); the re-laid matrices are built lazily by the first
  graph and kept by the `Model` for every later graph (resizes included).

## 2. Backends and their selection

| backend | GPUs | arithmetic | status |
| --- | --- | --- | --- |
| `native` | Ada, Hopper, Blackwell (FP8 tensor cores, driver with `VK_EXT_shader_float8`) | bit-exact to the captures (upstream claim) | unchanged by this fork; not run here (no such GPU) |
| `compat` | NVIDIA GPU meeting DeviceRequirements | bit-exact on the tested WebGPU-derived fixtures | validated on sm86; other hardware unconfirmed |
| `sm86` | Ampere (compute capability 8.0-8.7) | not bit-exact (f16 Tensor Core sums), deviations measured | validated on sm86 only; experimental, opt-in |

- `DLSS5VK_BACKEND=auto|native|compat|sm86` (environment, read by `vk::requestedBackend()`; a host can set it
  before creating the context, or create the device itself, section 3). `auto` = native, else compat; it never
  picks `sm86`, and it refuses to leave `native` on a GPU of compute capability 8.9 or later (a driver problem).
  If architecture identification fails and native cannot run, auto refuses a downgrade; compat must be explicit.
- An explicit backend that the device cannot run is an error, never a fallback. The chosen backend and the reason
  are printed (`backend: <name> - <description> (<reason>)`); `vk::Context::backend()` returns it. A host should
  show it to the user, and should record it next to any capture or bug report.
- `DLSS5VK_DEVICE=<index or part of the name>` restricts the physical devices (hybrid laptops list the integrated
  GPU too).

## 3. Device and queue

Two ways to get a context:

1. **The engine's own device**: `vk::Context()` creates an instance, picks the device and backend (section 2),
   creates a device with one compute queue. Simplest; the host then shares images with it (section 7).
2. **A borrowed device** (the demo's way, and the one to use when the host already has a Vulkan device):
   `vk::Context(const vk::BorrowedDevice&)` with the instance, physical device, device, queue family and index,
   the backend the device was created for, and **the `VkDeviceCreateInfo` it was created with**. The device must
   have been created with `vk::DeviceRequirements(backend)`'s feature chain in its `pNext` and its extensions
   enabled; the constructor checks exactly that (enabled extensions and feature bits, not what the GPU supports)
   and throws, naming what is missing. Preserve the queue declarations too: the requested family/index must
   have been created without protected flags and provide compute/timestamps. Pick the backend first with
   `vk::selectDevice(instance, extraExtensions)`. All declaration pointers must stay valid for the constructor.

Queue use: per-frame work is **recorded into the host's command buffer** (no submission, no host wait). The
context's own queue is used only for synchronous setup work: uploads (model, parameters), resizes, captures. Those
calls submit and wait; they must not run while another thread submits to the same `VkQueue` (the demo gives the NR
side the family's second queue when there is one).

## 4. Inputs per frame

| input | format | notes |
| --- | --- | --- |
| scene colour | HDR (rgba16f in the demo) | clamped to >= 0 and turned into the display proxy (frame.md, "The proxy") with the host's `paperWhite` |
| motion | uv units, y down, current -> previous, plus a history-valid flag | frame.md, "Motion vectors": the flag is separate from the motion (zero motion is a claim); off-screen previous positions have no history |
| conditioning | `localTone`, `localStructure`, `skinStructure`, `style` (0-2), `autoMask` | the network's five scalars; `intensity` and the style operator are composition, not network inputs |
| seed | u32 | the three Gaussian noise lanes; the same seed and inputs give the same output on a given backend |
| reset | bool | section 6 |
| size | the valid width and height (>= 33 each) | the field is padded by `nr::Geometry::fromValid` (e.g. 1920x1080 -> 1920x1152); the network runs at the field size |

The features the network reads are f32 `[field][16]` (`nr_preprocess.comp` builds them from the above; `image` and
`parity` build them from a proxy with `Kernels::preprocessFromProxy`). A host that only has an LDR image can feed
the proxy directly, as `dlss5vk image` does (no temporal history).

## 5. Outputs

- The head: f32 `[field][4]` (`Graph::head()`): RGB residual in proxy code space and the per-pixel blend logit.
- The composition (frame.md, "Composition"): `neural = clamp(proxy + rgb / 4, 0, 1)`, blended with the reprojected
  history by `clamp(sigmoid(logit) * blendScale, 0, 1)` where there is a history; the stored history is the
  result **truncated** to the half grid. The demo's `nr_composite.comp` then applies the style operator, the
  intensity, a tone upgrade back to HDR and its display transform; a host substitutes its own display path.
- Images are left in `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` (`NrPass::shaderReadOnlyLayout()`).

## 6. History

- Two history images alternate by frame parity. Frames using this graph must execute in order on one queue with
  the pass's reuse barriers. Parameter buffers follow the same parity. The demo records secondary commands and
  their descriptor sets once; their pools remain alive until GPU completion before a rebuild. A host must not
  reset a descriptor pool while commands using its sets are pending (`vk::Context::resetDescriptorPool(slot)`).
- Reset (`NrPass::resetHistory()`): on a camera cut, a scene change, a resize, NR or temporal toggled, and after a
  chained-wait timeout (section 8). After a reset the composite blends with weight 0 for that frame.
- With NR off the composite keeps the history primed with the proxy, so turning NR back on does not flash.
- What is not covered: disocclusion (a surface uncovered by motion keeps a valid motion vector and samples what
  was in front of it); the network's blend logit is relied upon there (frame.md).

## 7. Synchronization and resources

- Recorded work: the host records a barrier that makes its colour and motion writes visible to compute reads
  before the NR pass, and reads the outputs after a compute -> its-stage barrier. Inside the pass every launch is
  separated by a compute barrier (`DLSS5VK_CHAIN=0`, the default on `sm86`), or chained through device counters
  (the native default).
- A host on another API (D3D12) has two options, neither implemented: a separate Vulkan device sharing the images
  and a timeline semaphore with the game's device (`VK_KHR_external_memory_win32`,
  `VK_KHR_external_semaphore_win32` and `VK_KHR_timeline_semaphore` are all exposed on the RTX 3070 Ti, driver
  617.14), or a port of the graph's host code to D3D12. The first adds a cross-API wait per frame; it has not
  been measured.
- Lifetimes: `vk::Context` > `nr::Model` > `nr::Kernels` > `nr::Graph`. A resize rebuilds the `Graph` (its
  activations) and the host's images; context, model (and its re-laid matrices), kernels and pipelines survive.
  `vk::Context::memoryUse()` reports buffer memory allocation bytes in use and peak, excluding images/driver allocations.
- Preparation vs recurring cost: model load, kernel creation, graph build and the first frame (pipeline and PTX
  compilation) happen once; `dlss5vk bench` prints them apart (RESULTS.md).

## 8. Failures

- The engine reports failures with messages/exceptions; the renderer may also abort on a lost device. A host must
  treat device requirements, model checks, missing kernels and chained timeouts as "no NR this frame / this session"; it must never present
  a partial or zero output as a processed frame.
- Chained waits have a watchdog (execution.md): a timeout marks the frame wrong; with the renderer idle the host
  calls `NrPass::fallBackToBarriers()` and resets the history. The demo retains a failing session status and
  refuses to capture a known failed frame. On `sm86` chaining is off unless `DLSS5VK_CHAIN=1`.

## 9. What to gate on

- `compat`: `dlss5vk parity` on fixtures (bit-exact) - it is the arithmetic reference on Ampere.
- `sm86`: `scripts/compare_backends.py` against independently qualified `compat`, plus the conversion/MMA probe.
  Report byte differences and numerical metrics, then assess quality separately. Seed variation is descriptive;
  it is not a justified quality threshold or a full-network error bound. Do not make sm86 automatic on that basis.
- `native`: `dlss5vk parity` on the upstream fixtures, on an Ada or newer GPU.

## 10. Open points for the integration

- The D3D12 bridge (section 7) and its cost per frame.
- The game's HDR format (scRGB / HDR10) to the proxy's paper-white-relative input.
- Performance on Ampere against the engine Transfusion runs today: not compared here, as it needs the same game
  frames, size, passes and cadence on both engines (RESULTS.md gives this engine's cost alone).
