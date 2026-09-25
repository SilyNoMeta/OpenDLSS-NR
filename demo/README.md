# DLSS 5 NR: native demo on Filament

The NR network running live inside a [Filament](https://github.com/google/filament) frame: real scenes, real
per-object motion vectors, and the temporal history the network was trained to use.

Windows and Linux/X11. See [Platform support](#platform-support).

## Build and run

Linux:

```bash
scripts/fetch_tools.sh --npm
scripts/fetch_filament.sh
scripts/build_filament.sh
scripts/build.sh
scripts/build_demo.sh
build/demo/dlss5-demo [scene.gltf|.glb [environment.hdr]] --model <nr model dir>
```

Windows:

```
scripts\fetch_tools.ps1 -Npm        # once: the toolchain under tools\ (and the scene converter's npm modules)
scripts\fetch_filament.ps1          # once: clone Filament v1.77.0, apply third_party\filament.patch
scripts\build_filament.ps1          # Filament + SDL2 + filagui -> third_party\filament-install (~15 min, once)
scripts\build.ps1                   # NR kernels / PTX (build\shaders, build\ptx) + the dlss5vk tool
scripts\build_demo.ps1              # demo shaders + materials (build\data) + build\demo\dlss5-demo.exe
build\demo\dlss5-demo.exe [scene.gltf|.glb [environment.hdr]] --model <nr model dir>
```

`--model` is the network's model directory (see the top-level README; it is not part of this repository), or
`DLSS5VK_MODEL`, or `models\nr` next to the repository. Without a scene argument the demo starts on the first
directory under `build\scenes` that holds a `view.json`. `scripts\run_demo.ps1 -Model <dir>` does the same.

## Controls

The window can be resized; the network re-fits to the new size.

- **WASD / QE** move, **shift** faster
- **left drag** look around
- **N** NR on/off, **T** temporal history, **R** reset history
- **B** background
- **C** capture: a PPM of the output and of the HDR scene
- **Esc** quit

## Command line

- `--width w --height h`, `--scenes <dir>`, `--kernels <dir>`
- `--view px,py,pz,tx,ty,tz[,fov]` places the camera
- `--nr 0` runs without the network, `--temporal 0` without history
- `--style 1|2`, `--intensity x`, `--animation n`
- `--frames N --capture prefix` is the scripted run: it captures at frames 40, 100, 160 and 200 of a
  static, orbit, settle, NR-off sequence, and dumps the raw scene and velocity buffers at frames 99 and 100 and
  the unpacked motion (with its history flag) at 100. `--orbit deg` yaws the camera by that much per frame from
  frame 40.
- `DLSS5_DEMO_VALIDATION=1` enables the Khronos validation layer and refuses to start without it (a Vulkan SDK, or
  `VK_LAYER_PATH` at a build of Vulkan-ValidationLayers; `VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true` adds
  synchronization validation). The NR pass runs clean under both; the remaining messages are Filament's.

## Scenes

No scenes come with this repository. gltfio reads glTF 2.0 with Draco and meshopt compression, KTX2 textures
and the KHR material extensions (clearcoat, sheen, transmission, volume, IOR, specular, emissive strength).

### Getting one

- **Lone Monk**, Carlo Bergonzini / Monorender, CC BY 4.0, in the Cycles section of
  [blender.org/download/demo-files](https://www.blender.org/download/demo-files/). That download is a `.blend`,
  so it needs a glTF export first; the commands below take the web build of it.
- **Bistro**, Amazon Lumberyard, CC BY 4.0, from
  [ORCA](https://developer.nvidia.com/orca/amazon-lumberyard-bistro). That download is FBX, so it needs a glTF
  export too, and it is large: about 2.8 million triangles for the exterior, 381 MB once converted.
- The Filament checkout brings in two small assets that need **no conversion**, under
  `third_party\filament\third_party\models`: `Fox\Fox.glb` (CC0 model, CC BY 4.0 rig and animation) and
  `AnimatedMorphCube\AnimatedMorphCube.glb` (CC0). `demo\views` has a view file for each.

### Converting one

Web-style scene packages (a `view.json` next to a `.glb`, `lighting.hdr`, optional `instances.json/.bin`
scatter transforms and `groom.json` hair grooms, WebP textures) go through `scripts\convert_scene.mjs`. It
needs `@gltf-transform/core`, `@gltf-transform/extensions`, `@gltf-transform/functions`, `meshoptimizer` and
`draco3dgltf` from npm, plus Pillow for the WebP step. Lone Monk, as distributed for the web:

```
scripts\fetch_tools.ps1 -Npm        # the modules under tools\gltf (or npm install them anywhere and pass --modules)
node scripts\convert_scene.mjs <lone-monk.glb> build\scenes\lone-monk\lone-monk.gltf --modules tools\gltf --drop-material sky1
python scripts\convert_scene_textures.py build\scenes\lone-monk\lone-monk.gltf
copy demo\views\lone-monk.view.json build\scenes\lone-monk\view.json
```

`scripts\import_scene.py` does the same from a scene package directory:

```
python scripts\import_scene.py <package dir> build\scenes\<id> --modules tools\gltf [--name "Label"] [--model x.glb]
       [--environment x.hdr] [--drop-material name]... [--no-occlusion]
```

### What the importer changes, and why

Each of these is a Filament constraint rather than a preference:

- **Occlusion textures** are dropped by `--no-occlusion`. Exports whose occlusion channel is empty render
  black otherwise.
- **Hair grooms** become alpha-blended vertex-colored tube meshes drawn with depth writes, so the front
  strands win. Without depth writes a pile of translucent strands accumulates into an opaque mass.
- **The scene gets `msaa: 4`** when it has grooms. Strands are thinner than a pixel, so without multisampling every strand
  covers whole pixels and fuzz turns into speckle. With multisampling the coverage is geometric, and the patch
  interpolates normals, tangents and colors at the centroid, which keeps thin triangles from extrapolating
  into glints.
- **Scatter instances** become one node per instance.
- **UV sets** a material samples are renumbered to 0 and 1, and spare color and uv sets are dropped. Filament
  reads two uv sets and eight vertex buffers per primitive.
- **Zero-strength clearcoats** are removed. They would stop Filament's ubershaders applying transmission, IOR
  or sheen.
- **Thin transmission** (a cornea, a tear film: `KHR_materials_transmission` without a volume) is baked into
  alpha blending, because Filament's screen-space refraction cannot show an object's own inside through its
  shell.

One consequence worth knowing: blended (translucent) renderables are not part of the structure pass, so they
carry no motion of their own and the network sees the camera's. Masked ones do carry motion.

### view.json

Every directory under `build\scenes` holding a `view.json` appears in the **Demo scene** dropdown
(`--scenes <dir>` points elsewhere); selecting one loads its model, camera and lighting. Without a `view.json`
the model's bounds set the camera and the command-line environment lights it.

```
name                    label in the dropdown            model          .gltf/.glb in the directory (optional if unique)
environment             .hdr in the directory            environmentRotation   radians about y (the lighting; the visible sky does not rotate)
camera {position, target, fov, near, far}   first-person camera in the scene's own units (glTF space)
moveSpeed               WASD speed (default: the distance to the target per second)
exposure                the camera's exposure            environmentIntensity  IBL scale
msaa                    samples of the scene pass (1, 2, 4, 8; default 1): sub-pixel strands (hair grooms) need 4+
sun {direction, intensity}   directional light towards the sun, casting cascaded shadow maps
background "#rrggbb"    clear color; backgroundEnvironment true shows the environment instead
```

### Lighting units

The IBL intensity and the sun intensity are plain multipliers of the environment's radiance and of the sun's
illuminance (Filament's lux, with its 1/π Lambert), and the camera exposure multiplies both. The `view.json`
values of the previous renderer therefore carry over; retune by eye when a scene comes from another renderer.

### Animation

Animated glTF assets play their first clip; the **Animation** panel or `--animation n` picks another. The
motion vectors follow the skinning, the morph targets and the node transforms. `demo\views\fox.view.json`
and `morph-cube.view.json` are views for the two animated assets above: copy the model, an `.hdr` and the view
file as `view.json` into a directory under `build\scenes`.

## Styles

`Style` off / natural / cinematic are presets: exposure, contrast and saturation offsets scaled by the local
tone, applied to the temporal result. The id also feeds the network.

`custom` exposes the operator's knobs: exposure (EV), contrast (smoothstep blend, -1..1), gamma, saturation
(multiplier offset), hue shift (turns), vibrance (saturation power) and an overall strength (0 neutral, 1 the
knob values), plus the style id the network sees ("Network style"). "From natural" and "From cinematic" load a
preset into the knobs, with strength set to the tone, as a starting point.

Command line: `--style-knobs exposure,contrast,gamma,saturation,hue,vibrance,strength[,networkStyle]`.

## The Filament patch

Filament v1.77.0 gives the demo physically based materials, image-based lighting, cascaded shadow maps,
forward+ punctual lights, glTF 2.0 through gltfio, skinning and morph targets. `third_party/filament.patch`
adds the three things the network needs and Filament does not have.

**Per-object motion vectors.** The picking depth variant of every material also outputs, per pixel, the
screen-space motion of the surface point since the previous frame (`View::setVelocityBuffers`). The engine
keeps, per renderable, the world transform it was rendered with in the previous frame; for skinned and morphed
renderables the previous bones and morph weights (a pair of uniform buffers, swapped on the first write after
a frame consumed them); per instance buffer the previous local transforms; per view the previous un-jittered
clip-from-world. The structure pass, run at full resolution into user textures, writes
`(object id, depth bits, motion x bits, motion y bits)` as RGBA32UI. Cost: one depth-only pass.

**Vulkan interop.** `Engine::queueVulkanCommand` runs a callback on Filament's backend thread, at that point
of its command stream, with its current command buffer and the `VkImage` and layout of any texture
(`backend/VulkanInterop.h`). The NR pass records into Filament's own frame between the scene and the present
views: no extra queue submissions, no host synchronization.

**A shared device.** Filament runs on the demo's `VkDevice` through its shared-context path, so the device can
carry the NR kernels' features and extensions (`vk::DeviceRequirements`) next to what the renderer needs.

## The frame

Filament's scene view renders into an HDR target (rgba16f, post-processing off, so linear and un-tonemapped),
and the structure pass writes the velocity buffer. `NrPass` then runs motion unpack, feature preprocess, the
DLSS-NR graph from `src/`, and a temporal composite with the neural history, writing rgba8 into a Filament
texture. Filament's present view draws that as a fullscreen quad, with the ImGui overlay (filagui), into the
swapchain. The NR work is pre-recorded once per history parity as secondary command buffers.

## Platform support

The demo builds on Windows and Linux/X11. It requires an NVIDIA GPU exposing a graphics + compute queue because
Filament needs a swapchain; compute-only A100/H100 servers can run the top-level `dlss5vk` tool but not the
interactive demo. The macOS pieces (`native_window_cocoa.mm`, the CMake framework list) remain scaffolding for a
later Metal route.

Ada and Hopper use native FP8 cooperative matrices and PTX. Ampere A100 uses the slower software-E4M3
compatibility path in the compute tool; fused kernels and PTX are disabled automatically.

## Verifying the motion vectors

`--frames 220 --capture verify --orbit 0.4` on a static scene yaws the camera by 0.4 degrees per frame from
frame 40 and dumps, at frames 99 and 100, the HDR scene color and the motion (`verify-pre-scene.raw`,
`verify-moving-scene.raw`, `verify-moving-motion.raw`, plus the raw velocity buffer).

Reprojecting frame 99's color with frame 100's motion must reproduce frame 100. On Bistro at 1280x720 that is
a mean error of 0.006, against 0.030 without reprojection and 0.037 with the sign flipped, and no sub-pixel
offset. With `--orbit 0` and an animated asset the same check covers the skinning and morphing: the Fox's
"Run" clip with `--animation 2` gives 0.007 against 0.014.
