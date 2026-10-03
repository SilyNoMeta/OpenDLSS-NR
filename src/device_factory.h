// The device a tool or a test runs the NR graph on: an adapter of exec.h that owns its own device and queue.
//
//   DLSS5VK_API       vulkan (the default), d3d12 or d3d11 (Windows): the API that executes
//   DLSS5VK_BACKEND   the numeric route. On Vulkan: auto, native, compat, sm86 (vk_context.h). On Direct3D: native
//                     or sm86; auto is the one the GPU's compute capability allows (it has exactly one), and compat
//                     is refused (the scalar reference route runs on Vulkan).
//   DLSS5VK_SHADER_FP8=0   Vulkan, native backend: no cooperative-matrix GLSL, i.e. the kernel set Direct3D runs (FP8
//                     PTX plus the exact scalar kernels), so that two APIs can be compared on identical kernels.
//   DLSS5VK_VALIDATION=1   the API's validation: the Khronos layer, or the Direct3D debug layer; refuses to start
//                     without it, and any error it reports fails the run.
#pragma once
#include <memory>

#include "exec.h"

std::unique_ptr<exec::Device> makeDevice();
// Error-severity validation / debug-layer messages over every device made so far, destroyed ones included.
uint32_t deviceValidationErrors();
