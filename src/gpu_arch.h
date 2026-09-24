// The CUDA compute capability of a Vulkan physical device, read from the NVIDIA driver's own CUDA library
// (nvcuda.dll / libcuda.so.1, part of every NVIDIA driver; no CUDA toolkit involved) and matched to the Vulkan
// device by UUID. Vulkan has no portable way to say "this GPU has FP8 tensor cores": the backend selection uses
// this to tell an Ampere GPU (no FP8 hardware) from an Ada or newer GPU whose driver merely lacks an extension.
#pragma once
#include <volk.h>

#include <cstdint>
#include <optional>
#include <string>

namespace vk {

struct ComputeCapability {
  int major = 0, minor = 0;
  bool atLeast(int ma, int mi) const { return major > ma || (major == ma && minor >= mi); }
  std::string text() const { return std::to_string(major) + "." + std::to_string(minor); }
};

// Empty when the device is not NVIDIA, the driver library cannot be loaded, or no CUDA device has its UUID.
std::optional<ComputeCapability> computeCapability(VkPhysicalDevice physical);

}  // namespace vk
