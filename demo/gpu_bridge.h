// The handles that cross between the demo's two Vulkan worlds without either one's headers: the NR side
// (volk + the repository's Vulkan headers: device creation, the NR pass) and the Filament side (bluevk +
// Filament's bundled headers: the shared context, the interop callbacks). Vulkan handles are pointers or 64-bit
// integers in both, so they travel as such.
#pragma once
#include <cstdint>

struct GpuDevice {
  void* instance = nullptr;        // VkInstance
  void* physicalDevice = nullptr;  // VkPhysicalDevice
  void* device = nullptr;          // VkDevice
  uint32_t queueFamily = 0;        // the graphics + compute family
  uint32_t rendererQueueIndex = 0; // the queue Filament submits on
  uint32_t nrQueueIndex = 0;       // the queue the NR side submits its own (rare, synchronous) work on
  bool debugUtils = false;
  uint32_t backend = 0;            // vk::Backend the device was created for
  const void* createInfo = nullptr;  // the VkDeviceCreateInfo it was created with (features and extensions enabled)
};

// A Filament texture as the backend holds it: image (VkImage), format (VkFormat), layout (VkImageLayout)
struct GpuImage {
  uint64_t image = 0;
  uint32_t format = 0;
  uint32_t layout = 0;
  uint32_t width = 0, height = 0;
};
