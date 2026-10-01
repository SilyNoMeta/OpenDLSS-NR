// The demo's Vulkan instance and device (volk): the NR kernels' features and extensions (vk::DeviceRequirements)
// plus what a renderer needs (a swapchain, every supported core feature, two graphics queues when the family has
// them). Filament adopts the result through its shared-context path and the NR side through vk::Context.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "gpu_bridge.h"
#include "vk_context.h"

class VulkanDevice {
 public:
  VulkanDevice();
  ~VulkanDevice();
  const GpuDevice& handles() const { return handles_; }
  const std::string& deviceName() const { return deviceName_; }
  vk::Backend backend() const { return backend_; }
  static uint32_t validationErrors();

 private:
  GpuDevice handles_;
  std::string deviceName_;
  vk::Backend backend_ = vk::Backend::Native;
  // What the device was created with, kept alive for the NR side's check (vk::BorrowedDevice::createInfo)
  std::unique_ptr<vk::DeviceRequirements> requirements_;
  std::vector<const char*> enabledExtensions_;
  VkDeviceCreateInfo createInfo_{};
  uint64_t messenger_ = 0;   // VkDebugUtilsMessengerEXT, with DLSS5_DEMO_VALIDATION=1
};
