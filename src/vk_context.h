// Minimal Vulkan compute host for the DLSS-NR port: one device, one queue,
// storage buffers, compute pipelines with specialization constants, a generic
// twelve-binding descriptor layout and GPU timestamps. No graphics, no windows.
// The Vulkan adapter of the execution surface in exec.h.
#pragma once
#include <volk.h>

#include "exec.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#define VK_CHECK(expr)                                                                 \
  do {                                                                                 \
    VkResult vk_check_result_ = (expr);                                                \
    if (vk_check_result_ != VK_SUCCESS)                                                \
      throw std::runtime_error(std::string(#expr) + " failed: " +                      \
                               std::to_string((int)vk_check_result_));                 \
  } while (0)

namespace vk {

// The neutral types of exec.h under their Vulkan names. A Buffer's `buffer` / `memory` hold the VkBuffer and
// VkDeviceMemory, a Pipeline's `pipeline` the VkPipeline (handles are pointers on every 64-bit target).
using Buffer = exec::Buffer;
using SpecConstants = exec::SpecConstants;
using Pipeline = exec::Pipeline;
using Backend = exec::Backend;
using exec::backendDescription;
using exec::backendName;
using exec::kGenericBindings;
using exec::kPushConstantBytes;
static_assert(sizeof(VkBuffer) == sizeof(void*) && sizeof(VkPipeline) == sizeof(void*), "64-bit Vulkan handles");
inline VkBuffer handle(const Buffer& buffer) { return static_cast<VkBuffer>(buffer.buffer); }
inline VkDeviceMemory memoryOf(const Buffer& buffer) { return static_cast<VkDeviceMemory>(buffer.memory); }
inline VkPipeline handle(const Pipeline& pipeline) { return static_cast<VkPipeline>(pipeline.pipeline); }
inline VkCommandBuffer handle(exec::Commands commands) { return static_cast<VkCommandBuffer>(commands.stream); }

// DLSS5VK_BACKEND=auto (the default) takes the fastest route the device supports and says which and why; an
// explicit name is that route or an error.
std::optional<Backend> requestedBackend();          // DLSS5VK_BACKEND, empty for auto; throws on an unknown name

// The device features / extensions a backend needs, as a stable pNext chain (the demo hands it to the renderer's
// device creation; Context::Context() uses it for its own device). With enable = false the chain is the same with
// nothing set, ready for vkGetPhysicalDeviceFeatures2. Not copyable: the chain points into the object.
// Only what the kernels use is required: the native SPIR-V declares subgroup-scope matrices of flexible dimensions,
// conversions, per-element operations and tensor addressing (no workgroup scope, reductions or block loads), and
// both PTX backends launch through VK_NV_cuda_kernel_launch, whose feature bit they enable. Diagnostics are never
// a requirement: enableStatistics() adds VK_KHR_pipeline_executable_properties where the device has it.
struct DeviceRequirements {
  Backend backend;
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceCooperativeMatrix2FeaturesNV coop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV};
  VkPhysicalDeviceShaderFloat8FeaturesEXT fp8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR executable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};   // optional: enableStatistics()
  VkPhysicalDeviceShaderSMBuiltinsFeaturesNV sm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV};
  VkPhysicalDeviceCudaKernelLaunchFeaturesNV cuda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV};
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  std::vector<const char*> extensions;
  explicit DeviceRequirements(Backend backend, bool enable = true);
  DeviceRequirements(const DeviceRequirements&) = delete;
  DeviceRequirements& operator=(const DeviceRequirements&) = delete;
  void* pNextChain() { return features.pNext; }   // for a VkDeviceCreateInfo that carries VkPhysicalDeviceFeatures itself
  // Optional pipeline statistics (`dlss5vk stats`): links the feature and adds the extension. Only for a device
  // that supports both (statisticsSupported); no kernel depends on it.
  void enableStatistics();
  static bool statisticsSupported(VkPhysicalDevice physical);
};

// The physical device and backend for the NR kernels. Each device is checked against every candidate backend's
// extensions and features (plus `extraExtensions`, e.g. a swapchain); the fastest usable backend wins, then a
// discrete GPU, then NVIDIA, whatever the enumeration order (a hybrid laptop lists its integrated GPU too, often
// first when Windows prefers power saving). DLSS5VK_DEVICE=<index or part of the name> restricts the devices,
// DLSS5VK_BACKEND the backends. When none qualifies, the error names what each device lacks for each backend.
// auto refuses to fall back from the native route on a GPU with FP8 tensor cores (a driver problem).
struct DeviceChoice {
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  Backend backend = Backend::Native;
  std::string reason;   // why this backend
  std::string report;   // one line per device
};
DeviceChoice selectDevice(VkInstance instance, const std::vector<const char*>& extraExtensions = {});

// A device created elsewhere (a renderer, a game integration) for the NR kernels to adopt. `createInfo` is what it
// was created with: the backend's extensions and features must have been enabled there (a DeviceRequirements chain
// in its pNext does that), which Context checks instead of trusting what the GPU merely supports.
struct BorrowedDevice {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t queueFamily = 0;
  uint32_t queueIndex = 0;
  Backend backend = Backend::Native;
  const VkDeviceCreateInfo* createInfo = nullptr;   // only read during the constructor
};

// A device of the Context's own for a host that renders with another API (a Direct3D bridge). The GPU is the host's,
// named by its adapter LUID and never chosen by score; the backend is the one asked for, or an error that says what
// the device lacks; and the device is created with the extensions and features the sharing needs.
struct DedicatedDevice {
  uint64_t luid = 0;
  Backend backend = Backend::Native;
  std::vector<const char*> extensions;   // on top of the backend's (external memory, external semaphores...)
  bool timelineSemaphore = false;
};

class Context : public exec::Device {
 public:
  Context();   // its own instance and device, on selectDevice's choice
  explicit Context(const DedicatedDevice& dedicated);   // its own instance and device, on the GPU with that LUID
  // Adopt a device created elsewhere; the instance / device are not destroyed by this object.
  explicit Context(const BorrowedDevice& borrowed);
  ~Context() override;
  const char* api() const override { return "vulkan"; }
  bool chainSupported() const override { return true; }
  uint32_t queueFamily() const { return queueFamily_; }
  uint32_t queueIndex() const { return queueIndex_; }
  VkInstance instance() const { return instance_; }
  // Instance diagnostics of the tool's own instance. DLSS5VK_VALIDATION=1 enables VK_LAYER_KHRONOS_validation and
  // refuses to start without it; DLSS5VK_DEBUG=1 only installs a debug messenger for the driver's own messages
  // (PTX compiler diagnostics among them), which checks nothing.
  bool validationEnabled() const { return validation_; }
  static uint32_t validationErrors();   // error-severity validation messages so far, over every Context

  VkDevice device() const { return device_; }

  uint32_t smCount() const override { return smCount_; }
  VkQueue queue() const { return queue_; }
  VkCommandPool commandPool() const { return commandPool_; }   // the pool beginCommands allocates from
  VkPhysicalDevice physical() const { return physical_; }
  float timestampPeriodNs() const { return timestampPeriod_; }
  VkPipelineLayout pipelineLayout() const { return pipelineLayout_; }
  VkDescriptorSetLayout setLayout() const { return setLayout_; }
  const std::string& deviceName() const override { return deviceName_; }
  Backend backend() const override { return backend_; }
  bool nativeFp8() const override { return backend_ == Backend::Native && shaderFp8_; }   // FP8 cooperative matrices and conversions in GLSL
  // Run the native backend without its cooperative-matrix GLSL: FP8 PTX plus the exact scalar kernels, which is the
  // kernel set the Direct3D adapters run. For comparing APIs on identical kernels; set before the Kernels are made.
  void setShaderFp8(bool enabled) { shaderFp8_ = enabled; }
  uint32_t subgroupSize() const { return subgroupSize_; }

  // Buffers -----------------------------------------------------------------
  Buffer createBuffer(exec::Size size, bool hostVisible, const char* label, uint32_t extra = 0) override;   // extra: VkBufferUsageFlags
  void destroyBuffer(Buffer& buffer) override;
  // Upload through a staging buffer and wait for completion.
  void upload(const Buffer& target, const void* data, exec::Size size, exec::Size offset = 0) override;
  void fillZero(const Buffer& target) override;
  // Download via staging buffer and wait for completion.
  std::vector<uint8_t> download(const Buffer& source, exec::Size size, exec::Size offset = 0) override;
  const Buffer& dummyBuffer() const { return dummy_; }

  // Pipelines -----------------------------------------------------------------
  exec::Shader loadShaderModule(const std::string& spvPath) override;   // a VkShaderModule
  Pipeline createComputePipeline(exec::Shader module, const SpecConstants& constants,
                                 const char* label, uint32_t requiredSubgroupSize = 32) override;
  void destroyPipeline(Pipeline& pipeline) override;
  void dispatch(exec::Commands commands, exec::PipelineHandle pipeline, const Buffer* const bindings[kGenericBindings],
                const void* push, uint32_t pushBytes, uint32_t x, uint32_t y, uint32_t z) override;
  // VK_KHR_pipeline_executable_properties: register/spill statistics (and SASS when available).
  void setCaptureStatistics(bool enabled) { captureStatistics_ = enabled; }
  std::string pipelineStatistics(const Pipeline& pipeline, bool includeInternal = false);

  // Descriptors: one generic layout with kGenericBindings storage buffers.
  VkDescriptorSet allocateSet(const Buffer* const bindings[kGenericBindings],
                              const VkDeviceSize offsets[kGenericBindings] = nullptr,
                              const VkDeviceSize ranges[kGenericBindings] = nullptr);
  void resetDescriptorPool();              // advance to the next pool and reset it
  void resetDescriptorPool(uint32_t slot); // reset and use a specific pool (the caller's frame-in-flight slot)

  // Commands --------------------------------------------------------------------
  exec::Commands beginCommands() override;   // a VkCommandBuffer
  void endAndSubmit(exec::Commands commands, bool wait = true) override;
  void nextFrame() override { resetDescriptorPool(); }
  void waitIdle() { VK_CHECK(vkQueueWaitIdle(queue_)); }
  void computeBarrier(exec::Commands commands) override;   // compute -> compute
  // VK_NV_cuda_kernel_launch: PTX modules launched from the command buffer on buffer device addresses.
  exec::Address deviceAddress(const Buffer& buffer) const override;
  exec::KernelModule createCudaModule(const std::string& ptx) override;                        // a VkCudaModuleNV
  exec::Kernel createCudaFunction(exec::KernelModule module, const char* name) override;       // a VkCudaFunctionNV
  void destroyCudaFunction(exec::Kernel function) override;
  void destroyCudaModule(exec::KernelModule module) override;
  void cudaLaunch(exec::Commands commands, exec::Kernel function, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                  uint32_t blockX, uint32_t sharedBytes, const void* const* params, size_t paramCount) override;
  void transferBarrier(exec::Commands commands) override;  // compute/transfer -> compute/transfer (captures, uploads)
  void zeroBuffer(exec::Commands commands, const Buffer& target) override;
  void copyBuffer(exec::Commands commands, const Buffer& source, const Buffer& target, exec::Size bytes) override;

  // Timestamps -----------------------------------------------------------------
  exec::Timer createTimestampPool(uint32_t count) override;                                    // a VkQueryPool
  void destroyTimestampPool(exec::Timer pool) override;
  void resetTimestamps(exec::Commands commands, exec::Timer pool, uint32_t count) override;
  void writeTimestamp(exec::Commands commands, exec::Timer pool, uint32_t index, bool first) override;
  std::vector<double> readTimestampsMs(exec::Timer pool, uint32_t count) override;

  uint32_t maxComputeSharedMemory() const { return maxSharedMemory_; }

  // Memory accounting of every buffer this Context created (device-local and host-visible apart), current and peak.
  MemoryUse memoryUse() const override { return memoryUse_; }

 private:
  uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required);
  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  bool validation_ = false;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t queueFamily_ = 0;
  uint32_t queueIndex_ = 0;
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
  std::array<VkDescriptorPool, 2> descriptorPools_{};
  size_t descriptorPoolIndex_ = 0;
  VkPhysicalDeviceMemoryProperties memoryProperties_{};
  float timestampPeriod_ = 1.0f;
  uint32_t maxSharedMemory_ = 0;
  uint32_t smCount_ = 0;   // streaming multiprocessors (co-residency bound of spinning grids)
  bool captureStatistics_ = false;
  uint32_t subgroupSize_ = 0;
  Backend backend_ = Backend::Native;
  bool shaderFp8_ = true;
  bool executableProperties_ = false;   // VK_KHR_pipeline_executable_properties enabled (statistics)
  MemoryUse memoryUse_;
  bool owned_ = true;
  std::string deviceName_;
  void createOwned(const DedicatedDevice* dedicated);
  void readDeviceProperties();   // SMs, subgroup, limits, memory types
  void initCommon();   // command pool, layouts, pools, staging
  Buffer dummy_;
  Buffer staging_;
  std::vector<VkShaderModule> modules_;
};

}  // namespace vk
