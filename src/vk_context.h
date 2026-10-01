// Minimal Vulkan compute host for the DLSS-NR port: one device, one queue,
// storage buffers, compute pipelines with specialization constants, a generic
// eight-binding descriptor layout and GPU timestamps. No graphics, no windows.
#pragma once
#include <volk.h>

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

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  bool hostVisible = false;
  void* mapped = nullptr;
  const char* label = "";
};

constexpr uint32_t kGenericBindings = 12;
constexpr uint32_t kPushConstantBytes = 128;

struct SpecConstants {
  std::vector<VkSpecializationMapEntry> entries;
  std::vector<uint32_t> data;
  void add(uint32_t id, uint32_t value) {
    entries.push_back({id, (uint32_t)(data.size() * 4), 4});
    data.push_back(value);
  }
  void addFloat(uint32_t id, float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    add(id, bits);
  }
};

struct Pipeline {
  VkPipeline pipeline = VK_NULL_HANDLE;
  const char* label = "";
};

// The routes the NR kernels run on. Each has its own device requirements and its own kernels, and none stands in for
// another unless asked to: DLSS5VK_BACKEND=auto (the default) takes the fastest route the device supports and says
// which and why; an explicit name is that route or an error.
enum class Backend {
  Native,   // E4M3 cooperative matrices + FP8 PTX, fused blocks, counter chaining (Ada, Hopper, Blackwell)
  Compat,   // software E4M3 in scalar GLSL: any NVIDIA Vulkan 1.3 GPU, no tensor cores, no fusion, no chaining; exact
  Sm86,     // the native PTX kernels lowered to f16 tensor cores (Ampere, compute capability 8.x) through
            // VK_NV_cuda_kernel_launch, the compatibility GLSL elsewhere; close to native, not bit-exact
};
const char* backendName(Backend backend);          // "native", "compat", "sm86"
const char* backendDescription(Backend backend);   // one line for logs
std::optional<Backend> requestedBackend();          // DLSS5VK_BACKEND, empty for auto; throws on an unknown name

// The device features / extensions a backend needs, as a stable pNext chain (the demo hands it to the renderer's
// device creation; Context::Context() uses it for its own device). With enable = false the chain is the same with
// nothing set, ready for vkGetPhysicalDeviceFeatures2. Not copyable: the chain points into the object.
struct DeviceRequirements {
  Backend backend;
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceCooperativeMatrix2FeaturesNV coop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV};
  VkPhysicalDeviceShaderFloat8FeaturesEXT fp8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR executable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  VkPhysicalDeviceShaderClockFeaturesKHR clock{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  VkPhysicalDeviceShaderSMBuiltinsFeaturesNV sm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV};
  VkPhysicalDeviceCudaKernelLaunchFeaturesNV cuda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV};
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  std::vector<const char*> extensions;
  explicit DeviceRequirements(Backend backend, bool enable = true);
  DeviceRequirements(const DeviceRequirements&) = delete;
  DeviceRequirements& operator=(const DeviceRequirements&) = delete;
  void* pNextChain() { return features.pNext; }   // for a VkDeviceCreateInfo that carries VkPhysicalDeviceFeatures itself
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

class Context {
 public:
  Context();   // its own instance and device, on selectDevice's choice
  // Adopt a device created elsewhere; the instance / device are not destroyed by this object.
  explicit Context(const BorrowedDevice& borrowed);
  ~Context();
  uint32_t queueFamily() const { return queueFamily_; }
  uint32_t queueIndex() const { return queueIndex_; }
  VkInstance instance() const { return instance_; }
  // Instance diagnostics of the tool's own instance. DLSS5VK_VALIDATION=1 enables VK_LAYER_KHRONOS_validation and
  // refuses to start without it; DLSS5VK_DEBUG=1 only installs a debug messenger for the driver's own messages
  // (PTX compiler diagnostics among them), which checks nothing.
  bool validationEnabled() const { return validation_; }
  static uint32_t validationErrors();   // error-severity validation messages so far, over every Context

  VkDevice device() const { return device_; }

  uint32_t smCount() const { return smCount_; }
  VkQueue queue() const { return queue_; }
  VkPhysicalDevice physical() const { return physical_; }
  float timestampPeriodNs() const { return timestampPeriod_; }
  VkPipelineLayout pipelineLayout() const { return pipelineLayout_; }
  VkDescriptorSetLayout setLayout() const { return setLayout_; }
  const std::string& deviceName() const { return deviceName_; }
  Backend backend() const { return backend_; }
  bool nativeFp8() const { return backend_ == Backend::Native; }   // FP8 cooperative matrices and conversions in GLSL
  bool ptxKernels() const { return backend_ == Backend::Native || backend_ == Backend::Sm86; }
  uint32_t subgroupSize() const { return subgroupSize_; }

  // Buffers -----------------------------------------------------------------
  Buffer createBuffer(VkDeviceSize size, bool hostVisible, const char* label,
                      VkBufferUsageFlags extra = 0);
  void destroyBuffer(Buffer& buffer);
  // Upload through a staging buffer and wait for completion.
  void upload(const Buffer& target, const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
  void fillZero(const Buffer& target);
  // Download via staging buffer and wait for completion.
  std::vector<uint8_t> download(const Buffer& source, VkDeviceSize size, VkDeviceSize offset = 0);
  const Buffer& dummyBuffer() const { return dummy_; }

  // Pipelines -----------------------------------------------------------------
  VkShaderModule loadShaderModule(const std::string& spvPath);
  Pipeline createComputePipeline(VkShaderModule module, const SpecConstants& constants,
                                 const char* label, uint32_t requiredSubgroupSize = 32);
  void destroyPipeline(Pipeline& pipeline);
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
  VkCommandBuffer beginCommands();
  void endAndSubmit(VkCommandBuffer commands, bool wait = true);
  void waitIdle() { VK_CHECK(vkQueueWaitIdle(queue_)); }
  void computeBarrier(VkCommandBuffer commands);   // compute -> compute
  // VK_NV_cuda_kernel_launch: PTX modules launched from the command buffer on buffer device addresses.
  VkDeviceAddress deviceAddress(const Buffer& buffer) const;
  VkCudaModuleNV createCudaModule(const std::string& ptx);
  VkCudaFunctionNV createCudaFunction(VkCudaModuleNV module, const char* name);
  void destroyCudaFunction(VkCudaFunctionNV function);
  void destroyCudaModule(VkCudaModuleNV module);
  void cudaLaunch(VkCommandBuffer commands, VkCudaFunctionNV function, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                  uint32_t blockX, uint32_t sharedBytes, const void* const* params, size_t paramCount);
  void transferBarrier(VkCommandBuffer commands);  // compute/transfer -> compute/transfer (captures, uploads)

  // Timestamps -----------------------------------------------------------------
  VkQueryPool createTimestampPool(uint32_t count);
  std::vector<double> readTimestampsMs(VkQueryPool pool, uint32_t count);

  uint32_t maxComputeSharedMemory() const { return maxSharedMemory_; }

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
  bool executableProperties_ = false;   // VK_KHR_pipeline_executable_properties enabled (statistics)
  bool owned_ = true;
  std::string deviceName_;
  void readDeviceProperties();   // SMs, subgroup, limits, memory types
  void initCommon();   // command pool, layouts, pools, staging
  Buffer dummy_;
  Buffer staging_;
  std::vector<VkShaderModule> modules_;
};

}  // namespace vk
