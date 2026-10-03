// The execution surface the NR model, kernels and graph are written against. One network, one graph and one set
// of kernel generators; an adapter per API owns allocation and execution:
//
//   vk::Context          Vulkan: compute pipelines + VK_NV_cuda_kernel_launch                (src/vk_context.h)
//   d3d::D3D12Device     Direct3D 12: HLSL compute + NvAPI CuModule / LaunchCuKernelChain    (src/d3d/d3d12_device.h)
//   d3d::D3D11Device     Direct3D 11: HLSL compute + NvAPI cubin compute shaders             (src/d3d/d3d11_device.h)
//
// Nothing here names an API. The PTX kernels are the same text everywhere: they take GPU virtual addresses, which
// each adapter gets its own way (a buffer device address, a D3D12 GPU virtual address, the driver's address of a
// D3D11 resource), and they are launched into the caller's command stream, in order with its other commands.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace exec {

using Address = uint64_t;   // a GPU virtual address a PTX kernel dereferences
using Size = uint64_t;

// Opaque objects, owned by the adapter that made them.
using Shader = void*;           // a compute kernel's code: a SPIR-V module, or HLSL compiled per specialization
using PipelineHandle = void*;   // that code specialized
using KernelModule = void*;     // a PTX module
using Kernel = void*;           // an entry point of a PTX module
using Timer = void*;            // a set of GPU timestamps

// Where commands are recorded: a VkCommandBuffer, an ID3D12GraphicsCommandList or an immediate ID3D11DeviceContext
// (on which "recording" is execution in order). Converts from the API's own pointer, so callers pass what they have.
struct Commands {
  void* stream = nullptr;
  Commands() = default;
  template <typename T>
  Commands(T* native) : stream(native) {}
  explicit operator bool() const { return stream != nullptr; }
};

struct Buffer {
  void* buffer = nullptr;   // VkBuffer | ID3D12Resource* | ID3D11Buffer*
  void* memory = nullptr;   // the adapter's own state for it (a VkDeviceMemory, a driver handle and views...)
  Size size = 0;
  bool hostVisible = false;
  void* mapped = nullptr;   // host-visible buffers only, and only where the adapter can map what a kernel writes
  const char* label = "";
  Size allocation = 0;      // bytes of the allocation behind it (what the accounting counts)
};

constexpr uint32_t kGenericBindings = 12;
constexpr uint32_t kPushConstantBytes = 128;

// Per-pipeline constants: SPIR-V specialization constants on Vulkan, SPEC_<id> macros of the HLSL twin on Direct3D.
struct SpecConstants {
  struct Entry { uint32_t id, offset; };
  std::vector<Entry> entries;
  std::vector<uint32_t> data;
  void add(uint32_t id, uint32_t value) {
    entries.push_back({id, (uint32_t)(data.size() * 4)});
    data.push_back(value);
  }
  void addFloat(uint32_t id, float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    add(id, bits);
  }
};

struct Pipeline {
  PipelineHandle pipeline = nullptr;
  const char* label = "";
};

// The numeric routes the NR kernels run on (independent of the API that executes them). Each has its own
// requirements and its own kernels, and none stands in for another unless asked to.
enum class Backend {
  Native,   // FP8 tensor cores: E4M3 PTX everywhere, plus E4M3 cooperative matrices where the API has them (Vulkan)
  Compat,   // software E4M3 in scalar shader code: no tensor cores, no fusion, no chaining; exact
  Sm86,     // the native PTX kernels lowered to f16 tensor cores (Ampere), the exact scalar kernels elsewhere;
            // close to native, not bit-exact
};
const char* backendName(Backend backend);          // "native", "compat", "sm86"
const char* backendDescription(Backend backend);   // one line for logs

class Device {
 public:
  virtual ~Device() = default;

  virtual const char* api() const = 0;   // "vulkan", "d3d12", "d3d11"
  virtual Backend backend() const = 0;
  // FP8 cooperative matrices and E4M3 conversions in shader code: the cooperative-matrix kernels and their fused
  // variants exist only there. Without them the same arithmetic runs as PTX, or as exact scalar shader code.
  virtual bool nativeFp8() const = 0;
  bool ptxKernels() const { return backend() == Backend::Native || backend() == Backend::Sm86; }
  // Counter chaining reads a status word the kernels write while the frame runs (host-visible memory a kernel can
  // address). An adapter without that runs every launch behind a barrier.
  virtual bool chainSupported() const = 0;
  virtual uint32_t smCount() const = 0;   // streaming multiprocessors (co-residency bound of spinning grids)
  virtual const std::string& deviceName() const = 0;

  // Buffers -------------------------------------------------------------------
  // `usage` is extra API-specific usage the caller needs (Vulkan buffer usage bits); 0 for the kernels' own buffers.
  virtual Buffer createBuffer(Size size, bool hostVisible, const char* label, uint32_t usage = 0) = 0;
  virtual void destroyBuffer(Buffer& buffer) = 0;
  virtual void upload(const Buffer& target, const void* data, Size size, Size offset = 0) = 0;   // waits
  virtual void fillZero(const Buffer& target) = 0;                                               // waits
  virtual std::vector<uint8_t> download(const Buffer& source, Size size, Size offset = 0) = 0;   // waits
  virtual Address deviceAddress(const Buffer& buffer) const = 0;

  // Shader kernels: kGenericBindings storage buffers and up to kPushConstantBytes of constants per dispatch -------
  // `path` is the kernel's SPIR-V file; an adapter that runs another language finds its twin next to it.
  virtual Shader loadShaderModule(const std::string& path) = 0;
  virtual Pipeline createComputePipeline(Shader module, const SpecConstants& constants, const char* label,
                                         uint32_t requiredSubgroupSize = 32) = 0;
  virtual void destroyPipeline(Pipeline& pipeline) = 0;
  // Bind, set the constants and dispatch; no barrier (the caller decides).
  virtual void dispatch(Commands commands, PipelineHandle pipeline, const Buffer* const bindings[kGenericBindings],
                        const void* push, uint32_t pushBytes, uint32_t x, uint32_t y, uint32_t z) = 0;

  // PTX kernels ------------------------------------------------------------------
  virtual KernelModule createCudaModule(const std::string& ptx) = 0;
  virtual Kernel createCudaFunction(KernelModule module, const char* name) = 0;
  virtual void destroyCudaFunction(Kernel function) = 0;
  virtual void destroyCudaModule(KernelModule module) = 0;
  // `params` points at each argument's value, in the entry's declaration order (64-bit addresses and 32-bit words).
  virtual void cudaLaunch(Commands commands, Kernel function, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                          uint32_t blockX, uint32_t sharedBytes, const void* const* params, size_t paramCount) = 0;

  // Ordering and transfers inside a command stream -----------------------------------
  virtual void computeBarrier(Commands commands) = 0;    // kernel -> kernel
  virtual void transferBarrier(Commands commands) = 0;   // kernel / transfer -> kernel / transfer
  virtual void zeroBuffer(Commands commands, const Buffer& target) = 0;
  virtual void copyBuffer(Commands commands, const Buffer& source, const Buffer& target, Size bytes) = 0;

  // A device that owns its queue (the tool, the tests, a bridge) -------------------------
  // One-shot command streams; a device borrowed from a host records into the host's stream instead and throws here.
  virtual Commands beginCommands() = 0;
  virtual void endAndSubmit(Commands commands, bool wait = true) = 0;
  virtual void nextFrame() {}   // transient per-frame state (descriptor sets) may be recycled

  // Bytes of every buffer this device created, device-local and host-visible apart, current and peak.
  struct MemoryUse { Size deviceLocal = 0, hostVisible = 0, peakDeviceLocal = 0; };
  virtual MemoryUse memoryUse() const = 0;

  // Timestamps --------------------------------------------------------------------
  virtual Timer createTimestampPool(uint32_t count) = 0;
  virtual void destroyTimestampPool(Timer pool) = 0;
  virtual void resetTimestamps(Commands commands, Timer pool, uint32_t count) = 0;
  // `first`: the stamp opens a span (taken as early as the stream allows) instead of closing one.
  virtual void writeTimestamp(Commands commands, Timer pool, uint32_t index, bool first) = 0;
  virtual std::vector<double> readTimestampsMs(Timer pool, uint32_t count) = 0;   // after the stream completed
};

}  // namespace exec
