// The Direct3D 11 adapter of the execution surface in exec.h: the NR graph issued on an immediate
// ID3D11DeviceContext, on D3D11 raw buffers. No Direct3D 12 and no Vulkan device is involved.
//
//   PTX kernels     NvAPI_D3D11_CreateCubinComputeShaderEx (the driver compiles the PTX text) and
//                   NvAPI_D3D11_LaunchCubinShader. The driver object fixes the block dimensions and the dynamic
//                   shared memory, so an entry gets one per (block, shared size) it is launched with. A launch
//                   names the resources it reads and writes by driver handle (from the entry's argument list and
//                   the address each pointer argument falls in); the kernels address them by the GPU virtual
//                   address NvAPI_D3D11_GetResourceGPUVirtualAddress reports.
//   shader kernels  the HLSL twins (shaders/hlsl) as cs_5_0: the bindings as raw UAVs u0..u7 (Direct3D 11.0 has
//                   eight UAV slots; no kernel twin binds beyond u7), the push constants as one constant buffer.
//   ordering        the immediate context executes in order and the driver tracks the declared resources, so the
//                   barriers are no-ops. "Recording" a graph here is issuing it: there is nothing to replay.
//
// Only an immediate context: a deferred context is refused at construction (nothing is executed elsewhere on its
// behalf). The adapter leaves its compute shader, constant buffer and UAV slots unbound after each dispatch; a
// host isolates the rest of its pipeline state around the graph (ID3DDeviceContextState).
#pragma once
#ifdef _WIN32
#include <d3d11_4.h>
#include <wrl/client.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "d3d_common.h"

namespace d3d {

class D3D11Device final : public exec::Device {
 public:
  // Its own device on the NVIDIA adapter (the tool, the tests, the comparison harness).
  D3D11Device(exec::Backend backend, bool debugLayer = false);
  // A host's device and its immediate context. The host destroys this object, and the kernels and buffers made on
  // it, only once the GPU has run everything it issued with them (an event query it polls): nothing here waits for
  // the host's context.
  D3D11Device(ID3D11Device* device, ID3D11DeviceContext* immediate, exec::Backend backend);
  ~D3D11Device() override;

  const char* api() const override { return "d3d11"; }
  exec::Backend backend() const override { return backend_; }
  bool nativeFp8() const override { return false; }
  bool chainSupported() const override { return false; }
  uint32_t smCount() const override { return gpu_.smCount; }
  const std::string& deviceName() const override { return gpu_.name; }
  const GpuInfo& gpu() const { return gpu_; }
  ID3D11Device* device() const { return device_.Get(); }
  ID3D11DeviceContext* context() const { return context_.Get(); }
  uint32_t debugErrors() const;
  bool debugLayer() const { return debugLayer_; }

  exec::Buffer createBuffer(exec::Size size, bool hostVisible, const char* label, uint32_t usage = 0) override;
  void destroyBuffer(exec::Buffer& buffer) override;
  void upload(const exec::Buffer& target, const void* data, exec::Size size, exec::Size offset = 0) override;
  void fillZero(const exec::Buffer& target) override;
  std::vector<uint8_t> download(const exec::Buffer& source, exec::Size size, exec::Size offset = 0) override;
  exec::Address deviceAddress(const exec::Buffer& buffer) const override;

  exec::Shader loadShaderModule(const std::string& path) override;
  exec::Pipeline createComputePipeline(exec::Shader module, const exec::SpecConstants& constants, const char* label,
                                       uint32_t requiredSubgroupSize = 32) override;
  void destroyPipeline(exec::Pipeline& pipeline) override;
  void dispatch(exec::Commands commands, exec::PipelineHandle pipeline, const exec::Buffer* const bindings[exec::kGenericBindings],
                const void* push, uint32_t pushBytes, uint32_t x, uint32_t y, uint32_t z) override;

  exec::KernelModule createCudaModule(const std::string& ptx) override;
  exec::Kernel createCudaFunction(exec::KernelModule module, const char* name) override;
  void destroyCudaFunction(exec::Kernel function) override;
  void destroyCudaModule(exec::KernelModule module) override;
  void cudaLaunch(exec::Commands commands, exec::Kernel function, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                  uint32_t blockX, uint32_t sharedBytes, const void* const* params, size_t paramCount) override;

  void computeBarrier(exec::Commands) override {}
  void transferBarrier(exec::Commands) override {}
  void zeroBuffer(exec::Commands commands, const exec::Buffer& target) override;
  void copyBuffer(exec::Commands commands, const exec::Buffer& source, const exec::Buffer& target, exec::Size bytes) override;

  exec::Commands beginCommands() override;   // the immediate context
  void endAndSubmit(exec::Commands commands, bool wait = true) override;   // flush; `wait`: until the GPU has run it
  MemoryUse memoryUse() const override { return memoryUse_; }

  exec::Timer createTimestampPool(uint32_t count) override;
  void destroyTimestampPool(exec::Timer pool) override;
  void resetTimestamps(exec::Commands commands, exec::Timer pool, uint32_t count) override;
  void writeTimestamp(exec::Commands commands, exec::Timer pool, uint32_t index, bool first) override;
  std::vector<double> readTimestampsMs(exec::Timer pool, uint32_t count) override;

  uint64_t recordedLaunches() const { return launches_; }
  uint64_t recordedDispatches() const { return dispatches_; }

 private:
  void init();
  void waitForGpu();
  ID3D11DeviceContext* contextOf(exec::Commands commands) const;
  const exec::Buffer& zeros(exec::Size size);

  const nvcuda::Api* nv_ = nullptr;
  Microsoft::WRL::ComPtr<ID3D11Device> device_;
  Microsoft::WRL::ComPtr<ID3D11DeviceContext> context_;
  Microsoft::WRL::ComPtr<ID3D11Buffer> constants_;
  AddressMap<nvcuda::Handle> addresses_;
  exec::Buffer zeros_;
  exec::Backend backend_;
  GpuInfo gpu_;
  MemoryUse memoryUse_;
  bool debugLayer_ = false;
  bool borrowed_ = false;
  uint64_t launches_ = 0, dispatches_ = 0;
};

}  // namespace d3d
#endif  // _WIN32
