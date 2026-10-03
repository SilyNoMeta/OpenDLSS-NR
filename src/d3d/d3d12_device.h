// The Direct3D 12 adapter of the execution surface in exec.h: the NR graph recorded into an
// ID3D12GraphicsCommandList, on committed D3D12 buffers.
//
//   PTX kernels     NvAPI_D3D12_CreateCuModule (the driver compiles the PTX text) / CreateCuFunction, and one
//                   NvAPI_D3D12_LaunchCuKernelChain entry per launch, in order with the list's other commands.
//                   Arguments are packed in the CUDA ABI; a buffer's kernel address is its GPU virtual address
//                   (tests/d3d/cuda_probe.cpp is the proof that it is one the kernels can dereference).
//   shader kernels  the HLSL twins (shaders/hlsl) as cs_5_0, one root signature for all: twelve root UAVs and 128
//                   bytes of root constants. Root descriptors, so no descriptor heap is bound or changed.
//   ordering        a UAV barrier where the Vulkan adapter has a compute barrier.
//
// Every buffer stays in the UNORDERED_ACCESS state, the one the kernels and the dispatches use; transfers move it
// to a copy state and back inside the stream that copies. Recording sets the list's compute root signature and
// pipeline state and leaves them set, as any pass does: a host that draws after it restores its own.
#pragma once
#ifdef _WIN32
#include <d3d12.h>
#include <wrl/client.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "d3d_common.h"

namespace d3d {

class D3D12Device final : public exec::Device {
 public:
  // Its own device on the NVIDIA adapter, with its own queue (the tool, the tests, the comparison harness).
  // `debugLayer` enables the D3D12 debug layer when it is installed.
  D3D12Device(exec::Backend backend, bool debugLayer = false);
  // A host's device: the host's command lists receive the graph. Uploads at preparation go through a queue of this
  // object's own on that device; beginCommands / endAndSubmit use it too (tests), never the host's queue.
  D3D12Device(ID3D12Device* device, exec::Backend backend);
  ~D3D12Device() override;

  const char* api() const override { return "d3d12"; }
  exec::Backend backend() const override { return backend_; }
  bool nativeFp8() const override { return false; }        // no FP8 cooperative matrices in HLSL: PTX or exact scalar code
  bool chainSupported() const override { return false; }   // no host-visible memory a kernel can write
  uint32_t smCount() const override { return gpu_.smCount; }
  const std::string& deviceName() const override { return gpu_.name; }
  const GpuInfo& gpu() const { return gpu_; }
  ID3D12Device* device() const { return device_.Get(); }
  ID3D12CommandQueue* queue() const { return queue_.Get(); }
  uint32_t debugErrors() const;   // error-severity messages of the debug layer so far (0 without it)
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

  void computeBarrier(exec::Commands commands) override;
  void transferBarrier(exec::Commands commands) override;
  void zeroBuffer(exec::Commands commands, const exec::Buffer& target) override;
  void copyBuffer(exec::Commands commands, const exec::Buffer& source, const exec::Buffer& target, exec::Size bytes) override;

  exec::Commands beginCommands() override;   // an ID3D12GraphicsCommandList on this object's own queue
  void endAndSubmit(exec::Commands commands, bool wait = true) override;
  MemoryUse memoryUse() const override { return memoryUse_; }

  exec::Timer createTimestampPool(uint32_t count) override;
  void destroyTimestampPool(exec::Timer pool) override;
  void resetTimestamps(exec::Commands commands, exec::Timer pool, uint32_t count) override;
  void writeTimestamp(exec::Commands commands, exec::Timer pool, uint32_t index, bool first) override;
  std::vector<double> readTimestampsMs(exec::Timer pool, uint32_t count) override;

  // Launch counters since creation (recording, not GPU completion).
  uint64_t recordedLaunches() const { return launches_; }
  uint64_t recordedDispatches() const { return dispatches_; }

 private:
  struct Stream;   // an allocator and its list on this object's queue
  void init(bool debugLayer);
  void waitFence(uint64_t value);
  void collect();
  Microsoft::WRL::ComPtr<ID3D12Resource> allocate(exec::Size size, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags,
                                                  D3D12_RESOURCE_STATES state);
  const exec::Buffer& zeros(exec::Size size);

  const nvcuda::Api* nv_ = nullptr;
  Microsoft::WRL::ComPtr<ID3D12Device> device_;
  Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
  Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
  HANDLE fenceEvent_ = nullptr;
  uint64_t fenceValue_ = 0;
  Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature_;
  Microsoft::WRL::ComPtr<ID3D12Resource> uploadStaging_, readbackStaging_;
  void* uploadMapped_ = nullptr;
  std::vector<std::unique_ptr<Stream>> streams_;
  exec::Buffer dummy_, zeros_;
  exec::Backend backend_;
  GpuInfo gpu_;
  MemoryUse memoryUse_;
  bool debugLayer_ = false;
  uint64_t launches_ = 0, dispatches_ = 0;
};

}  // namespace d3d
#endif  // _WIN32
