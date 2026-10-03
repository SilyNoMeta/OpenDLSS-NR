// NVIDIA's Direct3D CUDA interop, resolved at run time from the driver's nvapi64.dll (no import library, no SDK
// header): the D3D12 module / function / launch-chain interfaces and the D3D11 cubin compute-shader interfaces
// that put a CUDA kernel into a Direct3D command list or immediate context, on Direct3D resources.
//
// Interface ids and signatures are those of the public NVAPI SDK (github.com/NVIDIA/nvapi, nvapi_interface.h and
// nvapi.h, MIT), where every one of these is marked "Experimental API for internal use". An id that resolves and
// a creation that succeeds prove nothing about execution: tests/d3d/cuda_probe.cpp is the proof, per API and GPU.
#pragma once
#ifdef _WIN32
#include <windows.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11Resource;
struct ID3D12Device;
struct ID3D12GraphicsCommandList;

namespace nvcuda {

using Status = int;   // NvAPI_Status: 0 is NVAPI_OK
using Handle = void*; // NVDX_ObjectHandle (a pointer-sized opaque driver handle)

struct Dim3 { uint32_t x, y, z; };

// NVAPI_CU_KERNEL_LAUNCH_PARAMS. `params` is one packed buffer in the CUDA ABI (every argument at its natural
// alignment), not an array of pointers as VK_NV_cuda_kernel_launch takes.
struct KernelLaunch {
  Handle function;
  Dim3 grid;
  Dim3 block;
  uint32_t dynamicSharedBytes;
  const void* params;
  uint32_t paramBytes;
};
static_assert(sizeof(KernelLaunch) == 56, "NVAPI_CU_KERNEL_LAUNCH_PARAMS layout");

// NV_GET_GPU_VIRTUAL_ADDRESS_V1
struct D3D11GpuVirtualAddress {
  uint32_t version;
  Handle resource;
  uint64_t start;
  uint64_t size;
};
static_assert(sizeof(D3D11GpuVirtualAddress) == 32, "NV_GET_GPU_VIRTUAL_ADDRESS_V1 layout");
constexpr uint32_t kD3D11GpuVirtualAddressVersion = (uint32_t)sizeof(D3D11GpuVirtualAddress) | (1u << 16);

struct Api {
  // D3D12: a module from PTX text (NUL-terminated, the driver compiles it) or a compiled image; launches are
  // recorded into the command list in order with the list's other commands.
  Status(__cdecl* d3d12IsFatbinPtxSupported)(ID3D12Device*, bool* supported) = nullptr;
  Status(__cdecl* d3d12CreateCuModule)(ID3D12Device*, const void* blob, uint32_t size, Handle* module) = nullptr;
  Status(__cdecl* d3d12CreateCuFunction)(ID3D12Device*, Handle module, const char* name, Handle* function) = nullptr;
  Status(__cdecl* d3d12LaunchCuKernelChain)(ID3D12GraphicsCommandList*, const KernelLaunch* kernels,
                                            uint32_t count) = nullptr;
  Status(__cdecl* d3d12DestroyCuFunction)(ID3D12Device*, Handle function) = nullptr;
  Status(__cdecl* d3d12DestroyCuModule)(ID3D12Device*, Handle module) = nullptr;

  // D3D11: one "cubin compute shader" per (image, entry, block dimensions, dynamic shared bytes); a launch names the
  // resources it reads and writes by driver handle, and the kernel addresses them by GPU virtual address.
  Status(__cdecl* d3d11IsFatbinPtxSupported)(ID3D11Device*, bool* supported) = nullptr;
  Status(__cdecl* d3d11CreateCubinComputeShaderEx)(ID3D11Device*, const void* cubin, uint32_t size, uint32_t blockX,
                                                   uint32_t blockY, uint32_t blockZ, uint32_t dynamicSharedBytes,
                                                   const char* name, Handle* shader) = nullptr;
  Status(__cdecl* d3d11LaunchCubinShader)(ID3D11DeviceContext*, Handle shader, uint32_t gridX, uint32_t gridY,
                                          uint32_t gridZ, const void* params, uint32_t paramBytes,
                                          const Handle* readResources, uint32_t readCount,
                                          const Handle* writeResources, uint32_t writeCount) = nullptr;
  Status(__cdecl* d3d11DestroyCubinComputeShader)(ID3D11Device*, Handle shader) = nullptr;
  Status(__cdecl* d3d11GetResourceHandle)(ID3D11Device*, ID3D11Resource*, Handle* handle) = nullptr;
  Status(__cdecl* d3d11GetResourceGpuVirtualAddress)(ID3D11Device*, Handle resource, uint64_t* address) = nullptr;
  Status(__cdecl* d3d11GetResourceGpuVirtualAddressEx)(ID3D11Device*, D3D11GpuVirtualAddress* inOut) = nullptr;

  bool d3d12() const {
    return d3d12CreateCuModule && d3d12CreateCuFunction && d3d12LaunchCuKernelChain && d3d12DestroyCuFunction &&
           d3d12DestroyCuModule;
  }
  bool d3d11() const {
    return d3d11CreateCubinComputeShaderEx && d3d11LaunchCubinShader && d3d11DestroyCubinComputeShader &&
           d3d11GetResourceHandle && d3d11GetResourceGpuVirtualAddress;
  }
};

// The driver's interfaces, resolved once per process. Null with `error` when nvapi64.dll is absent (not an NVIDIA
// driver) or NvAPI_Initialize fails. A resolved interface is only a function pointer; see the note at the top.
inline const Api* load(std::string& error) {
  static Api api;
  static int state = 0;   // 0 untried, 1 loaded, 2 failed
  static std::string failure;
  if (state == 1) return &api;
  if (state == 2) { error = failure; return nullptr; }
  state = 2;
  HMODULE module = GetModuleHandleW(L"nvapi64.dll");
  if (!module) module = LoadLibraryExW(L"nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  using Query = void*(__cdecl*)(uint32_t id);
  const Query query = module ? reinterpret_cast<Query>(GetProcAddress(module, "nvapi_QueryInterface")) : nullptr;
  if (!query) { error = failure = "nvapi64.dll (NVIDIA driver) is not available"; return nullptr; }
  const auto initialize = reinterpret_cast<Status(__cdecl*)()>(query(0x0150e828u));   // NvAPI_Initialize
  const Status status = initialize ? initialize() : -1;
  if (status != 0) { error = failure = "NvAPI_Initialize failed (" + std::to_string(status) + ")"; return nullptr; }
  auto bind = [&](auto& target, uint32_t id) { target = reinterpret_cast<std::remove_reference_t<decltype(target)>>(query(id)); };
  bind(api.d3d12IsFatbinPtxSupported, 0x70c07832u);
  bind(api.d3d12CreateCuModule, 0xad1a677du);
  bind(api.d3d12CreateCuFunction, 0xe2436e22u);
  bind(api.d3d12LaunchCuKernelChain, 0x24973538u);
  bind(api.d3d12DestroyCuFunction, 0xdf295ea6u);
  bind(api.d3d12DestroyCuModule, 0x41c65285u);
  bind(api.d3d11IsFatbinPtxSupported, 0x6086bd93u);
  bind(api.d3d11CreateCubinComputeShaderEx, 0x32c2a0f6u);
  bind(api.d3d11LaunchCubinShader, 0x427e236du);
  bind(api.d3d11DestroyCubinComputeShader, 0x01682c86u);
  bind(api.d3d11GetResourceHandle, 0x09d52986u);
  bind(api.d3d11GetResourceGpuVirtualAddress, 0x1819b423u);
  bind(api.d3d11GetResourceGpuVirtualAddressEx, 0xaf6d14dau);
  state = 1;
  return &api;
}

// The packed argument buffer of a launch: u64 and u32 values at their natural alignment, in declaration order.
class Params {
 public:
  Params& u64(uint64_t value) { return put(&value, 8); }
  Params& u32(uint32_t value) { return put(&value, 4); }
  const void* data() const { return bytes_; }
  uint32_t size() const { return size_; }
  bool overflow() const { return overflow_; }   // more arguments than the buffer holds: the launch must not be made

 private:
  Params& put(const void* value, uint32_t width) {
    const uint32_t at = (size_ + width - 1) & ~(width - 1);
    if (at + width > sizeof(bytes_)) { overflow_ = true; return *this; }
    memcpy(bytes_ + at, value, width);
    size_ = at + width;
    return *this;
  }
  alignas(8) uint8_t bytes_[512]{};
  uint32_t size_ = 0;
  bool overflow_ = false;
};

}  // namespace nvcuda
#endif  // _WIN32
