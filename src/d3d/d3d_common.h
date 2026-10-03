// What the Direct3D 12 and Direct3D 11 adapters share: the GPU's CUDA facts, a PTX entry's argument list, and the
// HLSL twins of the shader kernels.
#pragma once
#ifdef _WIN32
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../exec.h"
#include "nvapi_cuda.h"

namespace d3d {

// The adapter's CUDA facts, from the driver's own CUDA library (nvcuda.dll, part of every NVIDIA driver) matched by
// adapter LUID. No CUDA context is created. Throws when the adapter is not an NVIDIA GPU the driver lists.
struct GpuInfo {
  std::string name;
  uint64_t luid = 0;
  uint32_t smCount = 0;        // streaming multiprocessors
  int major = 0, minor = 0;    // compute capability
};
GpuInfo gpuInfo(uint64_t luid);

// The numeric routes a Direct3D adapter runs, checked against the GPU; throws with the reason otherwise. sm86 needs
// an Ampere GPU (compute capability 8.0 to 8.7), native FP8 tensor cores (8.9 or later). compat, the scalar
// reference route, has HLSL twins only for the few kernels the PTX routes leave to shader code: it is refused.
void requireBackend(exec::Backend backend, const GpuInfo& gpu, const char* api);

// A PTX entry's arguments, from its .param declarations: each one's width in bytes (4 or 8) for the packed launch
// buffer, and for the 64-bit ones whether the kernel writes through the address.
struct KernelSignature {
  std::string entry;
  std::vector<uint8_t> widths;
  std::vector<std::string> names;
  std::vector<bool> written;
};
KernelSignature parseSignature(const std::string& ptx, const char* entry);
// The launch's arguments in the CUDA ABI. `params` points at each value, as exec::Device::cudaLaunch takes them.
nvcuda::Params packArguments(const KernelSignature& signature, const void* const* params, size_t paramCount);

// A shader kernel's HLSL twin: next to the SPIR-V the caller names, <dir>/hlsl/<name>.hlsl. Loaded lazily, so a
// kernel without a twin fails only if a graph route actually dispatches it.
struct ShaderSource {
  std::string name;   // "ops_compat"
  std::string path;   // the .hlsl file
};
ShaderSource shaderSourceFor(const std::string& spvPath);
// cs_5_0 bytecode (it runs on Direct3D 11 and 12), the constants as SPEC_<id> macros.
std::vector<uint8_t> compileShader(const ShaderSource& source, const exec::SpecConstants& constants);

// Error-severity debug-layer messages of the Direct3D devices destroyed so far (a tool reports them at exit).
uint32_t debugErrorTotal();
void addDebugErrors(uint32_t errors);

// Live buffers by GPU address range: which resource a kernel's pointer argument addresses.
template <typename Resource>
class AddressMap {
 public:
  void add(uint64_t start, uint64_t size, Resource resource) { ranges_[start] = {size, resource}; }
  void remove(uint64_t start) { ranges_.erase(start); }
  bool find(uint64_t address, Resource& resource) const {
    auto it = ranges_.upper_bound(address);
    if (it == ranges_.begin()) return false;
    --it;
    if (address - it->first >= it->second.size) return false;
    resource = it->second.resource;
    return true;
  }

 private:
  struct Range { uint64_t size; Resource resource; };
  std::map<uint64_t, Range> ranges_;
};

}  // namespace d3d
#endif  // _WIN32
