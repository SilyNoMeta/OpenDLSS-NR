#ifdef _WIN32
#include "d3d_common.h"

#include <d3dcompiler.h>
#include <wrl/client.h>

#include <atomic>
#include <cstring>
#include <fstream>
#include <sstream>

namespace d3d {

namespace {
// The few CUDA driver API entry points used, declared here so that no CUDA header is needed (as src/gpu_arch.cpp).
using CUresult = int;
using CUdevice = int;
constexpr int kAttributeMultiprocessorCount = 16;      // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
constexpr int kAttributeComputeCapabilityMajor = 75;   // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
constexpr int kAttributeComputeCapabilityMinor = 76;   // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
}  // namespace

namespace {
std::atomic<uint32_t> g_debugErrors{0};
}
uint32_t debugErrorTotal() { return g_debugErrors.load(); }
void addDebugErrors(uint32_t errors) { g_debugErrors += errors; }

GpuInfo gpuInfo(uint64_t luid) {
  // The library stays loaded for the life of the process (the driver keeps its own references anyway).
  static HMODULE library = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!library) throw std::runtime_error("nvcuda.dll (NVIDIA driver) is not available");
  auto cuInit = reinterpret_cast<CUresult (*)(unsigned int)>(GetProcAddress(library, "cuInit"));
  auto cuDeviceGetCount = reinterpret_cast<CUresult (*)(int*)>(GetProcAddress(library, "cuDeviceGetCount"));
  auto cuDeviceGet = reinterpret_cast<CUresult (*)(CUdevice*, int)>(GetProcAddress(library, "cuDeviceGet"));
  auto cuDeviceGetLuid = reinterpret_cast<CUresult (*)(char*, unsigned int*, CUdevice)>(GetProcAddress(library, "cuDeviceGetLuid"));
  auto cuDeviceGetName = reinterpret_cast<CUresult (*)(char*, int, CUdevice)>(GetProcAddress(library, "cuDeviceGetName"));
  auto cuDeviceGetAttribute = reinterpret_cast<CUresult (*)(int*, int, CUdevice)>(GetProcAddress(library, "cuDeviceGetAttribute"));
  if (!cuInit || !cuDeviceGetCount || !cuDeviceGet || !cuDeviceGetLuid || !cuDeviceGetName || !cuDeviceGetAttribute)
    throw std::runtime_error("nvcuda.dll lacks the device query entry points");
  if (cuInit(0) != 0) throw std::runtime_error("cuInit failed");
  int count = 0;
  if (cuDeviceGetCount(&count) != 0) throw std::runtime_error("cuDeviceGetCount failed");
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    CUdevice device = 0;
    char deviceLuid[8]{};
    unsigned int nodeMask = 0;
    if (cuDeviceGet(&device, ordinal) != 0 || cuDeviceGetLuid(deviceLuid, &nodeMask, device) != 0) continue;
    uint64_t value = 0;
    memcpy(&value, deviceLuid, 8);
    if (value != luid) continue;
    GpuInfo info;
    info.luid = luid;
    char name[128]{};
    cuDeviceGetName(name, sizeof(name) - 1, device);
    info.name = name;
    int sms = 0;
    if (cuDeviceGetAttribute(&sms, kAttributeMultiprocessorCount, device) != 0 ||
        cuDeviceGetAttribute(&info.major, kAttributeComputeCapabilityMajor, device) != 0 ||
        cuDeviceGetAttribute(&info.minor, kAttributeComputeCapabilityMinor, device) != 0 || sms <= 0)
      throw std::runtime_error("the CUDA device attributes of the adapter cannot be read");
    info.smCount = (uint32_t)sms;
    return info;
  }
  throw std::runtime_error("no CUDA device has the adapter's LUID (not an NVIDIA GPU?)");
}

void requireBackend(exec::Backend backend, const GpuInfo& gpu, const char* api) {
  const std::string capability = std::to_string(gpu.major) + "." + std::to_string(gpu.minor);
  const bool atLeast89 = gpu.major > 8 || (gpu.major == 8 && gpu.minor >= 9);
  if (backend == exec::Backend::Compat)
    throw std::runtime_error(std::string("the compat backend is the scalar reference route and runs on Vulkan only: the ") + api +
                             " adapter has HLSL twins for the kernels the PTX routes leave to shader code, not for the attention kernels");
  if (backend == exec::Backend::Sm86 && (gpu.major != 8 || atLeast89))
    throw std::runtime_error("the sm86 backend needs an Ampere GPU (compute capability 8.0 to 8.7); this one is " + capability);
  if (backend == exec::Backend::Native && !atLeast89)
    throw std::runtime_error("the native backend needs FP8 tensor cores (compute capability 8.9 or later); this one is " + capability);
}

KernelSignature parseSignature(const std::string& ptx, const char* entry) {
  KernelSignature signature;
  signature.entry = entry;
  const std::string head = std::string(".entry ") + entry + "(";
  const size_t start = ptx.find(head);
  if (start == std::string::npos) throw std::runtime_error(std::string("PTX has no entry ") + entry);
  const size_t end = ptx.find(')', start);
  if (end == std::string::npos) throw std::runtime_error(std::string("PTX entry without an argument list: ") + entry);
  const bool reduce = signature.entry.rfind("reduce_", 0) == 0;   // the one kernel that reads the split-K partials
  size_t at = start + head.size();
  while (true) {
    at = ptx.find(".param", at);
    if (at == std::string::npos || at >= end) break;
    std::istringstream words(ptx.substr(at, end - at));
    std::string keyword, type, name;
    words >> keyword >> type >> name;
    while (!name.empty() && (name.back() == ',' || name.back() == ')')) name.pop_back();
    uint8_t width = 0;
    if (type == ".u64" || type == ".s64" || type == ".b64" || type == ".f64") width = 8;
    else if (type == ".u32" || type == ".s32" || type == ".b32" || type == ".f32") width = 4;
    else throw std::runtime_error(std::string("PTX entry ") + entry + ": unsupported argument type " + type);
    // What a kernel writes through, by the generators' naming: its outputs, the chaining counters it signals, the
    // status word, the split-K tile counters and (except in the reduce) the split-K partials.
    const bool written = width == 8 && (name.rfind("pOut", 0) == 0 || name == "pD" || name == "pStateOut" || name == "pSignal" ||
                                        name == "pError" || name == "pCount" || (name == "pPartial" && !reduce));
    signature.widths.push_back(width);
    signature.names.push_back(name);
    signature.written.push_back(written);
    at += 6;
  }
  if (signature.widths.empty()) throw std::runtime_error(std::string("PTX entry without arguments: ") + entry);
  return signature;
}

nvcuda::Params packArguments(const KernelSignature& signature, const void* const* params, size_t paramCount) {
  if (paramCount != signature.widths.size())
    throw std::runtime_error("launch of " + signature.entry + " with " + std::to_string(paramCount) + " arguments, the entry declares " +
                             std::to_string(signature.widths.size()));
  nvcuda::Params packed;
  for (size_t i = 0; i < paramCount; ++i) {
    if (signature.widths[i] == 8) { uint64_t value; memcpy(&value, params[i], 8); packed.u64(value); }
    else { uint32_t value; memcpy(&value, params[i], 4); packed.u32(value); }
  }
  if (packed.overflow()) throw std::runtime_error("launch of " + signature.entry + ": the arguments exceed the packed buffer");
  return packed;
}

ShaderSource shaderSourceFor(const std::string& spvPath) {
  const size_t slash = spvPath.find_last_of("/\\");
  const std::string directory = slash == std::string::npos ? "." : spvPath.substr(0, slash);
  std::string name = slash == std::string::npos ? spvPath : spvPath.substr(slash + 1);
  const size_t dot = name.rfind('.');
  if (dot != std::string::npos) name = name.substr(0, dot);
  return {name, directory + "/hlsl/" + name + ".hlsl"};
}

std::vector<uint8_t> compileShader(const ShaderSource& source, const exec::SpecConstants& constants) {
  std::ifstream file(source.path, std::ios::binary);
  if (!file)
    throw std::runtime_error("the shader kernel " + source.name + " has no HLSL twin (" + source.path +
                             "): a Direct3D graph route dispatched a kernel that only exists for Vulkan");
  std::stringstream buffer;
  buffer << file.rdbuf();
  const std::string text = buffer.str();
  std::vector<std::string> names, values;
  for (const exec::SpecConstants::Entry& entry : constants.entries) {
    names.push_back("SPEC_" + std::to_string(entry.id));
    values.push_back(std::to_string(constants.data[entry.offset / 4]));
  }
  std::vector<D3D_SHADER_MACRO> macros;
  for (size_t i = 0; i < names.size(); ++i) macros.push_back({names[i].c_str(), values[i].c_str()});
  macros.push_back({nullptr, nullptr});
  Microsoft::WRL::ComPtr<ID3DBlob> code, errors;
  // IEEE strictness: the kernels' arithmetic is spelled operation by operation.
  const HRESULT hr = D3DCompile(text.data(), text.size(), source.path.c_str(), macros.data(), D3D_COMPILE_STANDARD_FILE_INCLUDE, "main",
                                "cs_5_0", D3DCOMPILE_IEEE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
  if (FAILED(hr))
    throw std::runtime_error("HLSL " + source.name + ": " + (errors ? std::string((const char*)errors->GetBufferPointer(), errors->GetBufferSize()) : "compile failed"));
  const uint8_t* bytes = static_cast<const uint8_t*>(code->GetBufferPointer());
  return std::vector<uint8_t>(bytes, bytes + code->GetBufferSize());
}

}  // namespace d3d
#endif  // _WIN32
