#include "device_factory.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>

#include "vk_context.h"
#ifdef _WIN32
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "d3d/d3d11_device.h"
#include "d3d/d3d12_device.h"
#endif

namespace {
std::string lowerEnv(const char* name) {
  std::string text = getenv(name) ? getenv(name) : "";
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return (char)std::tolower(c); });
  return text;
}

#ifdef _WIN32
// The numeric route of a Direct3D device: the one asked for, else the one the NVIDIA adapter's compute capability
// allows. There is nothing to fall back to: an Ampere GPU runs sm86, an FP8 GPU native.
exec::Backend direct3dBackend() {
  const std::string text = lowerEnv("DLSS5VK_BACKEND");
  if (text == "native") return exec::Backend::Native;
  if (text == "sm86") return exec::Backend::Sm86;
  if (text == "compat") return exec::Backend::Compat;   // refused by the adapter, with the reason
  if (!text.empty() && text != "auto") throw std::runtime_error("DLSS5VK_BACKEND=" + text + ": expected auto, native or sm86 on Direct3D");
  DXGI_ADAPTER_DESC1 desc{};
  d3d::nvidiaAdapter()->GetDesc1(&desc);
  const d3d::GpuInfo gpu = d3d::gpuInfo((uint64_t)(uint32_t)desc.AdapterLuid.LowPart | ((uint64_t)(uint32_t)desc.AdapterLuid.HighPart << 32));
  return (gpu.major > 8 || (gpu.major == 8 && gpu.minor >= 9)) ? exec::Backend::Native : exec::Backend::Sm86;
}
#endif
}  // namespace

std::unique_ptr<exec::Device> makeDevice() {
  const std::string api = lowerEnv("DLSS5VK_API");
  if (api.empty() || api == "vulkan") {
    auto context = std::make_unique<vk::Context>();
    if (lowerEnv("DLSS5VK_SHADER_FP8") == "0") {
      context->setShaderFp8(false);
      fprintf(stderr, "kernels: no cooperative-matrix GLSL (DLSS5VK_SHADER_FP8=0): the Direct3D kernel set\n");
    }
    return context;
  }
#ifdef _WIN32
  const std::string validation = lowerEnv("DLSS5VK_VALIDATION");
  const bool debug = !validation.empty() && validation != "0";
  if (api == "d3d12") {
    auto device = std::make_unique<d3d::D3D12Device>(direct3dBackend(), debug);
    fprintf(stderr, "backend: %s - %s (d3d12, adapter LUID %016llX, compute capability %d.%d, %u SMs%s)\n", exec::backendName(device->backend()),
            exec::backendDescription(device->backend()), (unsigned long long)device->gpu().luid, device->gpu().major, device->gpu().minor,
            device->gpu().smCount, debug ? ", debug layer" : "");
    return device;
  }
  if (api == "d3d11") {
    auto device = std::make_unique<d3d::D3D11Device>(direct3dBackend(), debug);
    fprintf(stderr, "backend: %s - %s (d3d11, adapter LUID %016llX, compute capability %d.%d, %u SMs%s)\n", exec::backendName(device->backend()),
            exec::backendDescription(device->backend()), (unsigned long long)device->gpu().luid, device->gpu().major, device->gpu().minor,
            device->gpu().smCount, debug ? ", debug layer" : "");
    return device;
  }
#endif
  throw std::runtime_error("DLSS5VK_API=" + api + ": expected vulkan, d3d12 or d3d11 (Direct3D on Windows only)");
}

uint32_t deviceValidationErrors() {
  uint32_t errors = vk::Context::validationErrors();
#ifdef _WIN32
  errors += d3d::debugErrorTotal();
#endif
  return errors;
}
