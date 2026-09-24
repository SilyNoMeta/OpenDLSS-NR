#include "gpu_arch.h"

#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace vk {

namespace {
// The few CUDA driver API entry points used, declared here so that no CUDA header is needed.
using CUresult = int;
using CUdevice = int;
struct CUuuid { char bytes[16]; };
using PFN_cuInit = CUresult (*)(unsigned int);
using PFN_cuDeviceGetCount = CUresult (*)(int*);
using PFN_cuDeviceGet = CUresult (*)(CUdevice*, int);
using PFN_cuDeviceGetUuid = CUresult (*)(CUuuid*, CUdevice);
using PFN_cuDeviceGetAttribute = CUresult (*)(int*, int, CUdevice);
constexpr int kAttributeComputeCapabilityMajor = 75;   // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
constexpr int kAttributeComputeCapabilityMinor = 76;   // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

void* loadDriverLibrary() {
#if defined(_WIN32)
  return (void*)LoadLibraryA("nvcuda.dll");
#else
  void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  return library ? library : dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
#endif
}

void* symbol(void* library, const char* name) {
#if defined(_WIN32)
  return (void*)GetProcAddress((HMODULE)library, name);
#else
  return dlsym(library, name);
#endif
}
}  // namespace

std::optional<ComputeCapability> computeCapability(VkPhysicalDevice physical) {
  VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &id;
  vkGetPhysicalDeviceProperties2(physical, &properties);
  if (properties.properties.vendorID != 0x10DE) return std::nullopt;

  // The library stays loaded for the life of the process (the driver keeps its own references anyway).
  static void* library = loadDriverLibrary();
  if (!library) return std::nullopt;
  auto cuInit = (PFN_cuInit)symbol(library, "cuInit");
  auto cuDeviceGetCount = (PFN_cuDeviceGetCount)symbol(library, "cuDeviceGetCount");
  auto cuDeviceGet = (PFN_cuDeviceGet)symbol(library, "cuDeviceGet");
  auto cuDeviceGetUuid = (PFN_cuDeviceGetUuid)symbol(library, "cuDeviceGetUuid_v2");
  if (!cuDeviceGetUuid) cuDeviceGetUuid = (PFN_cuDeviceGetUuid)symbol(library, "cuDeviceGetUuid");
  auto cuDeviceGetAttribute = (PFN_cuDeviceGetAttribute)symbol(library, "cuDeviceGetAttribute");
  if (!cuInit || !cuDeviceGetCount || !cuDeviceGet || !cuDeviceGetUuid || !cuDeviceGetAttribute) return std::nullopt;
  if (cuInit(0) != 0) return std::nullopt;
  int count = 0;
  if (cuDeviceGetCount(&count) != 0) return std::nullopt;
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    CUdevice device = 0;
    CUuuid uuid{};
    if (cuDeviceGet(&device, ordinal) != 0 || cuDeviceGetUuid(&uuid, device) != 0) continue;
    if (memcmp(uuid.bytes, id.deviceUUID, sizeof(uuid.bytes)) != 0) continue;
    ComputeCapability capability;
    if (cuDeviceGetAttribute(&capability.major, kAttributeComputeCapabilityMajor, device) != 0 ||
        cuDeviceGetAttribute(&capability.minor, kAttributeComputeCapabilityMinor, device) != 0)
      return std::nullopt;
    return capability;
  }
  return std::nullopt;
}

}  // namespace vk
