#include "vk_context.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <fstream>

#include "gpu_arch.h"

namespace vk {

namespace {
constexpr VkDeviceSize kStagingBytes = 256ull << 20;  // 256 MiB staging window
constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
constexpr uint32_t kVendorNvidia = 0x10DE;
std::atomic<uint32_t> g_validationErrors{0};

bool envFlag(const char* name) {
  const char* value = getenv(name);
  return value && *value && strcmp(value, "0") != 0;
}

VkBool32 VKAPI_PTR onDebugMessage(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type,
                                  const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
  if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) && (type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT))
    ++g_validationErrors;
  fprintf(stderr, "[vk] %s\n", data->pMessage ? data->pMessage : "");
  return VK_FALSE;
}

const char* deviceTypeName(VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
    default: return "other";
  }
}

std::string driverVersionText(const VkPhysicalDeviceProperties& p) {
  const uint32_t v = p.driverVersion;
  if (p.vendorID == kVendorNvidia) return std::to_string((v >> 22) & 0x3ff) + "." + std::to_string((v >> 14) & 0xff);
  return std::to_string(VK_API_VERSION_MAJOR(v)) + "." + std::to_string(VK_API_VERSION_MINOR(v)) + "." + std::to_string(VK_API_VERSION_PATCH(v));
}

std::vector<VkExtensionProperties> deviceExtensions(VkPhysicalDevice physical) {
  uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> available(count);
  vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, available.data());
  return available;
}

// The structure of type `type` in a pNext chain, or null.
const VkBaseInStructure* findInChain(const void* chain, VkStructureType type) {
  for (auto* s = static_cast<const VkBaseInStructure*>(chain); s; s = s->pNext)
    if (s->sType == type) return s;
  return nullptr;
}

// Every VkBool32 a DeviceRequirements can enable, by structure: the one table both checks below walk (what a
// physical device supports, and what a borrowed device was created with).
struct FeatureField {
  VkStructureType type;
  size_t offset;   // of the VkBool32 inside the structure of that type
  const char* name;
};
#define NR_FEATURE(Struct, sType, member) {sType, offsetof(Struct, member), #member}
const FeatureField kFeatureFields[] = {
    NR_FEATURE(VkPhysicalDeviceFeatures2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, features.shaderInt16),
    NR_FEATURE(VkPhysicalDeviceFeatures2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, features.shaderInt64),
    NR_FEATURE(VkPhysicalDeviceVulkan11Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, storageBuffer16BitAccess),
    NR_FEATURE(VkPhysicalDeviceVulkan11Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, uniformAndStorageBuffer16BitAccess),
    NR_FEATURE(VkPhysicalDeviceVulkan12Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, storageBuffer8BitAccess),
    NR_FEATURE(VkPhysicalDeviceVulkan12Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, uniformAndStorageBuffer8BitAccess),
    NR_FEATURE(VkPhysicalDeviceVulkan12Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, shaderFloat16),
    NR_FEATURE(VkPhysicalDeviceVulkan12Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, shaderInt8),
    NR_FEATURE(VkPhysicalDeviceVulkan12Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, vulkanMemoryModel),
    NR_FEATURE(VkPhysicalDeviceVulkan12Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, vulkanMemoryModelDeviceScope),
    NR_FEATURE(VkPhysicalDeviceVulkan12Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, hostQueryReset),
    NR_FEATURE(VkPhysicalDeviceVulkan12Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, bufferDeviceAddress),
    NR_FEATURE(VkPhysicalDeviceVulkan13Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, subgroupSizeControl),
    NR_FEATURE(VkPhysicalDeviceVulkan13Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, computeFullSubgroups),
    NR_FEATURE(VkPhysicalDeviceVulkan13Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, synchronization2),
    NR_FEATURE(VkPhysicalDeviceVulkan13Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, maintenance4),
    NR_FEATURE(VkPhysicalDeviceCooperativeMatrixFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, cooperativeMatrix),
    NR_FEATURE(VkPhysicalDeviceCooperativeMatrix2FeaturesNV, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV, cooperativeMatrixFlexibleDimensions),
    NR_FEATURE(VkPhysicalDeviceCooperativeMatrix2FeaturesNV, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV, cooperativeMatrixConversions),
    NR_FEATURE(VkPhysicalDeviceCooperativeMatrix2FeaturesNV, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV, cooperativeMatrixPerElementOperations),
    NR_FEATURE(VkPhysicalDeviceCooperativeMatrix2FeaturesNV, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV, cooperativeMatrixTensorAddressing),
    NR_FEATURE(VkPhysicalDeviceShaderFloat8FeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT, shaderFloat8),
    NR_FEATURE(VkPhysicalDeviceShaderFloat8FeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT, shaderFloat8CooperativeMatrix),
    NR_FEATURE(VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, pipelineExecutableInfo),
    NR_FEATURE(VkPhysicalDeviceShaderSMBuiltinsFeaturesNV, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV, shaderSMBuiltins),
    NR_FEATURE(VkPhysicalDeviceCudaKernelLaunchFeaturesNV, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV, cudaKernelLaunchFeatures),
};
#undef NR_FEATURE

VkBool32 fieldValue(const VkBaseInStructure* structure, const FeatureField& field) {
  VkBool32 value = VK_FALSE;
  if (structure) memcpy(&value, reinterpret_cast<const uint8_t*>(structure) + field.offset, sizeof(value));
  return value;
}

// The features `wanted` enables that `chain` does not have. `core` stands in for a VkPhysicalDeviceFeatures2 that
// is not in the chain (VkDeviceCreateInfo::pEnabledFeatures).
std::vector<std::string> missingFeatures(const DeviceRequirements& wanted, const void* chain,
                                         const VkPhysicalDeviceFeatures* core) {
  std::vector<std::string> missing;
  VkPhysicalDeviceFeatures2 coreHolder{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  if (core) coreHolder.features = *core;
  for (const FeatureField& field : kFeatureFields) {
    if (!fieldValue(findInChain(&wanted.features, field.type), field)) continue;
    const VkBaseInStructure* have = findInChain(chain, field.type);
    if (!have && core && field.type == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
      have = reinterpret_cast<const VkBaseInStructure*>(&coreHolder);
    if (!fieldValue(have, field)) missing.push_back(field.name);
  }
  return missing;
}

// What `physical` lacks of the backend's extensions and features plus `extra` extensions; empty when it has all.
std::vector<std::string> missingRequirements(VkPhysicalDevice physical, Backend backend,
                                             const std::vector<const char*>& extra) {
  std::vector<std::string> missing;
  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(physical, &properties);
  if (properties.apiVersion < VK_API_VERSION_1_3) missing.push_back("Vulkan 1.3");
  const std::vector<VkExtensionProperties> available = deviceExtensions(physical);
  auto has = [&](const char* name) {
    return std::any_of(available.begin(), available.end(), [&](const VkExtensionProperties& e) { return !strcmp(e.extensionName, name); });
  };
  DeviceRequirements wanted(backend);
  std::vector<const char*> extensions = wanted.extensions;
  extensions.insert(extensions.end(), extra.begin(), extra.end());
  for (const char* name : extensions)
    if (!has(name)) missing.push_back(name);
  if (!missing.empty()) return missing;   // the feature structures below are only defined with their extensions

  DeviceRequirements supported(backend, /*enable=*/false);   // the same chain with nothing set, filled by the driver
  vkGetPhysicalDeviceFeatures2(physical, &supported.features);
  return missingFeatures(wanted, &supported.features, nullptr);
}

std::string lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return (char)std::tolower(c); });
  return text;
}
}  // namespace

std::optional<Backend> requestedBackend() {
  const std::string text = lower(getenv("DLSS5VK_BACKEND") ? getenv("DLSS5VK_BACKEND") : "");
  if (text.empty() || text == "auto") return std::nullopt;
  for (Backend backend : {Backend::Native, Backend::Compat, Backend::Sm86})
    if (text == backendName(backend)) return backend;
  throw std::runtime_error("DLSS5VK_BACKEND=" + text + ": expected auto, native, compat or sm86");
}

uint32_t Context::validationErrors() { return g_validationErrors.load(); }

DeviceRequirements::DeviceRequirements(Backend backend, bool enable) : backend(backend) {
  // The chain is the same whether or not anything is enabled, so a copy with nothing set can be handed to
  // vkGetPhysicalDeviceFeatures2 and compared field by field (missingRequirements).
  const VkBool32 on = enable ? VK_TRUE : VK_FALSE;
  f11.storageBuffer16BitAccess = on;
  f11.uniformAndStorageBuffer16BitAccess = on;
  f12.pNext = &f11;
  f12.storageBuffer8BitAccess = on;
  f12.uniformAndStorageBuffer8BitAccess = on;
  f12.shaderFloat16 = on;
  f12.shaderInt8 = on;
  f12.vulkanMemoryModel = on;
  f12.vulkanMemoryModelDeviceScope = on;
  f12.hostQueryReset = on;
  f12.bufferDeviceAddress = on;
  f13.pNext = &f12;
  f13.subgroupSizeControl = on;
  f13.computeFullSubgroups = on;
  f13.synchronization2 = on;
  f13.maintenance4 = on;   // LocalSizeId: gemm_fp8.comp takes its workgroup size from a specialization constant
  features.pNext = &f13;
  features.features.shaderInt16 = on;
  features.features.shaderInt64 = on;
  // robustBufferAccess stays off (15% cost, masks bugs); experiment shaders must check their own ranges.
  features.features.robustBufferAccess = VK_FALSE;
  auto link = [this](auto& structure) {
    structure.pNext = features.pNext;
    features.pNext = &structure;
  };
  sm.shaderSMBuiltins = on;
  link(sm);
  extensions = {VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME};
  if (backend == Backend::Native) {
    // PTX launches: the extension's feature bit is part of the contract, as on sm86 (a device created without it
    // names the extension but may not launch).
    cuda.cudaKernelLaunchFeatures = on;
    link(cuda);
    coop.cooperativeMatrix = on;
    link(coop);
    // What the native SPIR-V declares, and nothing else of the extension.
    coop2.cooperativeMatrixFlexibleDimensions = on;
    coop2.cooperativeMatrixConversions = on;
    coop2.cooperativeMatrixPerElementOperations = on;
    coop2.cooperativeMatrixTensorAddressing = on;
    link(coop2);
    fp8.shaderFloat8 = on;
    fp8.shaderFloat8CooperativeMatrix = on;
    link(fp8);
    extensions.insert(extensions.end(),
                      {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME,
                       VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME, VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME});
  } else if (backend == Backend::Sm86) {
    // The compatibility GLSL plus PTX launches.
    cuda.cudaKernelLaunchFeatures = on;
    link(cuda);
    extensions.insert(extensions.end(), {VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME});
  }
}

void DeviceRequirements::enableStatistics() {
  if (executable.pipelineExecutableInfo) return;
  executable.pipelineExecutableInfo = VK_TRUE;
  executable.pNext = features.pNext;
  features.pNext = &executable;
  extensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
}

bool DeviceRequirements::statisticsSupported(VkPhysicalDevice physical) {
  const std::vector<VkExtensionProperties> available = deviceExtensions(physical);
  if (!std::any_of(available.begin(), available.end(), [](const VkExtensionProperties& e) {
        return !strcmp(e.extensionName, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
      }))
    return false;
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR query{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  supported.pNext = &query;
  vkGetPhysicalDeviceFeatures2(physical, &supported);
  return query.pipelineExecutableInfo == VK_TRUE;
}

DeviceChoice selectDevice(VkInstance instance, const std::vector<const char*>& extraExtensions) {
  uint32_t count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
  if (!count) throw std::runtime_error("no Vulkan physical devices (is a GPU driver installed?)");
  std::vector<VkPhysicalDevice> devices(count);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));

  // DLSS5VK_DEVICE: an index into the enumeration, or a case-insensitive part of the device name
  const std::string pick = lower(getenv("DLSS5VK_DEVICE") ? getenv("DLSS5VK_DEVICE") : "");
  const bool pickIndex = !pick.empty() && std::all_of(pick.begin(), pick.end(), [](unsigned char c) { return std::isdigit(c); });
  // DLSS5VK_BACKEND: only that route is considered; auto tries them fastest first
  // auto: native, else compat (exact); sm86 is chosen explicitly (DLSS5VK_BACKEND=sm86): it is not bit-exact
  const std::optional<Backend> requested = requestedBackend();
  std::vector<Backend> candidates = requested ? std::vector<Backend>{*requested}
                                              : std::vector<Backend>{Backend::Native, Backend::Compat};

  DeviceChoice best;
  int bestScore = -1;
  std::string report, nativeMissingOnBest;
  for (uint32_t index = 0; index < count; ++index) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(devices[index], &properties);
    const std::string name = properties.deviceName;
    const std::string label = "  [" + std::to_string(index) + "] " + name + " (" + deviceTypeName(properties.deviceType) +
                              ", driver " + driverVersionText(properties) + ")";
    if (!pick.empty() && (pickIndex ? (uint32_t)std::stoul(pick) != index : lower(name).find(pick) == std::string::npos)) {
      report += label + ": not selected by DLSS5VK_DEVICE\n";
      continue;
    }
    std::string line = label, nativeMissing;
    std::optional<Backend> usable;
    for (Backend backend : candidates) {
      const std::vector<std::string> missing = missingRequirements(devices[index], backend, extraExtensions);
      if (missing.empty()) {
        line += std::string(": ") + backendName(backend) + " usable";
        usable = backend;
        break;
      }
      std::string list;
      for (const std::string& m : missing) list += " " + m;
      if (backend == Backend::Native) nativeMissing = list;
      line += std::string(": ") + backendName(backend) + " missing" + list;
    }
    report += line + "\n";
    if (!usable) continue;
    // a faster route first, then a discrete GPU, then NVIDIA (a hybrid laptop also lists its integrated GPU,
    // often first when Windows prefers power saving)
    const int score = (*usable == Backend::Native ? 8 : *usable == Backend::Sm86 ? 4 : 0) +
                      (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 0) +
                      (properties.vendorID == kVendorNvidia ? 1 : 0);
    if (score > bestScore) {
      best.physical = devices[index];
      best.backend = *usable;
      bestScore = score;
      nativeMissingOnBest = nativeMissing;
    }
  }
  best.report = report;
  if (!best.physical)
    throw std::runtime_error(
        std::string("no Vulkan device can run the NR kernels") +
        (requested ? std::string(" on the ") + backendName(*requested) + " backend (DLSS5VK_BACKEND)" : std::string()) +
        ":\n" + report +
        "The native route needs an NVIDIA RTX 40 / 50 series GPU (desktop or laptop) and a driver exposing FP8 cooperative "
        "matrices; the compatibility route any NVIDIA Vulkan 1.3 GPU. Laptop drivers from the manufacturer often lag behind: "
        "install the current Game Ready or Studio driver from nvidia.com. On a hybrid (Optimus) laptop also set this program "
        "to \"High performance\" under Windows Settings > System > Display > Graphics, or pick a device with "
        "DLSS5VK_DEVICE=<index or name>.");
  if (count > 1 || getenv("DLSS5VK_DEVICE") || requested) fprintf(stderr, "Vulkan devices:\n%s", report.c_str());
  if (best.backend == Backend::Sm86) {
    // The lowered PTX targets sm_86 (f16 mma.sync, cp.async: compute capability 8.0 and up). On an Ada or newer GPU
    // the native route is the one to run; refusing here keeps a slower, inexact route from being taken by mistake.
    const std::optional<ComputeCapability> capability = computeCapability(best.physical);
    if (!capability || !capability->atLeast(8, 0) || capability->atLeast(8, 9))
      throw std::runtime_error("the sm86 backend needs an Ampere GPU (compute capability 8.0 to 8.7); this one is " +
                               (capability ? capability->text() : std::string("of unknown compute capability")));
  }

  // auto never falls back silently. A GPU that has FP8 tensor cores (compute capability 8.9 or later) but no native
  // route is a driver problem, not a reason to run the slow route: refuse unless the fallback was asked for.
  if (!requested && best.backend != Backend::Native) {
    const std::optional<ComputeCapability> capability = computeCapability(best.physical);
    if (!capability)
      throw std::runtime_error("native route unavailable and GPU compute capability unknown; refusing an automatic downgrade. "
                               "Check the NVIDIA driver, or explicitly select DLSS5VK_BACKEND=compat.");
    if (capability && capability->atLeast(8, 9))
      throw std::runtime_error("this GPU (compute capability " + capability->text() +
                               ") has FP8 tensor cores but its driver does not expose the native route:" +
                               nativeMissingOnBest +
                               ". Update the driver, or set DLSS5VK_BACKEND=compat to run the slow compatibility route anyway.");
    best.reason = "native route unavailable on this device (missing" + nativeMissingOnBest + ")" +
                  (capability ? ", compute capability " + capability->text() : std::string(", compute capability unknown"));
  } else {
    best.reason = requested ? "requested by DLSS5VK_BACKEND" : "fastest route this device supports";
  }
  return best;
}

Context::Context() {
  if (volkInitialize() != VK_SUCCESS) throw std::runtime_error("the Vulkan loader is unavailable");

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "dlss5-vulkan";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instanceInfo.pApplicationInfo = &app;
  validation_ = envFlag("DLSS5VK_VALIDATION");
  const bool debug = envFlag("DLSS5VK_DEBUG");
  const char* instanceExtensions[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
  if (validation_ || debug) { instanceInfo.enabledExtensionCount = 1; instanceInfo.ppEnabledExtensionNames = instanceExtensions; }
  if (validation_) {
    uint32_t count = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
    std::vector<VkLayerProperties> layers(count);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, layers.data()));
    auto layer = std::find_if(layers.begin(), layers.end(), [](const VkLayerProperties& l) { return !strcmp(l.layerName, kValidationLayer); });
    if (layer == layers.end())
      throw std::runtime_error(std::string("DLSS5VK_VALIDATION=1 but ") + kValidationLayer +
                               " is not installed (a Vulkan SDK, or VK_LAYER_PATH at a Vulkan-ValidationLayers build)");
    instanceInfo.enabledLayerCount = 1;
    instanceInfo.ppEnabledLayerNames = &kValidationLayer;
    fprintf(stderr, "validation: %s enabled (API %u.%u.%u)\n", kValidationLayer, VK_API_VERSION_MAJOR(layer->specVersion),
            VK_API_VERSION_MINOR(layer->specVersion), VK_API_VERSION_PATCH(layer->specVersion));
  }
  VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance_));
  volkLoadInstance(instance_);
  if (validation_ || debug) {
    VkDebugUtilsMessengerCreateInfoEXT messengerInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    if (debug) messengerInfo.messageSeverity |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
    messengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messengerInfo.pfnUserCallback = onDebugMessage;
    VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance_, &messengerInfo, nullptr, &messenger_));
  }

  const DeviceChoice choice = selectDevice(instance_);
  physical_ = choice.physical;
  backend_ = choice.backend;
  if (getenv("DLSS5VK_LIST_EXTENSIONS")) {
    for (const auto& extension : deviceExtensions(physical_))
      if (strstr(extension.extensionName, "cooperative") || strstr(extension.extensionName, "NV_"))
        printf("  device extension: %s (v%u)\n", extension.extensionName, extension.specVersion);
  }
  if (getenv("DLSS5VK_LIST_EXTENSIONS") && backend_ == Backend::Native) {
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV cm2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &cm2;
    vkGetPhysicalDeviceFeatures2(physical_, &f2);
    printf("coopmat2 features: workgroupScope %u flexibleDimensions %u reductions %u conversions %u perElement %u tensorAddressing %u blockLoads %u\n",
           cm2.cooperativeMatrixWorkgroupScope, cm2.cooperativeMatrixFlexibleDimensions, cm2.cooperativeMatrixReductions,
           cm2.cooperativeMatrixConversions, cm2.cooperativeMatrixPerElementOperations, cm2.cooperativeMatrixTensorAddressing,
           cm2.cooperativeMatrixBlockLoads);
    VkPhysicalDeviceCooperativeMatrix2PropertiesNV p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES_NV};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &p2;
    vkGetPhysicalDeviceProperties2(physical_, &props2);
    printf("coopmat2 properties: maxWorkgroupSize %u maxFlexibleDimension %u reservedShared %u\n",
           p2.cooperativeMatrixWorkgroupScopeMaxWorkgroupSize, p2.cooperativeMatrixFlexibleDimensionsMaxDimension,
           p2.cooperativeMatrixWorkgroupScopeReservedSharedMemory);
    auto getFlex = (PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV)vkGetInstanceProcAddr(
        instance_, "vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");
    if (getFlex) {
      uint32_t n = 0;
      getFlex(physical_, &n, nullptr);
      std::vector<VkCooperativeMatrixFlexibleDimensionsPropertiesNV> flex(n);
      for (auto& f : flex) f.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
      getFlex(physical_, &n, flex.data());
      for (auto& f : flex)
        printf("  flexible: gran M%u N%u K%u  A %u B %u C %u R %u  sat %u scope %u wgInvocations %u\n", f.MGranularity, f.NGranularity,
               f.KGranularity, f.AType, f.BType, f.CType, f.ResultType, f.saturatingAccumulation, f.scope, f.workgroupInvocations);
    }
  }

  readDeviceProperties();
  if (subgroupSize_ != 32)
    fprintf(stderr, "warning: subgroup size %u (cooperative kernels assume 32)\n", subgroupSize_);

  uint32_t familyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &familyCount, nullptr);
  std::vector<VkQueueFamilyProperties> families(familyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &familyCount, families.data());
  queueFamily_ = UINT32_MAX;
  for (uint32_t index = 0; index < familyCount; ++index) {
    if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) && families[index].timestampValidBits) {
      // Prefer a family that also has graphics (the "main" queue) for widest support.
      if (queueFamily_ == UINT32_MAX || (families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT)) queueFamily_ = index;
    }
  }
  if (queueFamily_ == UINT32_MAX) throw std::runtime_error("no compute queue with timestamps");

  float priority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queueInfo.queueFamilyIndex = queueFamily_;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &priority;

  DeviceRequirements req(backend_);
  if (DeviceRequirements::statisticsSupported(physical_)) req.enableStatistics();   // diagnostics only
  VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  deviceInfo.pNext = &req.features;
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = &queueInfo;
  deviceInfo.enabledExtensionCount = (uint32_t)req.extensions.size();
  deviceInfo.ppEnabledExtensionNames = req.extensions.data();
  VK_CHECK(vkCreateDevice(physical_, &deviceInfo, nullptr, &device_));
  volkLoadDevice(device_);
  executableProperties_ = req.executable.pipelineExecutableInfo;
  fprintf(stderr, "backend: %s - %s (%s)\n", backendName(backend_), backendDescription(backend_), choice.reason.c_str());
  initCommon();
}

Context::Context(const BorrowedDevice& borrowed) {
  if (volkInitialize() != VK_SUCCESS) throw std::runtime_error("the Vulkan loader is unavailable");
  if (!borrowed.createInfo)
    throw std::runtime_error("a borrowed device must come with the VkDeviceCreateInfo it was created with");
  instance_ = borrowed.instance; physical_ = borrowed.physical; device_ = borrowed.device;
  queueFamily_ = borrowed.queueFamily; queueIndex_ = borrowed.queueIndex; backend_ = borrowed.backend; owned_ = false;
  volkLoadInstance(instance_);
  volkLoadDevice(device_);
  // A PTX backend records launches: the device must dispatch the extension, not merely name it.
  if (ptxKernels() && (!vkCreateCudaModuleNV || !vkCreateCudaFunctionNV || !vkDestroyCudaModuleNV ||
                       !vkDestroyCudaFunctionNV || !vkCmdCudaLaunchKernelNV))
    throw std::runtime_error(std::string("the borrowed device does not dispatch VK_NV_cuda_kernel_launch, which the ") +
                             backendName(backend_) + " backend launches its PTX kernels through");
  bool createdQueue = false;
  const auto& ci = *borrowed.createInfo;
  if (ci.pQueueCreateInfos) {
    for (uint32_t i = 0; i < ci.queueCreateInfoCount; ++i) {
      const auto& q = ci.pQueueCreateInfos[i];
      createdQueue = createdQueue || (q.queueFamilyIndex == queueFamily_ && queueIndex_ < q.queueCount && q.flags == 0);
    }
  }
  if (!createdQueue) throw std::runtime_error("the borrowed queue was not created as an unprotected device queue");
  uint32_t familyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &familyCount, nullptr);
  std::vector<VkQueueFamilyProperties> families(familyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &familyCount, families.data());
  if (queueFamily_ >= familyCount || !(families[queueFamily_].queueFlags & VK_QUEUE_COMPUTE_BIT) ||
      !families[queueFamily_].timestampValidBits)
    throw std::runtime_error("the borrowed queue needs compute and timestamps");
  if (backend_ == Backend::Sm86) {
    const auto capability = computeCapability(physical_);
    if (!capability || !capability->atLeast(8, 0) || capability->atLeast(8, 9))
      throw std::runtime_error("the borrowed sm86 backend needs a verified Ampere compute capability");
  }
  // The backend follows what the device was created with, not what the GPU could do: a renderer that did not
  // enable an extension or feature cannot have the kernels use it.
  const DeviceRequirements wanted(backend_);
  std::vector<std::string> missing;
  for (const char* name : wanted.extensions) {
    bool enabled = false;
    for (uint32_t i = 0; i < borrowed.createInfo->enabledExtensionCount; ++i)
      enabled = enabled || !strcmp(borrowed.createInfo->ppEnabledExtensionNames[i], name);
    if (!enabled) missing.push_back(name);
  }
  for (const std::string& name : missingFeatures(wanted, borrowed.createInfo->pNext, borrowed.createInfo->pEnabledFeatures))
    missing.push_back(name);
  if (!missing.empty()) {
    std::string list;
    for (const std::string& m : missing) list += " " + m;
    throw std::runtime_error(std::string("the borrowed device was not created with what the ") + backendName(backend_) +
                             " backend needs:" + list);
  }
  const VkBaseInStructure* executable =
      findInChain(borrowed.createInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR);
  bool executableExtension = false;
  for (uint32_t i = 0; i < ci.enabledExtensionCount; ++i)
    executableExtension = executableExtension || !strcmp(ci.ppEnabledExtensionNames[i], VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
  executableProperties_ = executableExtension && executable &&
      reinterpret_cast<const VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR*>(executable)->pipelineExecutableInfo;
  readDeviceProperties();
  fprintf(stderr, "backend: %s - %s (borrowed device)\n", backendName(backend_), backendDescription(backend_));
  initCommon();
}

void Context::readDeviceProperties() {
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceShaderSMBuiltinsPropertiesNV smBuiltins{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV};
  properties.pNext = &subgroup;
  subgroup.pNext = &smBuiltins;
  vkGetPhysicalDeviceProperties2(physical_, &properties);
  smCount_ = smBuiltins.shaderSMCount;
  subgroupSize_ = subgroup.subgroupSize;
  deviceName_ = properties.properties.deviceName;
  timestampPeriod_ = properties.properties.limits.timestampPeriod;
  maxSharedMemory_ = properties.properties.limits.maxComputeSharedMemorySize;
  vkGetPhysicalDeviceMemoryProperties(physical_, &memoryProperties_);
}

void Context::initCommon() {
  vkGetDeviceQueue(device_, queueFamily_, queueIndex_, &queue_);

  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = queueFamily_;
  VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_));

  VkDescriptorSetLayoutBinding bindings[kGenericBindings];
  for (uint32_t index = 0; index < kGenericBindings; ++index) {
    bindings[index] = {index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  }
  VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layoutInfo.bindingCount = kGenericBindings;
  layoutInfo.pBindings = bindings;
  VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &setLayout_));

  VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushConstantBytes};
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &setLayout_;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushRange;
  VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_));

  VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192 * kGenericBindings};
  VkDescriptorPoolCreateInfo descriptorPoolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  descriptorPoolInfo.maxSets = 8192;
  descriptorPoolInfo.poolSizeCount = 1;
  descriptorPoolInfo.pPoolSizes = &poolSize;
  for (VkDescriptorPool& pool : descriptorPools_) VK_CHECK(vkCreateDescriptorPool(device_, &descriptorPoolInfo, nullptr, &pool));
  descriptorPool_ = descriptorPools_[0];

  dummy_ = createBuffer(256, false, "dummy binding");
  fillZero(dummy_);
  staging_ = createBuffer(kStagingBytes, true, "staging");
}

Context::~Context() {
  if (!device_) return;
  vkDeviceWaitIdle(device_);
  for (VkShaderModule module : modules_) vkDestroyShaderModule(device_, module, nullptr);
  destroyBuffer(dummy_);
  destroyBuffer(staging_);
  for (VkDescriptorPool pool : descriptorPools_) vkDestroyDescriptorPool(device_, pool, nullptr);
  vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
  vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
  vkDestroyCommandPool(device_, commandPool_, nullptr);
  if (owned_) {
    vkDestroyDevice(device_, nullptr);
    if (messenger_) vkDestroyDebugUtilsMessengerEXT(instance_, messenger_, nullptr);
    vkDestroyInstance(instance_, nullptr);
  }
}

uint32_t Context::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required) {
  for (uint32_t index = 0; index < memoryProperties_.memoryTypeCount; ++index) {
    if ((typeBits & (1u << index)) &&
        (memoryProperties_.memoryTypes[index].propertyFlags & required) == required)
      return index;
  }
  throw std::runtime_error("no suitable memory type");
}

Buffer Context::createBuffer(exec::Size size, bool hostVisible, const char* label, uint32_t extra) {
  Buffer result;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  result.size = std::max<VkDeviceSize>(size, 16);
  result.hostVisible = hostVisible;
  result.label = label;
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = result.size;
  info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | extra;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device_, &info, nullptr, &buffer));
  result.buffer = buffer;
  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(device_, buffer, &requirements);
  VkMemoryAllocateFlagsInfo allocateFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  allocateFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocateInfo.pNext = &allocateFlags;
  allocateInfo.allocationSize = requirements.size;
  allocateInfo.memoryTypeIndex = findMemoryType(
      requirements.memoryTypeBits,
      hostVisible ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                  : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(device_, &allocateInfo, nullptr, &memory));
  result.memory = memory;
  VK_CHECK(vkBindBufferMemory(device_, buffer, memory, 0));
  result.allocation = requirements.size;
  (hostVisible ? memoryUse_.hostVisible : memoryUse_.deviceLocal) += result.allocation;
  memoryUse_.peakDeviceLocal = std::max(memoryUse_.peakDeviceLocal, memoryUse_.deviceLocal);
  if (hostVisible) VK_CHECK(vkMapMemory(device_, memory, 0, VK_WHOLE_SIZE, 0, &result.mapped));
  return result;
}

void Context::destroyBuffer(Buffer& buffer) {
  if (buffer.memory) (buffer.hostVisible ? memoryUse_.hostVisible : memoryUse_.deviceLocal) -= buffer.allocation;
  if (buffer.mapped) vkUnmapMemory(device_, memoryOf(buffer));
  if (buffer.buffer) vkDestroyBuffer(device_, handle(buffer), nullptr);
  if (buffer.memory) vkFreeMemory(device_, memoryOf(buffer), nullptr);
  buffer = Buffer{};
}

void Context::upload(const Buffer& target, const void* data, exec::Size size, exec::Size offset) {
  if (offset + size > target.size) throw std::runtime_error(std::string("upload overflows ") + target.label);
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  VkDeviceSize done = 0;
  while (done < size) {
    VkDeviceSize chunk = std::min(size - done, staging_.size);
    memcpy(staging_.mapped, bytes + done, chunk);
    VkCommandBuffer commands = handle(beginCommands());
    VkBufferCopy region{0, offset + done, chunk};
    vkCmdCopyBuffer(commands, handle(staging_), handle(target), 1, &region);
    endAndSubmit(commands, true);
    done += chunk;
  }
}

void Context::fillZero(const Buffer& target) {
  VkCommandBuffer commands = handle(beginCommands());
  vkCmdFillBuffer(commands, handle(target), 0, VK_WHOLE_SIZE, 0);
  endAndSubmit(commands, true);
}

std::vector<uint8_t> Context::download(const Buffer& source, exec::Size size, exec::Size offset) {
  if (offset + size > source.size) throw std::runtime_error(std::string("download overflows ") + source.label);
  std::vector<uint8_t> result(size);
  VkDeviceSize done = 0;
  while (done < size) {
    VkDeviceSize chunk = std::min(size - done, staging_.size);
    VkCommandBuffer commands = handle(beginCommands());
    VkBufferCopy region{offset + done, 0, chunk};
    vkCmdCopyBuffer(commands, handle(source), handle(staging_), 1, &region);
    endAndSubmit(commands, true);
    memcpy(result.data() + done, staging_.mapped, chunk);
    done += chunk;
  }
  return result;
}

exec::Shader Context::loadShaderModule(const std::string& spvPath) {
  std::ifstream file(spvPath, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("missing shader " + spvPath);
  std::streamsize size = file.tellg();
  file.seekg(0);
  std::vector<uint32_t> words((size + 3) / 4);
  file.read(reinterpret_cast<char*>(words.data()), size);
  VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  info.codeSize = size;
  info.pCode = words.data();
  VkShaderModule module;
  VK_CHECK(vkCreateShaderModule(device_, &info, nullptr, &module));
  modules_.push_back(module);
  return module;
}

Pipeline Context::createComputePipeline(exec::Shader module, const SpecConstants& constants,
                                        const char* label, uint32_t requiredSubgroupSize) {
  std::vector<VkSpecializationMapEntry> entries;
  for (const SpecConstants::Entry& entry : constants.entries) entries.push_back({entry.id, entry.offset, 4});
  VkSpecializationInfo specialization{};
  specialization.mapEntryCount = (uint32_t)entries.size();
  specialization.pMapEntries = entries.data();
  specialization.dataSize = constants.data.size() * 4;
  specialization.pData = constants.data.data();
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
  subgroupInfo.requiredSubgroupSize = requiredSubgroupSize;
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.pNext = requiredSubgroupSize ? &subgroupInfo : nullptr;
  stage.flags = requiredSubgroupSize ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = static_cast<VkShaderModule>(module);
  stage.pName = "main";
  stage.pSpecializationInfo = constants.entries.empty() ? nullptr : &specialization;
  VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.stage = stage;
  info.layout = pipelineLayout_;
  if (captureStatistics_ && executableProperties_)
    info.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
  Pipeline result;
  result.label = label;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
  result.pipeline = pipeline;
  return result;
}

std::string Context::pipelineStatistics(const Pipeline& pipeline, bool includeInternal) {
  std::string report;
  VkPipelineInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
  pipelineInfo.pipeline = handle(pipeline);
  uint32_t executableCount = 0;
  if (!executableProperties_) return "n/a (VK_KHR_pipeline_executable_properties not enabled on this backend)";
  if (vkGetPipelineExecutablePropertiesKHR(device_, &pipelineInfo, &executableCount, nullptr) != VK_SUCCESS) return "n/a";
  std::vector<VkPipelineExecutablePropertiesKHR> executables(executableCount,
                                                             {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
  vkGetPipelineExecutablePropertiesKHR(device_, &pipelineInfo, &executableCount, executables.data());
  for (uint32_t e = 0; e < executableCount; ++e) {
    report += std::string(executables[e].name) + " (" + executables[e].description + ") subgroup " +
              std::to_string(executables[e].subgroupSize) + "\n";
    VkPipelineExecutableInfoKHR executableInfo{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    executableInfo.pipeline = handle(pipeline);
    executableInfo.executableIndex = e;
    uint32_t statisticCount = 0;
    vkGetPipelineExecutableStatisticsKHR(device_, &executableInfo, &statisticCount, nullptr);
    std::vector<VkPipelineExecutableStatisticKHR> statistics(statisticCount,
                                                             {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
    vkGetPipelineExecutableStatisticsKHR(device_, &executableInfo, &statisticCount, statistics.data());
    for (const auto& statistic : statistics) {
      report += "  " + std::string(statistic.name) + " = ";
      switch (statistic.format) {
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: report += statistic.value.b32 ? "true" : "false"; break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: report += std::to_string(statistic.value.i64); break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: report += std::to_string(statistic.value.u64); break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: report += std::to_string(statistic.value.f64); break;
        default: break;
      }
      report += "  (" + std::string(statistic.description) + ")\n";
    }
    if (includeInternal) {
      uint32_t irCount = 0;
      vkGetPipelineExecutableInternalRepresentationsKHR(device_, &executableInfo, &irCount, nullptr);
      std::vector<VkPipelineExecutableInternalRepresentationKHR> irs(
          irCount, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR});
      vkGetPipelineExecutableInternalRepresentationsKHR(device_, &executableInfo, &irCount, irs.data());
      std::vector<std::vector<char>> storage(irCount);
      for (uint32_t i = 0; i < irCount; ++i) { storage[i].resize(irs[i].dataSize + 1); irs[i].pData = storage[i].data(); }
      vkGetPipelineExecutableInternalRepresentationsKHR(device_, &executableInfo, &irCount, irs.data());
      for (uint32_t i = 0; i < irCount; ++i) {
        report += "  --- " + std::string(irs[i].name) + " (" + irs[i].description + ", " + std::to_string(irs[i].dataSize) +
                  " bytes, text=" + (irs[i].isText ? "yes" : "no") + ")\n";
        if (irs[i].isText) report += std::string(storage[i].data(), irs[i].dataSize) + "\n";
      }
    }
  }
  return report;
}

void Context::destroyPipeline(Pipeline& pipeline) {
  if (pipeline.pipeline) vkDestroyPipeline(device_, handle(pipeline), nullptr);
  pipeline = Pipeline{};
}

VkDescriptorSet Context::allocateSet(const Buffer* const bindings[kGenericBindings],
                                     const VkDeviceSize offsets[kGenericBindings],
                                     const VkDeviceSize ranges[kGenericBindings]) {
  VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocateInfo.descriptorPool = descriptorPool_;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts = &setLayout_;
  VkDescriptorSet set;
  VK_CHECK(vkAllocateDescriptorSets(device_, &allocateInfo, &set));
  VkDescriptorBufferInfo infos[kGenericBindings];
  VkWriteDescriptorSet writes[kGenericBindings];
  for (uint32_t index = 0; index < kGenericBindings; ++index) {
    const Buffer* buffer = bindings[index] ? bindings[index] : &dummy_;
    infos[index] = {handle(*buffer), offsets ? offsets[index] : 0,
                    ranges && ranges[index] ? ranges[index] : VK_WHOLE_SIZE};
    writes[index] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[index].dstSet = set;
    writes[index].dstBinding = index;
    writes[index].descriptorCount = 1;
    writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[index].pBufferInfo = &infos[index];
  }
  vkUpdateDescriptorSets(device_, kGenericBindings, writes, 0, nullptr);
  return set;
}

// Two pools in rotation: a frame's sets stay valid while the next frame is recorded (the caller waits for the
// frame before the previous one, as the demo's two frames in flight do).
void Context::resetDescriptorPool() { resetDescriptorPool((uint32_t)((descriptorPoolIndex_ + 1) % descriptorPools_.size())); }

void Context::resetDescriptorPool(uint32_t slot) {
  descriptorPoolIndex_ = slot % descriptorPools_.size();
  descriptorPool_ = descriptorPools_[descriptorPoolIndex_];
  VK_CHECK(vkResetDescriptorPool(device_, descriptorPool_, 0));
}

exec::Commands Context::beginCommands() {
  VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocateInfo.commandPool = commandPool_;
  allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocateInfo.commandBufferCount = 1;
  VkCommandBuffer commands;
  VK_CHECK(vkAllocateCommandBuffers(device_, &allocateInfo, &commands));
  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(commands, &beginInfo));
  return commands;
}

void Context::endAndSubmit(exec::Commands stream, bool wait) {
  VkCommandBuffer commands = handle(stream);
  VK_CHECK(vkEndCommandBuffer(commands));
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &commands;
  VK_CHECK(vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE));
  if (wait) {
    VK_CHECK(vkQueueWaitIdle(queue_));
    vkFreeCommandBuffers(device_, commandPool_, 1, &commands);
  }
}

void Context::dispatch(exec::Commands stream, exec::PipelineHandle pipeline, const Buffer* const bindings[kGenericBindings],
                       const void* push, uint32_t pushBytes, uint32_t x, uint32_t y, uint32_t z) {
  VkCommandBuffer commands = handle(stream);
  VkDescriptorSet set = allocateSet(bindings);
  vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, static_cast<VkPipeline>(pipeline));
  vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &set, 0, nullptr);
  vkCmdPushConstants(commands, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes, push);
  vkCmdDispatch(commands, x, y, z);
}

void Context::zeroBuffer(exec::Commands commands, const Buffer& target) {
  vkCmdFillBuffer(handle(commands), handle(target), 0, VK_WHOLE_SIZE, 0);
}

void Context::copyBuffer(exec::Commands commands, const Buffer& source, const Buffer& target, exec::Size bytes) {
  VkBufferCopy region{0, 0, bytes};
  vkCmdCopyBuffer(handle(commands), handle(source), handle(target), 1, &region);
}

void Context::computeBarrier(exec::Commands stream) {
  VkCommandBuffer commands = handle(stream);
  // Compute -> compute only. Including the transfer stages here made NVIDIA flush
  // caches between every dispatch (tens of microseconds per barrier once L2 is dirty).
  VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(commands, &dependency);
}

// Captures and uploads only.
void Context::transferBarrier(exec::Commands stream) {
  VkCommandBuffer commands = handle(stream);
  VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                          VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(commands, &dependency);
}

exec::Timer Context::createTimestampPool(uint32_t count) {
  VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  info.queryCount = count;
  VkQueryPool pool;
  VK_CHECK(vkCreateQueryPool(device_, &info, nullptr, &pool));
  vkResetQueryPool(device_, pool, 0, count);
  return pool;
}

void Context::destroyTimestampPool(exec::Timer pool) {
  if (pool) vkDestroyQueryPool(device_, static_cast<VkQueryPool>(pool), nullptr);
}

void Context::resetTimestamps(exec::Commands commands, exec::Timer pool, uint32_t count) {
  vkCmdResetQueryPool(handle(commands), static_cast<VkQueryPool>(pool), 0, count);
}

void Context::writeTimestamp(exec::Commands commands, exec::Timer pool, uint32_t index, bool first) {
  vkCmdWriteTimestamp(handle(commands), first ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      static_cast<VkQueryPool>(pool), index);
}

std::vector<double> Context::readTimestampsMs(exec::Timer timer, uint32_t count) {
  const VkQueryPool pool = static_cast<VkQueryPool>(timer);
  std::vector<uint64_t> ticks(count);
  VK_CHECK(vkGetQueryPoolResults(device_, pool, 0, count, ticks.size() * 8, ticks.data(), 8,
                                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
  std::vector<double> result(count);
  for (uint32_t index = 0; index < count; ++index) result[index] = ticks[index] * (double)timestampPeriod_ * 1e-6;
  return result;
}

}  // namespace vk

namespace vk {

exec::Address Context::deviceAddress(const Buffer& buffer) const {
  VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  info.buffer = handle(buffer);
  return vkGetBufferDeviceAddress(device_, &info);
}

exec::KernelModule Context::createCudaModule(const std::string& ptx) {
  VkCudaModuleCreateInfoNV info{VK_STRUCTURE_TYPE_CUDA_MODULE_CREATE_INFO_NV};
  info.dataSize = ptx.size() + 1;
  info.pData = ptx.c_str();
  VkCudaModuleNV module = VK_NULL_HANDLE;
  VK_CHECK(vkCreateCudaModuleNV(device_, &info, nullptr, &module));
  return module;
}

exec::Kernel Context::createCudaFunction(exec::KernelModule module, const char* name) {
  VkCudaFunctionCreateInfoNV info{VK_STRUCTURE_TYPE_CUDA_FUNCTION_CREATE_INFO_NV};
  info.module = static_cast<VkCudaModuleNV>(module);
  info.pName = name;
  VkCudaFunctionNV function = VK_NULL_HANDLE;
  VK_CHECK(vkCreateCudaFunctionNV(device_, &info, nullptr, &function));
  return function;
}

void Context::destroyCudaFunction(exec::Kernel function) {
  if (function) vkDestroyCudaFunctionNV(device_, static_cast<VkCudaFunctionNV>(function), nullptr);
}

void Context::destroyCudaModule(exec::KernelModule module) {
  if (module) vkDestroyCudaModuleNV(device_, static_cast<VkCudaModuleNV>(module), nullptr);
}

void Context::cudaLaunch(exec::Commands stream, exec::Kernel function, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                         uint32_t blockX, uint32_t sharedBytes, const void* const* params, size_t paramCount) {
  VkCommandBuffer commands = handle(stream);
  VkCudaLaunchInfoNV info{VK_STRUCTURE_TYPE_CUDA_LAUNCH_INFO_NV};
  info.function = static_cast<VkCudaFunctionNV>(function);
  info.gridDimX = gridX; info.gridDimY = gridY; info.gridDimZ = gridZ;
  info.blockDimX = blockX; info.blockDimY = 1; info.blockDimZ = 1;
  info.sharedMemBytes = sharedBytes;
  info.paramCount = paramCount;
  info.pParams = params;
  vkCmdCudaLaunchKernelNV(commands, &info);
}

}  // namespace vk
