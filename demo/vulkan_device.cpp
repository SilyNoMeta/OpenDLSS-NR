#include "vulkan_device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "vk_context.h"

namespace {

// DLSS5_DEMO_VALIDATION=1: the Khronos validation layer (needs a Vulkan SDK) with a debug messenger
bool wantsValidation() { return getenv("DLSS5_DEMO_VALIDATION") != nullptr; }

bool hasExtension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
  for (const auto& extension : extensions) if (!strcmp(extension.extensionName, name)) return true;
  return false;
}

bool hasNativeFp8(VkPhysicalDevice physical, const std::vector<VkExtensionProperties>& extensions) {
  if (!hasExtension(extensions, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) ||
      !hasExtension(extensions, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME) ||
      !hasExtension(extensions, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME) ||
      !hasExtension(extensions, VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME))
    return false;
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceCooperativeMatrix2FeaturesNV coop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV};
  VkPhysicalDeviceShaderFloat8FeaturesEXT fp8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &fp8;
  fp8.pNext = &coop2;
  coop2.pNext = &coop;
  vkGetPhysicalDeviceFeatures2(physical, &features);
  return coop.cooperativeMatrix && fp8.shaderFloat8 && fp8.shaderFloat8CooperativeMatrix &&
         coop2.cooperativeMatrixWorkgroupScope && coop2.cooperativeMatrixFlexibleDimensions &&
         coop2.cooperativeMatrixReductions && coop2.cooperativeMatrixConversions &&
         coop2.cooperativeMatrixPerElementOperations && coop2.cooperativeMatrixTensorAddressing &&
         coop2.cooperativeMatrixBlockLoads;
}

VkBool32 VKAPI_PTR debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                                 const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
  if (severity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT))
    fprintf(stderr, "[vk] %s\n", data->pMessage ? data->pMessage : "");
  return VK_FALSE;
}

}  // namespace

VulkanDevice::VulkanDevice() {
  if (volkInitialize() != VK_SUCCESS) throw std::runtime_error("the Vulkan loader (vulkan-1) is unavailable");

  // ---- instance: the surface extensions of the platform (the renderer creates the surface from the window)
  std::vector<const char*> instanceExtensions = {VK_KHR_SURFACE_EXTENSION_NAME};
#if defined(_WIN32)
  instanceExtensions.push_back("VK_KHR_win32_surface");
#elif defined(__APPLE__)
  instanceExtensions.push_back("VK_EXT_metal_surface");
  instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#else
  instanceExtensions.push_back("VK_KHR_xlib_surface");
#endif
  const bool validation = wantsValidation();
  if (validation) instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
  if (validation) {
    uint32_t layerCount = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, nullptr));
    std::vector<VkLayerProperties> available(layerCount);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, available.data()));
    bool found = false;
    for (const VkLayerProperties& layer : available) found = found || !strcmp(layer.layerName, layers[0]);
    if (!found) throw std::runtime_error("DLSS5_DEMO_VALIDATION=1 but VK_LAYER_KHRONOS_validation is not installed");
    fprintf(stderr, "[vk] validation: %s enabled\n", layers[0]);
  }
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "dlss5-demo";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instanceInfo.pApplicationInfo = &app;
  instanceInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
  instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();
  if (validation) { instanceInfo.enabledLayerCount = 1; instanceInfo.ppEnabledLayerNames = layers; }
#if defined(__APPLE__)
  instanceInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
  VkInstance instance = VK_NULL_HANDLE;
  VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance));
  volkLoadInstance(instance);
  if (validation) {
    VkDebugUtilsMessengerCreateInfoEXT messengerInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messengerInfo.pfnUserCallback = debugCallback;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance, &messengerInfo, nullptr, &messenger));
    messenger_ = (uint64_t)messenger;
  }

  // ---- the physical device: the first one exposing the NR kernels' extensions
  uint32_t count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
  std::vector<VkPhysicalDevice> devices(count);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
  bool nativeFp8 = false;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkPhysicalDevice compatibility = VK_NULL_HANDLE;
  for (VkPhysicalDevice candidate : devices) {
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, extensions.data());
    VkPhysicalDeviceProperties candidateProperties{};
    vkGetPhysicalDeviceProperties(candidate, &candidateProperties);
    if (hasNativeFp8(candidate, extensions)) { physical = candidate; nativeFp8 = true; break; }
    if (!compatibility && candidateProperties.vendorID == 0x10de) compatibility = candidate;
  }
  if (!physical) physical = compatibility;
  if (!physical) throw std::runtime_error("no NVIDIA Vulkan device");
  vk::DeviceRequirements requirements(nativeFp8);
  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(physical, &properties);
  deviceName_ = properties.deviceName;

  // ---- queues: one graphics + compute family; a second queue for the NR side's own submissions when available.
  // The NR pass records into Filament's command buffer; vk::Context only submits on its own queue for uploads,
  // resizes and captures, all of which happen with the renderer idle - which is also what makes the one-queue
  // fallback below safe (two threads must not submit to one VkQueue concurrently).
  uint32_t familyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, nullptr);
  std::vector<VkQueueFamilyProperties> families(familyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, families.data());
  uint32_t family = UINT32_MAX;
  for (uint32_t index = 0; index < familyCount; ++index) {
    const VkQueueFlags flags = families[index].queueFlags;
    if ((flags & VK_QUEUE_GRAPHICS_BIT) && (flags & VK_QUEUE_COMPUTE_BIT) && families[index].timestampValidBits) { family = index; break; }
  }
  if (family == UINT32_MAX) throw std::runtime_error("no graphics + compute queue family with timestamps");
  const uint32_t queueCount = families[family].queueCount >= 2 ? 2 : 1;
  const float priorities[2] = {1.0f, 1.0f};
  VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queueInfo.queueFamilyIndex = family;
  queueInfo.queueCount = queueCount;
  queueInfo.pQueuePriorities = priorities;

  // ---- features: the kernels' chain, plus every supported core / 1.1 feature (the renderer assumes what the
  // device supports is enabled)
  VkPhysicalDeviceVulkan11Features supported11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  supported.pNext = &supported11;
  vkGetPhysicalDeviceFeatures2(physical, &supported);
  requirements.features.features = supported.features;
  requirements.f11.multiview = supported11.multiview;
  requirements.f11.shaderDrawParameters = supported11.shaderDrawParameters;
  requirements.f11.samplerYcbcrConversion = supported11.samplerYcbcrConversion;
  std::vector<const char*> extensions = requirements.extensions;
  extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#if defined(__APPLE__)
  extensions.push_back("VK_KHR_portability_subset");
#endif
  VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  deviceInfo.pNext = &requirements.features;
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = &queueInfo;
  deviceInfo.enabledExtensionCount = (uint32_t)extensions.size();
  deviceInfo.ppEnabledExtensionNames = extensions.data();
  VkDevice device = VK_NULL_HANDLE;
  VK_CHECK(vkCreateDevice(physical, &deviceInfo, nullptr, &device));
  volkLoadDevice(device);

  handles_.instance = instance;
  handles_.physicalDevice = physical;
  handles_.device = device;
  handles_.queueFamily = family;
  handles_.rendererQueueIndex = 0;
  handles_.nrQueueIndex = queueCount - 1;
  handles_.debugUtils = validation;
  fprintf(stderr, "[vk] %s, %s, queue family %u (%u queue%s)\n", deviceName_.c_str(),
          nativeFp8 ? "native FP8" : "software E4M3 compatibility", family, queueCount, queueCount > 1 ? "s" : "");
}

VulkanDevice::~VulkanDevice() {
  if (handles_.device) vkDestroyDevice((VkDevice)handles_.device, nullptr);
  if (messenger_) vkDestroyDebugUtilsMessengerEXT((VkInstance)handles_.instance, (VkDebugUtilsMessengerEXT)messenger_, nullptr);
  if (handles_.instance) vkDestroyInstance((VkInstance)handles_.instance, nullptr);
}
