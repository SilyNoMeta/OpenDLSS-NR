// Public runtime regression test: two backends in one process, plus borrowed enabled-feature checks.
#include "kernels.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void backend(const char* value) {
#ifdef _WIN32
  _putenv_s("DLSS5VK_BACKEND", value);
#else
  setenv("DLSS5VK_BACKEND", value, 1);
#endif
}

int main(int argc, char** argv) {
  try {
    const std::string shaders = argc > 1 ? argv[1] : "build/shaders";
    backend("sm86");
    vk::Context optimized;
    nr::Kernels a(optimized, shaders);
    const bool initialChain = a.chainEnabled();
    require(a.ptxGemmEnabled(), "sm86 PTX disabled before second context");
    backend("compat");
    vk::Context exact;
    nr::Kernels b(exact, shaders);
    require(!b.ptxGemmEnabled(), "compat must use software GEMM");
    require(a.ptxGemmEnabled(), "compat context disabled existing sm86 PTX");
    b.setChainEnabled(!initialChain);
    require(a.chainEnabled() == initialChain, "chain setting leaked across contexts");

    vk::DeviceRequirements enabled(vk::Backend::Sm86);
    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.pNext = &enabled.features;
    ci.enabledExtensionCount = (uint32_t)enabled.extensions.size();
    ci.ppEnabledExtensionNames = enabled.extensions.data();
    vk::BorrowedDevice device;
    device.instance = optimized.instance(); device.physical = optimized.physical();
    device.device = optimized.device(); device.queueFamily = optimized.queueFamily();
    device.backend = vk::Backend::Sm86; device.createInfo = &ci;
    { vk::Context borrowed(device); require(borrowed.ptxKernels(), "valid borrowed sm86 rejected"); }
    // Physical support remains unchanged; a declaration without an actually enabled bit must fail.
    enabled.f12.bufferDeviceAddress = VK_FALSE;
    bool rejected = false;
    try { vk::Context missingFeature(device); }
    catch (const std::exception& e) { rejected = std::string(e.what()).find("bufferDeviceAddress") != std::string::npos; }
    require(rejected, "borrowed supported-but-disabled feature accepted");
    enabled.f12.bufferDeviceAddress = VK_TRUE;
    ci.enabledExtensionCount = 0;
    rejected = false;
    try { vk::Context missingExtension(device); }
    catch (const std::exception& e) { rejected = std::string(e.what()).find("VK_NV_cuda_kernel_launch") != std::string::npos; }
    require(rejected, "borrowed disabled CUDA extension accepted");
    puts("PASS: backend isolation and supported-vs-enabled borrowed-device requirements");
    return 0;
  } catch (const std::exception& e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
