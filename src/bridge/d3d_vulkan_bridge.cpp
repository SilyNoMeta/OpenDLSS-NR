#ifdef _WIN32
#include "d3d_vulkan_bridge.h"

#include <dxgi1_6.h>

using Microsoft::WRL::ComPtr;

namespace bridge {

namespace {
constexpr VkFormat kVkFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
constexpr DXGI_FORMAT kDxgiFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
constexpr VkImageUsageFlags kUsage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                     VK_IMAGE_USAGE_STORAGE_BIT;

void check(HRESULT hr, const char* what) {
  if (FAILED(hr)) {
    char text[200];
    snprintf(text, sizeof(text), "%s failed (0x%08lX)", what, (unsigned long)hr);
    throw std::runtime_error(text);
  }
}
}  // namespace

Bridge::Bridge(ID3D12Device* device, ID3D12CommandQueue* queue, exec::Backend backend) : device12_(device), queue12_(queue) {
  if (!device || !queue) throw std::runtime_error("a d3d12 bridge needs the host's device and queue");
  const LUID luid = device->GetAdapterLuid();
  luid_ = (uint64_t)(uint32_t)luid.LowPart | ((uint64_t)(uint32_t)luid.HighPart << 32);
  init(backend);
  check(device12_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence12_)), "CreateFence(shared)");
  HANDLE handle = nullptr;
  check(device12_->CreateSharedHandle(fence12_.Get(), nullptr, GENERIC_ALL, nullptr, &handle), "CreateSharedHandle(fence)");
  importFence(handle);
  CloseHandle(handle);
}

Bridge::Bridge(ID3D11Device* device, ID3D11DeviceContext* immediate, exec::Backend backend) {
  if (!device || !immediate) throw std::runtime_error("a d3d11 bridge needs the host's device and immediate context");
  if (immediate->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) throw std::runtime_error("a d3d11 bridge needs the immediate context: deferred contexts are not supported");
  check(device->QueryInterface(IID_PPV_ARGS(&device11_)), "ID3D11Device5 (shared fences)");
  check(immediate->QueryInterface(IID_PPV_ARGS(&context11_)), "ID3D11DeviceContext4 (shared fences)");
  ComPtr<IDXGIDevice> dxgi;
  ComPtr<IDXGIAdapter> adapter;
  DXGI_ADAPTER_DESC desc{};
  check(device->QueryInterface(IID_PPV_ARGS(&dxgi)), "IDXGIDevice");
  check(dxgi->GetAdapter(&adapter), "IDXGIDevice::GetAdapter");
  check(adapter->GetDesc(&desc), "IDXGIAdapter::GetDesc");
  luid_ = (uint64_t)(uint32_t)desc.AdapterLuid.LowPart | ((uint64_t)(uint32_t)desc.AdapterLuid.HighPart << 32);
  init(backend);
  check(device11_->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence11_)), "ID3D11Device5::CreateFence(shared)");
  HANDLE handle = nullptr;
  check(fence11_->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &handle), "ID3D11Fence::CreateSharedHandle");
  importFence(handle);
  CloseHandle(handle);
}

void Bridge::init(exec::Backend backend) {
  vk::DedicatedDevice dedicated;
  dedicated.luid = luid_;
  dedicated.backend = backend;
  dedicated.extensions = {VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
  dedicated.timelineSemaphore = true;
  vulkan_ = std::make_unique<vk::Context>(dedicated);
}

void Bridge::importFence(HANDLE handle) {
  // A Direct3D fence is a monotonic 64-bit counter: a timeline semaphore. The driver says whether one can be imported.
  const VkExternalSemaphoreHandleTypeFlagBits type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;   // = D3D11_FENCE_BIT
  VkSemaphoreTypeCreateInfo timeline{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkPhysicalDeviceExternalSemaphoreInfo query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
  query.pNext = &timeline;
  query.handleType = type;
  VkExternalSemaphoreProperties properties{VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
  vkGetPhysicalDeviceExternalSemaphoreProperties(vulkan_->physical(), &query, &properties);
  if (!(properties.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT))
    throw std::runtime_error("the Vulkan device cannot import a Direct3D fence as a timeline semaphore");
  VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  info.pNext = &timeline;
  VK_CHECK(vkCreateSemaphore(vulkan_->device(), &info, nullptr, &semaphore_));
  VkImportSemaphoreWin32HandleInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
  import.semaphore = semaphore_;
  import.handleType = type;
  import.handle = handle;
  VK_CHECK(vkImportSemaphoreWin32HandleKHR(vulkan_->device(), &import));
}

Bridge::~Bridge() {
  if (!vulkan_) return;
  // The owner has waited for the last frame; nothing of this object is in flight when it goes.
  vkQueueWaitIdle(vulkan_->queue());
  for (const Pending& pending : pending_) vkFreeCommandBuffers(vulkan_->device(), vulkan_->commandPool(), 1, &pending.commands);
  if (semaphore_) vkDestroySemaphore(vulkan_->device(), semaphore_, nullptr);
}

SharedTexture Bridge::createTexture(uint32_t width, uint32_t height) {
  SharedTexture texture;
  texture.width = width;
  texture.height = height;
  HANDLE handle = nullptr;
  VkExternalMemoryHandleTypeFlagBits type;
  if (queue12_) {
    type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Format = kDxgiFormat;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    check(device12_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&texture.d3d12)),
          "CreateCommittedResource(shared texture)");
    check(device12_->CreateSharedHandle(texture.d3d12.Get(), nullptr, GENERIC_ALL, nullptr, &handle), "CreateSharedHandle(texture)");
  } else {
    type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = desc.ArraySize = 1;
    desc.Format = kDxgiFormat;
    desc.SampleDesc.Count = 1;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED;
    check(device11_->CreateTexture2D(&desc, nullptr, &texture.d3d11), "CreateTexture2D(shared)");
    ComPtr<IDXGIResource1> resource;
    check(texture.d3d11.As(&resource), "IDXGIResource1");
    check(resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &handle), "CreateSharedHandle(texture)");
  }

  VkPhysicalDeviceExternalImageFormatInfo external{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
  external.handleType = type;
  VkPhysicalDeviceImageFormatInfo2 format{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
  format.pNext = &external;
  format.format = kVkFormat;
  format.type = VK_IMAGE_TYPE_2D;
  format.tiling = VK_IMAGE_TILING_OPTIMAL;
  format.usage = kUsage;
  VkExternalImageFormatProperties externalProperties{VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
  VkImageFormatProperties2 formatProperties{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
  formatProperties.pNext = &externalProperties;
  const VkResult supported = vkGetPhysicalDeviceImageFormatProperties2(vulkan_->physical(), &format, &formatProperties);
  if (supported != VK_SUCCESS || !(externalProperties.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) {
    CloseHandle(handle);
    throw std::runtime_error(std::string("the Vulkan device cannot import an RGBA32F ") + host() + " texture");
  }

  VkExternalMemoryImageCreateInfo externalInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
  externalInfo.handleTypes = type;
  VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.pNext = &externalInfo;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = kVkFormat;
  info.extent = {width, height, 1};
  info.mipLevels = info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = kUsage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkDevice device = vulkan_->device();
  VK_CHECK(vkCreateImage(device, &info, nullptr, &texture.image));
  VkMemoryRequirements requirements{};
  vkGetImageMemoryRequirements(device, texture.image, &requirements);
  VkMemoryWin32HandlePropertiesKHR handleProperties{VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
  VK_CHECK(vkGetMemoryWin32HandlePropertiesKHR(device, type, handle, &handleProperties));
  const uint32_t typeBits = requirements.memoryTypeBits & handleProperties.memoryTypeBits;
  uint32_t memoryType = UINT32_MAX;
  for (uint32_t index = 0; index < 32; ++index)
    if (typeBits & (1u << index)) { memoryType = index; break; }
  if (memoryType == UINT32_MAX) {
    CloseHandle(handle);
    throw std::runtime_error("no Vulkan memory type backs the shared texture");
  }
  VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
  dedicated.image = texture.image;
  VkImportMemoryWin32HandleInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
  import.pNext = &dedicated;
  import.handleType = type;
  import.handle = handle;
  VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocate.pNext = &import;
  allocate.allocationSize = requirements.size;
  allocate.memoryTypeIndex = memoryType;
  const VkResult allocated = vkAllocateMemory(device, &allocate, nullptr, &texture.memory);
  CloseHandle(handle);   // the import holds its own reference to the resource
  VK_CHECK(allocated);
  VK_CHECK(vkBindImageMemory(device, texture.image, texture.memory, 0));

  // UNDEFINED -> GENERAL once, while the texture holds nothing; GENERAL from then on, so no later transition can
  // discard what Direct3D wrote.
  exec::Commands commands = vulkan_->beginCommands();
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = texture.image;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(vk::handle(commands), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  vulkan_->endAndSubmit(commands, true);
  return texture;
}

void Bridge::destroyTexture(SharedTexture& texture) {
  if (texture.image) vkDestroyImage(vulkan_->device(), texture.image, nullptr);
  if (texture.memory) vkFreeMemory(vulkan_->device(), texture.memory, nullptr);
  texture = SharedTexture{};
}

uint64_t Bridge::signalReady() {
  const uint64_t ready = ++value_;
  if (queue12_) {
    check(queue12_->Signal(fence12_.Get(), ready), "ID3D12CommandQueue::Signal");
  } else {
    check(context11_->Signal(fence11_.Get(), ready), "ID3D11DeviceContext4::Signal");
    context11_->Flush();   // the signal reaches the GPU now; nothing here waits for it
  }
  return ready;
}

uint64_t Bridge::submit(exec::Commands stream, uint64_t ready) {
  VkCommandBuffer commands = vk::handle(stream);
  VK_CHECK(vkEndCommandBuffer(commands));
  const uint64_t done = ++value_;
  VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline.waitSemaphoreValueCount = 1;
  timeline.pWaitSemaphoreValues = &ready;
  timeline.signalSemaphoreValueCount = 1;
  timeline.pSignalSemaphoreValues = &done;
  const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  info.pNext = &timeline;
  info.waitSemaphoreCount = 1;
  info.pWaitSemaphores = &semaphore_;
  info.pWaitDstStageMask = &stage;
  info.commandBufferCount = 1;
  info.pCommandBuffers = &commands;
  info.signalSemaphoreCount = 1;
  info.pSignalSemaphores = &semaphore_;
  VK_CHECK(vkQueueSubmit(vulkan_->queue(), 1, &info, VK_NULL_HANDLE));
  pending_.push_back({commands, done});
  return done;
}

void Bridge::waitDone(uint64_t done) {
  if (queue12_) check(queue12_->Wait(fence12_.Get(), done), "ID3D12CommandQueue::Wait");
  else check(context11_->Wait(fence11_.Get(), done), "ID3D11DeviceContext4::Wait");
}

uint64_t Bridge::completed() const {
  // Read on the Vulkan side: it is the same counter as the Direct3D fence, and it is what tells Vulkan (and its
  // validation) that a submission's signal value has been reached.
  uint64_t value = 0;
  VK_CHECK(vkGetSemaphoreCounterValue(vulkan_->device(), semaphore_, &value));
  return value;
}

void Bridge::collect() {
  const uint64_t reached = completed();
  while (!pending_.empty() && pending_.front().done <= reached) {
    vkFreeCommandBuffers(vulkan_->device(), vulkan_->commandPool(), 1, &pending_.front().commands);
    pending_.pop_front();
  }
}

}  // namespace bridge
#endif  // _WIN32
