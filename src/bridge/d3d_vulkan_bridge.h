// A Direct3D host and a dedicated Vulkan device on the same GPU, sharing textures and one fence.
//
// The other way to run the NR graph for a Direct3D host: instead of recording it into the host's own command stream
// (src/d3d), the host hands its frame to a Vulkan device that runs the Vulkan graph, and takes the result back.
//
//   GPU        the Vulkan device is created on the physical device whose LUID is the host adapter's
//              (vk::DedicatedDevice); no other GPU is ever taken.
//   textures   allocated by Direct3D (the host's API owns them), imported by Vulkan through their NT handle
//              (VK_KHR_external_memory_win32: D3D12_RESOURCE / D3D11_TEXTURE). Vulkan moves each one to the GENERAL
//              layout once, before it holds anything, and never transitions it again, so nothing it holds is lost.
//   ordering   one fence shared by both sides (a D3D12 or D3D11 fence imported as a Vulkan timeline semaphore). The
//              host signals a value after the commands that write its textures; the Vulkan submission waits for
//              that value and signals the next; the host's later commands wait for that one. Every wait is queued
//              on the GPU side: no CPU wait, no flush of the Vulkan queue, nothing copied through the CPU.
//
// Direct3D 11 goes to Vulkan directly: no Direct3D 12 device is involved.
#pragma once
#ifdef _WIN32
#include <d3d11_4.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <deque>
#include <memory>
#include <vector>

#include "../vk_context.h"

namespace bridge {

// RGBA 32-bit float, one mip, addressed by both APIs.
struct SharedTexture {
  uint32_t width = 0, height = 0;
  Microsoft::WRL::ComPtr<ID3D12Resource> d3d12;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> d3d11;
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
};

class Bridge {
 public:
  // The host's device, and the queue (D3D12) or immediate context (D3D11) its frame is issued on.
  Bridge(ID3D12Device* device, ID3D12CommandQueue* queue, exec::Backend backend);
  Bridge(ID3D11Device* device, ID3D11DeviceContext* immediate, exec::Backend backend);
  ~Bridge();

  vk::Context& vulkan() { return *vulkan_; }
  uint64_t luid() const { return luid_; }
  const char* host() const { return queue12_ ? "d3d12" : "d3d11"; }

  SharedTexture createTexture(uint32_t width, uint32_t height);
  void destroyTexture(SharedTexture& texture);

  // Host side, in the order of its stream. signalReady: after the commands that wrote the shared textures (they
  // must already be submitted on D3D12; the D3D11 context is flushed here). Returns the value the Vulkan work waits.
  uint64_t signalReady();
  // Vulkan side: end `commands` (from vulkan().beginCommands()) and submit them to run once `ready` is reached.
  // Returns the value signalled when they have run.
  uint64_t submit(exec::Commands commands, uint64_t ready);
  // Host side: every host command issued after this runs after the Vulkan work that signals `done`.
  void waitDone(uint64_t done);
  // The fence's completed value (polled, never waited): what has actually run on the GPU.
  uint64_t completed() const;
  // Retire the command buffers of submissions the GPU has completed.
  void collect();

 private:
  void init(exec::Backend backend);
  void importFence(HANDLE handle);
  uint64_t luid_ = 0;
  std::unique_ptr<vk::Context> vulkan_;
  Microsoft::WRL::ComPtr<ID3D12Device> device12_;
  Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue12_;
  Microsoft::WRL::ComPtr<ID3D12Fence> fence12_;
  Microsoft::WRL::ComPtr<ID3D11Device5> device11_;
  Microsoft::WRL::ComPtr<ID3D11DeviceContext4> context11_;
  Microsoft::WRL::ComPtr<ID3D11Fence> fence11_;
  VkSemaphore semaphore_ = VK_NULL_HANDLE;
  uint64_t value_ = 0;   // last value handed out on the shared fence
  struct Pending { VkCommandBuffer commands; uint64_t done; };
  std::deque<Pending> pending_;
};

}  // namespace bridge
#endif  // _WIN32
