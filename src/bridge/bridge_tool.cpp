#ifdef _WIN32
#include "bridge_tool.h"

#include <d3d11_4.h>
#include <d3d11sdklayers.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>

#include "../d3d/d3d_common.h"
#include "../exec_tape.h"
#include "../gpu_arch.h"
#include "../kernels.h"
#include "../nr_graph.h"
#include "../nr_model.h"
#include "../reference.h"
#include "../vk_context.h"
#include "d3d_vulkan_bridge.h"

using Microsoft::WRL::ComPtr;

namespace bridge {

namespace {
using Clock = std::chrono::high_resolution_clock;
double since(Clock::time_point from) { return std::chrono::duration<double, std::milli>(Clock::now() - from).count(); }

void check(HRESULT hr, const char* what) {
  if (FAILED(hr)) {
    char text[200];
    snprintf(text, sizeof(text), "%s failed (0x%08lX)", what, (unsigned long)hr);
    throw std::runtime_error(text);
  }
}

constexpr uint32_t kTexelBytes = 16;   // RGBA 32-bit float

// The Direct3D side of the tool. Inside a frame it does what a game does: GPU copies between its own textures and
// the shared ones, in its own stream. The CPU only prepares the source before the frame and reads the result after
// it (to check it); neither is part of the frame.
struct Host {
  virtual ~Host() = default;
  virtual uint64_t luid() const = 0;
  virtual std::unique_ptr<Bridge> makeBridge(exec::Backend backend) = 0;
  virtual void upload(const uint8_t* texels, uint32_t width, uint32_t height) = 0;   // CPU -> the host's source texture; waits
  virtual void write(const SharedTexture& texture) = 0;    // frame: source -> shared (GPU copy, issued and submitted)
  virtual void read(const SharedTexture& texture) = 0;     // frame: shared -> the host's result texture (GPU copy)
  virtual void wait() = 0;                                  // frame: its one CPU wait, where the output is usable
  virtual std::vector<uint8_t> result() = 0;                // the host's result texture -> CPU; waits
  virtual uint32_t debugErrors() = 0;
};

struct HostD3D12 final : Host {
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12CommandAllocator> allocators[3];   // write, read, and the transfers outside the frame
  ComPtr<ID3D12GraphicsCommandList> lists[3];
  ComPtr<ID3D12Fence> fence;
  ComPtr<ID3D12Resource> source, target, staging[2];   // staging: upload, readback
  HANDLE event = nullptr;
  uint64_t value = 0, stagingBytes = 0;
  uint32_t sourceWidth = 0, sourceHeight = 0, targetWidth = 0, targetHeight = 0;
  bool debug = false;

  explicit HostD3D12(bool debugLayer) {
    if (debugLayer) {
      ComPtr<ID3D12Debug> layer;
      if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&layer)))) throw std::runtime_error("the D3D12 debug layer was asked for and is not installed");
      layer->EnableDebugLayer();
      debug = true;
    }
    check(D3D12CreateDevice(d3d::nvidiaAdapter().Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "D3D12CreateDevice");
    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    check(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
    for (int i = 0; i < 3; ++i) {
      check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocators[i])), "CreateCommandAllocator");
      check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators[i].Get(), nullptr, IID_PPV_ARGS(&lists[i])), "CreateCommandList");
      check(lists[i]->Close(), "Close");
    }
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  }
  ~HostD3D12() override { if (event) CloseHandle(event); }
  uint64_t luid() const override {
    const LUID id = device->GetAdapterLuid();
    return (uint64_t)(uint32_t)id.LowPart | ((uint64_t)(uint32_t)id.HighPart << 32);
  }
  std::unique_ptr<Bridge> makeBridge(exec::Backend backend) override { return std::make_unique<Bridge>(device.Get(), queue.Get(), backend); }

  ComPtr<ID3D12Resource> texture(uint32_t width, uint32_t height, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    ComPtr<ID3D12Resource> resource;
    check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&resource)), "CreateCommittedResource(host texture)");
    return resource;
  }
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint(ID3D12Resource* resource) {
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &placed, nullptr, nullptr, &total);
    if (total > stagingBytes) {
      stagingBytes = total;
      for (int kind = 0; kind < 2; ++kind) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = kind ? D3D12_HEAP_TYPE_READBACK : D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = total;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        staging[kind].Reset();
        check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, kind ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_GENERIC_READ,
                                              nullptr, IID_PPV_ARGS(&staging[kind])),
              "CreateCommittedResource(staging)");
      }
    }
    return placed;
  }
  ID3D12GraphicsCommandList* begin(int index) {
    check(allocators[index]->Reset(), "ID3D12CommandAllocator::Reset");
    check(lists[index]->Reset(allocators[index].Get(), nullptr), "ID3D12GraphicsCommandList::Reset");
    return lists[index].Get();
  }
  void submit(int index) {
    check(lists[index]->Close(), "Close");
    ID3D12CommandList* list[] = {lists[index].Get()};
    queue->ExecuteCommandLists(1, list);
  }
  void drain() {
    check(queue->Signal(fence.Get(), ++value), "Signal");
    wait();
  }
  static void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &barrier);
  }
  void upload(const uint8_t* texels, uint32_t width, uint32_t height) override {
    if (!source || sourceWidth != width || sourceHeight != height) {
      source = texture(width, height, D3D12_RESOURCE_STATE_COPY_SOURCE);
      sourceWidth = width; sourceHeight = height;
    }
    const D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed = footprint(source.Get());
    uint8_t* mapped = nullptr;
    check(staging[0]->Map(0, nullptr, reinterpret_cast<void**>(&mapped)), "Map(upload)");
    const size_t rowBytes = (size_t)width * kTexelBytes;
    for (uint32_t y = 0; y < height; ++y) memcpy(mapped + placed.Offset + (size_t)y * placed.Footprint.RowPitch, texels + y * rowBytes, rowBytes);
    staging[0]->Unmap(0, nullptr);
    ID3D12GraphicsCommandList* list = begin(2);
    transition(list, source.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION from{staging[0].Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    from.PlacedFootprint = placed;
    D3D12_TEXTURE_COPY_LOCATION to{source.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    transition(list, source.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    submit(2);
    drain();
  }
  void write(const SharedTexture& shared) override {
    ID3D12GraphicsCommandList* list = begin(0);
    transition(list, shared.d3d12.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyResource(shared.d3d12.Get(), source.Get());
    transition(list, shared.d3d12.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    submit(0);
  }
  void read(const SharedTexture& shared) override {
    if (!target || targetWidth != shared.width || targetHeight != shared.height) {
      target = texture(shared.width, shared.height, D3D12_RESOURCE_STATE_COPY_DEST);
      targetWidth = shared.width; targetHeight = shared.height;
    }
    ID3D12GraphicsCommandList* list = begin(1);
    transition(list, shared.d3d12.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyResource(target.Get(), shared.d3d12.Get());
    transition(list, shared.d3d12.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    submit(1);
    check(queue->Signal(fence.Get(), ++value), "Signal");
  }
  void wait() override {
    if (fence->GetCompletedValue() < value) {
      check(fence->SetEventOnCompletion(value, event), "SetEventOnCompletion");
      if (WaitForSingleObject(event, 120000) != WAIT_OBJECT_0) throw std::runtime_error("the d3d12 host stream did not complete within 120 s");
    }
    check(device->GetDeviceRemovedReason(), "the d3d12 host device was removed");
  }
  std::vector<uint8_t> result() override {
    const D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed = footprint(target.Get());
    ID3D12GraphicsCommandList* list = begin(2);
    transition(list, target.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    D3D12_TEXTURE_COPY_LOCATION to{staging[1].Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    to.PlacedFootprint = placed;
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    transition(list, target.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    submit(2);
    drain();
    const size_t rowBytes = (size_t)targetWidth * kTexelBytes;
    std::vector<uint8_t> texels(rowBytes * targetHeight);
    uint8_t* mapped = nullptr;
    check(staging[1]->Map(0, nullptr, reinterpret_cast<void**>(&mapped)), "Map(readback)");
    for (uint32_t y = 0; y < targetHeight; ++y) memcpy(texels.data() + y * rowBytes, mapped + placed.Offset + (size_t)y * placed.Footprint.RowPitch, rowBytes);
    const D3D12_RANGE none{0, 0};
    staging[1]->Unmap(0, &none);
    return texels;
  }
  uint32_t debugErrors() override {
    ComPtr<ID3D12InfoQueue> info;
    if (!debug || FAILED(device.As(&info))) return 0;
    uint32_t errors = 0;
    for (UINT64 index = 0; index < info->GetNumStoredMessages(); ++index) {
      SIZE_T size = 0;
      info->GetMessage(index, nullptr, &size);
      std::vector<uint8_t> storage(size);
      auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
      if (SUCCEEDED(info->GetMessage(index, message, &size)) && message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
        if (errors < 8) fprintf(stderr, "[d3d12 host] %s\n", message->pDescription);
        ++errors;
      }
    }
    return errors;
  }
};

struct HostD3D11 final : Host {
  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  ComPtr<ID3D11Texture2D> source, target, staging;
  ComPtr<ID3D11Query> marker;
  uint32_t sourceWidth = 0, sourceHeight = 0, targetWidth = 0, targetHeight = 0;
  bool debug = false;

  explicit HostD3D11(bool debugLayer) : debug(debugLayer) {
    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL level{};
    const HRESULT hr = D3D11CreateDevice(d3d::nvidiaAdapter().Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, debugLayer ? D3D11_CREATE_DEVICE_DEBUG : 0, levels, 2,
                                         D3D11_SDK_VERSION, &device, &level, &context);
    if (FAILED(hr) && debugLayer) throw std::runtime_error("the D3D11 debug layer was asked for and is not installed");
    check(hr, "D3D11CreateDevice");
    D3D11_QUERY_DESC desc{D3D11_QUERY_EVENT, 0};
    check(device->CreateQuery(&desc, &marker), "CreateQuery(event)");
  }
  uint64_t luid() const override {
    ComPtr<IDXGIDevice> dxgi;
    ComPtr<IDXGIAdapter> adapter;
    DXGI_ADAPTER_DESC desc{};
    device.As(&dxgi);
    dxgi->GetAdapter(&adapter);
    adapter->GetDesc(&desc);
    return (uint64_t)(uint32_t)desc.AdapterLuid.LowPart | ((uint64_t)(uint32_t)desc.AdapterLuid.HighPart << 32);
  }
  std::unique_ptr<Bridge> makeBridge(exec::Backend backend) override { return std::make_unique<Bridge>(device.Get(), context.Get(), backend); }
  ComPtr<ID3D11Texture2D> texture(uint32_t width, uint32_t height, bool cpuRead) {
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = cpuRead ? D3D11_USAGE_STAGING : D3D11_USAGE_DEFAULT;
    desc.BindFlags = cpuRead ? 0 : D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = cpuRead ? D3D11_CPU_ACCESS_READ : 0;
    ComPtr<ID3D11Texture2D> resource;
    check(device->CreateTexture2D(&desc, nullptr, &resource), "CreateTexture2D(host texture)");
    return resource;
  }
  void upload(const uint8_t* texels, uint32_t width, uint32_t height) override {
    if (!source || sourceWidth != width || sourceHeight != height) {
      source = texture(width, height, false);
      sourceWidth = width; sourceHeight = height;
    }
    context->UpdateSubresource(source.Get(), 0, nullptr, texels, width * kTexelBytes, 0);
    context->End(marker.Get());
    wait();
  }
  void write(const SharedTexture& shared) override { context->CopyResource(shared.d3d11.Get(), source.Get()); }
  void read(const SharedTexture& shared) override {
    if (!target || targetWidth != shared.width || targetHeight != shared.height) {
      target = texture(shared.width, shared.height, false);
      staging = texture(shared.width, shared.height, true);
      targetWidth = shared.width; targetHeight = shared.height;
    }
    context->CopyResource(target.Get(), shared.d3d11.Get());
    context->End(marker.Get());
  }
  void wait() override {
    context->Flush();
    BOOL done = FALSE;
    const ULONGLONG start = GetTickCount64();
    while (context->GetData(marker.Get(), &done, sizeof(done), 0) != S_OK || !done) {
      if (GetTickCount64() - start > 120000) throw std::runtime_error("the d3d11 host stream did not complete within 120 s");
      SwitchToThread();
    }
    check(device->GetDeviceRemovedReason(), "the d3d11 host device was removed");
  }
  std::vector<uint8_t> result() override {
    context->CopyResource(staging.Get(), target.Get());
    D3D11_MAPPED_SUBRESOURCE mapped{};
    check(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map(staging)");
    const size_t rowBytes = (size_t)targetWidth * kTexelBytes;
    std::vector<uint8_t> texels(rowBytes * targetHeight);
    for (uint32_t y = 0; y < targetHeight; ++y) memcpy(texels.data() + y * rowBytes, static_cast<const uint8_t*>(mapped.pData) + (size_t)y * mapped.RowPitch, rowBytes);
    context->Unmap(staging.Get(), 0);
    return texels;
  }
  uint32_t debugErrors() override {
    ComPtr<ID3D11InfoQueue> info;
    if (!debug || FAILED(device.As(&info))) return 0;
    uint32_t errors = 0;
    for (UINT64 index = 0; index < info->GetNumStoredMessages(); ++index) {
      SIZE_T size = 0;
      info->GetMessage(index, nullptr, &size);
      std::vector<uint8_t> storage(size);
      auto* message = reinterpret_cast<D3D11_MESSAGE*>(storage.data());
      if (SUCCEEDED(info->GetMessage(index, message, &size)) && message->Severity <= D3D11_MESSAGE_SEVERITY_ERROR) {
        if (errors < 8) fprintf(stderr, "[d3d11 host] %s\n", message->pDescription);
        ++errors;
      }
    }
    return errors;
  }
};

double median(std::vector<double> values) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void writeList(std::ofstream& json, const char* name, const std::vector<double>& values, bool last = false) {
  json << "  \"" << name << "\": [";
  for (size_t i = 0; i < values.size(); ++i) json << (i ? ", " : "") << values[i];
  json << "]" << (last ? "\n" : ",\n");
}
}  // namespace

ToolResult runTool(const ToolInput& input) {
  ToolResult result;
  std::unique_ptr<Host> host;
  if (input.host == "d3d12") host = std::make_unique<HostD3D12>(input.debugLayer);
  else if (input.host == "d3d11") host = std::make_unique<HostD3D11>(input.debugLayer);
  else throw std::runtime_error("--host must be d3d12 or d3d11");

  // The numeric route: the one asked for (DLSS5VK_BACKEND), else the one the GPU's compute capability allows.
  const d3d::GpuInfo gpu = d3d::gpuInfo(host->luid());
  const std::optional<vk::Backend> requested = vk::requestedBackend();
  const exec::Backend backend = requested ? *requested : (gpu.major > 8 || (gpu.major == 8 && gpu.minor >= 9)) ? exec::Backend::Native : exec::Backend::Sm86;

  const auto prepareStart = Clock::now();
  std::unique_ptr<Bridge> link = host->makeBridge(backend);
  vk::Context& vulkan = link->vulkan();
  printf("bridge: %s host on LUID %016llX -> dedicated Vulkan device %s, %s backend%s\n", input.host.c_str(), (unsigned long long)link->luid(),
         vulkan.deviceName().c_str(), exec::backendName(vulkan.backend()), input.transportOnly ? " (transport only)" : "");

  std::vector<double> hostWriteMs, vulkanCpuMs, hostReadMs, wallMs, gpuTotalMs, gpuGraphMs;
  uint32_t failures = 0;
  auto require = [&](bool ok, const std::string& what) {
    printf("%s: %s\n", what.c_str(), ok ? "identical" : "DIFFERENT");
    if (!ok) ++failures;
  };

  if (input.transportOnly) {
    const uint32_t width = input.proxyWidth, height = input.proxyHeight;
    SharedTexture source = link->createTexture(width, height), target = link->createTexture(width, height);
    // The moved rectangle: a quarter of the image, from the upper left region to the lower right one.
    const uint32_t rectX = width / 8, rectY = height / 8, rectW = std::max(1u, width / 4), rectH = std::max(1u, height / 4);
    const uint32_t toX = width / 2, toY = height / 2;
    std::vector<uint8_t> previous;
    for (int frame = 0; frame < input.frames; ++frame) {
      std::vector<float> texels((size_t)width * height * 4);
      uint32_t state = 0x9e3779b9u * (uint32_t)(frame + 1);
      for (float& value : texels) { state = state * 1664525u + 1013904223u; value = (float)(state >> 8) * (1.0f / 16777216.0f) - 0.5f; }
      std::vector<float> expected = texels;
      for (uint32_t y = 0; y < rectH && toY + y < height; ++y)
        for (uint32_t x = 0; x < rectW && toX + x < width; ++x)
          memcpy(&expected[((size_t)(toY + y) * width + toX + x) * 4], &texels[((size_t)(rectY + y) * width + rectX + x) * 4], kTexelBytes);
      host->upload(reinterpret_cast<const uint8_t*>(texels.data()), width, height);   // before the frame
      const auto frameStart = Clock::now();
      host->write(source);
      const uint64_t ready = link->signalReady();
      hostWriteMs.push_back(since(frameStart));
      const auto vulkanStart = Clock::now();
      exec::Commands commands = vulkan.beginCommands();
      VkImageCopy whole{};
      whole.srcSubresource = whole.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      whole.extent = {width, height, 1};
      vkCmdCopyImage(vk::handle(commands), source.image, VK_IMAGE_LAYOUT_GENERAL, target.image, VK_IMAGE_LAYOUT_GENERAL, 1, &whole);
      VkImageCopy part = whole;
      part.srcOffset = {(int32_t)rectX, (int32_t)rectY, 0};
      part.dstOffset = {(int32_t)toX, (int32_t)toY, 0};
      part.extent = {std::min(rectW, width - toX), std::min(rectH, height - toY), 1};
      VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      vkCmdPipelineBarrier(vk::handle(commands), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
      vkCmdCopyImage(vk::handle(commands), source.image, VK_IMAGE_LAYOUT_GENERAL, target.image, VK_IMAGE_LAYOUT_GENERAL, 1, &part);
      const uint64_t done = link->submit(commands, ready);
      vulkanCpuMs.push_back(since(vulkanStart));
      const auto readStart = Clock::now();
      link->waitDone(done);
      host->read(target);
      hostReadMs.push_back(since(readStart));
      host->wait();
      wallMs.push_back(since(frameStart));
      link->collect();
      const std::vector<uint8_t> got = host->result();   // after the frame
      size_t differing = 0;
      for (size_t i = 0; i < expected.size(); ++i) {
        uint32_t a, b;
        memcpy(&a, &expected[i], 4);
        memcpy(&b, got.data() + i * 4, 4);
        differing += a != b;
      }
      if (differing) { printf("frame %d: %zu of %zu values differ from what Direct3D wrote and Vulkan moved\n", frame, differing, expected.size()); ++failures; }
      result.head = got;
    }
    printf("transport %ux%u RGBA32F, %d frames: Direct3D wrote, Vulkan copied the image and moved a %ux%u rectangle from (%u,%u) to (%u,%u), Direct3D read: %s\n",
           width, height, input.frames, rectW, rectH, rectX, rectY, toX, toY, failures ? "MISMATCH" : "every texel as expected");
    link->destroyTexture(source);
    link->destroyTexture(target);
  } else {
    if (input.sameKernels) vulkan.setShaderFp8(false);
    nr::Model model(vulkan, input.modelDir, false);
    exec::TapeRecorder recorder(vulkan);
    nr::Kernels kernels(recorder, input.shaderDir);
    kernels.setChainEnabled(false);   // the host schedule: a barrier after every launch, as an embedded graph runs
    kernels.setSiluTable(ref::siluTable());
    const nr::Geometry geometry = nr::Geometry::fromValid(input.validWidth, input.validHeight);
    if (geometry.fullWidth != input.fullWidth || geometry.fullHeight != input.fullHeight) throw std::runtime_error("fixture full dimensions disagree with the runtime profile");
    const uint32_t fullRows = geometry.fullWidth * geometry.fullHeight;
    nr::Graph graph(recorder, model, kernels, geometry, nr::Graph::Options{});
    nr::Activation* features = graph.allocate("input features", fullRows, 16, nr::Format::F32);
    exec::Buffer proxyBuffer = vulkan.createBuffer((exec::Size)input.proxyWidth * input.proxyHeight * kTexelBytes, false, "proxy");
    const nr::Kernels::PreprocessArgs preprocess{geometry.fullWidth, geometry.fullHeight, input.validWidth, input.validHeight, input.proxyWidth,
                                                 input.proxyHeight, input.seed, input.autoMask, input.localTone, input.localStructure,
                                                 input.skinStructure, input.style};
    kernels.preprocessFromProxy(recorder.stream(), proxyBuffer, *features, preprocess);
    const exec::Tape preprocessTape = recorder.take();
    graph.record(recorder.stream(), *features);
    const exec::Tape graphTape = recorder.take();
    SharedTexture proxyTexture = link->createTexture(input.proxyWidth, input.proxyHeight);
    SharedTexture headTexture = link->createTexture(geometry.fullWidth, geometry.fullHeight);
    exec::Timer stamps = vulkan.createTimestampPool(4);
    const double prepareSeconds = since(prepareStart) / 1000.0;
    printf("preparation %.2f s (device, model, kernels, graph, shared textures; kernel compilation falls in the first frame)\n", prepareSeconds);
    printf("graph: %zu launches and %zu dispatches per frame, %s kernels, chaining off\n", graphTape.launches(), graphTape.dispatches(),
           vulkan.nativeFp8() ? "cooperative-matrix GLSL and PTX" : "PTX and exact scalar");

    host->upload(input.proxy.data(), input.proxyWidth, input.proxyHeight);   // the host's own image, once
    std::vector<uint8_t> first;
    for (int frame = 0; frame < input.frames; ++frame) {
      vulkan.nextFrame();
      const auto frameStart = Clock::now();
      host->write(proxyTexture);
      const uint64_t ready = link->signalReady();
      hostWriteMs.push_back(since(frameStart));

      const auto vulkanStart = Clock::now();
      exec::Commands commands = vulkan.beginCommands();
      vulkan.resetTimestamps(commands, stamps, 4);
      vulkan.writeTimestamp(commands, stamps, 0, true);
      VkBufferImageCopy proxyRegion{};
      proxyRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      proxyRegion.imageExtent = {input.proxyWidth, input.proxyHeight, 1};
      vkCmdCopyImageToBuffer(vk::handle(commands), proxyTexture.image, VK_IMAGE_LAYOUT_GENERAL, vk::handle(proxyBuffer), 1, &proxyRegion);
      vulkan.transferBarrier(commands);
      preprocessTape.replay(vulkan, commands);
      vulkan.writeTimestamp(commands, stamps, 1, false);
      graphTape.replay(vulkan, commands);
      vulkan.writeTimestamp(commands, stamps, 2, false);
      vulkan.transferBarrier(commands);
      VkBufferImageCopy headRegion{};
      headRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      headRegion.imageExtent = {geometry.fullWidth, geometry.fullHeight, 1};
      vkCmdCopyBufferToImage(vk::handle(commands), vk::handle(graph.head().buffer), headTexture.image, VK_IMAGE_LAYOUT_GENERAL, 1, &headRegion);
      vulkan.writeTimestamp(commands, stamps, 3, false);
      const uint64_t done = link->submit(commands, ready);
      vulkanCpuMs.push_back(since(vulkanStart));

      const auto readStart = Clock::now();
      link->waitDone(done);
      host->read(headTexture);
      hostReadMs.push_back(since(readStart));
      host->wait();
      wallMs.push_back(since(frameStart));
      link->collect();
      std::vector<uint8_t> head = host->result();   // after the frame
      const std::vector<double> times = vulkan.readTimestampsMs(stamps, 4);
      gpuTotalMs.push_back(times[3] - times[0]);
      gpuGraphMs.push_back(times[2] - times[1]);
      if (frame == 0) first = head;
      else if (head != first) { printf("frame %d: the head Direct3D read differs from frame 0\n", frame); ++failures; }
      result.head = std::move(head);
    }
    require(failures == 0, "head read by Direct3D over " + std::to_string(input.frames) + " frames");
    // The transport is exact when Direct3D read what the Vulkan graph holds.
    require(vulkan.download(graph.head().buffer, graph.head().validBytes()) == result.head, "head read by Direct3D vs the Vulkan head buffer");
    if (!input.referenceHead.empty()) {
      size_t differing = 0;
      const size_t values = result.head.size() / 4;
      for (size_t i = 0; i < values && i * 4 + 4 <= input.referenceHead.size(); ++i) differing += memcmp(result.head.data() + i * 4, input.referenceHead.data() + i * 4, 4) != 0;
      printf("head vs reference: %s (%zu of %zu values differ)\n", differing ? "MISMATCH" : "bit-exact", differing, values);
    }
    if (!input.dumpDir.empty())
      std::ofstream(input.dumpDir + "/head.f32", std::ios::binary).write(reinterpret_cast<const char*>(result.head.data()), result.head.size());
    vulkan.destroyTimestampPool(stamps);
    vulkan.destroyBuffer(proxyBuffer);
    link->destroyTexture(proxyTexture);
    link->destroyTexture(headTexture);
    printf("Vulkan GPU: median %.3f ms for the whole submission, %.3f ms for the graph alone\n", median(gpuTotalMs), median(gpuGraphMs));
  }
  printf("frame: median %.3f ms to a usable output (host copy in %.3f ms + Vulkan record and submit %.3f ms + host copy out %.3f ms of CPU, the rest GPU)\n",
         median(wallMs), median(hostWriteMs), median(vulkanCpuMs), median(hostReadMs));
  const uint32_t hostErrors = host->debugErrors();
  if (input.debugLayer) printf("Direct3D debug layer: %u error%s\n", hostErrors, hostErrors == 1 ? "" : "s");
  if (hostErrors) ++failures;

  if (!input.jsonPath.empty()) {
    std::ofstream json(input.jsonPath, std::ios::binary);
    json << "{\n  \"mode\": \"" << (input.transportOnly ? "transport" : "bridge") << "\", \"host\": \"" << input.host << "\", \"api\": \"vulkan\",\n";
    json << "  \"backend\": \"" << exec::backendName(vulkan.backend()) << "\", \"device\": \"" << vulkan.deviceName() << "\",\n";
    json << "  \"shader_fp8\": " << (vulkan.nativeFp8() ? "true" : "false") << ", \"taped\": true, \"chained\": false,\n";
    json << "  \"width\": " << input.validWidth << ", \"height\": " << input.validHeight << ", \"full_width\": " << input.fullWidth
         << ", \"full_height\": " << input.fullHeight << ", \"transport_width\": " << input.proxyWidth << ", \"transport_height\": " << input.proxyHeight << ",\n";
    const exec::Device::MemoryUse memory = vulkan.memoryUse();
    json << "  \"memory_bytes\": {\"device_local\": " << memory.deviceLocal << ", \"peak_device_local\": " << memory.peakDeviceLocal
         << ", \"host_visible\": " << memory.hostVisible << "},\n";
    json << "  \"debug_errors\": " << hostErrors << ", \"failures\": " << failures << ",\n";
    writeList(json, "gpu_ms", gpuGraphMs);
    writeList(json, "gpu_submission_ms", gpuTotalMs);
    writeList(json, "host_write_ms", hostWriteMs);
    writeList(json, "record_ms", vulkanCpuMs);
    writeList(json, "host_read_ms", hostReadMs);
    writeList(json, "host_frame_ms", wallMs, true);
    json << "}\n";
  }
  result.passed = failures == 0;
  return result;
}

}  // namespace bridge
#endif  // _WIN32
