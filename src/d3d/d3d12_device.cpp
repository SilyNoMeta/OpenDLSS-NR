#ifdef _WIN32
#include "d3d12_device.h"

#include <d3d12sdklayers.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <cstring>

using Microsoft::WRL::ComPtr;

namespace d3d {

namespace {
constexpr exec::Size kStagingBytes = 64ull << 20;   // upload / readback window
constexpr UINT kConstantsParameter = exec::kGenericBindings;   // root parameters 0..11 are the UAVs

struct ShaderModule { ShaderSource source; };
struct PipelineState { ComPtr<ID3D12PipelineState> state; };
struct Module { nvcuda::Handle handle = nullptr; std::string ptx; };
struct Function { nvcuda::Handle handle = nullptr; KernelSignature signature; };
struct TimerPool { ComPtr<ID3D12QueryHeap> heap; ComPtr<ID3D12Resource> readback; uint32_t count = 0; };

ID3D12GraphicsCommandList* listOf(exec::Commands commands) { return static_cast<ID3D12GraphicsCommandList*>(commands.stream); }
ID3D12Resource* resourceOf(const exec::Buffer& buffer) { return static_cast<ID3D12Resource*>(buffer.buffer); }

void check(HRESULT hr, const char* what) {
  if (FAILED(hr)) {
    char text[200];
    snprintf(text, sizeof(text), "%s failed (0x%08lX)", what, (unsigned long)hr);
    throw std::runtime_error(text);
  }
}
void checkNv(nvcuda::Status status, const std::string& what) {
  if (status != 0) throw std::runtime_error(what + " failed (NvAPI status " + std::to_string(status) + ")");
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
  list->ResourceBarrier(1, &barrier);
}
void uavBarrier(ID3D12GraphicsCommandList* list) {
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;   // every pending unordered access completes first
  list->ResourceBarrier(1, &barrier);
}
constexpr D3D12_RESOURCE_STATES kHome = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
}  // namespace

struct D3D12Device::Stream {
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> list;
  uint64_t fence = 0;   // the queue's fence value that retires it
  bool open = false;
};

D3D12Device::D3D12Device(exec::Backend backend, bool debugLayer) : backend_(backend) {
  ComPtr<IDXGIFactory6> factory;
  check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
  ComPtr<IDXGIAdapter1> adapter, chosen;
  for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    if (desc.VendorId == 0x10de && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { chosen = adapter; break; }
    adapter.Reset();
  }
  if (!chosen) throw std::runtime_error("no NVIDIA adapter for the d3d12 execution");
  if (debugLayer) {
    ComPtr<ID3D12Debug> layer;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&layer)))) { layer->EnableDebugLayer(); debugLayer_ = true; }
    else throw std::runtime_error("the D3D12 debug layer was asked for and is not installed (Graphics Tools)");
  }
  check(D3D12CreateDevice(chosen.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device_)), "D3D12CreateDevice");
  init(debugLayer);
}

D3D12Device::D3D12Device(ID3D12Device* device, exec::Backend backend) : device_(device), backend_(backend) {
  if (!device) throw std::runtime_error("a borrowed d3d12 execution needs the host's device");
  init(false);
}

void D3D12Device::init(bool) {
  const LUID luid = device_->GetAdapterLuid();
  gpu_ = gpuInfo((uint64_t)(uint32_t)luid.LowPart | ((uint64_t)(uint32_t)luid.HighPart << 32));
  requireBackend(backend_, gpu_, "d3d12");
  std::string error;
  nv_ = nvcuda::load(error);
  if (!nv_) throw std::runtime_error(error);
  if (!nv_->d3d12()) throw std::runtime_error("the driver does not resolve the D3D12 CuModule / CuFunction / LaunchCuKernelChain interfaces");

  D3D12_COMMAND_QUEUE_DESC queueDesc{};
  check(device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue_)), "CreateCommandQueue");
  check(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "CreateFence");
  fenceEvent_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);

  D3D12_ROOT_PARAMETER parameters[exec::kGenericBindings + 1]{};
  for (UINT index = 0; index < exec::kGenericBindings; ++index) {
    parameters[index].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    parameters[index].Descriptor.ShaderRegister = index;
  }
  parameters[kConstantsParameter].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  parameters[kConstantsParameter].Constants.Num32BitValues = exec::kPushConstantBytes / 4;
  D3D12_ROOT_SIGNATURE_DESC rootDesc{exec::kGenericBindings + 1, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
  ComPtr<ID3DBlob> blob, errors;
  check(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors), "D3D12SerializeRootSignature");
  check(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rootSignature_)), "CreateRootSignature");

  dummy_ = createBuffer(256, false, "dummy binding");
  fillZero(dummy_);
}

D3D12Device::~D3D12Device() {
  if (!device_) return;
  if (queue_ && fence_ && fenceValue_) {
    // Everything this object submitted has completed before its resources go.
    try { waitFence(fenceValue_); } catch (...) {}
  }
  addDebugErrors(debugErrors());
  destroyBuffer(dummy_);
  destroyBuffer(zeros_);
  if (uploadStaging_ && uploadMapped_) uploadStaging_->Unmap(0, nullptr);
  if (fenceEvent_) CloseHandle(fenceEvent_);
}

uint32_t D3D12Device::debugErrors() const {
  ComPtr<ID3D12InfoQueue> info;
  if (!debugLayer_ || FAILED(device_.As(&info))) return 0;
  uint32_t errors = 0;
  for (UINT64 index = 0; index < info->GetNumStoredMessages(); ++index) {
    SIZE_T size = 0;
    info->GetMessage(index, nullptr, &size);
    std::vector<uint8_t> storage(size);
    auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
    if (SUCCEEDED(info->GetMessage(index, message, &size)) && message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
      if (errors < 8) fprintf(stderr, "[d3d12] %s\n", message->pDescription);
      ++errors;
    }
  }
  return errors;
}

void D3D12Device::waitFence(uint64_t value) {
  if (fence_->GetCompletedValue() < value) {
    check(fence_->SetEventOnCompletion(value, fenceEvent_), "SetEventOnCompletion");
    if (WaitForSingleObject(fenceEvent_, 120000) != WAIT_OBJECT_0) throw std::runtime_error("the d3d12 queue did not complete within 120 s");
  }
  check(device_->GetDeviceRemovedReason(), "the d3d12 device was removed");
}

ComPtr<ID3D12Resource> D3D12Device::allocate(exec::Size size, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
  D3D12_HEAP_PROPERTIES properties{};
  properties.Type = heap;
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = size;
  desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags = flags;
  ComPtr<ID3D12Resource> resource;
  check(device_->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&resource)), "CreateCommittedResource");
  return resource;
}

exec::Buffer D3D12Device::createBuffer(exec::Size size, bool hostVisible, const char* label, uint32_t) {
  // Host-visible is a request the kernels' buffers never depend on here: an UPLOAD / READBACK heap cannot hold an
  // unordered-access buffer, so it is device-local and unmapped (chainSupported() is false for that reason).
  exec::Buffer result;
  result.size = std::max<exec::Size>(size, 16);
  result.hostVisible = hostVisible;
  result.label = label;
  const exec::Size width = (result.size + 15) & ~exec::Size(15);
  ComPtr<ID3D12Resource> resource = allocate(width, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, kHome);
  const D3D12_RESOURCE_DESC desc = resource->GetDesc();
  result.allocation = device_->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes;
  memoryUse_.deviceLocal += result.allocation;
  memoryUse_.peakDeviceLocal = std::max(memoryUse_.peakDeviceLocal, memoryUse_.deviceLocal);
  result.buffer = resource.Detach();   // the Buffer owns the reference until destroyBuffer
  return result;
}

void D3D12Device::destroyBuffer(exec::Buffer& buffer) {
  if (buffer.buffer) {
    memoryUse_.deviceLocal -= buffer.allocation;
    resourceOf(buffer)->Release();
  }
  buffer = exec::Buffer{};
}

exec::Address D3D12Device::deviceAddress(const exec::Buffer& buffer) const { return resourceOf(buffer)->GetGPUVirtualAddress(); }

void D3D12Device::upload(const exec::Buffer& target, const void* data, exec::Size size, exec::Size offset) {
  if (offset + size > target.size) throw std::runtime_error(std::string("upload overflows ") + target.label);
  if (!uploadStaging_) {
    uploadStaging_ = allocate(kStagingBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    check(uploadStaging_->Map(0, nullptr, &uploadMapped_), "Map(upload staging)");
  }
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  for (exec::Size done = 0; done < size;) {
    const exec::Size chunk = std::min(size - done, kStagingBytes);
    if (bytes) memcpy(uploadMapped_, bytes + done, chunk);
    else memset(uploadMapped_, 0, chunk);
    exec::Commands commands = beginCommands();
    ID3D12GraphicsCommandList* list = listOf(commands);
    transition(list, resourceOf(target), kHome, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(resourceOf(target), offset + done, uploadStaging_.Get(), 0, chunk);
    transition(list, resourceOf(target), D3D12_RESOURCE_STATE_COPY_DEST, kHome);
    endAndSubmit(commands, true);
    done += chunk;
  }
}

void D3D12Device::fillZero(const exec::Buffer& target) { upload(target, nullptr, target.size, 0); }

std::vector<uint8_t> D3D12Device::download(const exec::Buffer& source, exec::Size size, exec::Size offset) {
  if (offset + size > source.size) throw std::runtime_error(std::string("download overflows ") + source.label);
  if (!readbackStaging_) readbackStaging_ = allocate(kStagingBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  std::vector<uint8_t> result(size);
  for (exec::Size done = 0; done < size;) {
    const exec::Size chunk = std::min(size - done, kStagingBytes);
    exec::Commands commands = beginCommands();
    ID3D12GraphicsCommandList* list = listOf(commands);
    transition(list, resourceOf(source), kHome, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readbackStaging_.Get(), 0, resourceOf(source), offset + done, chunk);
    transition(list, resourceOf(source), D3D12_RESOURCE_STATE_COPY_SOURCE, kHome);
    endAndSubmit(commands, true);
    void* mapped = nullptr;
    const D3D12_RANGE range{0, (SIZE_T)chunk};
    check(readbackStaging_->Map(0, &range, &mapped), "Map(readback staging)");
    memcpy(result.data() + done, mapped, chunk);
    const D3D12_RANGE none{0, 0};
    readbackStaging_->Unmap(0, &none);
    done += chunk;
  }
  return result;
}

exec::Shader D3D12Device::loadShaderModule(const std::string& path) { return new ShaderModule{shaderSourceFor(path)}; }

exec::Pipeline D3D12Device::createComputePipeline(exec::Shader module, const exec::SpecConstants& constants, const char* label, uint32_t) {
  const std::vector<uint8_t> code = compileShader(static_cast<ShaderModule*>(module)->source, constants);
  D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
  desc.pRootSignature = rootSignature_.Get();
  desc.CS = {code.data(), code.size()};
  auto state = std::make_unique<PipelineState>();
  check(device_->CreateComputePipelineState(&desc, IID_PPV_ARGS(&state->state)), "CreateComputePipelineState");
  exec::Pipeline result;
  result.label = label;
  result.pipeline = state.release();
  return result;
}

void D3D12Device::destroyPipeline(exec::Pipeline& pipeline) {
  delete static_cast<PipelineState*>(pipeline.pipeline);
  pipeline = exec::Pipeline{};
}

void D3D12Device::dispatch(exec::Commands commands, exec::PipelineHandle pipeline, const exec::Buffer* const bindings[exec::kGenericBindings],
                           const void* push, uint32_t pushBytes, uint32_t x, uint32_t y, uint32_t z) {
  if (pushBytes > exec::kPushConstantBytes) throw std::runtime_error("push constants exceed 128 bytes");
  ID3D12GraphicsCommandList* list = listOf(commands);
  // A kernel launch may leave its own compute state behind: set ours every time.
  list->SetComputeRootSignature(rootSignature_.Get());
  list->SetPipelineState(static_cast<PipelineState*>(pipeline)->state.Get());
  for (UINT index = 0; index < exec::kGenericBindings; ++index)
    list->SetComputeRootUnorderedAccessView(index, resourceOf(bindings[index] ? *bindings[index] : dummy_)->GetGPUVirtualAddress());
  uint32_t constants[exec::kPushConstantBytes / 4]{};
  memcpy(constants, push, pushBytes);
  list->SetComputeRoot32BitConstants(kConstantsParameter, exec::kPushConstantBytes / 4, constants, 0);
  list->Dispatch(x, y, z);
  ++dispatches_;
}

exec::KernelModule D3D12Device::createCudaModule(const std::string& ptx) {
  auto module = std::make_unique<Module>();
  module->ptx = ptx;
  // The driver takes PTX as a C string, terminator included in the size.
  checkNv(nv_->d3d12CreateCuModule(device_.Get(), module->ptx.c_str(), (uint32_t)module->ptx.size() + 1, &module->handle), "NvAPI_D3D12_CreateCuModule");
  if (!module->handle) throw std::runtime_error("NvAPI_D3D12_CreateCuModule returned no module");
  return module.release();
}

exec::Kernel D3D12Device::createCudaFunction(exec::KernelModule module, const char* name) {
  auto function = std::make_unique<Function>();
  function->signature = parseSignature(static_cast<Module*>(module)->ptx, name);
  checkNv(nv_->d3d12CreateCuFunction(device_.Get(), static_cast<Module*>(module)->handle, name, &function->handle),
          std::string("NvAPI_D3D12_CreateCuFunction(") + name + ")");
  if (!function->handle) throw std::runtime_error(std::string("NvAPI_D3D12_CreateCuFunction returned no function for ") + name);
  return function.release();
}

void D3D12Device::destroyCudaFunction(exec::Kernel function) {
  if (!function) return;
  nv_->d3d12DestroyCuFunction(device_.Get(), static_cast<Function*>(function)->handle);
  delete static_cast<Function*>(function);
}

void D3D12Device::destroyCudaModule(exec::KernelModule module) {
  if (!module) return;
  nv_->d3d12DestroyCuModule(device_.Get(), static_cast<Module*>(module)->handle);
  delete static_cast<Module*>(module);
}

void D3D12Device::cudaLaunch(exec::Commands commands, exec::Kernel function, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                             uint32_t blockX, uint32_t sharedBytes, const void* const* params, size_t paramCount) {
  const Function* kernel = static_cast<const Function*>(function);
  const nvcuda::Params packed = packArguments(kernel->signature, params, paramCount);
  // Diagnostics: DLSS5VK_D3D_TRACE=1 names every launch, DLSS5VK_D3D_MAX_LAUNCHES=N records only the first N
  // (to find the launch a hang or a wrong result starts at).
  static const bool trace = getenv("DLSS5VK_D3D_TRACE") != nullptr;
  static const long long limit = getenv("DLSS5VK_D3D_MAX_LAUNCHES") ? atoll(getenv("DLSS5VK_D3D_MAX_LAUNCHES")) : -1;
  if (trace)
    fprintf(stderr, "[d3d12] launch %llu %s grid %ux%ux%u block %u shared %u args %u bytes\n", (unsigned long long)launches_,
            kernel->signature.entry.c_str(), gridX, gridY, gridZ, blockX, sharedBytes, packed.size());
  if (limit >= 0 && (long long)launches_ >= limit) { ++launches_; return; }
  nvcuda::KernelLaunch launch{};
  launch.function = kernel->handle;
  launch.grid = {gridX, gridY, gridZ};
  launch.block = {blockX, 1, 1};
  launch.dynamicSharedBytes = sharedBytes;
  launch.params = packed.data();   // copied by the driver while recording (tests/d3d/cuda_probe.cpp --scribble)
  launch.paramBytes = packed.size();
  checkNv(nv_->d3d12LaunchCuKernelChain(listOf(commands), &launch, 1), "NvAPI_D3D12_LaunchCuKernelChain(" + kernel->signature.entry + ")");
  ++launches_;
}

void D3D12Device::computeBarrier(exec::Commands commands) { uavBarrier(listOf(commands)); }
void D3D12Device::transferBarrier(exec::Commands commands) { uavBarrier(listOf(commands)); }

const exec::Buffer& D3D12Device::zeros(exec::Size size) {
  if (zeros_.size < size) {
    // Lists already recorded may still copy from the old one: it is retired with this object, not here.
    if (zeros_.buffer) throw std::runtime_error("the zero source would have to grow after it was recorded");
    zeros_ = createBuffer(std::max<exec::Size>(size, 1u << 20), false, "zero source");
    fillZero(zeros_);
  }
  return zeros_;
}

void D3D12Device::zeroBuffer(exec::Commands commands, const exec::Buffer& target) { copyBuffer(commands, zeros(target.size), target, target.size); }

void D3D12Device::copyBuffer(exec::Commands commands, const exec::Buffer& source, const exec::Buffer& target, exec::Size bytes) {
  ID3D12GraphicsCommandList* list = listOf(commands);
  D3D12_RESOURCE_BARRIER barriers[2]{};
  barriers[0].Type = barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barriers[0].Transition = {resourceOf(source), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, kHome, D3D12_RESOURCE_STATE_COPY_SOURCE};
  barriers[1].Transition = {resourceOf(target), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, kHome, D3D12_RESOURCE_STATE_COPY_DEST};
  list->ResourceBarrier(2, barriers);
  list->CopyBufferRegion(resourceOf(target), 0, resourceOf(source), 0, bytes);
  std::swap(barriers[0].Transition.StateBefore, barriers[0].Transition.StateAfter);
  std::swap(barriers[1].Transition.StateBefore, barriers[1].Transition.StateAfter);
  list->ResourceBarrier(2, barriers);
}

void D3D12Device::collect() {
  const uint64_t done = fence_->GetCompletedValue();
  for (auto& stream : streams_)
    if (!stream->open && stream->fence && stream->fence <= done) stream->fence = 0;   // reusable
}

exec::Commands D3D12Device::beginCommands() {
  collect();
  Stream* stream = nullptr;
  for (auto& candidate : streams_)
    if (!candidate->open && !candidate->fence) { stream = candidate.get(); break; }
  if (!stream) {
    auto created = std::make_unique<Stream>();
    check(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&created->allocator)), "CreateCommandAllocator");
    check(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, created->allocator.Get(), nullptr, IID_PPV_ARGS(&created->list)), "CreateCommandList");
    stream = created.get();
    streams_.push_back(std::move(created));
  } else {
    check(stream->allocator->Reset(), "ID3D12CommandAllocator::Reset");
    check(stream->list->Reset(stream->allocator.Get(), nullptr), "ID3D12GraphicsCommandList::Reset");
  }
  stream->open = true;
  return exec::Commands(stream->list.Get());
}

void D3D12Device::endAndSubmit(exec::Commands commands, bool wait) {
  ID3D12GraphicsCommandList* list = listOf(commands);
  Stream* stream = nullptr;
  for (auto& candidate : streams_)
    if (candidate->list.Get() == list) stream = candidate.get();
  if (!stream || !stream->open) throw std::runtime_error("endAndSubmit: not a stream of this d3d12 device");
  check(list->Close(), "ID3D12GraphicsCommandList::Close");
  ID3D12CommandList* lists[] = {list};
  queue_->ExecuteCommandLists(1, lists);
  check(queue_->Signal(fence_.Get(), ++fenceValue_), "ID3D12CommandQueue::Signal");
  stream->fence = fenceValue_;
  stream->open = false;
  if (wait) waitFence(fenceValue_);
}

exec::Timer D3D12Device::createTimestampPool(uint32_t count) {
  auto pool = std::make_unique<TimerPool>();
  pool->count = count;
  D3D12_QUERY_HEAP_DESC desc{};
  desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
  desc.Count = count;
  check(device_->CreateQueryHeap(&desc, IID_PPV_ARGS(&pool->heap)), "CreateQueryHeap");
  pool->readback = allocate((exec::Size)count * 8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  return pool.release();
}

void D3D12Device::destroyTimestampPool(exec::Timer pool) { delete static_cast<TimerPool*>(pool); }
void D3D12Device::resetTimestamps(exec::Commands, exec::Timer, uint32_t) {}

void D3D12Device::writeTimestamp(exec::Commands commands, exec::Timer pool, uint32_t index, bool) {
  listOf(commands)->EndQuery(static_cast<TimerPool*>(pool)->heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, index);
}

std::vector<double> D3D12Device::readTimestampsMs(exec::Timer timer, uint32_t count) {
  TimerPool* pool = static_cast<TimerPool*>(timer);
  exec::Commands commands = beginCommands();
  listOf(commands)->ResolveQueryData(pool->heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, count, pool->readback.Get(), 0);
  endAndSubmit(commands, true);
  UINT64 frequency = 0;
  check(queue_->GetTimestampFrequency(&frequency), "GetTimestampFrequency");
  void* mapped = nullptr;
  const D3D12_RANGE range{0, (SIZE_T)count * 8};
  check(pool->readback->Map(0, &range, &mapped), "Map(timestamps)");
  std::vector<double> result(count);
  for (uint32_t index = 0; index < count; ++index) result[index] = (double)static_cast<const uint64_t*>(mapped)[index] * 1000.0 / (double)frequency;
  const D3D12_RANGE none{0, 0};
  pool->readback->Unmap(0, &none);
  return result;
}

}  // namespace d3d
#endif  // _WIN32
