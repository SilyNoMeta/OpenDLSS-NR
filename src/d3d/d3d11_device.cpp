#ifdef _WIN32
#include "d3d11_device.h"

#include <d3d11sdklayers.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <cstring>

using Microsoft::WRL::ComPtr;

namespace d3d {

namespace {
constexpr UINT kUavSlots = 8;   // D3D11_PS_CS_UAV_REGISTER_COUNT: what a feature level 11_0 device has

// What stands behind an exec::Buffer here (its `memory`).
struct BufferState {
  ComPtr<ID3D11Buffer> buffer;
  ComPtr<ID3D11UnorderedAccessView> view;   // raw
  nvcuda::Handle handle = nullptr;          // the driver's handle, named in a launch's read / write lists
  uint64_t address = 0;                     // what the kernels dereference
};
struct ShaderModule { ShaderSource source; };
struct PipelineState { ComPtr<ID3D11ComputeShader> shader; };
struct Module { std::string ptx; };
// One driver object per (block size, dynamic shared bytes) the entry is launched with.
struct Function {
  const Module* module = nullptr;
  KernelSignature signature;
  std::map<std::pair<uint32_t, uint32_t>, nvcuda::Handle> shaders;
};
struct TimerPool {
  std::vector<ComPtr<ID3D11Query>> stamps;
  ComPtr<ID3D11Query> disjoint;
  bool open = false;
};

BufferState* stateOf(const exec::Buffer& buffer) { return static_cast<BufferState*>(buffer.memory); }

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
}  // namespace

D3D11Device::D3D11Device(exec::Backend backend, bool debugLayer) : backend_(backend) {
  const ComPtr<IDXGIAdapter1> chosen = nvidiaAdapter();
  const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
  D3D_FEATURE_LEVEL level{};
  const HRESULT hr = D3D11CreateDevice(chosen.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, debugLayer ? D3D11_CREATE_DEVICE_DEBUG : 0, levels, 2,
                                       D3D11_SDK_VERSION, &device_, &level, &context_);
  if (FAILED(hr) && debugLayer) throw std::runtime_error("the D3D11 debug layer was asked for and is not installed (Graphics Tools)");
  check(hr, "D3D11CreateDevice");
  debugLayer_ = debugLayer;
  init();
}

D3D11Device::D3D11Device(ID3D11Device* device, ID3D11DeviceContext* immediate, exec::Backend backend)
    : device_(device), context_(immediate), backend_(backend) {
  if (!device || !immediate) throw std::runtime_error("a borrowed d3d11 execution needs the host's device and immediate context");
  init();
}

void D3D11Device::init() {
  // Kernels are launched in order on the one stream that executes in order. A deferred context would only record:
  // it is refused rather than having its work issued somewhere else.
  if (context_->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE)
    throw std::runtime_error("the d3d11 execution needs the immediate context: deferred contexts are not supported");
  ComPtr<ID3D11Device> owner;
  context_->GetDevice(&owner);
  if (owner.Get() != device_.Get()) throw std::runtime_error("the d3d11 context does not belong to the device");
  ComPtr<IDXGIDevice> dxgi;
  ComPtr<IDXGIAdapter> adapter;
  DXGI_ADAPTER_DESC desc{};
  check(device_.As(&dxgi), "IDXGIDevice");
  check(dxgi->GetAdapter(&adapter), "IDXGIDevice::GetAdapter");
  check(adapter->GetDesc(&desc), "IDXGIAdapter::GetDesc");
  gpu_ = gpuInfo((uint64_t)(uint32_t)desc.AdapterLuid.LowPart | ((uint64_t)(uint32_t)desc.AdapterLuid.HighPart << 32));
  requireBackend(backend_, gpu_, "d3d11");
  std::string error;
  nv_ = nvcuda::load(error);
  if (!nv_) throw std::runtime_error(error);
  if (!nv_->d3d11()) throw std::runtime_error("the driver does not resolve the D3D11 cubin shader / resource address interfaces");

  D3D11_BUFFER_DESC constantDesc{};
  constantDesc.ByteWidth = exec::kPushConstantBytes;
  constantDesc.Usage = D3D11_USAGE_DEFAULT;
  constantDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  check(device_->CreateBuffer(&constantDesc, nullptr, &constants_), "CreateBuffer(push constants)");
}

D3D11Device::~D3D11Device() {
  if (!device_) return;
  try { waitForGpu(); } catch (...) {}   // everything issued has run before its resources go
  destroyBuffer(zeros_);
  addDebugErrors(debugErrors());
}

uint32_t D3D11Device::debugErrors() const {
  ComPtr<ID3D11InfoQueue> info;
  if (!debugLayer_ || FAILED(device_.As(&info))) return 0;
  uint32_t errors = 0;
  for (UINT64 index = 0; index < info->GetNumStoredMessages(); ++index) {
    SIZE_T size = 0;
    info->GetMessage(index, nullptr, &size);
    std::vector<uint8_t> storage(size);
    auto* message = reinterpret_cast<D3D11_MESSAGE*>(storage.data());
    if (SUCCEEDED(info->GetMessage(index, message, &size)) && message->Severity <= D3D11_MESSAGE_SEVERITY_ERROR) {
      if (errors < 8) fprintf(stderr, "[d3d11] %s\n", message->pDescription);
      ++errors;
    }
  }
  return errors;
}

ID3D11DeviceContext* D3D11Device::contextOf(exec::Commands commands) const {
  ID3D11DeviceContext* context = static_cast<ID3D11DeviceContext*>(commands.stream);
  if (context != context_.Get()) throw std::runtime_error("d3d11 commands must be issued on the device's immediate context");
  return context;
}

void D3D11Device::waitForGpu() {
  D3D11_QUERY_DESC desc{D3D11_QUERY_EVENT, 0};
  ComPtr<ID3D11Query> query;
  check(device_->CreateQuery(&desc, &query), "CreateQuery(event)");
  context_->End(query.Get());
  context_->Flush();
  BOOL done = FALSE;
  const ULONGLONG start = GetTickCount64();
  while (context_->GetData(query.Get(), &done, sizeof(done), 0) != S_OK || !done) {
    if (GetTickCount64() - start > 120000) throw std::runtime_error("the d3d11 context did not complete within 120 s");
    SwitchToThread();
  }
  check(device_->GetDeviceRemovedReason(), "the d3d11 device was removed");
}

exec::Buffer D3D11Device::createBuffer(exec::Size size, bool hostVisible, const char* label, uint32_t) {
  exec::Buffer result;
  result.size = std::max<exec::Size>(size, 16);
  result.hostVisible = hostVisible;   // device-local and unmapped, as on d3d12 (chainSupported() is false)
  result.label = label;
  const exec::Size width = (result.size + 15) & ~exec::Size(15);
  if (width > 0xffffffffull) throw std::runtime_error(std::string("d3d11 buffer too large: ") + label);
  auto state = std::make_unique<BufferState>();
  D3D11_BUFFER_DESC desc{};
  desc.ByteWidth = (UINT)width;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
  desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
  check(device_->CreateBuffer(&desc, nullptr, &state->buffer), "CreateBuffer");
  D3D11_UNORDERED_ACCESS_VIEW_DESC viewDesc{};
  viewDesc.Format = DXGI_FORMAT_R32_TYPELESS;
  viewDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
  viewDesc.Buffer.NumElements = (UINT)(width / 4);
  viewDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
  check(device_->CreateUnorderedAccessView(state->buffer.Get(), &viewDesc, &state->view), "CreateUnorderedAccessView");
  checkNv(nv_->d3d11GetResourceHandle(device_.Get(), state->buffer.Get(), &state->handle), "NvAPI_D3D11_GetResourceHandle");
  checkNv(nv_->d3d11GetResourceGpuVirtualAddress(device_.Get(), state->handle, &state->address), "NvAPI_D3D11_GetResourceGPUVirtualAddress");
  if (!state->handle || !state->address) throw std::runtime_error(std::string("the driver gave no handle / GPU address for ") + label);
  addresses_.add(state->address, width, state->handle);
  result.allocation = width;
  memoryUse_.deviceLocal += result.allocation;
  memoryUse_.peakDeviceLocal = std::max(memoryUse_.peakDeviceLocal, memoryUse_.deviceLocal);
  result.buffer = state->buffer.Get();
  result.memory = state.release();
  return result;
}

void D3D11Device::destroyBuffer(exec::Buffer& buffer) {
  if (BufferState* state = stateOf(buffer)) {
    memoryUse_.deviceLocal -= buffer.allocation;
    addresses_.remove(state->address);
    delete state;
  }
  buffer = exec::Buffer{};
}

exec::Address D3D11Device::deviceAddress(const exec::Buffer& buffer) const {
  // Diagnostics: DLSS5VK_D3D_TRACE=1 re-reads the driver's address and reports a resource that moved.
  static const bool trace = getenv("DLSS5VK_D3D_TRACE") != nullptr;
  BufferState* state = stateOf(buffer);
  if (trace) {
    uint64_t now = 0;
    nv_->d3d11GetResourceGpuVirtualAddress(device_.Get(), state->handle, &now);
    if (now != state->address)
      fprintf(stderr, "[d3d11] %s moved: 0x%llX -> 0x%llX\n", buffer.label, (unsigned long long)state->address, (unsigned long long)now);
  }
  return state->address;
}

void D3D11Device::upload(const exec::Buffer& target, const void* data, exec::Size size, exec::Size offset) {
  if (offset + size > target.size) throw std::runtime_error(std::string("upload overflows ") + target.label);
  if (!size) return;
  const D3D11_BOX box{(UINT)offset, 0, 0, (UINT)(offset + size), 1, 1};
  context_->UpdateSubresource(stateOf(target)->buffer.Get(), 0, &box, data, 0, 0);
}

void D3D11Device::fillZero(const exec::Buffer& target) {
  const std::vector<uint8_t> zero(target.size, 0);
  upload(target, zero.data(), target.size, 0);
}

std::vector<uint8_t> D3D11Device::download(const exec::Buffer& source, exec::Size size, exec::Size offset) {
  if (offset + size > source.size) throw std::runtime_error(std::string("download overflows ") + source.label);
  D3D11_BUFFER_DESC desc{};
  desc.ByteWidth = (UINT)((size + 15) & ~exec::Size(15));
  desc.Usage = D3D11_USAGE_STAGING;
  desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  ComPtr<ID3D11Buffer> staging;
  check(device_->CreateBuffer(&desc, nullptr, &staging), "CreateBuffer(staging)");
  const D3D11_BOX box{(UINT)offset, 0, 0, (UINT)(offset + size), 1, 1};
  context_->CopySubresourceRegion(staging.Get(), 0, 0, 0, 0, stateOf(source)->buffer.Get(), 0, &box);
  D3D11_MAPPED_SUBRESOURCE mapped{};
  check(context_->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map(staging)");   // waits for the GPU
  std::vector<uint8_t> result(size);
  memcpy(result.data(), mapped.pData, size);
  context_->Unmap(staging.Get(), 0);
  return result;
}

exec::Shader D3D11Device::loadShaderModule(const std::string& path) { return new ShaderModule{shaderSourceFor(path)}; }

exec::Pipeline D3D11Device::createComputePipeline(exec::Shader module, const exec::SpecConstants& constants, const char* label, uint32_t) {
  const std::vector<uint8_t> code = compileShader(static_cast<ShaderModule*>(module)->source, constants);
  auto state = std::make_unique<PipelineState>();
  check(device_->CreateComputeShader(code.data(), code.size(), nullptr, &state->shader), "CreateComputeShader");
  exec::Pipeline result;
  result.label = label;
  result.pipeline = state.release();
  return result;
}

void D3D11Device::destroyPipeline(exec::Pipeline& pipeline) {
  delete static_cast<PipelineState*>(pipeline.pipeline);
  pipeline = exec::Pipeline{};
}

void D3D11Device::dispatch(exec::Commands commands, exec::PipelineHandle pipeline, const exec::Buffer* const bindings[exec::kGenericBindings],
                           const void* push, uint32_t pushBytes, uint32_t x, uint32_t y, uint32_t z) {
  if (pushBytes > exec::kPushConstantBytes) throw std::runtime_error("push constants exceed 128 bytes");
  ID3D11DeviceContext* context = contextOf(commands);
  ID3D11UnorderedAccessView* views[kUavSlots]{};
  for (UINT index = 0; index < exec::kGenericBindings; ++index) {
    if (!bindings[index]) continue;
    if (index >= kUavSlots) throw std::runtime_error("a shader kernel binds beyond the eight Direct3D 11 UAV slots");
    views[index] = stateOf(*bindings[index])->view.Get();
  }
  uint8_t constants[exec::kPushConstantBytes]{};
  memcpy(constants, push, pushBytes);
  context->UpdateSubresource(constants_.Get(), 0, nullptr, constants, 0, 0);
  ID3D11Buffer* constantBuffers[] = {constants_.Get()};
  context->CSSetShader(static_cast<PipelineState*>(pipeline)->shader.Get(), nullptr, 0);
  context->CSSetConstantBuffers(0, 1, constantBuffers);
  context->CSSetUnorderedAccessViews(0, kUavSlots, views, nullptr);
  context->Dispatch(x, y, z);
  // Nothing of ours stays bound: a view left in a slot conflicts with the next kernel that addresses the resource.
  ID3D11UnorderedAccessView* none[kUavSlots]{};
  context->CSSetUnorderedAccessViews(0, kUavSlots, none, nullptr);
  ++dispatches_;
}

exec::KernelModule D3D11Device::createCudaModule(const std::string& ptx) { return new Module{ptx}; }

exec::Kernel D3D11Device::createCudaFunction(exec::KernelModule module, const char* name) {
  auto function = std::make_unique<Function>();
  function->module = static_cast<const Module*>(module);
  function->signature = parseSignature(function->module->ptx, name);
  return function.release();
}

void D3D11Device::destroyCudaFunction(exec::Kernel function) {
  if (!function) return;
  for (auto& [shape, shader] : static_cast<Function*>(function)->shaders) nv_->d3d11DestroyCubinComputeShader(device_.Get(), shader);
  delete static_cast<Function*>(function);
}

void D3D11Device::destroyCudaModule(exec::KernelModule module) { delete static_cast<Module*>(module); }

void D3D11Device::cudaLaunch(exec::Commands commands, exec::Kernel function, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                             uint32_t blockX, uint32_t sharedBytes, const void* const* params, size_t paramCount) {
  Function* kernel = static_cast<Function*>(function);
  ID3D11DeviceContext* context = contextOf(commands);
  nvcuda::Handle& shader = kernel->shaders[{blockX, sharedBytes}];
  if (!shader) {
    // The driver takes the PTX text here too (NvAPI_D3D11_IsFatbinPTXSupported), terminator included in the size.
    const std::string& ptx = kernel->module->ptx;
    checkNv(nv_->d3d11CreateCubinComputeShaderEx(device_.Get(), ptx.c_str(), (uint32_t)ptx.size() + 1, blockX, 1, 1, sharedBytes,
                                                 kernel->signature.entry.c_str(), &shader),
            "NvAPI_D3D11_CreateCubinComputeShaderEx(" + kernel->signature.entry + ")");
    if (!shader) throw std::runtime_error("NvAPI_D3D11_CreateCubinComputeShaderEx returned no shader for " + kernel->signature.entry);
  }
  const nvcuda::Params packed = packArguments(kernel->signature, params, paramCount);
  // The resources behind the pointer arguments. A resource the kernel writes is declared read as well: kernels
  // write parts of a buffer (a column range, a batch, counters they also read), and the driver orders a launch after
  // the earlier writers of a resource only when the launch reads it. Declared written only, a resource is taken as
  // overwritten whole and the launch is not held back for it: measured as a head that changes between submissions.
  // An argument that addresses nothing of ours (a null counter address when chaining is off) names no resource.
  std::vector<nvcuda::Handle> reads, writes;
  auto add = [](std::vector<nvcuda::Handle>& list, nvcuda::Handle handle) {
    if (std::find(list.begin(), list.end(), handle) == list.end()) list.push_back(handle);
  };
  for (size_t index = 0; index < paramCount; ++index) {
    if (kernel->signature.widths[index] != 8) continue;
    uint64_t address;
    memcpy(&address, params[index], 8);
    nvcuda::Handle handle = nullptr;
    if (!address || !addresses_.find(address, handle)) continue;
    add(reads, handle);
    if (kernel->signature.written[index]) add(writes, handle);
  }
  checkNv(nv_->d3d11LaunchCubinShader(context, shader, gridX, gridY, gridZ, packed.data(), packed.size(), reads.data(), (uint32_t)reads.size(),
                                      writes.data(), (uint32_t)writes.size()),
          "NvAPI_D3D11_LaunchCubinShader(" + kernel->signature.entry + ")");
  // Diagnostics: DLSS5VK_D3D_TRACE=1 names every launch and its resources, DLSS5VK_D3D_SERIAL=1 waits for the GPU
  // after each one (a result that only then becomes right is an ordering fault, not an arithmetic one).
  static const bool trace = getenv("DLSS5VK_D3D_TRACE") != nullptr, serial = getenv("DLSS5VK_D3D_SERIAL") != nullptr;
  if (trace)
    fprintf(stderr, "[d3d11] launch %llu %s grid %ux%ux%u block %u shared %u args %u bytes reads %zu writes %zu\n",
            (unsigned long long)launches_, kernel->signature.entry.c_str(), gridX, gridY, gridZ, blockX, sharedBytes, packed.size(),
            reads.size(), writes.size());
  if (serial) waitForGpu();
  ++launches_;
}

const exec::Buffer& D3D11Device::zeros(exec::Size size) {
  if (zeros_.size < size) {
    destroyBuffer(zeros_);   // the immediate context has already issued every copy from the old one
    zeros_ = createBuffer(std::max<exec::Size>(size, 1u << 20), false, "zero source");
    fillZero(zeros_);
  }
  return zeros_;
}

void D3D11Device::zeroBuffer(exec::Commands commands, const exec::Buffer& target) { copyBuffer(commands, zeros(target.size), target, target.size); }

void D3D11Device::copyBuffer(exec::Commands commands, const exec::Buffer& source, const exec::Buffer& target, exec::Size bytes) {
  const D3D11_BOX box{0, 0, 0, (UINT)bytes, 1, 1};
  contextOf(commands)->CopySubresourceRegion(stateOf(target)->buffer.Get(), 0, 0, 0, 0, stateOf(source)->buffer.Get(), 0, &box);
}

exec::Commands D3D11Device::beginCommands() { return exec::Commands(context_.Get()); }

void D3D11Device::endAndSubmit(exec::Commands commands, bool wait) {
  contextOf(commands);
  if (wait) waitForGpu();
  else context_->Flush();
}

exec::Timer D3D11Device::createTimestampPool(uint32_t count) {
  auto pool = std::make_unique<TimerPool>();
  pool->stamps.resize(count);
  D3D11_QUERY_DESC stamp{D3D11_QUERY_TIMESTAMP, 0}, disjoint{D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
  for (auto& query : pool->stamps) check(device_->CreateQuery(&stamp, &query), "CreateQuery(timestamp)");
  check(device_->CreateQuery(&disjoint, &pool->disjoint), "CreateQuery(disjoint)");
  return pool.release();
}

void D3D11Device::destroyTimestampPool(exec::Timer pool) { delete static_cast<TimerPool*>(pool); }

void D3D11Device::resetTimestamps(exec::Commands commands, exec::Timer timer, uint32_t) {
  TimerPool* pool = static_cast<TimerPool*>(timer);
  if (pool->open) contextOf(commands)->End(pool->disjoint.Get());
  contextOf(commands)->Begin(pool->disjoint.Get());
  pool->open = true;
}

void D3D11Device::writeTimestamp(exec::Commands commands, exec::Timer timer, uint32_t index, bool) {
  TimerPool* pool = static_cast<TimerPool*>(timer);
  if (!pool->open) { contextOf(commands)->Begin(pool->disjoint.Get()); pool->open = true; }
  contextOf(commands)->End(pool->stamps.at(index).Get());
}

std::vector<double> D3D11Device::readTimestampsMs(exec::Timer timer, uint32_t count) {
  TimerPool* pool = static_cast<TimerPool*>(timer);
  if (pool->open) { context_->End(pool->disjoint.Get()); pool->open = false; }
  context_->Flush();
  auto wait = [&](ID3D11Query* query, void* data, UINT size) {
    const ULONGLONG start = GetTickCount64();
    while (context_->GetData(query, data, size, 0) != S_OK) {
      if (GetTickCount64() - start > 120000) throw std::runtime_error("d3d11 timestamps did not resolve within 120 s");
      SwitchToThread();
    }
  };
  D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint{};
  wait(pool->disjoint.Get(), &disjoint, sizeof(disjoint));
  if (disjoint.Disjoint) throw std::runtime_error("d3d11 timestamps are disjoint (the GPU clock changed during the span)");
  std::vector<double> result(count);
  for (uint32_t index = 0; index < count; ++index) {
    UINT64 ticks = 0;
    wait(pool->stamps.at(index).Get(), &ticks, sizeof(ticks));
    result[index] = (double)ticks * 1000.0 / (double)disjoint.Frequency;
  }
  return result;
}

}  // namespace d3d
#endif  // _WIN32
