// GPU proof that CUDA kernels can be launched from Direct3D on Direct3D resources, per API:
//
//   cuda-probe --api d3d12 --ptx tests/d3d/probe_kernels.ptx
//   cuda-probe --api d3d11 --ptx tests/d3d/probe_kernels.ptx --cubin build/d3d/probe_kernels.sm_80.cubin
//
// One command stream, in this order, on four raw buffers of `count` 32-bit words:
//
//   upload -> In          an ordinary Direct3D copy
//   In = In + 3           an ordinary Direct3D compute dispatch (HLSL), before any kernel
//   Tmp = affine(In)      CUDA kernel 1: mixed 64 / 32-bit arguments (packing), addresses of two resources
//   Out = shared(Tmp)     CUDA kernel 2: depends on kernel 1's output, dynamic shared memory, a block barrier
//   Copy = Out            an ordinary Direct3D copy, after the kernels
//   Copy = Copy * 2 + 1   an ordinary Direct3D compute dispatch (HLSL), after the kernels
//
// then a GPU completion (a fence on D3D12, an event query on D3D11) and a readback of every buffer, each compared
// with what the CPU computes for it. The stages are compared separately, so the first wrong one is named. Nothing
// here assumes that a Direct3D GPU address is a valid CUDA pointer: that is what the kernels' output proves or not.
#include <d3d11_4.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/d3d/nvapi_cuda.h"

using Microsoft::WRL::ComPtr;

namespace {

constexpr uint32_t kCount = 100003;          // not a multiple of the block size: the last block is partial
constexpr uint32_t kBlock = 128;
constexpr uint32_t kGrid = (kCount + kBlock - 1) / kBlock;
constexpr uint32_t kScale = 2654435761u, kBias = 0x9e3779b9u;
constexpr uint64_t kMask = 0xa5a5f00d12345678ull;
constexpr uint32_t kBytes = kCount * 4;
constexpr uint32_t kBufferBytes = (kBytes + 255) & ~255u;

const char kHlsl[] = R"(
RWByteAddressBuffer data : register(u0);
cbuffer Constants : register(b0) { uint count; uint multiply; uint add; uint unused; };
[numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {
  if (id.x < count) data.Store(id.x * 4, data.Load(id.x * 4) * multiply + add);
}
)";

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}
void requireHr(HRESULT hr, const char* what) {
  if (FAILED(hr)) {
    char text[160];
    snprintf(text, sizeof(text), "%s failed (0x%08lX)", what, (unsigned long)hr);
    throw std::runtime_error(text);
  }
}
void requireNv(nvcuda::Status status, const char* what) {
  if (status != 0) throw std::runtime_error(std::string(what) + " failed (NvAPI status " + std::to_string(status) + ")");
}

std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path);
  std::stringstream buffer;
  buffer << file.rdbuf();
  const std::string text = buffer.str();
  return std::vector<uint8_t>(text.begin(), text.end());
}

std::string arg(int argc, char** argv, const char* name, const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], name)) return argv[i + 1];
  return fallback;
}
bool flag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], name)) return true;
  return false;
}

// ---- what every stage must hold, computed on the CPU ------------------------------------------------------------
struct Expected {
  std::vector<uint32_t> upload, in, tmp, out, copy;
  Expected() : upload(kCount), in(kCount), tmp(kCount), out(kCount), copy(kCount) {
    uint32_t state = 0x243f6a88u;
    for (uint32_t i = 0; i < kCount; ++i) {
      state = state * 1664525u + 1013904223u;
      upload[i] = state ^ (i * 0x9e3779b1u);
    }
    for (uint32_t i = 0; i < kCount; ++i) in[i] = upload[i] + 3u;
    for (uint32_t i = 0; i < kCount; ++i)
      tmp[i] = (in[i] * kScale + kBias + i) ^ (uint32_t)kMask ^ (uint32_t)(kMask >> 32);
    for (uint32_t i = 0; i < kCount; ++i) {
      const uint32_t block = i / kBlock, neighbour = block * kBlock + (i % kBlock + 1) % kBlock;
      out[i] = (neighbour < kCount ? tmp[neighbour] : 0u) + (block << 16);
    }
    for (uint32_t i = 0; i < kCount; ++i) copy[i] = out[i] * 2u + 1u;
  }
};

struct Stage {
  const char* name;
  const char* producer;
  uint32_t mismatches = 0, first = 0, got = 0, want = 0;
};

Stage compare(const char* name, const char* producer, const uint32_t* got, const std::vector<uint32_t>& want) {
  Stage stage{name, producer};
  for (uint32_t i = 0; i < kCount; ++i)
    if (got[i] != want[i]) {
      if (!stage.mismatches) { stage.first = i; stage.got = got[i]; stage.want = want[i]; }
      ++stage.mismatches;
    }
  return stage;
}

struct Report {
  std::string api, adapter, driverNote, cubinSource;
  uint64_t luid = 0;
  uint64_t addresses[4]{};   // In, Tmp, Out, Copy as the kernels were given them
  bool ptxImage = false;     // the kernel image handed to the driver was PTX text
  std::vector<Stage> stages;
  std::vector<std::string> notes;
  uint32_t debugErrors = 0;
  bool debugLayer = false;
  bool passed() const {
    if (stages.empty() || debugErrors) return false;
    for (const Stage& stage : stages)
      if (stage.mismatches) return false;
    return true;
  }
};

ComPtr<IDXGIAdapter1> nvidiaAdapter(Report& report) {
  ComPtr<IDXGIFactory6> factory;
  requireHr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
  ComPtr<IDXGIAdapter1> adapter;
  for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    if (desc.VendorId == 0x10de && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
      char name[160];
      WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name), nullptr, nullptr);
      report.adapter = name;
      report.luid = (uint64_t)(uint32_t)desc.AdapterLuid.LowPart | ((uint64_t)(uint32_t)desc.AdapterLuid.HighPart << 32);
      return adapter;
    }
    adapter.Reset();
  }
  throw std::runtime_error("no NVIDIA adapter");
}

ComPtr<ID3DBlob> compileHlsl() {
  ComPtr<ID3DBlob> code, errors;
  const HRESULT hr = D3DCompile(kHlsl, sizeof(kHlsl) - 1, "probe", nullptr, nullptr, "main", "cs_5_0", 0, 0, &code, &errors);
  if (FAILED(hr))
    throw std::runtime_error(std::string("HLSL: ") + (errors ? (const char*)errors->GetBufferPointer() : "compile failed"));
  return code;
}

// ---- D3D12 ------------------------------------------------------------------------------------------------------
void runD3D12(const nvcuda::Api& nv, const std::vector<uint8_t>& ptx, bool debug, const Expected& expected, Report& report) {
  require(nv.d3d12(), "the driver does not resolve the D3D12 CuModule / CuFunction / LaunchCuKernelChain interfaces");
  ComPtr<IDXGIAdapter1> adapter = nvidiaAdapter(report);
  if (debug) {
    ComPtr<ID3D12Debug> layer;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&layer)))) { layer->EnableDebugLayer(); report.debugLayer = true; }
    else report.notes.push_back("D3D12 debug layer requested but not installed (Graphics Tools)");
  }
  ComPtr<ID3D12Device> device;
  requireHr(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "D3D12CreateDevice");
  D3D12_COMMAND_QUEUE_DESC queueDesc{};
  ComPtr<ID3D12CommandQueue> queue;
  requireHr(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
  ComPtr<ID3D12CommandAllocator> allocator;
  requireHr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "CreateCommandAllocator");
  ComPtr<ID3D12GraphicsCommandList> list;
  requireHr(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "CreateCommandList");
  ComPtr<ID3D12Fence> fence;
  requireHr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");

  auto buffer = [&](D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES properties{};
    properties.Type = heap;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = kBufferBytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    ComPtr<ID3D12Resource> resource;
    requireHr(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&resource)),
              "CreateCommittedResource");
    return resource;
  };
  const auto uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  ComPtr<ID3D12Resource> upload = buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
  ComPtr<ID3D12Resource> in = buffer(D3D12_HEAP_TYPE_DEFAULT, uav, D3D12_RESOURCE_STATE_COPY_DEST);
  ComPtr<ID3D12Resource> tmp = buffer(D3D12_HEAP_TYPE_DEFAULT, uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  ComPtr<ID3D12Resource> out = buffer(D3D12_HEAP_TYPE_DEFAULT, uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  ComPtr<ID3D12Resource> copy = buffer(D3D12_HEAP_TYPE_DEFAULT, uav, D3D12_RESOURCE_STATE_COPY_DEST);
  ComPtr<ID3D12Resource> readback[4];
  for (auto& resource : readback) resource = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  {
    void* mapped = nullptr;
    requireHr(upload->Map(0, nullptr, &mapped), "Map(upload)");
    memcpy(mapped, expected.upload.data(), kBytes);
    upload->Unmap(0, nullptr);
  }

  // The ordinary compute pass: a root UAV and four root constants.
  D3D12_ROOT_PARAMETER rootParameters[2]{};
  rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
  rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  rootParameters[1].Constants.Num32BitValues = 4;
  D3D12_ROOT_SIGNATURE_DESC rootDesc{2, rootParameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
  ComPtr<ID3DBlob> rootBlob, rootErrors;
  requireHr(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rootBlob, &rootErrors), "D3D12SerializeRootSignature");
  ComPtr<ID3D12RootSignature> rootSignature;
  requireHr(device->CreateRootSignature(0, rootBlob->GetBufferPointer(), rootBlob->GetBufferSize(), IID_PPV_ARGS(&rootSignature)), "CreateRootSignature");
  ComPtr<ID3DBlob> shader = compileHlsl();
  D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDesc{};
  pipelineDesc.pRootSignature = rootSignature.Get();
  pipelineDesc.CS = {shader->GetBufferPointer(), shader->GetBufferSize()};
  ComPtr<ID3D12PipelineState> pipeline;
  requireHr(device->CreateComputePipelineState(&pipelineDesc, IID_PPV_ARGS(&pipeline)), "CreateComputePipelineState");

  // The kernels: one module from the PTX text (NUL-terminated), two functions.
  if (nv.d3d12IsFatbinPtxSupported) {
    bool supported = false;
    const nvcuda::Status status = nv.d3d12IsFatbinPtxSupported(device.Get(), &supported);
    report.notes.push_back("NvAPI_D3D12_IsFatbinPTXSupported: status " + std::to_string(status) + ", supported " + (supported ? "true" : "false"));
  }
  std::vector<uint8_t> image = ptx;
  image.push_back(0);
  report.ptxImage = true;
  nvcuda::Handle module = nullptr, affine = nullptr, shared = nullptr;
  requireNv(nv.d3d12CreateCuModule(device.Get(), image.data(), (uint32_t)image.size(), &module), "NvAPI_D3D12_CreateCuModule");
  require(module != nullptr, "NvAPI_D3D12_CreateCuModule returned no module");
  requireNv(nv.d3d12CreateCuFunction(device.Get(), module, "probe_affine", &affine), "NvAPI_D3D12_CreateCuFunction(probe_affine)");
  requireNv(nv.d3d12CreateCuFunction(device.Get(), module, "probe_shared", &shared), "NvAPI_D3D12_CreateCuFunction(probe_shared)");

  auto transition = [&](ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &barrier);
  };
  auto uavBarrier = [&] {   // every pending unordered access completes before the next command
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    list->ResourceBarrier(1, &barrier);
  };
  auto dispatch = [&](ID3D12Resource* resource, uint32_t multiply, uint32_t add) {
    list->SetComputeRootSignature(rootSignature.Get());
    list->SetPipelineState(pipeline.Get());
    list->SetComputeRootUnorderedAccessView(0, resource->GetGPUVirtualAddress());
    const uint32_t constants[4] = {kCount, multiply, add, 0};
    list->SetComputeRoot32BitConstants(1, 4, constants, 0);
    list->Dispatch((kCount + 63) / 64, 1, 1);
    uavBarrier();
  };

  const uint64_t inAddress = in->GetGPUVirtualAddress(), tmpAddress = tmp->GetGPUVirtualAddress(),
                 outAddress = out->GetGPUVirtualAddress();
  report.addresses[0] = inAddress; report.addresses[1] = tmpAddress; report.addresses[2] = outAddress;
  report.addresses[3] = copy->GetGPUVirtualAddress();

  // ---- the command stream ----
  list->CopyBufferRegion(in.Get(), 0, upload.Get(), 0, kBufferBytes);
  transition(in.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  dispatch(in.Get(), 1, 3);

  nvcuda::Params affineParams, sharedParams;
  affineParams.u64(inAddress).u32(kCount).u64(tmpAddress).u32(kScale).u32(kBias).u64(kMask);
  sharedParams.u64(tmpAddress).u64(outAddress).u32(kCount);
  nvcuda::KernelLaunch launch{};
  launch.function = affine;
  launch.grid = {kGrid, 1, 1};
  launch.block = {kBlock, 1, 1};
  launch.params = affineParams.data();
  launch.paramBytes = affineParams.size();
  requireNv(nv.d3d12LaunchCuKernelChain(list.Get(), &launch, 1), "NvAPI_D3D12_LaunchCuKernelChain(probe_affine)");
  uavBarrier();
  launch.function = shared;
  launch.dynamicSharedBytes = kBlock * 4;
  launch.params = sharedParams.data();
  launch.paramBytes = sharedParams.size();
  requireNv(nv.d3d12LaunchCuKernelChain(list.Get(), &launch, 1), "NvAPI_D3D12_LaunchCuKernelChain(probe_shared)");
  uavBarrier();

  transition(out.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
  list->CopyBufferRegion(copy.Get(), 0, out.Get(), 0, kBufferBytes);
  transition(copy.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  dispatch(copy.Get(), 2, 1);

  ID3D12Resource* sources[4] = {in.Get(), tmp.Get(), out.Get(), copy.Get()};
  for (int i = 0; i < 4; ++i) {
    if (i != 2) transition(sources[i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback[i].Get(), 0, sources[i], 0, kBufferBytes);
  }
  requireHr(list->Close(), "Close");
  ID3D12CommandList* lists[] = {list.Get()};
  queue->ExecuteCommandLists(1, lists);
  requireHr(queue->Signal(fence.Get(), 1), "Signal");
  HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  requireHr(fence->SetEventOnCompletion(1, event), "SetEventOnCompletion");
  const DWORD waited = WaitForSingleObject(event, 60000);
  CloseHandle(event);
  require(waited == WAIT_OBJECT_0 && fence->GetCompletedValue() == 1, "the D3D12 fence did not complete within 60 s");
  requireHr(device->GetDeviceRemovedReason(), "the device was removed");

  const std::vector<uint32_t>* wanted[4] = {&expected.in, &expected.tmp, &expected.out, &expected.copy};
  const char* names[4] = {"In", "Tmp", "Out", "Copy"};
  const char* producers[4] = {"D3D12 copy + HLSL dispatch", "CUDA probe_affine", "CUDA probe_shared", "D3D12 copy + HLSL dispatch"};
  for (int i = 0; i < 4; ++i) {
    void* mapped = nullptr;
    D3D12_RANGE range{0, kBytes};
    requireHr(readback[i]->Map(0, &range, &mapped), "Map(readback)");
    report.stages.push_back(compare(names[i], producers[i], static_cast<const uint32_t*>(mapped), *wanted[i]));
    readback[i]->Unmap(0, nullptr);
  }
  ComPtr<ID3D12InfoQueue> infoQueue;
  if (report.debugLayer && SUCCEEDED(device.As(&infoQueue))) {
    for (UINT64 i = 0; i < infoQueue->GetNumStoredMessages(); ++i) {
      SIZE_T size = 0;
      infoQueue->GetMessage(i, nullptr, &size);
      std::vector<uint8_t> storage(size);
      auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
      if (SUCCEEDED(infoQueue->GetMessage(i, message, &size)) && message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
        ++report.debugErrors;
        report.notes.push_back(std::string("D3D12 debug: ") + message->pDescription);
      }
    }
  }
  nv.d3d12DestroyCuFunction(device.Get(), affine);
  nv.d3d12DestroyCuFunction(device.Get(), shared);
  nv.d3d12DestroyCuModule(device.Get(), module);
}

// ---- D3D11 ------------------------------------------------------------------------------------------------------
void runD3D11(const nvcuda::Api& nv, const std::vector<uint8_t>& ptx, const std::vector<uint8_t>& cubin, bool debug,
              const Expected& expected, Report& report) {
  require(nv.d3d11(), "the driver does not resolve the D3D11 cubin shader / resource address interfaces");
  ComPtr<IDXGIAdapter1> adapter = nvidiaAdapter(report);
  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
  D3D_FEATURE_LEVEL level{};
  UINT flags = 0;
  if (debug) flags |= D3D11_CREATE_DEVICE_DEBUG;
  HRESULT hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags, levels, 2, D3D11_SDK_VERSION, &device, &level, &context);
  if (FAILED(hr) && debug) {
    report.notes.push_back("D3D11 debug layer requested but not installed (Graphics Tools)");
    hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, levels, 2, D3D11_SDK_VERSION, &device, &level, &context);
  } else if (debug) {
    report.debugLayer = true;
  }
  requireHr(hr, "D3D11CreateDevice");
  require(context->GetType() == D3D11_DEVICE_CONTEXT_IMMEDIATE, "not an immediate context");

  auto buffer = [&](ComPtr<ID3D11UnorderedAccessView>* view) {
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = kBufferBytes;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    ComPtr<ID3D11Buffer> resource;
    requireHr(device->CreateBuffer(&desc, nullptr, &resource), "CreateBuffer");
    if (view) {
      D3D11_UNORDERED_ACCESS_VIEW_DESC viewDesc{};
      viewDesc.Format = DXGI_FORMAT_R32_TYPELESS;
      viewDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      viewDesc.Buffer.NumElements = kBufferBytes / 4;
      viewDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      requireHr(device->CreateUnorderedAccessView(resource.Get(), &viewDesc, view->GetAddressOf()), "CreateUnorderedAccessView");
    }
    return resource;
  };
  ComPtr<ID3D11UnorderedAccessView> inView, copyView;
  ComPtr<ID3D11Buffer> in = buffer(&inView), tmp = buffer(nullptr), out = buffer(nullptr), copy = buffer(&copyView);
  ComPtr<ID3D11Buffer> staging[4];
  for (auto& resource : staging) {
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = kBufferBytes;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    requireHr(device->CreateBuffer(&desc, nullptr, &resource), "CreateBuffer(staging)");
  }
  ComPtr<ID3DBlob> shaderCode = compileHlsl();
  ComPtr<ID3D11ComputeShader> shader;
  requireHr(device->CreateComputeShader(shaderCode->GetBufferPointer(), shaderCode->GetBufferSize(), nullptr, &shader), "CreateComputeShader");
  D3D11_BUFFER_DESC constantDesc{};
  constantDesc.ByteWidth = 16;
  constantDesc.Usage = D3D11_USAGE_DEFAULT;
  constantDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  ComPtr<ID3D11Buffer> constants;
  requireHr(device->CreateBuffer(&constantDesc, nullptr, &constants), "CreateBuffer(constants)");

  // Driver handles and GPU addresses of the three resources the kernels touch.
  ID3D11Buffer* touched[4] = {in.Get(), tmp.Get(), out.Get(), copy.Get()};
  nvcuda::Handle handles[4]{};
  for (int i = 0; i < 4; ++i) {
    requireNv(nv.d3d11GetResourceHandle(device.Get(), touched[i], &handles[i]), "NvAPI_D3D11_GetResourceHandle");
    require(handles[i] != nullptr, "NvAPI_D3D11_GetResourceHandle returned no handle");
    requireNv(nv.d3d11GetResourceGpuVirtualAddress(device.Get(), handles[i], &report.addresses[i]), "NvAPI_D3D11_GetResourceGPUVirtualAddress");
    require(report.addresses[i] != 0, "NvAPI_D3D11_GetResourceGPUVirtualAddress returned a null address");
    if (nv.d3d11GetResourceGpuVirtualAddressEx) {
      nvcuda::D3D11GpuVirtualAddress query{nvcuda::kD3D11GpuVirtualAddressVersion, handles[i], 0, 0};
      const nvcuda::Status status = nv.d3d11GetResourceGpuVirtualAddressEx(device.Get(), &query);
      if (i == 0) {
        char text[160];
        snprintf(text, sizeof(text), "GetResourceGPUVirtualAddressEx(In): status %d, start 0x%llX, size %llu (buffer %u)", status,
                 (unsigned long long)query.start, (unsigned long long)query.size, kBufferBytes);
        report.notes.push_back(text);
      }
      if (status == 0) {
        require(query.start == report.addresses[i], "the two GPU address queries disagree");
        require(query.size >= kBufferBytes, "the resource's GPU range is smaller than the buffer");
      }
    }
  }
  if (nv.d3d11IsFatbinPtxSupported) {
    bool supported = false;
    const nvcuda::Status status = nv.d3d11IsFatbinPtxSupported(device.Get(), &supported);
    report.notes.push_back("NvAPI_D3D11_IsFatbinPTXSupported: status " + std::to_string(status) + ", supported " + (supported ? "true" : "false"));
  }

  // The kernel image. PTX text is tried first only to record the driver's answer; a refusal is not an error when a
  // cubin was given.
  nvcuda::Handle affine = nullptr, shared = nullptr;
  auto create = [&](const std::vector<uint8_t>& image, const char* what) {
    nvcuda::Handle a = nullptr, s = nullptr;
    const nvcuda::Status first = nv.d3d11CreateCubinComputeShaderEx(device.Get(), image.data(), (uint32_t)image.size(), kBlock, 1, 1, 0, "probe_affine", &a);
    const nvcuda::Status second = first == 0 ? nv.d3d11CreateCubinComputeShaderEx(device.Get(), image.data(), (uint32_t)image.size(), kBlock, 1, 1, kBlock * 4, "probe_shared", &s) : first;
    report.notes.push_back(std::string("NvAPI_D3D11_CreateCubinComputeShaderEx with ") + what + ": status " + std::to_string(first) + " / " + std::to_string(second));
    if (first == 0 && second == 0 && a && s) { affine = a; shared = s; return true; }
    if (a) nv.d3d11DestroyCubinComputeShader(device.Get(), a);
    if (s) nv.d3d11DestroyCubinComputeShader(device.Get(), s);
    return false;
  };
  std::vector<uint8_t> text = ptx;
  text.push_back(0);
  if (create(text, "PTX text")) { report.ptxImage = true; report.cubinSource = "PTX text compiled by the driver"; }
  else {
    require(!cubin.empty(), "the driver refused PTX text and no --cubin was given");
    require(create(cubin, "a ptxas cubin"), "the driver refused the cubin");
    report.cubinSource = "cubin assembled by ptxas";
  }

  auto dispatch = [&](ID3D11UnorderedAccessView* view, uint32_t multiply, uint32_t add) {
    const uint32_t values[4] = {kCount, multiply, add, 0};
    context->UpdateSubresource(constants.Get(), 0, nullptr, values, 0, 0);
    context->CSSetShader(shader.Get(), nullptr, 0);
    ID3D11Buffer* constantBuffers[] = {constants.Get()};
    context->CSSetConstantBuffers(0, 1, constantBuffers);
    context->CSSetUnorderedAccessViews(0, 1, &view, nullptr);
    context->Dispatch((kCount + 63) / 64, 1, 1);
    ID3D11UnorderedAccessView* none = nullptr;   // no view stays bound while a kernel addresses the resource
    context->CSSetUnorderedAccessViews(0, 1, &none, nullptr);
  };

  // ---- the command stream ----
  {
    std::vector<uint32_t> padded(kBufferBytes / 4, 0);
    memcpy(padded.data(), expected.upload.data(), kBytes);
    context->UpdateSubresource(in.Get(), 0, nullptr, padded.data(), 0, 0);
  }
  dispatch(inView.Get(), 1, 3);

  nvcuda::Params affineParams, sharedParams;
  affineParams.u64(report.addresses[0]).u32(kCount).u64(report.addresses[1]).u32(kScale).u32(kBias).u64(kMask);
  sharedParams.u64(report.addresses[1]).u64(report.addresses[2]).u32(kCount);
  requireNv(nv.d3d11LaunchCubinShader(context.Get(), affine, kGrid, 1, 1, affineParams.data(), affineParams.size(), &handles[0], 1, &handles[1], 1),
            "NvAPI_D3D11_LaunchCubinShader(probe_affine)");
  requireNv(nv.d3d11LaunchCubinShader(context.Get(), shared, kGrid, 1, 1, sharedParams.data(), sharedParams.size(), &handles[1], 1, &handles[2], 1),
            "NvAPI_D3D11_LaunchCubinShader(probe_shared)");

  context->CopyResource(copy.Get(), out.Get());
  dispatch(copyView.Get(), 2, 1);
  for (int i = 0; i < 4; ++i) context->CopyResource(staging[i].Get(), touched[i]);

  // GPU completion: an event query after the last command, polled without blocking the device.
  D3D11_QUERY_DESC queryDesc{D3D11_QUERY_EVENT, 0};
  ComPtr<ID3D11Query> query;
  requireHr(device->CreateQuery(&queryDesc, &query), "CreateQuery");
  context->End(query.Get());
  context->Flush();
  BOOL done = FALSE;
  const ULONGLONG start = GetTickCount64();
  while (context->GetData(query.Get(), &done, sizeof(done), 0) != S_OK || !done) {
    require(GetTickCount64() - start < 60000, "the D3D11 event query did not complete within 60 s");
    Sleep(1);
  }
  requireHr(device->GetDeviceRemovedReason(), "the device was removed");

  const std::vector<uint32_t>* wanted[4] = {&expected.in, &expected.tmp, &expected.out, &expected.copy};
  const char* names[4] = {"In", "Tmp", "Out", "Copy"};
  const char* producers[4] = {"D3D11 update + HLSL dispatch", "CUDA probe_affine", "CUDA probe_shared", "D3D11 copy + HLSL dispatch"};
  for (int i = 0; i < 4; ++i) {
    D3D11_MAPPED_SUBRESOURCE mapped{};
    requireHr(context->Map(staging[i].Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map(staging)");
    report.stages.push_back(compare(names[i], producers[i], static_cast<const uint32_t*>(mapped.pData), *wanted[i]));
    context->Unmap(staging[i].Get(), 0);
  }
  ComPtr<ID3D11InfoQueue> infoQueue;
  if (report.debugLayer && SUCCEEDED(device.As(&infoQueue))) {
    for (UINT64 i = 0; i < infoQueue->GetNumStoredMessages(); ++i) {
      SIZE_T size = 0;
      infoQueue->GetMessage(i, nullptr, &size);
      std::vector<uint8_t> storage(size);
      auto* message = reinterpret_cast<D3D11_MESSAGE*>(storage.data());
      if (SUCCEEDED(infoQueue->GetMessage(i, message, &size)) && message->Severity <= D3D11_MESSAGE_SEVERITY_ERROR) {
        ++report.debugErrors;
        report.notes.push_back(std::string("D3D11 debug: ") + message->pDescription);
      }
    }
  }
  nv.d3d11DestroyCubinComputeShader(device.Get(), affine);
  nv.d3d11DestroyCubinComputeShader(device.Get(), shared);
}

std::string jsonEscape(const std::string& text) {
  std::string out;
  for (char c : text) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c == '\n') out += "\\n";
    else if ((unsigned char)c < 0x20) out += ' ';
    else out += c;
  }
  return out;
}

void writeJson(const std::string& path, const Report& report, const std::string& error) {
  std::ofstream file(path, std::ios::binary);
  char buffer[256];
  file << "{\n  \"probe\": \"elementary CUDA kernels on Direct3D resources\",\n";
  file << "  \"api\": \"" << report.api << "\",\n  \"adapter\": \"" << jsonEscape(report.adapter) << "\",\n";
  snprintf(buffer, sizeof(buffer), "  \"adapter_luid\": \"0x%016llX\",\n", (unsigned long long)report.luid);
  file << buffer;
  file << "  \"kernel_image\": \"" << (report.ptxImage ? "PTX text" : "cubin") << "\",\n";
  file << "  \"count\": " << kCount << ", \"block\": " << kBlock << ", \"grid\": " << kGrid << ",\n";
  file << "  \"debug_layer\": " << (report.debugLayer ? "true" : "false") << ", \"debug_errors\": " << report.debugErrors << ",\n";
  file << "  \"gpu_addresses\": [";
  for (int i = 0; i < 4; ++i) {
    snprintf(buffer, sizeof(buffer), "%s\"0x%llX\"", i ? ", " : "", (unsigned long long)report.addresses[i]);
    file << buffer;
  }
  file << "],\n  \"stages\": [\n";
  for (size_t i = 0; i < report.stages.size(); ++i) {
    const Stage& s = report.stages[i];
    file << "    {\"buffer\": \"" << s.name << "\", \"producer\": \"" << s.producer << "\", \"values\": " << kCount
         << ", \"mismatches\": " << s.mismatches << "}" << (i + 1 < report.stages.size() ? "," : "") << "\n";
  }
  file << "  ],\n  \"notes\": [";
  for (size_t i = 0; i < report.notes.size(); ++i) file << (i ? ", " : "") << "\"" << jsonEscape(report.notes[i]) << "\"";
  file << "],\n  \"error\": \"" << jsonEscape(error) << "\",\n  \"passed\": " << (error.empty() && report.passed() ? "true" : "false") << "\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  Report report;
  report.api = arg(argc, argv, "--api");
  const std::string ptxPath = arg(argc, argv, "--ptx"), cubinPath = arg(argc, argv, "--cubin"), jsonPath = arg(argc, argv, "--json");
  const bool debug = flag(argc, argv, "--debug");
  if ((report.api != "d3d12" && report.api != "d3d11") || ptxPath.empty()) {
    fprintf(stderr, "usage: cuda-probe --api d3d12|d3d11 --ptx <probe_kernels.ptx> [--cubin <probe_kernels.cubin>] [--debug] [--json <file>]\n");
    return 2;
  }
  std::string error;
  try {
    const Expected expected;
    const std::vector<uint8_t> ptx = readFile(ptxPath);
    std::vector<uint8_t> cubin;
    if (!cubinPath.empty()) cubin = readFile(cubinPath);
    std::string nvError;
    const nvcuda::Api* nv = nvcuda::load(nvError);
    require(nv != nullptr, nvError);
    if (report.api == "d3d12") runD3D12(*nv, ptx, debug, expected, report);
    else runD3D11(*nv, ptx, cubin, debug, expected, report);
  } catch (const std::exception& e) {
    error = e.what();
  }
  printf("api %s, adapter %s (LUID 0x%016llX), kernel image: %s\n", report.api.c_str(), report.adapter.c_str(),
         (unsigned long long)report.luid, report.ptxImage ? "PTX text" : report.cubinSource.empty() ? "none" : report.cubinSource.c_str());
  printf("GPU addresses given to the kernels: In 0x%llX, Tmp 0x%llX, Out 0x%llX\n", (unsigned long long)report.addresses[0],
         (unsigned long long)report.addresses[1], (unsigned long long)report.addresses[2]);
  for (const std::string& note : report.notes) printf("note: %s\n", note.c_str());
  for (const Stage& s : report.stages) {
    if (!s.mismatches) printf("  %-5s equal to the CPU (%u values)   <- %s\n", s.name, kCount, s.producer);
    else printf("  %-5s MISMATCH %u/%u, first at %u: gpu 0x%08X cpu 0x%08X   <- %s\n", s.name, s.mismatches, kCount, s.first, s.got, s.want, s.producer);
  }
  if (report.debugLayer) printf("debug layer: %u error%s\n", report.debugErrors, report.debugErrors == 1 ? "" : "s");
  if (!error.empty()) printf("error: %s\n", error.c_str());
  const bool passed = error.empty() && report.passed();
  if (!jsonPath.empty()) writeJson(jsonPath, report, error);
  printf("%s\n", passed ? "PASS: Direct3D commands, CUDA kernels on Direct3D resources, GPU completion and readback"
                        : "FAIL");
  return passed ? 0 : 1;
}
