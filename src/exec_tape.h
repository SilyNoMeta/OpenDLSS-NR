// Record a graph once, replay it every frame.
//
// Walking the graph costs host time (routes, weight layouts, labels: about nine milliseconds for a 512x512 field)
// that a frame should not pay again: what the walk emits depends on the geometry only, never on the frame's data,
// which lives in the buffers. A TapeRecorder stands in for the device while the graph is walked and keeps the
// command-stream operations (launches, dispatches, barriers, in-stream zero / copy) as a Tape; everything else it
// forwards, so the kernels, pipelines and buffers are the real device's. Tape::replay then issues the same
// operations, in the same order, into any stream of that device.
//
// This is what a Vulkan host gets from a secondary command buffer; Direct3D has no equivalent that carries kernel
// launches, and an immediate Direct3D 11 context has nothing recorded at all.
//
//   exec::TapeRecorder recorder(device);
//   nr::Kernels kernels(recorder, shaders);  nr::Graph graph(recorder, model, kernels, geometry, options);
//   graph.record(recorder.stream(), features);
//   const exec::Tape tape = recorder.take();
//   ... every frame:  tape.replay(device, commands);
//
// The recorder must outlive the Kernels and the Graph made on it (they destroy their objects through it).
#pragma once
#include <map>
#include <string>
#include <vector>

#include "exec.h"

namespace exec {

class Tape {
 public:
  void replay(Device& device, Commands commands) const;
  size_t launches() const { return launches_.size(); }
  size_t dispatches() const { return dispatches_.size(); }
  bool empty() const { return order_.empty(); }

 private:
  friend class TapeRecorder;
  enum class Op : uint8_t { Launch, Dispatch, ComputeBarrier, TransferBarrier, Zero, Copy };
  struct Step { Op op; uint32_t index; };
  struct Launch { Kernel kernel; uint32_t grid[3], blockX, sharedBytes, first, count; };
  struct Dispatch {
    PipelineHandle pipeline;
    Buffer bindings[kGenericBindings];
    bool bound[kGenericBindings];
    uint8_t push[kPushConstantBytes];
    uint32_t pushBytes, x, y, z;
  };
  struct Transfer { Buffer source, target; Size bytes; };
  std::vector<Step> order_;
  std::vector<Launch> launches_;
  std::vector<uint64_t> values_;   // launch arguments, one 8-byte slot each (a 32-bit argument in its low half)
  std::vector<Dispatch> dispatches_;
  std::vector<Transfer> transfers_;
};

class TapeRecorder final : public Device {
 public:
  explicit TapeRecorder(Device& target) : target_(target) {}
  Commands stream() { return Commands(this); }   // what the graph is walked with; nothing is issued to it
  Tape take();                                    // the operations recorded so far; the recorder starts over

  const char* api() const override { return target_.api(); }
  Backend backend() const override { return target_.backend(); }
  bool nativeFp8() const override { return target_.nativeFp8(); }
  bool chainSupported() const override { return target_.chainSupported(); }
  uint32_t smCount() const override { return target_.smCount(); }
  const std::string& deviceName() const override { return target_.deviceName(); }

  Buffer createBuffer(Size size, bool hostVisible, const char* label, uint32_t usage = 0) override { return target_.createBuffer(size, hostVisible, label, usage); }
  void destroyBuffer(Buffer& buffer) override { target_.destroyBuffer(buffer); }
  void upload(const Buffer& target, const void* data, Size size, Size offset = 0) override { target_.upload(target, data, size, offset); }
  void fillZero(const Buffer& target) override { target_.fillZero(target); }
  std::vector<uint8_t> download(const Buffer& source, Size size, Size offset = 0) override { return target_.download(source, size, offset); }
  Address deviceAddress(const Buffer& buffer) const override { return target_.deviceAddress(buffer); }

  Shader loadShaderModule(const std::string& path) override { return target_.loadShaderModule(path); }
  Pipeline createComputePipeline(Shader module, const SpecConstants& constants, const char* label, uint32_t requiredSubgroupSize = 32) override {
    return target_.createComputePipeline(module, constants, label, requiredSubgroupSize);
  }
  void destroyPipeline(Pipeline& pipeline) override { target_.destroyPipeline(pipeline); }
  void dispatch(Commands commands, PipelineHandle pipeline, const Buffer* const bindings[kGenericBindings], const void* push,
                uint32_t pushBytes, uint32_t x, uint32_t y, uint32_t z) override;

  KernelModule createCudaModule(const std::string& ptx) override;
  Kernel createCudaFunction(KernelModule module, const char* name) override;
  void destroyCudaFunction(Kernel function) override;
  void destroyCudaModule(KernelModule module) override;
  void cudaLaunch(Commands commands, Kernel function, uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                  uint32_t sharedBytes, const void* const* params, size_t paramCount) override;

  void computeBarrier(Commands commands) override;
  void transferBarrier(Commands commands) override;
  void zeroBuffer(Commands commands, const Buffer& target) override;
  void copyBuffer(Commands commands, const Buffer& source, const Buffer& target, Size bytes) override;

  Commands beginCommands() override { return target_.beginCommands(); }
  void endAndSubmit(Commands commands, bool wait = true) override { target_.endAndSubmit(commands, wait); }
  void nextFrame() override { target_.nextFrame(); }
  MemoryUse memoryUse() const override { return target_.memoryUse(); }

  // Timestamps belong to a real stream: per-dispatch profiling walks the graph on the device itself.
  Timer createTimestampPool(uint32_t count) override { return target_.createTimestampPool(count); }
  void destroyTimestampPool(Timer pool) override { target_.destroyTimestampPool(pool); }
  void resetTimestamps(Commands commands, Timer pool, uint32_t count) override;
  void writeTimestamp(Commands commands, Timer pool, uint32_t index, bool first) override;
  std::vector<double> readTimestampsMs(Timer pool, uint32_t count) override { return target_.readTimestampsMs(pool, count); }

 private:
  void step(Tape::Op op, uint32_t index) { tape_.order_.push_back({op, index}); }
  Device& target_;
  Tape tape_;
  std::map<KernelModule, std::string> modules_;          // PTX text, for the entries' argument widths
  std::map<Kernel, std::vector<uint8_t>> widths_;        // 4 or 8 bytes per argument
};

// The widths in bytes (4 or 8) of a PTX entry's arguments, from its .param declarations.
std::vector<uint8_t> ptxArgumentWidths(const std::string& ptx, const char* entry);

}  // namespace exec
