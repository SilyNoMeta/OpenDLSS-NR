#include "exec_tape.h"

#include <sstream>

namespace exec {

std::vector<uint8_t> ptxArgumentWidths(const std::string& ptx, const char* entry) {
  const std::string head = std::string(".entry ") + entry + "(";
  const size_t start = ptx.find(head);
  if (start == std::string::npos) throw std::runtime_error(std::string("PTX has no entry ") + entry);
  const size_t end = ptx.find(')', start);
  if (end == std::string::npos) throw std::runtime_error(std::string("PTX entry without an argument list: ") + entry);
  std::vector<uint8_t> widths;
  for (size_t at = ptx.find(".param", start + head.size()); at != std::string::npos && at < end; at = ptx.find(".param", at + 6)) {
    std::istringstream words(ptx.substr(at, end - at));
    std::string keyword, type;
    words >> keyword >> type;
    if (type == ".u64" || type == ".s64" || type == ".b64" || type == ".f64") widths.push_back(8);
    else if (type == ".u32" || type == ".s32" || type == ".b32" || type == ".f32") widths.push_back(4);
    else throw std::runtime_error(std::string("PTX entry ") + entry + ": unsupported argument type " + type);
  }
  return widths;
}

Tape TapeRecorder::take() {
  Tape taken = std::move(tape_);
  tape_ = Tape{};
  return taken;
}

KernelModule TapeRecorder::createCudaModule(const std::string& ptx) {
  const KernelModule module = target_.createCudaModule(ptx);
  modules_[module] = ptx;
  return module;
}

Kernel TapeRecorder::createCudaFunction(KernelModule module, const char* name) {
  const Kernel kernel = target_.createCudaFunction(module, name);
  widths_[kernel] = ptxArgumentWidths(modules_.at(module), name);
  return kernel;
}

void TapeRecorder::destroyCudaFunction(Kernel function) {
  widths_.erase(function);
  target_.destroyCudaFunction(function);
}

void TapeRecorder::destroyCudaModule(KernelModule module) {
  modules_.erase(module);
  target_.destroyCudaModule(module);
}

void TapeRecorder::cudaLaunch(Commands, Kernel function, uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                              uint32_t sharedBytes, const void* const* params, size_t paramCount) {
  const std::vector<uint8_t>& widths = widths_.at(function);
  if (widths.size() != paramCount) throw std::runtime_error("a recorded launch does not match its entry's argument count");
  Tape::Launch launch{function, {gridX, gridY, gridZ}, blockX, sharedBytes, (uint32_t)tape_.values_.size(), (uint32_t)paramCount};
  for (size_t index = 0; index < paramCount; ++index) {
    uint64_t value = 0;
    memcpy(&value, params[index], widths[index]);
    tape_.values_.push_back(value);
  }
  step(Tape::Op::Launch, (uint32_t)tape_.launches_.size());
  tape_.launches_.push_back(launch);
}

void TapeRecorder::dispatch(Commands, PipelineHandle pipeline, const Buffer* const bindings[kGenericBindings], const void* push,
                            uint32_t pushBytes, uint32_t x, uint32_t y, uint32_t z) {
  if (pushBytes > kPushConstantBytes) throw std::runtime_error("push constants exceed 128 bytes");
  Tape::Dispatch recorded{};
  recorded.pipeline = pipeline;
  for (uint32_t index = 0; index < kGenericBindings; ++index) {
    recorded.bound[index] = bindings[index] != nullptr;
    if (bindings[index]) recorded.bindings[index] = *bindings[index];
  }
  memcpy(recorded.push, push, pushBytes);
  recorded.pushBytes = pushBytes;
  recorded.x = x; recorded.y = y; recorded.z = z;
  step(Tape::Op::Dispatch, (uint32_t)tape_.dispatches_.size());
  tape_.dispatches_.push_back(recorded);
}

void TapeRecorder::computeBarrier(Commands) { step(Tape::Op::ComputeBarrier, 0); }
void TapeRecorder::transferBarrier(Commands) { step(Tape::Op::TransferBarrier, 0); }

void TapeRecorder::zeroBuffer(Commands, const Buffer& target) {
  step(Tape::Op::Zero, (uint32_t)tape_.transfers_.size());
  tape_.transfers_.push_back({Buffer{}, target, target.size});
}

void TapeRecorder::copyBuffer(Commands, const Buffer& source, const Buffer& target, Size bytes) {
  step(Tape::Op::Copy, (uint32_t)tape_.transfers_.size());
  tape_.transfers_.push_back({source, target, bytes});
}

void TapeRecorder::resetTimestamps(Commands, Timer, uint32_t) {
  throw std::runtime_error("timestamps are not recorded on a tape: profile on the device itself");
}
void TapeRecorder::writeTimestamp(Commands, Timer, uint32_t, bool) {
  throw std::runtime_error("timestamps are not recorded on a tape: profile on the device itself");
}

void Tape::replay(Device& device, Commands commands) const {
  const void* params[64];
  for (const Step& step : order_) {
    switch (step.op) {
      case Op::Launch: {
        const Launch& launch = launches_[step.index];
        if (launch.count > 64) throw std::runtime_error("a recorded launch has more than 64 arguments");
        for (uint32_t index = 0; index < launch.count; ++index) params[index] = &values_[launch.first + index];
        device.cudaLaunch(commands, launch.kernel, launch.grid[0], launch.grid[1], launch.grid[2], launch.blockX, launch.sharedBytes, params,
                          launch.count);
        break;
      }
      case Op::Dispatch: {
        const Dispatch& dispatch = dispatches_[step.index];
        const Buffer* bindings[kGenericBindings];
        for (uint32_t index = 0; index < kGenericBindings; ++index) bindings[index] = dispatch.bound[index] ? &dispatch.bindings[index] : nullptr;
        device.dispatch(commands, dispatch.pipeline, bindings, dispatch.push, dispatch.pushBytes, dispatch.x, dispatch.y, dispatch.z);
        break;
      }
      case Op::ComputeBarrier: device.computeBarrier(commands); break;
      case Op::TransferBarrier: device.transferBarrier(commands); break;
      case Op::Zero: device.zeroBuffer(commands, transfers_[step.index].target); break;
      case Op::Copy: device.copyBuffer(commands, transfers_[step.index].source, transfers_[step.index].target, transfers_[step.index].bytes); break;
    }
  }
}

}  // namespace exec
