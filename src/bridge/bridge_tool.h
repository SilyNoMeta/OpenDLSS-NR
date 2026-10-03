// dlss5vk bridge: a Direct3D host that hands its frame to the Vulkan graph over src/bridge and takes the head back.
//
//   dlss5vk bridge --host d3d12|d3d11 --model <dir> --fixture <proxy fixture> [--frames N] [--same-kernels]
//                  [--dump <dir>] [--json <file>]
//   dlss5vk bridge --host d3d12|d3d11 --transport [--width W --height H] [--frames N] [--json <file>]
//
// Every frame, in the host's stream order: Direct3D copies its proxy image into the shared proxy texture (a GPU
// copy, as a game's producer is on the GPU); Vulkan (once the host's fence value is reached) copies it to the proxy
// buffer, generates the features, runs the graph and copies the head to the shared head texture; Direct3D (once
// Vulkan's fence value is reached) copies the head texture into a texture of its own. The only CPU wait is the one
// that ends the frame, where the time to a usable output is taken; reading the result back to check it comes after.
//
// --transport runs no network: Vulkan copies the input texture to the output texture and one sub-rectangle of it to
// another place, and Direct3D checks every texel. It is the cost and the proof of the transport alone.
// --same-kernels runs the Vulkan graph without its cooperative-matrix GLSL (the Direct3D kernel set).
#pragma once
#ifdef _WIN32
#include <cstdint>
#include <string>
#include <vector>

namespace bridge {

struct ToolInput {
  std::string host;            // "d3d12" or "d3d11"
  std::string modelDir, shaderDir, dumpDir, jsonPath;
  bool transportOnly = false, sameKernels = false, debugLayer = false;
  int frames = 10;
  // The fixture, already read by the caller (empty in --transport).
  uint32_t validWidth = 0, validHeight = 0, fullWidth = 0, fullHeight = 0, proxyWidth = 0, proxyHeight = 0, seed = 0;
  bool autoMask = false;
  float localTone = 0, localStructure = 0, skinStructure = 0, style = 0;
  std::vector<uint8_t> proxy;            // RGBA f32
  std::vector<uint8_t> referenceHead;    // f32 [full][4], empty when the fixture gates no head
};

struct ToolResult {
  bool passed = false;
  std::vector<uint8_t> head;   // what Direct3D read back on the last frame
};

ToolResult runTool(const ToolInput& input);

}  // namespace bridge
#endif  // _WIN32
