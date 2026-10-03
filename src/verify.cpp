// dlss5vk verify: run the graph with intra-block captures on a fixture and compare each block-0 kernel output
// against the CPU reference, feeding every check the GPU's own inputs so a mismatch names one kernel.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "json.h"
#include "kernels.h"
#include "nr_graph.h"
#include "nr_model.h"
#include "numeric.h"
#include "reference.h"
#include "vk_context.h"

namespace {

std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("cannot read " + path);
  std::streamsize size = file.tellg();
  file.seekg(0);
  std::vector<uint8_t> bytes((size_t)size);
  file.read(reinterpret_cast<char*>(bytes.data()), size);
  return bytes;
}

std::string argValue(int argc, char** argv, const char* name, const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
  return fallback;
}

std::string executableDirectory(const char* argv0) {
  std::string path = argv0;
  size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

struct Tensor2D {
  std::vector<float> values;
  std::vector<uint16_t> halfBits;
  std::vector<uint8_t> codes;
  uint32_t rows = 0, channels = 0;
  nr::Format format = nr::Format::E4;
  const float* row(uint32_t r) const { return values.data() + (size_t)r * channels; }
};

Tensor2D fetch(vk::Context& context, const nr::Graph& graph, const std::string& name) {
  auto it = graph.boundaries().find(name);
  if (it == graph.boundaries().end()) throw std::runtime_error("missing capture " + name);
  const nr::Activation& activation = *it->second;
  Tensor2D t;
  t.rows = activation.rows;
  t.channels = activation.channels;
  t.format = activation.format;
  std::vector<uint8_t> bytes = context.download(activation.buffer, activation.validBytes());
  size_t count = (size_t)t.rows * t.channels;
  t.values.resize(count);
  if (t.format == nr::Format::E4) {
    t.codes.assign(bytes.begin(), bytes.end());
    for (size_t i = 0; i < count; ++i) t.values[i] = (t.codes[i] & 0x7f) == 0x7f ? 0.0f : num::e4m3ToF32(t.codes[i]);
  } else if (t.format == nr::Format::F16) {
    t.halfBits.resize(count);
    memcpy(t.halfBits.data(), bytes.data(), count * 2);
    for (size_t i = 0; i < count; ++i) t.values[i] = num::f16ToF32(t.halfBits[i]);
  } else {
    memcpy(t.values.data(), bytes.data(), count * 4);
  }
  return t;
}

// Value comparisons against the CPU reference (a NaN equals a NaN, +0 equals -0): verify localizes an arithmetic
// difference to one kernel; `parity` is the bit-exact gate.
size_t g_mismatchedChecks = 0;

struct Stats {
  size_t count = 0, mismatches = 0;
  size_t firstRow = 0, firstColumn = 0;
  float firstActual = 0, firstExpected = 0;
  void add(size_t row, size_t column, float actual, float expected) {
    ++count;
    bool same = actual == expected || (actual != actual && expected != expected) || (actual == 0 && expected == 0);
    if (same) return;
    if (!mismatches) { firstRow = row; firstColumn = column; firstActual = actual; firstExpected = expected; }
    ++mismatches;
  }
  void report(const char* label) const {
    if (!mismatches) printf("  %-28s equal (%zu values)\n", label, count);
    else printf("  %-28s MISMATCH %zu/%zu  first row %zu col %zu: gpu %.8g ref %.8g (gpu bits %04x ref bits %04x)\n",
                label, mismatches, count, firstRow, firstColumn, firstActual, firstExpected, num::f16Bits(firstActual),
                num::f16Bits(firstExpected));
    if (mismatches || !count) ++g_mismatchedChecks;
  }
};
}  // namespace

int runVerify(int argc, char** argv) {
  std::string modelDir = argValue(argc, argv, "--model");
  std::string fixtureDir = argValue(argc, argv, "--fixture");
  std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  uint32_t sampleRows = (uint32_t)atoi(argValue(argc, argv, "--rows", "2048").c_str());
  if (modelDir.empty() || fixtureDir.empty()) {
    fprintf(stderr, "usage: dlss5vk verify --model <dir> --fixture <dir> [--rows N]\n");
    return 2;
  }
  std::string manifestText;
  { std::vector<uint8_t> b = readFile(fixtureDir + "/manifest.json"); manifestText.assign(b.begin(), b.end()); }
  json::Value manifest = json::parse(manifestText);
  uint32_t validWidth = (uint32_t)manifest["sourceDimensions"][0].integer();
  uint32_t validHeight = (uint32_t)manifest["sourceDimensions"][1].integer();

  vk::Context context;
  nr::Model model(context, modelDir, false);
  nr::Kernels kernels(context, shaderDir);
  kernels.setSiluTable(ref::siluTable());
  nr::Geometry geometry = nr::Geometry::fromValid(validWidth, validHeight);
  nr::Graph::Options options;
  options.captureBoundaries = true;
  options.captureIntermediates = true;
  options.fusedBlocks = false;
  nr::Graph graph(context, model, kernels, geometry, options);
  const uint32_t fullRows = geometry.fullWidth * geometry.fullHeight;
  nr::Activation* features = graph.allocate("input features", fullRows, 16, nr::Format::F32);
  std::vector<uint8_t> featureBytes = readFile(fixtureDir + "/" + manifest["inputFeatures"]["file"].str());
  if (featureBytes.size() != features->validBytes()) throw std::runtime_error("input feature size mismatch");
  std::string block0File;
  for (const json::Value& entry : manifest["blocks"].array)
    if (entry["block"].integer() == 0) block0File = fixtureDir + "/" + entry["file"].str();
  if (block0File.empty()) throw std::runtime_error("verify needs the fixture's block-0 reference");
  context.upload(features->buffer, featureBytes.data(), featureBytes.size());
  exec::Commands commands = context.beginCommands();
  graph.record(commands, *features);
  context.endAndSubmit(commands, true);
  printf("graph executed; checking block 0 against the CPU reference on %u rows\n", sampleRows);

  const nr::Tensor& tensor = model.tensor(0);
  nr::FusedLayout layout = nr::preFusedLayout();
  const float* featureValues = reinterpret_cast<const float*>(featureBytes.data());
  const uint32_t width = geometry.fullWidth;
  sampleRows = std::min(sampleRows, fullRows);

  Tensor2D projectedFp16 = fetch(context, graph, "block-0/projectedFp16");
  Tensor2D projected = fetch(context, graph, "block-0/projected");
  {
    Stats raw, quantized;
    for (uint32_t r = 0; r < sampleRows; ++r) {
      for (uint32_t n = 0; n < 32; ++n) {
        float expected = ref::gemmF16Element(tensor, layout.inputAdapter, 16, 32, featureValues + (size_t)r * 16, n);
        raw.add(r, n, projectedFp16.values[(size_t)r * 32 + n], expected);
        quantized.add(r, n, projected.values[(size_t)r * 32 + n], ref::fp8Domain(expected));
      }
    }
    raw.report("pre adapter f16 GEMM");
    quantized.report("pre adapter quantize");
  }

  Tensor2D ffn = fetch(context, graph, "block-0/ffn");
  {
    Stats s;
    ref::GemmRef gemm{&tensor, layout.expand, 32, 128, 0, 0, true};
    for (uint32_t r = 0; r < sampleRows; ++r) {
      for (uint32_t n = 0; n < 128; ++n) {
        float pre = ref::gemmFp8Element(gemm, projected.row(r), n, 0.0f);
        s.add(r, n, ffn.values[(size_t)r * 128 + n], ref::fp8Domain(ref::mpCubicSilu(pre)));
      }
    }
    s.report("FFN expand + SiLU");
  }

  Tensor2D ffnResidual = fetch(context, graph, "block-0/ffnResidual");
  Tensor2D ffnQuantized = fetch(context, graph, "block-0/ffnQuantized");
  {
    Stats raw, quantized;
    ref::GemmRef gemm{&tensor, layout.contractWeights, 128, 32, 0, 0, true};
    for (uint32_t r = 0; r < sampleRows; ++r) {
      for (uint32_t n = 0; n < 32; ++n) {
        float scale = num::f16ToF32(nr::auxHalf(tensor, layout.ffnCosSkip, n));
        float initial = projectedFp16.values[(size_t)r * 32 + n] * scale;
        float expected = ref::gemmFp8Element(gemm, ffn.row(r), n, initial);
        raw.add(r, n, ffnResidual.values[(size_t)r * 32 + n], expected);
        quantized.add(r, n, ffnQuantized.values[(size_t)r * 32 + n], ref::fp8Domain(expected));
      }
    }
    raw.report("FFN contract (raw f16)");
    quantized.report("FFN contract (E4 dual)");
  }

  Tensor2D qkv = fetch(context, graph, "block-0/qkv");
  {
    Stats s;
    ref::GemmRef gemm{&tensor, layout.qkv, 32, 96, 0, 0, true};
    for (uint32_t r = 0; r < sampleRows; ++r)
      for (uint32_t n = 0; n < 96; ++n)
        s.add(r, n, qkv.values[(size_t)r * 96 + n], ref::gemmFp8Element(gemm, ffnQuantized.row(r), n, 0.0f));
    s.report("QKV GEMM");
  }

  Tensor2D normalized = fetch(context, graph, "block-0/normalized");
  {
    Stats s;
    float scale = nr::auxF32(tensor, layout.scale);
    float out[96];
    for (uint32_t r = 0; r < sampleRows; ++r) {
      ref::windowNormalizeRef(qkv.row(r), 0, scale, out);
      for (uint32_t c = 0; c < 96; ++c) s.add(r, c, normalized.values[(size_t)r * 96 + c], out[c]);
    }
    s.report("window normalize");
  }

  Tensor2D attended = fetch(context, graph, "block-0/attended");
  {
    Stats s;
    std::vector<float> out(64 * 32);
    uint32_t windowsX = (width + 7) / 8;
    uint32_t sampleWindows = std::min((sampleRows + width * 8 - 1) / (width * 8) * windowsX,
                                      windowsX * ((geometry.fullHeight + 7) / 8));
    for (uint32_t w = 0; w < sampleWindows; ++w) {
      int windowX = (int)((w % windowsX) * 8), windowY = (int)((w / windowsX) * 8);
      ref::windowAttendRef(normalized.values, width, geometry.fullHeight, 32, 0, windowX, windowY, tensor,
                           layout.relative, out.data());
      for (uint32_t q = 0; q < 64; ++q) {
        int x = windowX + (int)(q & 7), y = windowY + (int)(q >> 3);
        if (x >= (int)width || y >= (int)geometry.fullHeight) continue;
        size_t token = (size_t)y * width + x;
        for (uint32_t c = 0; c < 32; ++c) s.add(token, c, attended.values[token * 32 + c], out[q * 32 + c]);
      }
    }
    s.report("window attention");
  }

  Tensor2D adapterRaw = fetch(context, graph, "block-0/adapterRaw");
  Tensor2D adapter = fetch(context, graph, "block-0");
  {
    Stats raw, quantized;
    ref::GemmRef gemm{&tensor, layout.projection, 32, 32, 0, 0, true};
    for (uint32_t r = 0; r < sampleRows; ++r) {
      for (uint32_t n = 0; n < 32; ++n) {
        float scale = num::f16ToF32(nr::auxHalf(tensor, layout.attnCosSkip, n));
        float initial = ffnResidual.values[(size_t)r * 32 + n] * scale;
        float expected = ref::gemmFp8Element(gemm, attended.row(r), n, initial);
        raw.add(r, n, adapterRaw.values[(size_t)r * 32 + n], expected);
        quantized.add(r, n, adapter.values[(size_t)r * 32 + n], ref::fp8Domain(expected));
      }
    }
    raw.report("projection (raw f16)");
    quantized.report("projection (E4 block-0)");
  }

  {
    std::vector<uint8_t> expected = readFile(block0File);
    if (expected.size() != (size_t)fullRows * 32) throw std::runtime_error("block-0 reference size mismatch");
    Stats s;
    for (uint32_t r = 0; r < sampleRows; ++r)
      for (uint32_t n = 0; n < 32; ++n) {
        uint8_t code = expected[(size_t)r * 32 + n];
        s.add(r, n, adapter.values[(size_t)r * 32 + n], (code & 0x7f) == 0x7f ? 0.0f : num::e4m3ToF32(code));
      }
    s.report("block-0 vs fixture");
  }
  printf("%s\n", g_mismatchedChecks ? "VERIFY: MISMATCH" : "VERIFY: every kernel of block 0 equals the CPU reference (values)");
  return g_mismatchedChecks ? 1 : 0;
}
