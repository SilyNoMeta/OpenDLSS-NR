// dlss5vk: standalone Vulkan DLSS-NR runner.
//   dlss5vk parity --model <nr model dir> --fixture <fixtures/nr512> [--repeat N] [--shaders <dir>] [--dump <dir>]
//   dlss5vk bench  --model <nr model dir> --width W --height H [--frames N]
//   dlss5vk image  --model <nr model dir> --input input.rgba-f32 --output output.rgba-f32 --width W --height H
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "json.h"
#include "kernels.h"
#include "nr_graph.h"
#include "nr_model.h"
#include "numeric.h"
#include "reference.h"
#include "device_factory.h"
#ifdef _WIN32
#include "bridge/bridge_tool.h"
#endif
#include "exec_tape.h"
#include "vk_context.h"

#if defined(_WIN32)
// Hybrid-graphics laptops: ask the NVIDIA Optimus and AMD PowerXpress drivers to run this program on the discrete GPU
// (the only one that can run the network) instead of the power-saving integrated one.
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

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

std::string readText(const std::string& path) {
  std::vector<uint8_t> bytes = readFile(path);
  return std::string(bytes.begin(), bytes.end());
}

std::string argValue(int argc, char** argv, const char* name, const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
  return fallback;
}

bool hasFlag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], name)) return true;
  return false;
}

std::string executableDirectory(const char* argv0) {
  std::string path = argv0;
  size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

// After a frame: a chained wait that gave up means its output is wrong (docs/execution.md, "Barrier-free chaining").
void checkChainTimeouts(const nr::Kernels& kernels) {
  const nr::Kernels::ChainTimeouts timeouts = kernels.chainTimeouts();
  if (timeouts.waits)
    throw std::runtime_error(std::to_string(timeouts.waits) + " chained wait(s) timed out, the first on " + timeouts.counter +
                             ": the frame is invalid (DLSS5VK_CHAIN=0 runs without chaining)");
}

// DLSS5VK_UNFUSED=1 runs the least fused form of the graph (the GLSL reference kernels).
static bool fusedBlocksEnabled() {
  const char* unfused = getenv("DLSS5VK_UNFUSED");
  return !(unfused && !strcmp(unfused, "1"));
}

// ---- parity ----------------------------------------------------------------------------------------------------
// A fixture declares the comparisons it exists for ("checks") and carries one reference per comparison. The whole
// declaration is validated before the GPU runs and nothing is ever dropped from a count: a declared check that
// cannot run, a missing, malformed or unknown reference, or a reference no declared check uses rejects the fixture.
// Every verdict names the equality it proved (docs/numerics.md): bit-exact, equal only up to the sign of zero (a
// failure: not the same bytes), or within a stated tolerance.

struct BoundaryReference { std::string name, file; uint32_t width = 0, height = 0, channels = 0; };

struct FixturePlan {
  uint32_t validWidth = 0, validHeight = 0, fullWidth = 0, fullHeight = 0;
  std::string proxyFile, featuresFile;
  uint32_t proxyWidth = 0, proxyHeight = 0;
  bool checkBoundaries = false, checkHead = false, checkOutput = false;
  std::vector<BoundaryReference> boundaries;                  // graph order
  std::vector<std::pair<std::string, std::string>> omitted;   // boundary, why the fixture has no reference for it
  std::string headFile, outputFile;
  bool outputBytes = false;                                   // an RGBA8 image (a tolerance check), not RGBA f32 halves
};

int64_t fileSize(const std::string& path) {
  std::error_code error;
  const uintmax_t size = std::filesystem::file_size(path, error);
  return error ? -1 : (int64_t)size;
}

FixturePlan planFixture(const json::Value& manifest, const std::string& fixtureDir) {
  FixturePlan plan;
  std::vector<std::string> problems;
  auto fail = [&](const std::string& text) { problems.push_back(text); };
  auto dims = [&](const char* key, uint32_t& width, uint32_t& height) {
    if (!manifest.has(key) || manifest[key].kind != json::Value::Array || manifest[key].size() != 2) {
      fail(std::string("\"") + key + "\" must be [width, height]");
      return;
    }
    width = (uint32_t)manifest[key][0].integer(); height = (uint32_t)manifest[key][1].integer();
  };
  auto positive = [&](const std::string& what, const json::Value& entry, const char* key) -> uint32_t {
    if (!entry.has(key) || entry[key].integer() <= 0) { fail(what + ": \"" + key + "\" missing or not positive"); return 0; }
    return (uint32_t)entry[key].integer();
  };
  // A reference file must exist and hold exactly the bytes its shape says.
  auto reference = [&](const std::string& what, const json::Value& entry, int64_t bytes) -> std::string {
    if (!entry.has("file") || entry["file"].str().empty()) { fail(what + ": no \"file\""); return ""; }
    const std::string path = fixtureDir + "/" + entry["file"].str();
    const int64_t size = fileSize(path);
    if (size < 0) fail(what + ": cannot read " + path);
    else if (size != bytes) fail(what + ": " + path + " holds " + std::to_string(size) + " bytes, its shape needs " + std::to_string(bytes));
    return path;
  };
  dims("sourceDimensions", plan.validWidth, plan.validHeight);
  dims("fullDimensions", plan.fullWidth, plan.fullHeight);
  const int64_t fullRows = (int64_t)plan.fullWidth * plan.fullHeight;

  // The input: the proxy image the features are generated from, or the recorded features themselves.
  if (manifest.has("proxy") == manifest.has("inputFeatures")) fail("exactly one of \"proxy\" and \"inputFeatures\" is the input");
  if (manifest.has("proxy")) {
    const json::Value& proxy = manifest["proxy"];
    plan.proxyWidth = positive("proxy", proxy, "width");
    plan.proxyHeight = positive("proxy", proxy, "height");
    plan.proxyFile = reference("proxy", proxy, (int64_t)plan.proxyWidth * plan.proxyHeight * 16);
    for (const char* key : {"conditioning", "seed", "autoMask"})
      if (!manifest.has(key)) fail(std::string("a proxy fixture needs \"") + key + "\"");
  } else if (manifest.has("inputFeatures")) {
    plan.featuresFile = reference("inputFeatures", manifest["inputFeatures"], fullRows * 16 * 4);
  }

  // The declared check set.
  std::set<std::string> checks;
  if (!manifest.has("checks") || manifest["checks"].kind != json::Value::Array || manifest["checks"].size() == 0) {
    fail("\"checks\" must list what the fixture gates: \"boundaries\", \"head\", \"output\"");
  } else {
    for (const json::Value& check : manifest["checks"].array) {
      if (check.str() != "boundaries" && check.str() != "head" && check.str() != "output") fail("unknown check \"" + check.str() + "\"");
      else if (!checks.insert(check.str()).second) fail("check \"" + check.str() + "\" listed twice");
    }
  }
  plan.checkBoundaries = checks.count("boundaries") != 0;
  plan.checkHead = checks.count("head") != 0;
  plan.checkOutput = checks.count("output") != 0;

  // Boundaries: every stored output the graph has a reference name for is either compared or declared omitted,
  // with a reason, so a shortened export cannot pass as a smaller suite.
  const bool hasBoundaryKeys = (manifest.has("blocks") && manifest["blocks"].size()) ||
                               (manifest.has("transitions") && manifest["transitions"].size()) || manifest.has("omittedBoundaries");
  if (plan.checkBoundaries) {
    std::map<std::string, BoundaryReference> byName;
    auto add = [&](const std::string& name, const json::Value& entry) {
      BoundaryReference ref{name};
      ref.width = positive(name, entry, "width"); ref.height = positive(name, entry, "height");
      ref.channels = positive(name, entry, "channels");
      ref.file = reference(name, entry, (int64_t)ref.width * ref.height * ref.channels);
      if (!byName.emplace(name, ref).second) fail(name + " listed twice");
    };
    if (manifest.has("blocks"))
      for (const json::Value& entry : manifest["blocks"].array) {
        if (!entry.has("block")) { fail("a \"blocks\" entry has no \"block\""); continue; }
        add("block-" + std::to_string(entry["block"].integer()), entry);
      }
    if (manifest.has("transitions"))
      for (const json::Value& entry : manifest["transitions"].array) {
        if (!entry.has("id")) { fail("a \"transitions\" entry has no \"id\""); continue; }
        add("transition-" + entry["id"].str(), entry);
      }
    std::map<std::string, std::string> omitted;
    if (manifest.has("omittedBoundaries"))
      for (const auto& [name, reason] : manifest["omittedBoundaries"].object) {
        if (reason.str().empty()) fail("omitted boundary " + name + " gives no reason");
        omitted[name] = reason.str();
      }
    const std::vector<std::string>& names = nr::Graph::referenceBoundaryNames();
    for (const auto& [name, ref] : byName)
      if (std::find(names.begin(), names.end(), name) == names.end()) fail(name + " is not a boundary of this graph");
    for (const auto& [name, reason] : omitted) {
      if (std::find(names.begin(), names.end(), name) == names.end()) fail("omitted " + name + " is not a boundary of this graph");
      if (byName.count(name)) fail(name + " is both compared and declared omitted");
    }
    std::vector<std::string> unaccounted;
    for (const std::string& name : names) {
      if (byName.count(name)) plan.boundaries.push_back(byName[name]);
      else if (omitted.count(name)) plan.omitted.emplace_back(name, omitted[name]);
      else unaccounted.push_back(name);
    }
    if (!unaccounted.empty()) {
      std::string list;
      for (size_t i = 0; i < unaccounted.size() && i < 8; ++i) list += (i ? ", " : "") + unaccounted[i];
      fail(std::to_string(unaccounted.size()) + " of " + std::to_string(names.size()) + " graph boundaries have neither a reference nor an omission reason (" +
           list + (unaccounted.size() > 8 ? ", ..." : "") + ")");
    }
  } else if (hasBoundaryKeys) {
    fail("the fixture carries boundary references but does not declare the \"boundaries\" check");
  }

  if (plan.checkHead != manifest.has("referenceHead"))
    fail(plan.checkHead ? "the \"head\" check needs \"referenceHead\"" : "\"referenceHead\" is carried but the \"head\" check is not declared");
  else if (plan.checkHead)
    plan.headFile = reference("referenceHead", manifest["referenceHead"], fullRows * 16);

  if (plan.checkOutput != manifest.has("nativeOutput")) {
    fail(plan.checkOutput ? "the \"output\" check needs \"nativeOutput\"" : "\"nativeOutput\" is carried but the \"output\" check is not declared");
  } else if (plan.checkOutput) {
    const json::Value& output = manifest["nativeOutput"];
    const std::string dtype = output.has("dtype") ? output["dtype"].str() : "";
    const uint32_t width = positive("nativeOutput", output, "width"), height = positive("nativeOutput", output, "height");
    if (width != plan.validWidth || height != plan.validHeight) fail("nativeOutput is not the source size");
    if (dtype == "f32") {
      if (plan.proxyFile.empty()) fail("an f32 nativeOutput is composed from the proxy, and the fixture has none");
      plan.outputFile = reference("nativeOutput", output, (int64_t)width * height * 16);
    } else if (dtype == "u8") {
      plan.outputBytes = true;
      plan.outputFile = reference("nativeOutput", output, (int64_t)width * height * 4);
    } else {
      fail("nativeOutput \"dtype\" must be \"f32\" (RGBA f32 halves) or \"u8\" (RGBA8)");
    }
  }

  if (!problems.empty()) {
    for (const std::string& problem : problems) fprintf(stderr, "fixture: %s\n", problem.c_str());
    throw std::runtime_error("fixture rejected (" + std::to_string(problems.size()) + " problem" + (problems.size() == 1 ? "" : "s") +
                             "); nothing was run");
  }
  return plan;
}

// One comparison, element by element. `signedZeros` counts elements that differ only in the sign of a zero;
// `tolerated` those within the check's stated tolerance; `differing` everything else.
struct Tally {
  size_t count = 0, differing = 0, signedZeros = 0, tolerated = 0;
  size_t first = SIZE_MAX;
  double maxAbs = 0, sumSquares = 0;
  std::map<int, size_t> codeDeltas;   // E4M3: signed code distance, clamped to +-9
  void miss(size_t index, double difference) {
    if (first == SIZE_MAX) first = index;
    ++differing;
    maxAbs = std::max(maxAbs, std::fabs(difference));
    sumSquares += difference * difference;
  }
};

Tally compareE4(const std::vector<uint8_t>& actual, const std::vector<uint8_t>& expected) {
  Tally t;
  t.count = expected.size();
  for (size_t i = 0; i < expected.size(); ++i) {
    const uint8_t a = actual[i], e = expected[i];
    if (a == e) continue;
    if ((a & 0x7f) == 0 && (e & 0x7f) == 0) { ++t.signedZeros; continue; }
    t.miss(i, (double)num::e4m3ToF32(a) - num::e4m3ToF32(e));
    const int signedA = (a & 0x80) ? -(int)(a & 0x7f) : (int)(a & 0x7f);
    const int signedE = (e & 0x80) ? -(int)(e & 0x7f) : (int)(e & 0x7f);
    t.codeDeltas[std::max(-9, std::min(9, signedA - signedE))]++;
  }
  return t;
}

// f32 elements by bit pattern: +0 and -0 differ, a NaN equals only the same NaN.
Tally compareBits(const float* actual, const float* expected, size_t count) {
  Tally t;
  t.count = count;
  for (size_t i = 0; i < count; ++i) {
    uint32_t a, e;
    memcpy(&a, actual + i, 4); memcpy(&e, expected + i, 4);
    if (a == e) continue;
    if (((a | e) & 0x7fffffffu) == 0) { ++t.signedZeros; continue; }
    t.miss(i, (double)actual[i] - expected[i]);
  }
  return t;
}

// Truncation (toward zero) to the half grid: the composite's publication of the neural result.
float truncateHalf(float value) {
  uint32_t bits; memcpy(&bits, &value, 4);
  const uint32_t signBit = (bits >> 16) & 0x8000u, exponent = (bits >> 23) & 0xffu, mantissa = bits & 0x7fffffu;
  uint32_t halfBits;
  if (exponent == 0xffu) halfBits = signBit | (mantissa ? 0x7e00u : 0x7c00u);
  else {
    const int halfExponent = (int)exponent - 112;
    if (halfExponent >= 31) halfBits = signBit | 0x7c00u;
    else if (halfExponent <= 0) halfBits = halfExponent < -10 ? signBit : signBit | ((mantissa | 0x800000u) >> (14 - halfExponent));
    else halfBits = signBit | ((uint32_t)halfExponent << 10) | (mantissa >> 13);
  }
  return num::f16ToF32((uint16_t)halfBits);
}

// The composed RGB of one frame with no history, as the demo composite publishes it:
// neural = clamp((head / 32 + centred) * 8 + 0.5, 0, 1), truncated to the half grid. `inner * 8` is exact, so there
// is one rounding whether or not the last multiply-add is contracted.
float composed(float head, float centred) {
  const float inner = std::fmaf(head, 0.03125f, centred);
  return truncateHalf(std::fmin(std::fmax(inner * 8.0f + 0.5f, 0.0f), 1.0f));
}

Tally compareOutput(const FixturePlan& plan, const float* head, const std::vector<uint8_t>& input, const std::vector<uint8_t>& reference) {
  const float* values = reinterpret_cast<const float*>(input.data());
  Tally t;
  t.count = (size_t)plan.validWidth * plan.validHeight * 3;
  for (uint32_t y = 0; y < plan.validHeight; ++y)
    for (uint32_t x = 0; x < plan.validWidth; ++x)
      for (uint32_t c = 0; c < 3; ++c) {
        const float h = head[((size_t)y * plan.fullWidth + x) * 4 + c];
        const size_t pixel = (size_t)y * plan.validWidth + x;
        if (plan.outputBytes) {
          // An RGBA8 capture is the same image quantized to eight bits by a rounding this repository does not know
          // exactly, so the check is a tolerance: one code either way. The centred proxy is the features' lane 4 + c.
          const float centred = values[((size_t)y * plan.fullWidth + x) * 16 + 4 + c];
          const int published = (int)std::fmin(255.0f, std::fmax(0.0f, std::floor(composed(h, centred) * 255.0f + 0.5f)));
          const int expected = reference[pixel * 4 + c];
          if (published == expected) continue;
          if (std::abs(published - expected) == 1) { ++t.tolerated; continue; }
          t.miss(pixel * 3 + c, published - expected);
        } else {
          const float centred = std::fmaf(values[pixel * 4 + c], 0.125f, -0.0625f);
          const float published = composed(h, centred);
          const float expected = reinterpret_cast<const float*>(reference.data())[pixel * 4 + c];
          const Tally one = compareBits(&published, &expected, 1);
          t.signedZeros += one.signedZeros;
          if (one.differing) t.miss(pixel * 3 + c, (double)published - expected);
        }
      }
  return t;
}

enum class Verdict { BitExact, SignedZeroOnly, WithinTolerance, Mismatch };

Verdict verdictOf(const Tally& t) {
  if (t.differing) return Verdict::Mismatch;
  if (t.signedZeros) return Verdict::SignedZeroOnly;
  return t.tolerated ? Verdict::WithinTolerance : Verdict::BitExact;
}

std::string describe(const Tally& t) {
  char text[256];
  switch (verdictOf(t)) {
    case Verdict::BitExact: snprintf(text, sizeof(text), "bit-exact (%zu)", t.count); break;
    case Verdict::SignedZeroOnly:
      snprintf(text, sizeof(text), "NOT BIT-EXACT: %zu of %zu differ only in the sign of zero", t.signedZeros, t.count); break;
    case Verdict::WithinTolerance:
      snprintf(text, sizeof(text), "within one code (%zu exact, %zu one code off)", t.count - t.tolerated, t.tolerated); break;
    case Verdict::Mismatch:
      snprintf(text, sizeof(text), "MISMATCH %zu/%zu (%.4f%%) max|d| %.6g rmse %.6g%s", t.differing, t.count, 100.0 * t.differing / t.count,
               t.maxAbs, std::sqrt(t.sumSquares / t.count), t.signedZeros ? " (+ signed zeros)" : "");
      break;
  }
  return text;
}

// A graph built for one schedule, recorded and submitted `submissions` times on the same buffers (as the demo
// resubmits its frame); `repeatable` says whether every submission gave the first one's head.
struct GraphRun {
  std::vector<uint8_t> head;
  std::map<std::string, std::vector<uint8_t>> boundaries;
  double minGpuMs = 1e30;
  uint32_t dispatches = 0;
  bool chained = false, repeatable = true;
};

// `recorder`: when the device the kernels were made on is a tape recorder, the graph is walked once into a tape and
// every submission replays it on the real device (the way a host runs it).
GraphRun runGraph(exec::Device& context, nr::Model& model, nr::Kernels& kernels, const nr::Geometry& geometry,
                  const nr::Activation& features, bool chain, bool capture, int submissions, exec::TapeRecorder* recorder = nullptr,
                  exec::Device* target = nullptr) {
  const bool chainBefore = kernels.chainEnabled();
  kernels.setChainEnabled(chainBefore && chain);
  GraphRun run;
  {
    nr::Graph::Options options;
    options.captureBoundaries = capture;
    // DLSS5VK_CAPTURE_BLOCK=N: the instrumented run also captures block N's intermediates (parity --dump writes them).
    // It takes the unfused route, like dlss5vk verify, so only the compatibility backend keeps its schedule.
    options.captureIntermediates = capture && getenv("DLSS5VK_CAPTURE_BLOCK") != nullptr;
    options.fusedBlocks = fusedBlocksEnabled();
    nr::Graph graph(context, model, kernels, geometry, options);
    run.chained = graph.chained();
    exec::Tape tape;
    if (recorder) { graph.record(recorder->stream(), features); tape = recorder->take(); }
    exec::Timer queries = context.createTimestampPool(2);
    for (int submission = 0; submission < submissions; ++submission) {
      context.nextFrame();
      exec::Commands commands = context.beginCommands();
      if (recorder) target->resetTimestamps(commands, queries, 2); else context.resetTimestamps(commands, queries, 2);
      if (recorder) target->writeTimestamp(commands, queries, 0, true); else context.writeTimestamp(commands, queries, 0, true);
      if (recorder) tape.replay(*target, commands); else graph.record(commands, features);
      if (recorder) target->writeTimestamp(commands, queries, 1, false); else context.writeTimestamp(commands, queries, 1, false);
      context.endAndSubmit(commands, true);
      checkChainTimeouts(kernels);
      std::vector<double> stamps = context.readTimestampsMs(queries, 2);
      run.minGpuMs = std::min(run.minGpuMs, stamps[1] - stamps[0]);
      run.dispatches = kernels.dispatchCount();
      std::vector<uint8_t> head = context.download(graph.head().buffer, graph.head().validBytes());
      if (submission == 0) run.head = std::move(head);
      else run.repeatable = run.repeatable && head == run.head;
    }
    context.destroyTimestampPool(queries);
    for (const auto& [name, activation] : graph.boundaries())
      run.boundaries[name] = context.download(activation->buffer, activation->validBytes());
  }
  kernels.setChainEnabled(chainBefore);
  return run;
}

int runParity(int argc, char** argv) {
  std::string modelDir = argValue(argc, argv, "--model");
  std::string fixtureDir = argValue(argc, argv, "--fixture");
  std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  std::string dumpDir = argValue(argc, argv, "--dump");
  const int repeats = std::max(2, atoi(argValue(argc, argv, "--repeat", "3").c_str()));
  if (modelDir.empty() || fixtureDir.empty()) {
    fprintf(stderr, "usage: dlss5vk parity --model <dir> --fixture <dir> [--repeat N] [--shaders <dir>] [--dump <dir>]\n");
    return 2;
  }
  json::Value manifest = json::parse(readText(fixtureDir + "/manifest.json"));
  const FixturePlan plan = planFixture(manifest, fixtureDir);

  const std::unique_ptr<exec::Device> owned = makeDevice();
  // DLSS5VK_TAPE=1: every graph is walked once into a tape and its submissions replay the tape.
  const bool taped = getenv("DLSS5VK_TAPE") && atoi(getenv("DLSS5VK_TAPE")) != 0;
  exec::TapeRecorder tapeRecorder(*owned);
  exec::Device& context = taped ? static_cast<exec::Device&>(tapeRecorder) : *owned;
  exec::TapeRecorder* const recorder = taped ? &tapeRecorder : nullptr;
  printf("device: %s (%s, %s backend%s)\n", context.deviceName().c_str(), context.api(), exec::backendName(context.backend()),
         taped ? ", graph replayed from a tape" : "");
  auto started = std::chrono::steady_clock::now();
  nr::Model model(context, modelDir, !hasFlag(argc, argv, "--no-verify"));
  printf("model loaded in %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
  nr::Kernels kernels(context, shaderDir);
  kernels.setSiluTable(ref::siluTable());
  nr::Geometry geometry = nr::Geometry::fromValid(plan.validWidth, plan.validHeight);
  printf("geometry %ux%u -> full %ux%u, levels", plan.validWidth, plan.validHeight, geometry.fullWidth, geometry.fullHeight);
  for (auto level : geometry.levels) printf(" %ux%u", level.width, level.height);
  printf("\n");
  if (geometry.fullWidth != plan.fullWidth || geometry.fullHeight != plan.fullHeight)
    throw std::runtime_error("fixture full dimensions disagree with the runtime profile");
  printf("checks:%s%s%s\n", plan.checkBoundaries ? " boundaries" : "", plan.checkHead ? " head" : "", plan.checkOutput ? " output" : "");
  if (plan.checkBoundaries) {
    printf("  boundaries: %zu references, %zu declared omitted\n", plan.boundaries.size(), plan.omitted.size());
    std::map<std::string, std::vector<std::string>> byReason;
    for (const auto& [name, reason] : plan.omitted) byReason[reason].push_back(name);
    for (const auto& [reason, names] : byReason)
      printf("    omitted (%s): %s%s%s\n", reason.c_str(), names.front().c_str(), names.size() > 1 ? " .. " : "",
             names.size() > 1 ? names.back().c_str() : "");
  }
  if (plan.checkOutput) printf("  output: %s\n", plan.outputBytes ? "RGBA8, compared within one code" : "RGBA f32 halves, bit-exact");

  // The features the graph reads, generated from the proxy on the GPU or uploaded as recorded.
  const uint32_t fullRows = geometry.fullWidth * geometry.fullHeight;
  nr::Activation features;
  features.format = nr::Format::F32; features.rows = fullRows; features.channels = 16; features.allocRows = nr::alignRows(fullRows);
  features.label = "input features";
  features.buffer = context.createBuffer((VkDeviceSize)features.allocRows * 16 * 4, false, "input features");
  context.fillZero(features.buffer);
  std::vector<uint8_t> inputBytes = readFile(plan.proxyFile.empty() ? plan.featuresFile : plan.proxyFile);
  vk::Buffer proxyBuffer;
  if (!plan.proxyFile.empty()) {
    proxyBuffer = context.createBuffer(inputBytes.size(), false, "proxy");
    context.upload(proxyBuffer, inputBytes.data(), inputBytes.size());
    const json::Value& conditioning = manifest["conditioning"];
    nr::Kernels::PreprocessArgs preprocess{geometry.fullWidth, geometry.fullHeight, plan.validWidth, plan.validHeight,
                                           plan.proxyWidth, plan.proxyHeight, (uint32_t)manifest["seed"].integer(),
                                           manifest["autoMask"].boolean, (float)conditioning["localTone"].number,
                                           (float)conditioning["localStructure"].number, (float)conditioning["skinStructure"].number,
                                           (float)conditioning["style"].number};
    exec::Commands commands = context.beginCommands();
    kernels.preprocessFromProxy(commands, proxyBuffer, features, preprocess);
    if (recorder) tapeRecorder.take().replay(*owned, commands);
    context.endAndSubmit(commands, true);
  } else {
    context.upload(features.buffer, inputBytes.data(), inputBytes.size());
  }
  if (!dumpDir.empty()) {
    std::vector<uint8_t> bytes = context.download(features.buffer, features.validBytes());
    std::ofstream(dumpDir + "/features.f32", std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  size_t failures = 0, tolerated = 0, bitExact = 0;
  auto require = [&](bool ok, const char* what) {
    printf("%s: %s\n", what, ok ? "identical" : "DIFFERENT");
    if (!ok) ++failures;
  };

  // The production schedule (no captures, chaining as configured) is what the head and output references are
  // compared against, resubmitted several times: its head must not change between submissions.
  const GraphRun production = runGraph(context, model, kernels, geometry, features, true, false, repeats, recorder, owned.get());
  printf("production schedule (%s, no captures): %u dispatches, GPU %.3f ms (min of %d)\n",
         production.chained ? "counter chaining" : "barriers", production.dispatches, production.minGpuMs, repeats);
  printf("NaN weight codes replaced: %zu\n", model.nanWeightsReplaced());
  require(production.repeatable, ("head over " + std::to_string(repeats) + " production submissions").c_str());
  // Chaining is scheduling, not arithmetic: the same graph with a barrier after every launch gives the same head.
  if (production.chained)
    require(runGraph(context, model, kernels, geometry, features, false, false, 1, recorder, owned.get()).head == production.head, "head with barriers instead of chaining");
  else
    printf("chaining is off in this configuration: the barrier schedule is the production schedule\n");
  // Boundaries come from an instrumented schedule, which adds a copy and two barriers at every boundary (and
  // materializes the deferred projections); that must not change the head either.
  GraphRun instrumented;
  if (plan.checkBoundaries) {
    instrumented = runGraph(context, model, kernels, geometry, features, true, true, 1, recorder, owned.get());
    require(instrumented.head == production.head, "head of the instrumented schedule (boundary captures) vs production");
  }

  printf("\n");
  auto count = [&](const Tally& t) {
    const Verdict v = verdictOf(t);
    if (v == Verdict::BitExact) ++bitExact;
    else if (v == Verdict::WithinTolerance) ++tolerated;
    else ++failures;
  };
  bool codeHistogramShown = false;
  for (const BoundaryReference& ref : plan.boundaries) {
    auto it = instrumented.boundaries.find(ref.name);
    if (it == instrumented.boundaries.end()) {
      printf("%-16s NOT CAPTURED by this route\n", ref.name.c_str());
      ++failures;
      continue;
    }
    const std::vector<uint8_t> expected = readFile(ref.file);
    if (it->second.size() != expected.size()) {
      printf("%-16s SHAPE: the graph has %zu bytes, the reference %zu\n", ref.name.c_str(), it->second.size(), expected.size());
      ++failures;
      continue;
    }
    if (!dumpDir.empty())
      std::ofstream(dumpDir + "/" + ref.name + ".u8", std::ios::binary).write(reinterpret_cast<const char*>(it->second.data()), it->second.size());
    const Tally t = compareE4(it->second, expected);
    count(t);
    printf("%-16s %ux%ux%u %s", ref.name.c_str(), ref.width, ref.height, ref.channels, describe(t).c_str());
    if (t.differing) {
      const size_t pixel = t.first / ref.channels;
      printf(" first@ x%zu y%zu c%zu: got 0x%02x exp 0x%02x", pixel % ref.width, pixel / ref.width, t.first % ref.channels,
             it->second[t.first], expected[t.first]);
      if (!codeHistogramShown) {
        printf("  code-delta histogram:");
        for (auto& [delta, n] : t.codeDeltas) printf(" %+d:%zu", delta, n);
        codeHistogramShown = true;
      }
    }
    printf("\n");
  }

  const float* head = reinterpret_cast<const float*>(production.head.data());
  const size_t headValues = (size_t)fullRows * 4;
  if (plan.checkHead) {
    const std::vector<uint8_t> reference = readFile(plan.headFile);
    const Tally t = compareBits(head, reinterpret_cast<const float*>(reference.data()), headValues);
    count(t);
    printf("head vs reference: %s", describe(t).c_str());
    if (t.differing) printf(" first@ x%zu y%zu c%zu", (t.first / 4) % geometry.fullWidth, (t.first / 4) / geometry.fullWidth, t.first % 4);
    printf("\n");
  }
  if (plan.checkOutput) {
    const Tally t = compareOutput(plan, head, inputBytes, readFile(plan.outputFile));
    count(t);
    printf("composed RGB vs native output: %s", describe(t).c_str());
    if (t.differing) printf(" first@ x%zu y%zu c%zu", (t.first / 3) % plan.validWidth, (t.first / 3) / plan.validWidth, t.first % 3);
    printf("\n");
  }
  double sum = 0, sumSquares = 0; size_t nonFinite = 0;
  for (size_t i = 0; i < headValues; ++i) {
    if (!std::isfinite(head[i])) { ++nonFinite; continue; }
    sum += head[i]; sumSquares += (double)head[i] * head[i];
  }
  printf("head: %zu values, mean %.6f, rms %.6f, non-finite %zu\n", headValues, sum / headValues, std::sqrt(sumSquares / headValues), nonFinite);
  if (!dumpDir.empty()) {
    std::ofstream(dumpDir + "/head.f32", std::ios::binary).write(reinterpret_cast<const char*>(production.head.data()), production.head.size());
    for (const auto& [name, bytes] : instrumented.boundaries) {
      if (name.find('/') == std::string::npos) continue;   // DLSS5VK_CAPTURE_BLOCK intermediates, as block-N_step.bin
      std::string file = name;
      std::replace(file.begin(), file.end(), '/', '_');
      std::ofstream(dumpDir + "/" + file + ".bin", std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  if (proxyBuffer.buffer) context.destroyBuffer(proxyBuffer);
  context.destroyBuffer(features.buffer);

  printf("\nVERDICT: %s - %zu bit-exact, %zu within tolerance, %zu failed\n", failures ? "FAIL" : "PASS", bitExact, tolerated, failures);
  return failures ? 1 : 0;
}

int runBench(int argc, char** argv) {
  std::string modelDir = argValue(argc, argv, "--model");
  std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  uint32_t width = (uint32_t)atoi(argValue(argc, argv, "--width", "768").c_str());
  uint32_t height = (uint32_t)atoi(argValue(argc, argv, "--height", "768").c_str());
  int frames = atoi(argValue(argc, argv, "--frames", "10").c_str());
  if (modelDir.empty()) {
    fprintf(stderr, "usage: dlss5vk bench --model <dir> [--width W --height H | --fixture <proxy fixture>] [--frames N] [--tape] [--json <file>]\n");
    return 2;
  }
  // --fixture: the frames a proxy fixture describes (its size, its features generated from its proxy), so that every
  // execution mode a harness compares runs the same frames. Without it, a synthetic field of --width x --height.
  const std::string fixtureDir = argValue(argc, argv, "--fixture");
  json::Value fixtureManifest;
  FixturePlan fixture;
  std::vector<uint8_t> fixtureProxy;
  if (!fixtureDir.empty()) {
    fixtureManifest = json::parse(readText(fixtureDir + "/manifest.json"));
    fixture = planFixture(fixtureManifest, fixtureDir);
    if (fixture.proxyFile.empty()) throw std::runtime_error("bench --fixture needs a proxy fixture");
    fixtureProxy = readFile(fixture.proxyFile);
    width = fixture.validWidth; height = fixture.validHeight;
  }
  // Preparation (once) and the recurring frame are reported apart: device and model setup, kernels, the graph
  // build, and the first frame, which pays for the pipeline and PTX compilation the later frames reuse.
  using Clock = std::chrono::steady_clock;
  auto seconds = [](Clock::time_point from) { return std::chrono::duration<double>(Clock::now() - from).count(); };
  auto mib = [](VkDeviceSize bytes) { return bytes / 1048576.0; };
  auto started = Clock::now();
  const std::unique_ptr<exec::Device> owned = makeDevice();
  // --tape: walk the graph once and replay its recorded operations every frame (what a host does), instead of
  // walking it every frame.
  const bool taped = hasFlag(argc, argv, "--tape");
  exec::TapeRecorder tapeRecorder(*owned);
  exec::Device& context = taped ? static_cast<exec::Device&>(tapeRecorder) : *owned;
  exec::Tape tape;
  vk::Context* const vulkan = dynamic_cast<vk::Context*>(owned.get());
  printf("device: %s (%s, %s backend)\n", context.deviceName().c_str(), context.api(), exec::backendName(context.backend()));
  const double contextSeconds = seconds(started);
  started = Clock::now();
  nr::Model model(context, modelDir, false);
  const double modelSeconds = seconds(started);
  const VkDeviceSize modelBytes = context.memoryUse().deviceLocal;
  started = Clock::now();
  nr::Kernels kernels(context, shaderDir);
  kernels.setSiluTable(ref::siluTable());
  const double kernelSeconds = seconds(started);
  nr::Geometry geometry = nr::Geometry::fromValid(width, height);
  const uint32_t fullRows = geometry.fullWidth * geometry.fullHeight;
  std::vector<float> synthetic((size_t)fullRows * 16);
  for (size_t i = 0; i < synthetic.size(); ++i) synthetic[i] = num::roundF16(std::sin(i * 0.0017f) * 0.125f);
  std::unique_ptr<nr::Graph> graph;
  nr::Activation* features = nullptr;
  exec::Buffer fixtureBuffer;
  std::optional<nr::Kernels::PreprocessArgs> preprocess;
  exec::Tape preprocessTape;
  auto build = [&]() {
    graph.reset();
    graph = std::make_unique<nr::Graph>(context, model, kernels, geometry, nr::Graph::Options{.fusedBlocks = fusedBlocksEnabled()});
    features = graph->allocate("input features", fullRows, 16, nr::Format::F32);
    if (fixtureProxy.empty()) {
      context.upload(features->buffer, synthetic.data(), synthetic.size() * 4);
    } else {
      // The fixture's frame is its proxy: every frame generates the features from it and then runs the graph, which
      // is what a host's frame does (and what dlss5vk bridge runs on the Vulkan side).
      if (!fixtureBuffer.buffer) {
        fixtureBuffer = context.createBuffer(fixtureProxy.size(), false, "proxy");
        context.upload(fixtureBuffer, fixtureProxy.data(), fixtureProxy.size());
      }
      const json::Value& conditioning = fixtureManifest["conditioning"];
      preprocess = nr::Kernels::PreprocessArgs{geometry.fullWidth, geometry.fullHeight, fixture.validWidth, fixture.validHeight,
                                               fixture.proxyWidth, fixture.proxyHeight, (uint32_t)fixtureManifest["seed"].integer(),
                                               fixtureManifest["autoMask"].boolean, (float)conditioning["localTone"].number,
                                               (float)conditioning["localStructure"].number, (float)conditioning["skinStructure"].number,
                                               (float)conditioning["style"].number};
      if (taped) { kernels.preprocessFromProxy(tapeRecorder.stream(), fixtureBuffer, *features, *preprocess); preprocessTape = tapeRecorder.take(); }
    }
    if (taped) { graph->record(tapeRecorder.stream(), *features); tape = tapeRecorder.take(); }
  };
  auto emitPreprocess = [&](exec::Commands commands) {
    if (!preprocess) return;
    if (taped) preprocessTape.replay(*owned, commands);
    else kernels.preprocessFromProxy(commands, fixtureBuffer, *features, *preprocess);
  };
  auto emitGraph = [&](exec::Commands commands) {
    if (taped) tape.replay(*owned, commands);
    else graph->record(commands, *features);
  };
  auto emit = [&](exec::Commands commands) { emitPreprocess(commands); emitGraph(commands); };
  started = Clock::now();
  build();
  const double graphSeconds = seconds(started);
  started = Clock::now();
  // Warm compile. A chained wait that timed out here (a GPU that schedules the launches differently) rebuilds the
  // graph with barriers, as the demo does, instead of failing the run; later frames still fail on one.
  {
    exec::Commands commands = context.beginCommands();
    emit(commands);
    context.endAndSubmit(commands, true);
    const nr::Kernels::ChainTimeouts timeouts = kernels.chainTimeouts();
    if (timeouts.waits && kernels.chainEnabled()) {
      fprintf(stderr, "%u chained wait(s) timed out, the first on %s: rebuilding with barriers (DLSS5VK_CHAIN=0)\n",
              timeouts.waits, timeouts.counter.c_str());
      kernels.setChainEnabled(false);
      kernels.resetChainTimeouts();
      build();
      context.nextFrame();
      commands = context.beginCommands();
      emit(commands);
      context.endAndSubmit(commands, true);
    }
    checkChainTimeouts(kernels);
  }
  const double firstFrameSeconds = seconds(started);
  printf("preparation: device %.2f s, model %.2f s, kernels %.2f s, graph %.2f s, first frame (compiles) %.2f s\n",
         contextSeconds, modelSeconds, kernelSeconds, graphSeconds, firstFrameSeconds);
  const exec::Device::MemoryUse memory = context.memoryUse();
  printf("memory: %.0f MiB device-local in use (raw tensors %.0f MiB, re-laid weights %.0f MiB, activations and scratch "
         "%.0f MiB), peak %.0f MiB; host-visible %.0f MiB\n",
         mib(memory.deviceLocal), mib(modelBytes), mib(model.matrixBytes()),
         mib(memory.deviceLocal - modelBytes - model.matrixBytes()), mib(memory.peakDeviceLocal), mib(memory.hostVisible));
  exec::Timer queries = context.createTimestampPool(3);
  std::vector<double> samples, submissionSamples;   // the graph alone, and everything the frame submits
  std::vector<double> wallSamples, recordSamples, submitSamples;
  auto milliseconds = [](std::chrono::high_resolution_clock::time_point from) {
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - from).count();
  };
  // --secondary: the demo's (and a host's) way, the graph recorded once into a secondary command buffer that every
  // frame's primary executes; the descriptor sets stay in one pool for the secondary's lifetime.
  const bool secondary = hasFlag(argc, argv, "--secondary");
  VkCommandPool secondaryPool = VK_NULL_HANDLE;
  VkCommandBuffer prerecorded = VK_NULL_HANDLE;
  if (secondary) {
    if (!vulkan) throw std::runtime_error("--secondary is a Vulkan secondary command buffer: not available on this API");
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = vulkan->queueFamily();
    VK_CHECK(vkCreateCommandPool(vulkan->device(), &poolInfo, nullptr, &secondaryPool));
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = secondaryPool; allocate.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY; allocate.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(vulkan->device(), &allocate, &prerecorded));
    VkCommandBufferInheritanceInfo inheritance{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    begin.pInheritanceInfo = &inheritance;
    vulkan->resetDescriptorPool(0);
    VK_CHECK(vkBeginCommandBuffer(prerecorded, &begin));
    graph->record(prerecorded, *features);
    VK_CHECK(vkEndCommandBuffer(prerecorded));
    printf("recorded once into a secondary command buffer\n");
  }
  for (int frame = 0; frame < frames; ++frame) {
    const auto frameStart = std::chrono::high_resolution_clock::now();
    if (!secondary) context.nextFrame();
    exec::Commands commands = context.beginCommands();
    owned->resetTimestamps(commands, queries, 3);
    owned->writeTimestamp(commands, queries, 0, true);
    // Recording is host time every frame (an immediate context executes while it records: there the split between
    // recording and waiting is where the driver chooses to flush, and only their sum means something).
    const auto recordStart = std::chrono::high_resolution_clock::now();
    emitPreprocess(commands);
    owned->writeTimestamp(commands, queries, 1, false);
    if (secondary) vkCmdExecuteCommands(vk::handle(commands), 1, &prerecorded);
    else emitGraph(commands);
    recordSamples.push_back(milliseconds(recordStart));
    owned->writeTimestamp(commands, queries, 2, false);
    const auto submitStart = std::chrono::high_resolution_clock::now();
    context.endAndSubmit(commands, true);
    submitSamples.push_back(milliseconds(submitStart));
    wallSamples.push_back(milliseconds(frameStart));
    checkChainTimeouts(kernels);
    std::vector<double> stamps = context.readTimestampsMs(queries, 3);
    samples.push_back(stamps[2] - stamps[1]);
    submissionSamples.push_back(stamps[2] - stamps[0]);
    printf("frame %d: %.3f ms GPU (%u dispatches)\n", frame, samples.back(), kernels.dispatchCount());
  }
  // Machine-readable samples, in frame order, for a harness that alternates configurations (--json <file>).
  const std::string jsonPath = argValue(argc, argv, "--json");
  if (!jsonPath.empty()) {
    std::ofstream json(jsonPath, std::ios::binary);
    auto list = [&](const char* name, const std::vector<double>& values) {
      json << "  \"" << name << "\": [";
      for (size_t i = 0; i < values.size(); ++i) json << (i ? ", " : "") << values[i];
      json << "],\n";
    };
    json << "{\n  \"api\": \"" << context.api() << "\", \"backend\": \"" << exec::backendName(context.backend()) << "\",\n";
    json << "  \"device\": \"" << context.deviceName() << "\", \"sm_count\": " << context.smCount() << ",\n";
    json << "  \"shader_fp8\": " << (context.nativeFp8() ? "true" : "false") << ", \"chained\": " << (graph->chained() ? "true" : "false")
         << ", \"secondary\": " << (secondary ? "true" : "false") << ", \"taped\": " << (taped ? "true" : "false") << ",\n";
    json << "  \"width\": " << width << ", \"height\": " << height << ", \"full_width\": " << geometry.fullWidth << ", \"full_height\": "
         << geometry.fullHeight << ", \"dispatches\": " << kernels.dispatchCount() << ",\n";
    json << "  \"preparation_s\": {\"device\": " << contextSeconds << ", \"model\": " << modelSeconds << ", \"kernels\": " << kernelSeconds
         << ", \"graph\": " << graphSeconds << ", \"first_frame\": " << firstFrameSeconds << "},\n";
    json << "  \"memory_bytes\": {\"device_local\": " << memory.deviceLocal << ", \"peak_device_local\": " << memory.peakDeviceLocal
         << ", \"host_visible\": " << memory.hostVisible << ", \"raw_tensors\": " << modelBytes << ", \"weights\": " << model.matrixBytes() << "},\n";
    // gpu_ms is the graph alone; gpu_submission_ms is everything the frame submits (with --fixture, the features
    // generated from the proxy and then the graph).
    json << "  \"frame_work\": \"" << (preprocess ? "preprocess+graph" : "graph") << "\",\n";
    list("gpu_ms", samples);
    list("gpu_submission_ms", submissionSamples);
    list("record_ms", recordSamples);
    list("submit_wait_ms", submitSamples);
    json << "  \"host_frame_ms\": [";
    for (size_t i = 0; i < wallSamples.size(); ++i) json << (i ? ", " : "") << wallSamples[i];
    json << "]\n}\n";
  }
  auto median = [](std::vector<double> values) { std::sort(values.begin(), values.end()); return values[values.size() / 2]; };
  std::sort(samples.begin(), samples.end());
  printf("median %.3f ms, min %.3f ms over %d frames at %ux%u (full %ux%u)\n", samples[samples.size() / 2],
         samples.front(), frames, width, height, geometry.fullWidth, geometry.fullHeight);
  if (preprocess) printf("with the features generated from the proxy: median %.3f ms GPU per frame\n", median(submissionSamples));
  printf("host frame: median %.3f ms (recording %.3f ms + submission and GPU wait %.3f ms; readback excluded)\n", median(wallSamples),
         median(recordSamples), median(submitSamples));
  context.destroyTimestampPool(queries);
  if (fixtureBuffer.buffer) context.destroyBuffer(fixtureBuffer);
  if (secondaryPool) vkDestroyCommandPool(vulkan->device(), secondaryPool, nullptr);
  return 0;
}

int runImage(int argc, char** argv) {
  const std::string modelDir = argValue(argc, argv, "--model");
  const std::string inputPath = argValue(argc, argv, "--input");
  const std::string outputPath = argValue(argc, argv, "--output");
  const std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  if (modelDir.empty() || inputPath.empty() || outputPath.empty() || argValue(argc, argv, "--width").empty() ||
      argValue(argc, argv, "--height").empty()) {
    fprintf(stderr,
            "usage: dlss5vk image --model <dir> --input <rgba-f32> --output <rgba-f32> --width W --height H "
            "[--tone F --structure F --skin F --style 0|1|2 --intensity F --auto-mask 0|1 --seed N --repeat N]\n");
    return 2;
  }
  auto integer = [&](const char* name, const char* fallback, uint64_t low, uint64_t high) {
    const std::string text = argValue(argc, argv, name, fallback);
    char* end = nullptr;
    const unsigned long long value = strtoull(text.c_str(), &end, 10);
    if (!end || *end || value < low || value > high)
      throw std::runtime_error(std::string(name) + " must be an integer in [" + std::to_string(low) + ", " +
                               std::to_string(high) + "]");
    return (uint32_t)value;
  };
  auto number = [&](const char* name, const char* fallback, float low, float high) {
    const std::string text = argValue(argc, argv, name, fallback);
    char* end = nullptr;
    const float value = strtof(text.c_str(), &end);
    if (!end || *end || !std::isfinite(value) || value < low || value > high)
      throw std::runtime_error(std::string(name) + " must be a number in [" + std::to_string(low) + ", " +
                               std::to_string(high) + "]");
    return value;
  };
  const uint32_t width = integer("--width", "0", 33, 16384);
  const uint32_t height = integer("--height", "0", 33, 16384);
  const int repeats = (int)integer("--repeat", "1", 1, 16);
  const uint32_t seed = integer("--seed", "0", 0, UINT32_MAX);
  const uint32_t autoMaskValue = integer("--auto-mask", "1", 0, 1);
  const bool autoMask = autoMaskValue != 0;
  const float tone = number("--tone", "1", 0.0f, 2.0f);
  const float structure = number("--structure", "1", 0.0f, 2.0f);
  const float skin = number("--skin", "-1", -1.0f, 2.0f);
  const float style = number("--style", "0", 0.0f, 2.0f);
  const float intensity = number("--intensity", "1", 0.0f, 2.0f);
  if (style != std::floor(style)) throw std::runtime_error("--style must be 0, 1 or 2");
  const uint64_t inputBytesExpected = (uint64_t)width * height * 4 * sizeof(float);
  if (inputBytesExpected > SIZE_MAX) throw std::runtime_error("image input is too large");
  std::vector<uint8_t> inputBytes = readFile(inputPath);
  if (inputBytes.size() != inputBytesExpected)
    throw std::runtime_error("image input must be tightly packed RGBA f32: expected " + std::to_string(inputBytesExpected) +
                             " bytes, got " + std::to_string(inputBytes.size()));
  const float* input = reinterpret_cast<const float*>(inputBytes.data());
  for (size_t i = 0; i < (size_t)width * height * 4; ++i) {
    if (!std::isfinite(input[i])) throw std::runtime_error("image input contains a non-finite value at float " + std::to_string(i));
    if (input[i] < 0.0f || input[i] > 1.0f)
      throw std::runtime_error("image input contains a value outside [0, 1] at float " + std::to_string(i));
  }

  const std::unique_ptr<exec::Device> owned = makeDevice();
  // DLSS5VK_TAPE=1: every graph is walked once into a tape and its submissions replay the tape.
  const bool taped = getenv("DLSS5VK_TAPE") && atoi(getenv("DLSS5VK_TAPE")) != 0;
  exec::TapeRecorder tapeRecorder(*owned);
  exec::Device& context = taped ? static_cast<exec::Device&>(tapeRecorder) : *owned;
  exec::TapeRecorder* const recorder = taped ? &tapeRecorder : nullptr;
  printf("device: %s (%s, %s backend%s)\n", context.deviceName().c_str(), context.api(), exec::backendName(context.backend()),
         taped ? ", graph replayed from a tape" : "");
  auto started = std::chrono::steady_clock::now();
  nr::Model model(context, modelDir, !hasFlag(argc, argv, "--no-verify"));
  printf("model loaded in %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
  nr::Kernels kernels(context, shaderDir);
  kernels.setSiluTable(ref::siluTable());
  const nr::Geometry geometry = nr::Geometry::fromValid(width, height);
  printf("geometry %ux%u -> full %ux%u\n", width, height, geometry.fullWidth, geometry.fullHeight);

  nr::Activation features;
  features.format = nr::Format::F32;
  features.rows = geometry.fullWidth * geometry.fullHeight;
  features.channels = 16;
  features.allocRows = nr::alignRows(features.rows);
  features.label = "image input features";
  features.buffer = context.createBuffer((VkDeviceSize)features.allocRows * features.channels * sizeof(float), false,
                                         "image input features");
  context.fillZero(features.buffer);
  vk::Buffer proxy = context.createBuffer(inputBytes.size(), false, "image proxy");
  context.upload(proxy, inputBytes.data(), inputBytes.size());
  nr::Kernels::PreprocessArgs preprocess{geometry.fullWidth, geometry.fullHeight, width, height, width, height, seed,
                                         autoMask, tone, structure, skin, style};
  exec::Commands commands = context.beginCommands();
  kernels.preprocessFromProxy(commands, proxy, features, preprocess);
  context.endAndSubmit(commands, true);

  GraphRun run = runGraph(context, model, kernels, geometry, features, true, false, repeats);
  printf("production schedule (%s): %u dispatches, GPU %.3f ms (min of %d)\n",
         run.chained ? "counter chaining" : "barriers", run.dispatches, run.minGpuMs, repeats);
  printf("head repeatable: %s; NaN weight codes replaced: %zu\n", run.repeatable ? "yes" : "NO", model.nanWeightsReplaced());
  if (!run.repeatable) throw std::runtime_error("image head changed across repeated submissions");

  const float* head = reinterpret_cast<const float*>(run.head.data());
  std::vector<float> output((size_t)width * height * 4);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      const size_t sourcePixel = (size_t)y * width + x;
      const size_t headPixel = (size_t)y * geometry.fullWidth + x;
      for (uint32_t c = 0; c < 3; ++c) {
        const float proxyValue = std::fmin(std::fmax(input[sourcePixel * 4 + c], 0.0f), 1.0f);
        const float centred = std::fmaf(proxyValue, 0.125f, -0.0625f);
        const float neural = composed(head[headPixel * 4 + c], centred);
        output[sourcePixel * 4 + c] =
            truncateHalf(std::fmin(std::fmax(std::fmaf(intensity, neural - proxyValue, proxyValue), 0.0f), 1.0f));
      }
      output[sourcePixel * 4 + 3] = std::fmin(std::fmax(input[sourcePixel * 4 + 3], 0.0f), 1.0f);
    }
  }
  std::ofstream file(outputPath, std::ios::binary);
  if (!file) throw std::runtime_error("cannot write " + outputPath);
  file.write(reinterpret_cast<const char*>(output.data()), (std::streamsize)(output.size() * sizeof(float)));
  if (!file) throw std::runtime_error("failed while writing " + outputPath);
  printf("wrote %s (%zu bytes, RGBA f32)\n", outputPath.c_str(), output.size() * sizeof(float));
  context.destroyBuffer(proxy);
  context.destroyBuffer(features.buffer);
  return 0;
}

}  // namespace

int runVerify(int argc, char** argv);

namespace {
int runShaderInfo(int argc, char** argv) {
  std::string modelDir = argValue(argc, argv, "--model");
  std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  std::string filter = argValue(argc, argv, "--filter", "");
  bool sass = hasFlag(argc, argv, "--sass");
  if (modelDir.empty()) { fprintf(stderr, "usage: dlss5vk shaderinfo --model <dir> [--filter name] [--sass]\n"); return 2; }
  vk::Context context;
  context.setCaptureStatistics(true);
  nr::Model model(context, modelDir, false);
  nr::Kernels kernels(context, shaderDir);
  kernels.setSiluTable(ref::siluTable());
  nr::Geometry geometry = nr::Geometry::fromValid(768, 768);
  nr::Graph graph(context, model, kernels, geometry, {.fusedBlocks = fusedBlocksEnabled()});
  nr::Activation* features = graph.allocate("input features", geometry.fullWidth * geometry.fullHeight, 16, nr::Format::F32);
  exec::Commands commands = context.beginCommands();
  graph.record(commands, *features);
  context.endAndSubmit(commands, true);
  checkChainTimeouts(kernels);
  for (const auto& [key, pipeline] : kernels.pipelines()) {
    if (!filter.empty() && key.find(filter) == std::string::npos) continue;
    printf("== %s\n%s\n", key.c_str(), context.pipelineStatistics(pipeline, sass).c_str());
  }
  return 0;
}
}  // namespace

namespace {
int runProfile(int argc, char** argv) {
  std::string modelDir = argValue(argc, argv, "--model");
  std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  uint32_t width = (uint32_t)atoi(argValue(argc, argv, "--width", "768").c_str());
  uint32_t height = (uint32_t)atoi(argValue(argc, argv, "--height", "768").c_str());
  if (modelDir.empty()) { fprintf(stderr, "usage: dlss5vk profile --model <dir> [--width W --height H]\n"); return 2; }
  const std::unique_ptr<exec::Device> owned = makeDevice();
  exec::Device& context = *owned;
  printf("device: %s (%s, %s backend)\n", context.deviceName().c_str(), context.api(), exec::backendName(context.backend()));
  if (const vk::Context* vulkan = dynamic_cast<const vk::Context*>(owned.get()))
    printf("maxComputeSharedMemorySize %u bytes\n", vulkan->maxComputeSharedMemory());
  nr::Model model(context, modelDir, false);
  nr::Kernels kernels(context, shaderDir);
  if (!getenv("DLSS5VK_CHAIN")) kernels.setChainEnabled(false);   // per-dispatch timings need the barriers
  kernels.setSiluTable(ref::siluTable());
  // Barrier/dispatch overhead: 400 trivial dispatches each followed by a full compute barrier.
  {
    nr::Activation tiny; tiny.format = nr::Format::F16; tiny.rows = 64; tiny.channels = 16; tiny.allocRows = 64;
    tiny.buffer = context.createBuffer(64 * 16 * 2, false, "tiny");
    nr::Activation tinyOut = tiny; tinyOut.format = nr::Format::E4;
    tinyOut.buffer = context.createBuffer(64 * 16, false, "tiny out");
    exec::Timer pool = context.createTimestampPool(2);
    exec::Commands commands = context.beginCommands();
    context.resetTimestamps(commands, pool, 2);
    context.writeTimestamp(commands, pool, 0, true);
    for (int i = 0; i < 400; ++i) kernels.quantize(commands, tiny, tinyOut);
    context.writeTimestamp(commands, pool, 1, false);
    context.endAndSubmit(commands, true);
    std::vector<double> stamps = context.readTimestampsMs(pool, 2);
    printf("400 trivial dispatches + barriers: %.3f ms (%.2f us each)\n", stamps[1] - stamps[0], (stamps[1] - stamps[0]) * 2.5);
    context.destroyTimestampPool(pool);
    context.destroyBuffer(tiny.buffer); context.destroyBuffer(tinyOut.buffer);
    context.nextFrame();
  }
  nr::Geometry geometry = nr::Geometry::fromValid(width, height);
  nr::Graph graph(context, model, kernels, geometry, {.fusedBlocks = fusedBlocksEnabled()});
  const uint32_t fullRows = geometry.fullWidth * geometry.fullHeight;
  nr::Activation* features = graph.allocate("input features", fullRows, 16, nr::Format::F32);
  std::vector<float> synthetic((size_t)fullRows * 16);
  for (size_t i = 0; i < synthetic.size(); ++i) synthetic[i] = num::roundF16(std::sin(i * 0.0017f) * 0.125f);
  context.upload(features->buffer, synthetic.data(), synthetic.size() * 4);
  {
    exec::Commands commands = context.beginCommands();
    graph.record(commands, *features);
    context.endAndSubmit(commands, true);
    checkChainTimeouts(kernels);
  }
  std::map<std::string, double> byLabel;
  std::map<std::string, double> byStage;
  double total = 0;
  // Other programs share the GPU: take the per-dispatch minimum over several frames.
  const int frames = atoi(argValue(argc, argv, "--frames", "7").c_str());
  std::vector<nr::Kernels::ProfileEntry> best;
  for (int frame = 0; frame < frames; ++frame) {
    context.nextFrame();
    exec::Commands commands = context.beginCommands();
    kernels.beginProfile(commands, 4096);
    graph.record(commands, *features);
    context.endAndSubmit(commands, true);
    checkChainTimeouts(kernels);
    std::vector<nr::Kernels::ProfileEntry> entries = kernels.endProfile();
    if (best.empty()) best = entries;
    for (size_t i = 0; i < entries.size() && i < best.size(); ++i)
      best[i].milliseconds = std::min(best[i].milliseconds, entries[i].milliseconds);
  }
  const std::string stageFilter = argValue(argc, argv, "--stage", "");
  for (size_t index = 0; index < best.size(); ++index) {
    const auto& entry = best[index];
    size_t bar = entry.label.find(" | ");
    std::string stage = entry.label.substr(0, bar), kernel = entry.label.substr(bar + 3);
    if (!stageFilter.empty() && stage.find(stageFilter) != std::string::npos)
      printf("  %8.2f us  [%3zu] %s | %s\n", entry.milliseconds * 1000.0, index, stage.c_str(), kernel.c_str());
    byLabel[kernel] += entry.milliseconds;
    byStage[stage] += entry.milliseconds;
    total += entry.milliseconds;
  }
  std::vector<std::pair<std::string, double>> sorted(byLabel.begin(), byLabel.end());
  std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });
  printf("total %.3f ms (sum of per-dispatch spans, %ux%u)\n\nby kernel:\n", total, width, height);
  for (size_t i = 0; i < sorted.size() && i < 40; ++i)
    printf("  %8.3f ms  %5.1f%%  %s\n", sorted[i].second, 100 * sorted[i].second / total, sorted[i].first.c_str());
  std::vector<std::pair<std::string, double>> stages(byStage.begin(), byStage.end());
  std::sort(stages.begin(), stages.end(), [](auto& a, auto& b) { return a.second > b.second; });
  printf("\nby stage:\n");
  for (size_t i = 0; i < stages.size(); ++i)
    printf("  %8.3f ms  %5.1f%%  %s\n", stages[i].second, 100 * stages[i].second / total, stages[i].first.c_str());
  return 0;
}
}  // namespace

#ifdef _WIN32
// dlss5vk bridge: see src/bridge/bridge_tool.h.
int runBridge(int argc, char** argv) {
  bridge::ToolInput input;
  input.host = argValue(argc, argv, "--host");
  input.modelDir = argValue(argc, argv, "--model");
  input.shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  input.dumpDir = argValue(argc, argv, "--dump");
  input.jsonPath = argValue(argc, argv, "--json");
  input.transportOnly = hasFlag(argc, argv, "--transport");
  input.sameKernels = hasFlag(argc, argv, "--same-kernels");
  input.frames = std::max(2, atoi(argValue(argc, argv, "--frames", "10").c_str()));
  const char* validation = getenv("DLSS5VK_VALIDATION");
  input.debugLayer = validation && *validation && strcmp(validation, "0");
  const std::string fixtureDir = argValue(argc, argv, "--fixture");
  if (input.host.empty() || (!input.transportOnly && (input.modelDir.empty() || fixtureDir.empty()))) {
    fprintf(stderr, "usage: dlss5vk bridge --host d3d12|d3d11 --model <dir> --fixture <proxy fixture> [--frames N] [--same-kernels] [--dump <dir>] [--json <file>]\n"
                    "       dlss5vk bridge --host d3d12|d3d11 --transport [--width W --height H] [--frames N] [--json <file>]\n");
    return 2;
  }
  if (input.transportOnly) {
    input.proxyWidth = (uint32_t)atoi(argValue(argc, argv, "--width", "512").c_str());
    input.proxyHeight = (uint32_t)atoi(argValue(argc, argv, "--height", "512").c_str());
    if (!input.proxyWidth || !input.proxyHeight) throw std::runtime_error("--width and --height must be positive");
  } else {
    json::Value manifest = json::parse(readText(fixtureDir + "/manifest.json"));
    const FixturePlan plan = planFixture(manifest, fixtureDir);
    if (plan.proxyFile.empty()) throw std::runtime_error("the bridge needs a proxy fixture (the host writes the proxy image)");
    input.validWidth = plan.validWidth; input.validHeight = plan.validHeight;
    input.fullWidth = plan.fullWidth; input.fullHeight = plan.fullHeight;
    input.proxyWidth = plan.proxyWidth; input.proxyHeight = plan.proxyHeight;
    input.proxy = readFile(plan.proxyFile);
    const json::Value& conditioning = manifest["conditioning"];
    input.seed = (uint32_t)manifest["seed"].integer();
    input.autoMask = manifest["autoMask"].boolean;
    input.localTone = (float)conditioning["localTone"].number;
    input.localStructure = (float)conditioning["localStructure"].number;
    input.skinStructure = (float)conditioning["skinStructure"].number;
    input.style = (float)conditioning["style"].number;
    if (plan.checkHead) input.referenceHead = readFile(plan.headFile);
  }
  const bridge::ToolResult result = bridge::runTool(input);
  printf("\nVERDICT: %s\n", result.passed ? "PASS" : "FAIL");
  return result.passed ? 0 : 1;
}
#endif

int runCommand(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "verify")) return runVerify(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "shaderinfo")) return runShaderInfo(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "profile")) return runProfile(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "parity")) return runParity(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "image")) return runImage(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "bench")) return runBench(argc, argv);
#ifdef _WIN32
  if (argc >= 2 && !strcmp(argv[1], "bridge")) return runBridge(argc, argv);
#endif
  fprintf(stderr, "usage: dlss5vk parity|verify|image|bench|profile|shaderinfo --model <dir> ...\n");
  return 2;
}

int main(int argc, char** argv) {
  int code = 1;
  try {
    code = runCommand(argc, argv);
  } catch (const std::exception& error) {
    fprintf(stderr, "error: %s\n", error.what());
  }
  // With the validation layer on, an error it reported fails the run, whatever the command concluded.
  const char* validation = getenv("DLSS5VK_VALIDATION");
  if (validation && *validation && strcmp(validation, "0")) {
    const uint32_t errors = deviceValidationErrors();
    printf("validation: %u error%s reported by the layer\n", errors, errors == 1 ? "" : "s");
    if (errors && code == 0) code = 1;
  }
  return code;
}
