// DLSS-NR inside the renderer's frame: the renderer (Filament) draws the scene into an HDR color texture and
// writes per-object motion vectors; this pass, recorded into the renderer's own command buffer between its passes,
// unpacks the motion vectors, runs the feature preprocess, the NR graph (src/) and the temporal composite with the
// neural history, and copies the rgba8 result into the renderer's output texture, which its present view draws.
// Runs on the renderer's device (vk::Context adopts it). The NR work is pre-recorded once per history parity and
// NR on/off in secondary command buffers.
#pragma once
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "gpu_bridge.h"
#include "kernels.h"
#include "nr_graph.h"
#include "nr_model.h"
#include "vk_context.h"

struct NrControls {
  bool enabled = true;
  float intensity = 1.0f, localTone = 1.0f, localStructure = 1.0f, skinStructure = -1.0f;
  bool autoMask = true;
  int style = 0;            // 0 off, 1 natural, 2 cinematic (the id the network sees, and the composite preset)
  float paperWhite = 1.0f, colorStrength = 1.0f;
  bool temporal = true;     // blend with the reprojected history
  // custom style: the presets' operator with its knobs exposed (used instead of the preset when styleCustom)
  bool styleCustom = false;
  float styleExposure = 0.0f, styleContrast = 0.0f, styleGamma = 1.0f, styleSaturation = 0.0f;
  float styleHue = 0.0f, styleVibrance = 1.0f, styleStrength = 1.0f;
};

struct NrTimings {
  double sceneMs = 0, preprocessMs = 0, networkMs = 0, compositeMs = 0, presentMs = 0, frameMs = 0;
};

class NrPass {
 public:
  NrPass(const GpuDevice& device, uint32_t width, uint32_t height, const std::string& modelDir, const std::string& kernelDir,
         const std::string& demoShaderDir);
  ~NrPass();

  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  // re-fit to a new size (window resize), with the renderer idle: the model / kernels / pipelines stay, the graph's
  // activations, the images, the descriptor sets and the pre-recorded command buffers are rebuilt (~50 ms)
  void resize(uint32_t width, uint32_t height);
  vk::Context& context() { return *context_; }

  // What record() needs of a frame, decided on the main thread (the renderer runs the recording on its own thread,
  // possibly after the main thread moved on to the next frame)
  struct Frame {
    uint32_t parity = 0;
    bool enabled = true;
    uint8_t params[128];
    float background[16];   // column-major: previous clip <- current clip for points at infinity (see beginFrame)
  };
  // per frame, main thread, before the renderer's frame: rotates the history parity, reads the timings of the frame
  // that used this parity last, snapshots the controls. `background` (column-major 4x4) maps the current clip space to
  // the previous frame's for points at infinity: previous projection x view rotation x inverse(current projection x
  // view rotation), the views without their translation. The motion unpack uses it where the renderer drew nothing,
  // or its skybox, which the renderer draws at infinity with a zero motion vector.
  Frame beginFrame(const NrControls& controls, const float background[16]);
  void endFrame() { ++frames_; ++framesSinceReset_; }
  void resetHistory() { framesSinceReset_ = 0; }
  // A chained wait gave up (the kernels' watchdog, docs/execution.md): that frame was wrong. Call with the renderer
  // idle; the graph is rebuilt with a barrier after every launch, as DLSS5VK_CHAIN=0 would have it.
  bool chainTimedOut() const { return kernels_->chainTimeouts().waits != 0; }
  void fallBackToBarriers();

  // The renderer's thread, inside its command buffer, outside of any render pass. `color` (rgba16f) and `velocity`
  // (rgba32ui: id, depth bits, motion x/y bits) are the renderer's textures as it left them; `output` (rgba8) is
  // the texture the renderer's present view samples. Stamps kSceneEnd .. kCompositeEnd. The images are left in
  // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL (the value of shaderReadOnlyLayout()) for the renderer's tracking.
  void record(void* commandBuffer, const Frame& frame, const GpuImage& color, const GpuImage& velocity, const GpuImage& output);
  // timestamps around the renderer's own passes, recorded the same way (kFrameStart resets the frame's queries)
  enum Stamp { kFrameStart = 0, kSceneEnd = 1, kPreprocessEnd = 2, kNetworkEnd = 3, kCompositeEnd = 4, kPresentEnd = 5, kStampCount = 6 };
  void recordStamp(void* commandBuffer, const Frame& frame, Stamp stamp);
  static uint32_t shaderReadOnlyLayout();

  const NrTimings& timings() const { return timings_; }
  uint32_t frameCount() const { return frames_; }
  const std::string& deviceName() const { return context_->deviceName(); }

  // debugging / verification, with the renderer idle: the composited rgba8 output (or the HDR scene color,
  // tone-mapped) as a binary PPM; raw dumps: kind 0 = scene color (rgb f32), 1 = motion (f32 x, y in uv units and
  // 1 / 0 = the previous position on / off screen), 2 = the renderer's velocity buffer (rgba32ui: id, depth bits, motion bits)
  void saveOutput(const std::string& path, bool sceneInstead = false);
  void saveRaw(const std::string& path, int kind);
  // Local verification data only, emitted by --capture with the renderer idle.
  void saveVerification(const std::string& prefix);

 private:
  struct Image {
    VkImage image = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED; VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  };
  // a view onto one of the renderer's images (re-made when the renderer's image changes, e.g. after a resize)
  struct ExternalView {
    VkImage image = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; VkFormat format = VK_FORMAT_UNDEFINED;
  };
  Image createImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect);
  void destroyImage(Image& image);
  void transition(VkCommandBuffer commands, Image& image, VkImageLayout layout, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                  VkPipelineStageFlags dstStage, VkAccessFlags dstAccess, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
  void barrier(VkCommandBuffer commands, VkImage image, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags srcStage,
               VkAccessFlags srcAccess, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);
  bool bindExternal(ExternalView& external, const GpuImage& image);
  VkPipeline computePipeline(const std::string& spv, VkPipelineLayout layout);
  void updateComputeSets();
  void buildComputeCommands();
  void createSized(uint32_t width, uint32_t height);    // everything that depends on the size
  void destroySized();
  void writeStamp(VkCommandBuffer commands, uint32_t parity, Stamp stamp, VkPipelineStageFlagBits stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  std::vector<uint8_t> readImage(VkImage image, uint32_t bytesPerPixel, VkImageLayout layout);

  std::unique_ptr<vk::Context> context_;
  std::unique_ptr<nr::Model> model_;
  std::unique_ptr<nr::Kernels> kernels_;
  std::unique_ptr<nr::Graph> graph_;
  nr::Geometry geometry_{};
  nr::Activation* features_ = nullptr;
  VkDevice device_ = VK_NULL_HANDLE;
  uint32_t width_ = 0, height_ = 0;
  float blendScale_ = 1.0f;

  Image sceneMotion_, history_[2], output_;
  ExternalView color_, velocity_;
  VkSampler linearSampler_ = VK_NULL_HANDLE, nearestSampler_ = VK_NULL_HANDLE;
  vk::Buffer params_[2];
  struct ParamsBlock;

  VkDescriptorSetLayout computeSetLayout_ = VK_NULL_HANDLE, unpackSetLayout_ = VK_NULL_HANDLE;
  VkPipelineLayout computeLayout_ = VK_NULL_HANDLE, unpackLayout_ = VK_NULL_HANDLE;
  VkPipeline preprocessPipeline_ = VK_NULL_HANDLE, compositePipeline_ = VK_NULL_HANDLE, unpackPipeline_ = VK_NULL_HANDLE;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  VkDescriptorSet computeSets_[2][2]{};   // [history index][0 preprocess, 1 composite]
  VkDescriptorSet unpackSet_ = VK_NULL_HANDLE;
  std::string demoShaderDir_;
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  VkCommandBuffer computeCommands_[2][2]{};   // [history parity][NR off, on]: secondary command buffers
  VkQueryPool queries_[2]{};       // per history parity: read when that parity comes around again
  std::atomic<bool> queriesWritten_[2]{};
  NrTimings timings_;
  Frame lastFrame_{};
  uint32_t frames_ = 0, framesSinceReset_ = 0, historyIndex_ = 0;
};
