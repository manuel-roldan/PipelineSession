#pragma once

#include "sima/nodes/common/AppSink.h"
#include "sima/pipeline/PipelineReport.h"
#include "sima/pipeline/TapStream.h"
#include "sima/pipeline/FrameStream.h"
#include "sima/pipeline/TensorTypes.h"

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sima {

struct RtspServerOptions {
  std::string mount = "image";
  int port = 8554;
};

struct ValidateOptions {
  bool parse_launch = true;  // build gst pipeline and verify element naming contract
  bool enforce_names = true; // ensure no unnamed/foreign elements exist
};

struct RunDebugOptions {
  int timeout_ms = 10000;
};

struct OutputTensorOptions {
  std::string format = "RGB";
  TensorLayout layout = TensorLayout::HWC;
  TensorDType dtype = TensorDType::UInt8;

  int target_width = -1;
  int target_height = -1;
  int target_fps = -1;

  bool use_videoconvert = true;
  bool use_videoscale = true;

  OutputAppSinkOptions sink;
};

struct RunDebugTap {
  std::string name;
  std::optional<TapPacket> packet;
  std::optional<FrameTensorRef> tensor;
  std::optional<FrameTensorRef> last_good_tensor;
  std::string error;
};

enum class RunOutputKind {
  Tensor,
  FrameNV12,
  Unknown,
};

struct RunInputResult {
  RunOutputKind kind = RunOutputKind::Unknown;

  std::optional<FrameTensor> tensor;
  std::optional<FrameNV12> frame_nv12;

  std::string caps_string;
  std::string media_type;
  std::string format;
};

struct RunDebugResult {
  std::optional<FrameNV12> first_frame; // tight packed copy
  std::vector<RunDebugTap> taps;
  std::unordered_map<std::string, FrameTensorRef> tensors;
  PipelineReport report;
};

} // namespace sima
