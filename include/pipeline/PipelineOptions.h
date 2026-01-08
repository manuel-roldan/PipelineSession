#pragma once

#include "sima/pipeline/PipelineReport.h"
#include "sima/pipeline/TapStream.h"

#include <optional>
#include <string>

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
  int timeout_ms = 1000;
};

struct RunDebugResult {
  std::optional<FrameNV12> first_frame; // tight packed copy
  PipelineReport report;
};

} // namespace sima
