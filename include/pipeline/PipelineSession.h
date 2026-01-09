#pragma once

#include "sima/builder/Node.h"
#include "sima/pipeline/PipelineOptions.h"
#include "sima/pipeline/TapStream.h"
#include "sima/nodes/common/AppSink.h"
#include "sima/nodes/common/Caps.h"
#include "sima/nodes/common/DebugPoint.h"
#include "sima/nodes/common/FileSrc.h"
#include "sima/nodes/common/JpegDec.h"
#include "sima/nodes/common/QtDemuxVideoPad.h"
#include "sima/nodes/common/Queue.h"
#include "sima/nodes/common/VideoConvert.h"
#include "sima/nodes/common/VideoScale.h"
#include "sima/nodes/io/AppSrcImage.h"
#include "sima/nodes/io/RTSPInput.h"
#include "sima/nodes/rtp/RtpH264Depay.h"
#include "sima/nodes/sima/H264DecodeSima.h"
#include "sima/nodes/sima/H264EncodeSima.h"
#include "sima/nodes/sima/H264Parse.h"
#include "sima/nodes/sima/RtpH264Pay.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sima {

class RtspServerHandle {
public:
  RtspServerHandle() = default;
  ~RtspServerHandle();

  RtspServerHandle(const RtspServerHandle&) = delete;
  RtspServerHandle& operator=(const RtspServerHandle&) = delete;

  RtspServerHandle(RtspServerHandle&&) noexcept;
  RtspServerHandle& operator=(RtspServerHandle&&) noexcept;

  const std::string& url() const { return url_; }
  void stop();
  bool running() const;

private:
  friend class PipelineSession;

  std::string url_;
  void* impl_ = nullptr;
};

class PipelineSession {
public:
  // Core: add a node (factory functions return std::shared_ptr<Node>)
  PipelineSession& add(std::shared_ptr<Node> node);

  // Explicit raw-string escape hatch (keeps "power user" obvious)
  PipelineSession& gst(std::string fragment);

  // Typed runner: last node must be OutputAppSink() and negotiated into NV12.
  FrameStream run();

  // Split at DebugPoint(point_name), ignore nodes after it, attach appsink and return TapStream.
  TapStream run_tap(const std::string& point_name);

  // Server-style run
  RtspServerHandle run_rtsp(const RtspServerOptions& opt);

  // Build + validate pipeline (no PLAYING). Returns machine-readable report.
  PipelineReport validate(const ValidateOptions& opt = {}) const;

  // Run once, pull 1 frame (copy) with full structured report.
  RunDebugResult run_debug(const RunDebugOptions& opt = {});

  const std::string& last_pipeline() const { return last_pipeline_; }

private:
  std::vector<std::shared_ptr<Node>> nodes_;
  std::string last_pipeline_;
};

} // namespace sima
