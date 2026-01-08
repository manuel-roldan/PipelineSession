#pragma once

#include "sima/builder/Node.h"
#include "sima/pipeline/PipelineOptions.h"
#include "sima/pipeline/TapStream.h"

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
