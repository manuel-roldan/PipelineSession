#pragma once

#include "pipeline/Deprecated.h"
#include "sima/builder/Node.h"
#include "sima/builder/NodeGroup.h"
#include "sima/pipeline/PipelineOptions.h"
#include "sima/pipeline/TapStream.h"
#include "sima/pipeline/TensorStream.h"
#include "sima/pipeline/PipelineRun.h"
#include "sima/builder/GraphPrinter.h"
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
#include "sima/nodes/io/InputAppSrc.h"
#include "sima/nodes/io/RTSPInput.h"
#include "sima/nodes/rtp/RtpH264Depay.h"
#include "sima/nodes/sima/H264DecodeSima.h"
#include "sima/nodes/sima/H264EncodeSima.h"
#include "sima/nodes/sima/H264Parse.h"
#include "sima/nodes/sima/RtpH264Pay.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

struct _GstElement;
using GstElement = struct _GstElement;

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
  void kill() { stop(); }
  bool running() const;

private:
  friend class PipelineSession;

  std::string url_;
  void* impl_ = nullptr;
  std::shared_ptr<void> guard_;
};

class PipelineSession {
public:
  using FrameCallback = std::function<bool(const NeatTensor&)>;
  using TensorCallback = std::function<bool(const NeatTensor&)>;

  explicit PipelineSession(const PipelineSessionOptions& opt = {});
  ~PipelineSession();
  PipelineSession(const PipelineSession&) = delete;
  PipelineSession& operator=(const PipelineSession&) = delete;
  PipelineSession(PipelineSession&&) noexcept;
  PipelineSession& operator=(PipelineSession&&) noexcept;

  // Core: add a node (factory functions return std::shared_ptr<Node>)
  PipelineSession& add(std::shared_ptr<Node> node);
  PipelineSession& add(const NodeGroup& group);
  PipelineSession& add(NodeGroup&& group);

  // Explicit raw-string escape hatch (keeps "power user" obvious)
  PipelineSession& gst(std::string fragment);
  PipelineSession& gst(std::string fragment, InputRole role);

  // Typed runner: last node must be OutputAppSink().
  void run();
  TensorStream run_tensor();
  TapStream run_packet_stream();
  RunInputResult run(const cv::Mat& input);
  RunInputResult run(const cv::Mat& input, const RunInputOptions& opt);
  RunInputResult run(const NeatTensor& input);
  RunInputResult run(const NeatTensor& input, const RunInputOptions& opt);
  PipelineRun build(const cv::Mat& input,
                    const PipelineRunOptions& opt = {});
  PipelineRun build(const cv::Mat& input, PipelineRunMode mode);
  PipelineRun build(const cv::Mat& input,
                    PipelineRunMode mode,
                    const PipelineRunOptions& opt);
  PipelineRun build(const NeatTensor& input,
                    const PipelineRunOptions& opt = {});
  PipelineRun build(const NeatTensor& input, PipelineRunMode mode);
  PipelineRun build(const NeatTensor& input,
                    PipelineRunMode mode,
                    const PipelineRunOptions& opt);
  RunDebugResult run_debug(const RunDebugOptions& opt, const cv::Mat& input);
  PipelineReport validate(const ValidateOptions& opt, const cv::Mat& input) const;

  // Split at DebugPoint(point_name), ignore nodes after it, attach appsink and return TapStream.
  TapStream run_tap(const std::string& point_name);

  // Server-style run
  RtspServerHandle run_rtsp(const RtspServerOptions& opt);

  // Build + validate pipeline (no PLAYING). Returns machine-readable report.
  PipelineReport validate(const ValidateOptions& opt = {}) const;

  // Run once, pull 1 frame (copy) with full structured report.
  RunDebugResult run_debug(const RunDebugOptions& opt = {});

  // Tensor-friendly output helper (adds convert/scale/caps + OutputAppSink).
  PipelineSession& add_output_tensor(const OutputTensorOptions& opt = {});
  PipelineSession& add_output_torch(const OutputTensorOptions& opt = {});
  PipelineSession& add_output_numpy(const OutputTensorOptions& opt = {});
  PipelineSession& add_output_tensorflow(const OutputTensorOptions& opt = {});

  // UX helpers: builder-only view and gst-launch string.
  std::string describe(const GraphPrinter::Options& opt = {}) const;
  std::string to_gst(bool insert_boundaries = false) const;

  // Optional external guard (e.g., provided by ModelSession).
  void set_guard(std::shared_ptr<void> guard);

  // Output callbacks for source pipelines (used by run()).
  void set_frame_callback(FrameCallback cb);
  void set_tensor_callback(TensorCallback cb);

  // Save/load pipeline config for reproducible runs.
  void save(const std::string& path) const;
  static PipelineSession load(const std::string& path);

  // Build pipeline once (no frames processed) and return an async runner.
  // Push pipelines use build(input) to configure appsrc caps.
  PipelineRun build(const PipelineRunOptions& opt = {});

  const std::string& last_pipeline() const { return last_pipeline_; }

private:
  struct BuiltState {
    GstElement* pipeline = nullptr;
    GstElement* sink = nullptr;
    std::shared_ptr<void> diag;
  };
  struct RunCache;

  std::vector<std::shared_ptr<Node>> nodes_;
  std::string last_pipeline_;
  std::shared_ptr<void> guard_;
  PipelineSessionOptions opt_{};
  FrameCallback frame_cb_;
  TensorCallback tensor_cb_;
  std::atomic<uint64_t> nodes_version_{0};
  std::unique_ptr<BuiltState> built_;
  std::unique_ptr<RunCache> run_cache_;
  uint64_t built_version_ = 0;

  void build_cached_source();
};

} // namespace sima
