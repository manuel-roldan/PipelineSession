#pragma once

#include "sima/pipeline/PipelineOptions.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cv {
class Mat;
} // namespace cv

namespace sima {

class InputStream;
struct InputStreamOptions;

enum class DropPolicy {
  Block = 0,
  DropNewest,
  DropOldest,
};

struct PipelineRunOptions {
  int input_queue = 24;
  int output_queue = 24;
  DropPolicy drop = DropPolicy::Block;
  bool copy_output = false;
  bool copy_input = false;
  bool allow_mismatched_input = false;
  bool no_map_tensor_ref = false;
  bool reuse_input_buffer = false;
  int timeout_ms = -1;
  int poll_ms = 0;
  bool enable_timings = false;
  int appsink_max_buffers = 24;
  bool appsink_drop = false;
  bool appsink_sync = false;
  bool preflight_run = false;
  bool auto_recover_dispatcher = true;
  std::function<void(const PipelineReport&)> on_dispatcher_error;
};

struct InputStreamStats {
  std::uint64_t push_count = 0;
  std::uint64_t push_failures = 0;
  std::uint64_t pull_count = 0;
  std::uint64_t poll_count = 0;
  double avg_alloc_us = 0.0;
  double avg_map_us = 0.0;
  double avg_copy_us = 0.0;
  double avg_push_us = 0.0;
  double avg_pull_wait_us = 0.0;
  double avg_decode_us = 0.0;
};

struct PipelineRunStats {
  std::uint64_t inputs_enqueued = 0;
  std::uint64_t inputs_dropped = 0;
  std::uint64_t inputs_pushed = 0;
  std::uint64_t outputs_ready = 0;
  std::uint64_t outputs_pulled = 0;
  double avg_latency_ms = 0.0;
  double min_latency_ms = 0.0;
  double max_latency_ms = 0.0;
};

struct PipelineRunStageStats {
  std::string stage_name;
  std::uint64_t samples = 0;
  std::uint64_t total_us = 0;
  std::uint64_t max_us = 0;
};

struct PipelineRunDiagSnapshot {
  std::vector<PipelineRunStageStats> stages;
  std::vector<BoundaryFlowStats> boundaries;
};

struct PipelineRunReportOptions {
  bool include_pipeline = true;
  bool include_stage_timings = true;
  bool include_boundaries = true;
  bool include_node_reports = false;
  bool include_next_cpu = false;
  bool include_queue_depth = true;
  bool include_num_buffers = true;
  bool include_run_stats = true;
  bool include_input_stats = true;
  bool include_system_info = false;
};

class PipelineRun {
public:
  PipelineRun() = default;
  PipelineRun(const PipelineRun&) = delete;
  PipelineRun& operator=(const PipelineRun&) = delete;

  PipelineRun(PipelineRun&&) noexcept;
  PipelineRun& operator=(PipelineRun&&) noexcept;
  ~PipelineRun();

  explicit operator bool() const noexcept;
  bool can_push() const;
  bool can_pull() const;
  bool running() const;

  bool push(const cv::Mat& input);
  bool try_push(const cv::Mat& input);
  bool push(const NeatTensor& input);
  bool try_push(const NeatTensor& input);
  // Internal: pushes a GstBuffer held by a tensor ref to preserve plugin metadata.
  bool push_holder(const std::shared_ptr<void>& holder);
  bool try_push_holder(const std::shared_ptr<void>& holder);
  void close_input();
  std::optional<RunInputResult> pull(int timeout_ms = -1);
  RunInputResult push_and_pull(const cv::Mat& input, int timeout_ms = -1);
  RunInputResult push_and_pull(const NeatTensor& input, int timeout_ms = -1);
  RunInputResult push_and_pull_holder(const std::shared_ptr<void>& holder, int timeout_ms = -1);
  int warmup(const cv::Mat& input, int warm = -1, int timeout_ms = -1);

  PipelineRunStats stats() const;
  InputStreamStats input_stats() const;
  PipelineRunDiagSnapshot diag_snapshot() const;
  std::string report(const PipelineRunReportOptions& opt = {}) const;
  std::string last_error() const;
  std::string diagnostics_summary() const;

  void stop();
  void close();

private:
  struct State;
  std::shared_ptr<State> state_;

  explicit PipelineRun(std::shared_ptr<State> state);
  bool push_impl(const cv::Mat& input, bool block);
  bool push_impl(const NeatTensor& input, bool block);
  bool push_holder_impl(const std::shared_ptr<void>& holder, bool block);
  static PipelineRun create(InputStream stream,
                            const PipelineRunOptions& opt,
                            const struct InputStreamOptions& stream_opt);
  friend class PipelineSession;
};

} // namespace sima
