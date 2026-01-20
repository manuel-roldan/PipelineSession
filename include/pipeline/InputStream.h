#pragma once

#include "sima/pipeline/PipelineOptions.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace cv {
class Mat;
} // namespace cv

namespace sima {

struct InputStreamOptions {
  int timeout_ms = 10000;
  bool appsink_sync = true;
  bool appsink_drop = false;
  int appsink_max_buffers = 1;
  bool enable_timings = false;
  bool preflight_run = false;
  bool allow_mismatched_input = false;
  bool copy_output = true;
  bool reuse_input_buffer = false;
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

class InputStream {
public:
  InputStream() = default;
  InputStream(const InputStream&) = delete;
  InputStream& operator=(const InputStream&) = delete;

  InputStream(InputStream&&) noexcept;
  InputStream& operator=(InputStream&&) noexcept;
  ~InputStream();

  explicit operator bool() const noexcept;

  RunInputResult push_and_pull(const cv::Mat& input, int timeout_ms = -1);
  void push(const cv::Mat& input);
  bool try_push(const cv::Mat& input);
  RunInputResult pull(int timeout_ms = -1);
  void signal_eos();

  void start(std::function<void(RunInputResult)> on_output);
  void stop();
  bool running() const;
  std::string last_error() const;
  InputStreamStats stats() const;
  std::string diagnostics_summary() const;

  void close();

private:
  struct State;
  std::shared_ptr<State> state_;

  explicit InputStream(std::shared_ptr<State> state);
  friend class PipelineSession;
};

} // namespace sima
