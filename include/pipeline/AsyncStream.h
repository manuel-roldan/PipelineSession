#pragma once

#include "pipeline/InputStream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace sima {

enum class DropPolicy {
  Block = 0,
  DropNewest,
  DropOldest,
};

struct AsyncOptions {
  int input_queue = 8;
  int output_queue = 8;
  DropPolicy drop = DropPolicy::Block;
  bool copy_output = false;
  bool allow_mismatched_input = false;
  bool copy_input = false;
  int timeout_ms = -1;
};

struct AsyncStats {
  std::uint64_t inputs_enqueued = 0;
  std::uint64_t inputs_dropped = 0;
  std::uint64_t inputs_pushed = 0;
  std::uint64_t outputs_ready = 0;
  std::uint64_t outputs_pulled = 0;
  double avg_latency_ms = 0.0;
  double min_latency_ms = 0.0;
  double max_latency_ms = 0.0;
};

class AsyncStream {
public:
  AsyncStream() = default;
  AsyncStream(const AsyncStream&) = delete;
  AsyncStream& operator=(const AsyncStream&) = delete;

  AsyncStream(AsyncStream&&) noexcept;
  AsyncStream& operator=(AsyncStream&&) noexcept;
  ~AsyncStream();

  explicit operator bool() const noexcept;
  bool running() const;

  bool push(const cv::Mat& input);
  bool try_push(const cv::Mat& input);
  void close_input();
  std::optional<RunInputResult> pull(int timeout_ms = -1);

  AsyncStats stats() const;
  std::string last_error() const;

  void stop();
  void close();

private:
  struct State;
  std::shared_ptr<State> state_;

  explicit AsyncStream(std::shared_ptr<State> state);
  bool push_impl(const cv::Mat& input, bool block);
  static AsyncStream create(InputStream stream,
                            const AsyncOptions& opt,
                            const InputStreamOptions& stream_opt);

  friend class PipelineSession;
};

} // namespace sima
