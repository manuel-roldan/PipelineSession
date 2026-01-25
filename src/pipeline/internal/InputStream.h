#pragma once

#include "pipeline/NeatTensor.h"
#include "pipeline/NeatTensorCore.h"
#include "pipeline/PipelineRun.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace cv {
class Mat;
} // namespace cv

struct _GstElement;
using GstElement = struct _GstElement;

namespace sima {

struct InputAppSrcOptions;
struct InputCapsConfig;
struct RunOutput;

namespace pipeline_internal {
struct DiagCtx;
} // namespace pipeline_internal

struct InputStreamOptions {
  int timeout_ms = 10000;
  int poll_ms = 0;
  bool appsink_sync = true;
  bool appsink_drop = false;
  int appsink_max_buffers = 1;
  bool enable_timings = false;
  bool preflight_run = false;
  bool allow_mismatched_input = false;
  bool copy_output = true;
  bool no_map_tensor_ref = false;
  bool reuse_input_buffer = false;
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
  bool can_push() const noexcept;
  bool can_pull() const noexcept;

  static InputStream create(GstElement* pipeline,
                            GstElement* appsrc,
                            GstElement* appsink,
                            const InputCapsConfig& cfg,
                            const InputAppSrcOptions& src_opt,
                            const InputStreamOptions& opt,
                            std::shared_ptr<pipeline_internal::DiagCtx> diag,
                            std::shared_ptr<void> guard);

  RunInputResult push_and_pull(const cv::Mat& input, int timeout_ms = -1);
  RunInputResult push_and_pull(const NeatTensor& input, int timeout_ms = -1);
  RunInputResult push_and_pull_holder(const std::shared_ptr<void>& holder, int timeout_ms = -1);
  void push(const cv::Mat& input);
  bool try_push(const cv::Mat& input);
  void push(const NeatTensor& input);
  bool try_push(const NeatTensor& input);
  void push_message(const RunOutput& msg);
  bool try_push_message(const RunOutput& msg);
  void push_holder(const std::shared_ptr<void>& holder);
  bool try_push_holder(const std::shared_ptr<void>& holder);
  RunInputResult pull(int timeout_ms = -1);
  void signal_eos();

  void start(std::function<void(RunInputResult)> on_output);
  void stop();
  bool running() const;
  std::string last_error() const;
  InputStreamStats stats() const;
  std::string diagnostics_summary() const;
  std::shared_ptr<pipeline_internal::DiagCtx> diag_ctx() const;

  void close();

private:
  struct State;
  std::shared_ptr<State> state_;

  explicit InputStream(std::shared_ptr<State> state);
  bool push_with_fill(const char* where,
                      const std::function<size_t(uint8_t*, size_t)>& fill,
                      const std::optional<int64_t>& frame_id_override,
                      const std::optional<std::string>& stream_id_override,
                      const std::optional<std::string>& buffer_name_override);
  friend class PipelineSession;
  friend class PipelineRun;
};

} // namespace sima
