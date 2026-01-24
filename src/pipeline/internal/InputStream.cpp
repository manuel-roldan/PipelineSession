// src/pipeline/internal/InputStream.cpp
#include "InputStream.h"
#include "InputStreamUtil.h"

#include "pipeline/internal/CapsBridge.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/NeatTensorAdapters.h"
#include "sima/nodes/io/InputAppSrc.h"

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace sima {

using pipeline_internal::DiagCtx;

struct InputStream::State {
  GstElement* pipeline = nullptr;
  GstElement* appsrc = nullptr;
  GstElement* appsink = nullptr;
  InputCapsConfig cfg;
  InputAppSrcOptions src_opt;
  InputStreamOptions opt;
  InputBufferPoolGuard pool_guard;
  GstBuffer* reusable_buffer = nullptr;
  size_t reusable_bytes = 0;
  std::shared_ptr<DiagCtx> diag;
  std::shared_ptr<void> guard;
  std::thread worker;
  std::atomic<bool> running{false};
  std::atomic<bool> stop_requested{false};
  std::function<void(RunInputResult)> callback;
  mutable std::mutex error_mu;
  std::string error;
  bool timing_enabled = false;
  std::atomic<std::uint64_t> push_count{0};
  std::atomic<std::uint64_t> push_failures{0};
  std::atomic<std::uint64_t> pull_count{0};
  std::atomic<std::uint64_t> poll_count{0};
  std::atomic<std::uint64_t> alloc_ns{0};
  std::atomic<std::uint64_t> map_ns{0};
  std::atomic<std::uint64_t> copy_ns{0};
  std::atomic<std::uint64_t> push_ns{0};
  std::atomic<std::uint64_t> pull_wait_ns{0};
  std::atomic<std::uint64_t> decode_ns{0};
  std::atomic<std::int64_t> last_push_ns{0};
  std::atomic<bool> eos_sent{false};
};

// Appsrc must carry GstSimaMeta; holder paths preserve it.
void attach_required_meta(GstBuffer* buffer,
                          const InputAppSrcOptions& opt,
                          InputBufferPoolGuard& guard,
                          const char* where) {
  if (!attach_simaai_meta_inplace(buffer, opt, guard, where)) {
    gst_buffer_unref(buffer);
    throw std::runtime_error(std::string(where) + ": failed to attach GstSimaMeta");
  }
}

bool InputStream::push_with_fill(
    const char* where,
    const std::function<size_t(uint8_t*, size_t)>& fill) {
  auto st = state_;
  if (!st || !st->pipeline) {
    throw std::runtime_error(std::string(where) + ": stream is closed");
  }
  if (!st->appsrc) {
    throw std::runtime_error(std::string(where) + ": appsrc not available (no InputAppSrc)");
  }
  const bool timings = st->timing_enabled;

  std::chrono::steady_clock::time_point t_alloc_start{};
  if (timings) t_alloc_start = std::chrono::steady_clock::now();
  GstBuffer* buf = nullptr;
  if (st->opt.reuse_input_buffer) {
    if (!st->reusable_buffer || st->reusable_bytes != st->cfg.bytes) {
      if (st->reusable_buffer) {
        gst_buffer_unref(st->reusable_buffer);
        st->reusable_buffer = nullptr;
        st->reusable_bytes = 0;
      }
      st->reusable_buffer =
          allocate_input_buffer(st->cfg.bytes, st->src_opt, st->pool_guard);
      st->reusable_bytes = st->cfg.bytes;
    }
    buf = st->reusable_buffer;
  } else {
    buf = allocate_input_buffer(st->cfg.bytes, st->src_opt, st->pool_guard);
  }
  std::chrono::steady_clock::time_point t_alloc_end{};
  if (timings) t_alloc_end = std::chrono::steady_clock::now();
  if (!buf) {
    throw std::runtime_error(std::string(where) + ": failed to allocate GstBuffer");
  }

  GstMapInfo mi{};
  std::chrono::steady_clock::time_point t_map_start{};
  if (timings) t_map_start = std::chrono::steady_clock::now();
  if (!gst_buffer_map(buf, &mi, GST_MAP_WRITE)) {
    gst_buffer_unref(buf);
    throw std::runtime_error(std::string(where) + ": failed to map GstBuffer");
  }
  std::chrono::steady_clock::time_point t_map_end{};
  if (timings) t_map_end = std::chrono::steady_clock::now();

  const size_t filled = fill(static_cast<uint8_t*>(mi.data), mi.size);
  if (filled < mi.size) {
    std::memset(static_cast<uint8_t*>(mi.data) + filled, 0, mi.size - filled);
  }
  gst_buffer_unmap(buf, &mi);
  std::chrono::steady_clock::time_point t_copy_end{};
  if (timings) t_copy_end = std::chrono::steady_clock::now();

  attach_required_meta(buf, st->src_opt, st->pool_guard, where);
  GstBuffer* push_src = buf;

  std::chrono::steady_clock::time_point t_push_start{};
  if (timings) t_push_start = std::chrono::steady_clock::now();
  GstBuffer* push_buf = push_src;
  if (st->opt.reuse_input_buffer && push_src == buf) {
    push_buf = gst_buffer_ref(buf);
  }
  GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(st->appsrc), push_buf);
  const auto t_push_end = std::chrono::steady_clock::now();

  if (timings) {
    st->push_count.fetch_add(1, std::memory_order_relaxed);
    st->alloc_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_alloc_end - t_alloc_start).count()),
        std::memory_order_relaxed);
    st->map_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_map_end - t_map_start).count()),
        std::memory_order_relaxed);
    st->copy_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_copy_end - t_map_end).count()),
        std::memory_order_relaxed);
    st->push_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end - t_push_start).count()),
        std::memory_order_relaxed);
  }

  if (ret != GST_FLOW_OK) {
    if (st->opt.reuse_input_buffer) {
      gst_buffer_unref(push_buf);
    } else {
      gst_buffer_unref(buf);
    }
    if (timings) {
      st->push_failures.fetch_add(1, std::memory_order_relaxed);
    }
    return false;
  }
  const auto push_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      t_push_end.time_since_epoch()).count();
  st->last_push_ns.store(static_cast<std::int64_t>(push_ns), std::memory_order_relaxed);
  return true;
}

static std::string image_format_name(NeatImageSpec::PixelFormat fmt) {
  switch (fmt) {
    case NeatImageSpec::PixelFormat::RGB: return "RGB";
    case NeatImageSpec::PixelFormat::BGR: return "BGR";
    case NeatImageSpec::PixelFormat::GRAY8: return "GRAY8";
    case NeatImageSpec::PixelFormat::NV12: return "NV12";
    case NeatImageSpec::PixelFormat::I420: return "I420";
    case NeatImageSpec::PixelFormat::UNKNOWN: return "";
  }
  return "";
}

static std::string neat_format_name(const NeatTensor& t) {
  if (t.semantic.image.has_value()) {
    const std::string fmt = image_format_name(t.semantic.image->format);
    if (!fmt.empty()) return fmt;
  }
  if (t.semantic.tess.has_value()) {
    if (!t.semantic.tess->format.empty()) return t.semantic.tess->format;
  }
  return "";
}

static RunInputResult output_from_sample_stream(GstSample* sample,
                                                const char* where,
                                                bool copy_output,
                                                bool /*map_tensor_ref*/) {
  const auto normalize_format = [](const std::string& fmt) {
    std::string out;
    out.reserve(fmt.size());
    for (char c : fmt) {
      out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    if (out == "YUV420P" || out == "YUV420") return std::string("I420");
    if (out == "GRAY") return std::string("GRAY8");
    return out;
  };

  RunInputResult out;
  GstCaps* out_caps = gst_sample_get_caps(sample);
  const GstStructure* st = out_caps ? gst_caps_get_structure(out_caps, 0) : nullptr;
  const char* media = st ? gst_structure_get_name(st) : nullptr;

  out.caps_string = pipeline_internal::gst_caps_to_string_safe(out_caps);
  out.media_type = media ? media : "";

  if (media && (std::string(media).rfind("video/x-raw", 0) == 0 ||
                std::string(media) == "application/vnd.simaai.tensor")) {
    try {
      out.neat = from_gst_sample(sample);
    } catch (const std::exception&) {
      // Best-effort: keep legacy paths untouched if NeatTensor conversion fails.
    }
  }

  NeatTensor neat = from_gst_sample(sample);
  if (pipeline_internal::env_bool("SIMA_NEAT_CAPS_TRACE", false)) {
    const auto constraint = pipeline_internal::neat_constraint_from_caps(out_caps);
    std::fprintf(stderr,
                 "[NEAT_CAPS] caps=%s constraint=%s tensor=%s\n",
                 out.caps_string.c_str(),
                 pipeline_internal::neat_constraint_debug_string(constraint).c_str(),
                 neat.debug_string().c_str());
  }
  out.kind = RunOutputKind::Tensor;
  out.format = normalize_format(neat_format_name(neat));
  if (copy_output) {
    out.neat = neat.clone();
    out.owned = true;
  } else {
    out.neat = neat;
    out.owned = false;
  }
  return out;
}

size_t dtype_bytes(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::UInt8: return 1;
    case TensorDType::Int8: return 1;
    case TensorDType::UInt16: return 2;
    case TensorDType::Int16: return 2;
    case TensorDType::Int32: return 4;
    case TensorDType::BFloat16: return 2;
    case TensorDType::Float32: return 4;
    case TensorDType::Float64: return 8;
  }
  return 1;
}

size_t neat_plane_bytes_tight(const NeatPlane& plane, TensorDType dtype) {
  if (plane.shape.size() < 2) return 0;
  const int64_t h = plane.shape[0];
  const int64_t w = plane.shape[1];
  if (h <= 0 || w <= 0) return 0;
  const size_t elem = dtype_bytes(dtype);
  return static_cast<size_t>(w) * static_cast<size_t>(h) * elem;
}

size_t neat_dense_bytes_tight(const NeatTensor& input) {
  if (!input.is_dense() || input.shape.empty()) return 0;
  size_t total = dtype_bytes(input.dtype);
  for (const auto dim : input.shape) {
    if (dim <= 0) return 0;
    total *= static_cast<size_t>(dim);
  }
  return total;
}

size_t neat_tensor_bytes_tight(const NeatTensor& input) {
  if (input.is_composite()) {
    size_t total = 0;
    for (const auto& plane : input.planes) {
      total += neat_plane_bytes_tight(plane, input.dtype);
    }
    return total;
  }
  return neat_dense_bytes_tight(input);
}

InputStream InputStream::create(GstElement* pipeline,
                                GstElement* appsrc,
                                GstElement* appsink,
                                const InputCapsConfig& cfg,
                                const InputAppSrcOptions& src_opt,
                                const InputStreamOptions& opt,
                                std::shared_ptr<DiagCtx> diag,
                                std::shared_ptr<void> guard) {
  auto state = std::make_shared<State>();
  state->pipeline = pipeline;
  state->appsrc = appsrc;
  state->appsink = appsink;
  state->cfg = cfg;
  state->src_opt = src_opt;
  state->opt = opt;
  state->diag = std::move(diag);
  state->guard = std::move(guard);
  state->timing_enabled = opt.enable_timings;
  return InputStream(std::move(state));
}

InputStream::InputStream(std::shared_ptr<State> state) : state_(std::move(state)) {}

InputStream::InputStream(InputStream&& other) noexcept : state_(std::move(other.state_)) {}

InputStream& InputStream::operator=(InputStream&& other) noexcept {
  if (this != &other) {
    close();
    state_ = std::move(other.state_);
  }
  return *this;
}

InputStream::~InputStream() {
  close();
}

InputStream::operator bool() const noexcept {
  return state_ && state_->pipeline;
}

bool InputStream::can_push() const noexcept {
  return state_ && state_->appsrc;
}

bool InputStream::can_pull() const noexcept {
  return state_ && state_->appsink;
}

bool InputStream::running() const {
  return state_ && state_->running.load();
}

std::string InputStream::last_error() const {
  if (!state_) return {};
  std::lock_guard<std::mutex> lock(state_->error_mu);
  return state_->error;
}

InputStreamStats InputStream::stats() const {
  InputStreamStats out;
  if (!state_) return out;
  out.push_count = state_->push_count.load();
  out.push_failures = state_->push_failures.load();
  out.pull_count = state_->pull_count.load();
  out.poll_count = state_->poll_count.load();
  const auto avg_us = [](std::uint64_t total_ns, std::uint64_t count) -> double {
    if (count == 0) return 0.0;
    return static_cast<double>(total_ns) / static_cast<double>(count) / 1000.0;
  };
  out.avg_alloc_us = avg_us(state_->alloc_ns.load(), out.push_count);
  out.avg_map_us = avg_us(state_->map_ns.load(), out.push_count);
  out.avg_copy_us = avg_us(state_->copy_ns.load(), out.push_count);
  out.avg_push_us = avg_us(state_->push_ns.load(), out.push_count);
  out.avg_pull_wait_us = avg_us(state_->pull_wait_ns.load(), out.pull_count);
  out.avg_decode_us = avg_us(state_->decode_ns.load(), out.pull_count);
  return out;
}

std::string InputStream::diagnostics_summary() const {
  if (!state_ || !state_->diag) return {};
  std::ostringstream oss;
  if (!state_->diag->pipeline_string.empty()) {
    oss << "Pipeline:\n" << state_->diag->pipeline_string << "\n";
  }
  const std::string boundary = pipeline_internal::boundary_summary(state_->diag);
  if (!boundary.empty()) oss << boundary;
  const std::string stages = pipeline_internal::stage_timing_summary(state_->diag);
  if (!stages.empty()) oss << stages;
  if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_DEBUG", false)) {
    PipelineReport rep = state_->diag->snapshot_basic();
    if (!rep.bus.empty()) {
      oss << "Bus:\n";
      const size_t max_lines = std::min<size_t>(rep.bus.size(), 10);
      for (size_t i = 0; i < max_lines; ++i) {
        const auto& msg = rep.bus[i];
        oss << "  - [" << msg.type << "] " << msg.src << ": " << msg.detail << "\n";
      }
    }
  }
  return oss.str();
}

std::shared_ptr<DiagCtx> InputStream::diag_ctx() const {
  if (!state_) return {};
  return state_->diag;
}

void InputStream::start(std::function<void(RunInputResult)> on_output) {
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::start: stream is closed");
  }
  if (!state_->appsink) {
    throw std::runtime_error("InputStream::start: appsink not available (no OutputAppSink)");
  }
  if (state_->running.load()) {
    throw std::runtime_error("InputStream::start: stream already running");
  }
  if (state_->opt.reuse_input_buffer) {
    std::fprintf(stderr,
                 "[WARN] InputStream::start: reuse_input_buffer is unsafe for async streams; "
                 "disabling to avoid data races.\n");
    state_->opt.reuse_input_buffer = false;
  }
  state_->callback = std::move(on_output);
  state_->stop_requested.store(false);
  state_->running.store(true);

  auto st = state_;
  st->worker = std::thread([st]() {
    const int poll_ms = (st->opt.poll_ms > 0)
        ? st->opt.poll_ms
        : std::max(10,
            std::atoi(pipeline_internal::env_str("SIMA_INPUTSTREAM_POLL_MS", "50").c_str()));
    const int timeout_ms = st->opt.timeout_ms;
    const bool timings = st->timing_enabled;
    auto cb = st->callback;
    std::int64_t last_output_ns = 0;
    try {
      while (!st->stop_requested.load()) {
        std::chrono::steady_clock::time_point t_wait_start{};
        if (timings) t_wait_start = std::chrono::steady_clock::now();
        auto sample_opt = pipeline_internal::try_pull_sample_sliced(
            st->pipeline, st->appsink, poll_ms, st->diag, "InputStream::start");
        std::chrono::steady_clock::time_point t_wait_end{};
        if (timings) t_wait_end = std::chrono::steady_clock::now();
        if (timings) {
          st->poll_count.fetch_add(1, std::memory_order_relaxed);
        }
        if (!sample_opt.has_value()) {
          if (timeout_ms > 0) {
            const std::int64_t last_push_ns =
                st->last_push_ns.load(std::memory_order_relaxed);
            if (last_push_ns > 0) {
              const auto now_tp = std::chrono::steady_clock::now();
              const std::int64_t now_ns = static_cast<std::int64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      now_tp.time_since_epoch()).count());
              const std::int64_t last_activity_ns =
                  (last_output_ns > last_push_ns) ? last_output_ns : last_push_ns;
              const std::int64_t timeout_ns =
                  static_cast<std::int64_t>(timeout_ms) * 1000000;
              if (now_ns - last_activity_ns > timeout_ns) {
                if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_DOT_ON_TIMEOUT", false)) {
                  pipeline_internal::maybe_dump_dot(st->pipeline, "inputstream_timeout");
                }
                std::lock_guard<std::mutex> lock(st->error_mu);
                st->error = "InputStream::start: timeout waiting for output";
                st->stop_requested.store(true);
                break;
              }
            }
          }
          continue;
        }
        if (timings) {
          st->pull_wait_ns.fetch_add(
              static_cast<std::uint64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      t_wait_end - t_wait_start).count()),
              std::memory_order_relaxed);
        }
        GstSample* sample = sample_opt.value();
        std::chrono::steady_clock::time_point t_decode_start{};
        if (timings) t_decode_start = std::chrono::steady_clock::now();
        RunInputResult out =
            output_from_sample_stream(sample,
                                      "InputStream::start",
                                      st->opt.copy_output,
                                      !st->opt.no_map_tensor_ref);
        std::chrono::steady_clock::time_point t_decode_end{};
        if (timings) t_decode_end = std::chrono::steady_clock::now();
        gst_sample_unref(sample);
        if (timings) {
          st->decode_ns.fetch_add(
              static_cast<std::uint64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      t_decode_end - t_decode_start).count()),
              std::memory_order_relaxed);
          st->pull_count.fetch_add(1, std::memory_order_relaxed);
          last_output_ns = static_cast<std::int64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  t_decode_end.time_since_epoch()).count());
        } else {
          const auto now_tp = std::chrono::steady_clock::now();
          last_output_ns = static_cast<std::int64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  now_tp.time_since_epoch()).count());
        }
        if (cb) cb(std::move(out));
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(st->error_mu);
      st->error = e.what();
    }
    st->running.store(false);
  });
}

void InputStream::stop() {
  if (!state_) return;
  state_->stop_requested.store(true);
  if (state_->worker.joinable()) {
    state_->worker.join();
  }
  state_->running.store(false);
}

void InputStream::close() {
  if (!state_) return;
  stop();
  if (state_->reusable_buffer) {
    gst_buffer_unref(state_->reusable_buffer);
    state_->reusable_buffer = nullptr;
    state_->reusable_bytes = 0;
  }
  if (state_->appsrc) {
    gst_object_unref(state_->appsrc);
    state_->appsrc = nullptr;
  }
  if (state_->appsink) {
    gst_object_unref(state_->appsink);
    state_->appsink = nullptr;
  }
  if (state_->pipeline) {
    pipeline_internal::stop_and_unref(state_->pipeline);
  }
  state_.reset();
}

void InputStream::push(const cv::Mat& input) {
  if (!try_push(input)) {
    throw std::runtime_error("InputStream::push: appsrc push failed");
  }
}

bool InputStream::try_push(const cv::Mat& input) {
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::try_push: stream is closed");
  }
  if (!state_->opt.allow_mismatched_input) {
    validate_input_matches_caps(state_->cfg, state_->src_opt, input, "InputStream::try_push");
  } else if (input.empty()) {
    throw std::invalid_argument("InputStream::try_push: input frame is empty");
  }

  cv::Mat contiguous = input;
  if (!contiguous.isContinuous()) {
    contiguous = input.clone();
  }
  const size_t input_bytes = contiguous.total() * contiguous.elemSize();
  return push_with_fill("InputStream::try_push(mat)",
                        [&](uint8_t* dst, size_t dst_bytes) -> size_t {
                          const size_t copy_bytes = std::min(input_bytes, dst_bytes);
                          if (copy_bytes > 0) {
                            std::memcpy(dst, contiguous.data, copy_bytes);
                          }
                          return copy_bytes;
                        });
}

void InputStream::push(const NeatTensor& input) {
  if (!try_push(input)) {
    throw std::runtime_error("InputStream::push: appsrc push failed");
  }
}

bool InputStream::try_push(const NeatTensor& input) {
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::try_push: stream is closed");
  }
  if (!state_->opt.allow_mismatched_input) {
    validate_input_matches_caps(state_->cfg, state_->src_opt, input, "InputStream::try_push");
  } else if (!input.storage) {
    throw std::invalid_argument("InputStream::try_push: NeatTensor missing storage");
  }

  const size_t input_bytes = neat_tensor_bytes_tight(input);
  return push_with_fill("InputStream::try_push(neat)",
                        [&](uint8_t* dst, size_t dst_bytes) -> size_t {
                          const size_t copy_bytes = std::min(input_bytes, dst_bytes);
                          if (!input.storage || copy_bytes == 0) return 0;

                          NeatMapping mapping = input.map(NeatMapMode::Read);
                          if (!mapping.data) {
                            throw std::runtime_error("InputStream::try_push: NeatTensor map failed");
                          }
                          const uint8_t* base = static_cast<const uint8_t*>(mapping.data);

                          if (input.is_composite()) {
                            size_t offset = 0;
                            for (const auto& plane : input.planes) {
                              if (offset >= copy_bytes) break;
                              const size_t plane_bytes =
                                  neat_plane_bytes_tight(plane, input.dtype);
                              const size_t remaining = copy_bytes - offset;
                              const size_t take = std::min(plane_bytes, remaining);
                              if (take == 0) continue;
                              const size_t plane_offset =
                                  static_cast<size_t>(plane.byte_offset);
                              if (plane_offset + take > mapping.size_bytes) {
                                throw std::runtime_error(
                                    "InputStream::try_push: NeatTensor plane out of range");
                              }
                              std::memcpy(dst + offset, base + plane_offset, take);
                              offset += take;
                            }
                            return offset;
                          }

                          const size_t base_offset = static_cast<size_t>(input.byte_offset);
                          if (base_offset + copy_bytes > mapping.size_bytes) {
                            throw std::runtime_error(
                                "InputStream::try_push: NeatTensor buffer out of range");
                          }
                          std::memcpy(dst, base + base_offset, copy_bytes);
                          return copy_bytes;
                        });
}

void InputStream::push_holder(const std::shared_ptr<void>& holder) {
  if (!try_push_holder(holder)) {
    throw std::runtime_error("InputStream::push_holder: appsrc push failed");
  }
}

bool InputStream::try_push_holder(const std::shared_ptr<void>& holder) {
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::try_push_holder: stream is closed");
  }
  if (!state_->appsrc) {
    throw std::runtime_error(
        "InputStream::try_push_holder: appsrc not available (no InputAppSrc)");
  }
  if (!holder) {
    throw std::invalid_argument("InputStream::try_push_holder: missing holder");
  }

  auto st = state_;
  const bool timings = st->timing_enabled;

  // This path reuses the original GstBuffer so plugin metadata (GstSimaMeta)
  // and layout assumptions survive standalone stage boundaries.
  GstBuffer* buf = pipeline_internal::buffer_from_tensor_holder(holder);
  if (!buf) {
    throw std::runtime_error("InputStream::try_push_holder: missing GstBuffer");
  }
  dump_buffer_memories(buf, "InputStream::try_push_holder");
  GstBuffer* push_src = buf;

  std::chrono::steady_clock::time_point t_push_start{};
  if (timings) t_push_start = std::chrono::steady_clock::now();
  GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(state_->appsrc), push_src);
  const auto t_push_end = std::chrono::steady_clock::now();

  if (timings) {
    st->push_count.fetch_add(1, std::memory_order_relaxed);
    st->push_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end - t_push_start).count()),
        std::memory_order_relaxed);
  }

  if (ret != GST_FLOW_OK) {
    gst_buffer_unref(push_src);
    if (timings) {
      st->push_failures.fetch_add(1, std::memory_order_relaxed);
    }
    return false;
  }
  return true;
}

void InputStream::signal_eos() {
  if (!state_ || !state_->appsrc) return;
  bool expected = false;
  if (!state_->eos_sent.compare_exchange_strong(expected, true)) return;
  GstFlowReturn ret = gst_app_src_end_of_stream(GST_APP_SRC(state_->appsrc));
  if (ret != GST_FLOW_OK) {
    throw std::runtime_error("InputStream::signal_eos: gst_app_src_end_of_stream failed");
  }
}

RunInputResult InputStream::pull(int timeout_ms) {
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::pull: stream is closed");
  }
  if (!state_->appsink) {
    throw std::runtime_error("InputStream::pull: appsink not available (no OutputAppSink)");
  }
  const int timeout = (timeout_ms >= 0) ? timeout_ms : state_->opt.timeout_ms;
  const bool timings = state_->timing_enabled;
  std::chrono::steady_clock::time_point t_wait_start{};
  if (timings) t_wait_start = std::chrono::steady_clock::now();
  auto sample_opt = pipeline_internal::try_pull_sample_sliced(
      state_->pipeline, state_->appsink, timeout, state_->diag, "InputStream::pull");
  std::chrono::steady_clock::time_point t_wait_end{};
  if (timings) t_wait_end = std::chrono::steady_clock::now();
  if (!sample_opt.has_value()) {
    throw std::runtime_error("InputStream::pull: timeout waiting for output");
  }
  GstSample* sample = sample_opt.value();
  std::chrono::steady_clock::time_point t_decode_start{};
  if (timings) t_decode_start = std::chrono::steady_clock::now();
  RunInputResult out = output_from_sample_stream(sample,
                                                 "InputStream::pull",
                                                 state_->opt.copy_output,
                                                 !state_->opt.no_map_tensor_ref);
  std::chrono::steady_clock::time_point t_decode_end{};
  if (timings) t_decode_end = std::chrono::steady_clock::now();
  gst_sample_unref(sample);
  if (timings) {
    state_->pull_count.fetch_add(1, std::memory_order_relaxed);
    state_->pull_wait_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                t_wait_end - t_wait_start).count()),
        std::memory_order_relaxed);
    state_->decode_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                t_decode_end - t_decode_start).count()),
        std::memory_order_relaxed);
  }
  return out;
}

RunInputResult InputStream::push_and_pull(const cv::Mat& input, int timeout_ms) {
  push(input);
  return pull(timeout_ms);
}

RunInputResult InputStream::push_and_pull(const NeatTensor& input, int timeout_ms) {
  push(input);
  return pull(timeout_ms);
}

RunInputResult InputStream::push_and_pull_holder(const std::shared_ptr<void>& holder,
                                                 int timeout_ms) {
  push_holder(holder);
  return pull(timeout_ms);
}

} // namespace sima
