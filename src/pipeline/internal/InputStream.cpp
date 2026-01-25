// src/pipeline/internal/InputStream.cpp
#include "InputStream.h"

#include "pipeline/PipelineOptions.h"
#include "InputStreamUtil.h"

#include "pipeline/internal/CapsBridge.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/SampleUtil.h"
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
#include <limits>
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
  if (!attach_simaai_meta_inplace(buffer, opt, guard, where,
                                  std::nullopt, std::nullopt, std::nullopt)) {
    gst_buffer_unref(buffer);
    throw std::runtime_error(std::string(where) + ": failed to attach GstSimaMeta");
  }
}

bool InputStream::push_with_fill(
    const char* where,
    const std::function<size_t(uint8_t*, size_t)>& fill,
    const std::optional<int64_t>& frame_id_override,
    const std::optional<std::string>& stream_id_override,
    const std::optional<std::string>& buffer_name_override) {
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

  if (!attach_simaai_meta_inplace(buf, st->src_opt, st->pool_guard, where,
                                  frame_id_override,
                                  stream_id_override,
                                  buffer_name_override)) {
    gst_buffer_unref(buf);
    throw std::runtime_error(std::string(where) + ": failed to attach GstSimaMeta");
  }
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

constexpr const char* kSampleMetaName = "GstSimaSampleMeta";

bool sample_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_SAMPLE_DEBUG", false);
}

bool sample_bytes_enabled() {
  return pipeline_internal::env_bool("SIMA_SAMPLE_BYTES", false);
}

bool flow_trace_enabled() {
  return pipeline_internal::env_bool("SIMA_FLOW_TRACE", false);
}

size_t neat_tensor_bytes_tight(const NeatTensor& input);

static std::optional<RunInputResult> bundle_from_sample_meta(GstSample* sample,
                                                             const char* where,
                                                             bool copy_output,
                                                             bool map_tensor_ref);

static std::string gst_time_to_string(GstClockTime t) {
  if (!GST_CLOCK_TIME_IS_VALID(t)) {
    return "none";
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%" GST_TIME_FORMAT, GST_TIME_ARGS(t));
  return std::string(buf);
}

static void fill_output_meta_from_sample(GstSample* sample, RunInputResult* out) {
  if (!sample || !out) return;
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) return;
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s) return;

  gint64 frame_id = -1;
  gint64 buffer_offset = -1;
  gst_structure_get_int64(s, "frame-id", &frame_id);
  gst_structure_get_int64(s, "buffer-offset", &buffer_offset);
  const char* stream_id = gst_structure_get_string(s, "stream-id");
  const char* buffer_name = gst_structure_get_string(s, "buffer-name");
  if (frame_id >= 0) out->frame_id = frame_id;
  if (stream_id) out->stream_id = stream_id;
  if (buffer_name) out->port_name = buffer_name;
  if (buffer_offset >= 0 && buffer_offset <= std::numeric_limits<int>::max()) {
    out->output_index = static_cast<int>(buffer_offset);
  }
}

static void log_sample_flow(const RunInputResult& out,
                            GstSample* sample,
                            const char* where,
                            int bundle_fields = -1) {
  if (!flow_trace_enabled()) return;

  GstBuffer* buffer = sample ? gst_sample_get_buffer(sample) : nullptr;
  const gsize size = buffer ? gst_buffer_get_size(buffer) : 0;
  const GstClockTime pts = buffer ? GST_BUFFER_PTS(buffer) : GST_CLOCK_TIME_NONE;
  const GstClockTime dts = buffer ? GST_BUFFER_DTS(buffer) : GST_CLOCK_TIME_NONE;
  const GstClockTime dur = buffer ? GST_BUFFER_DURATION(buffer) : GST_CLOCK_TIME_NONE;
  const guint flags = buffer ? GST_BUFFER_FLAGS(buffer) : 0;

  const std::string pts_s = gst_time_to_string(pts);
  const std::string dts_s = gst_time_to_string(dts);
  const std::string dur_s = gst_time_to_string(dur);

  const char* stream_id = out.stream_id.empty() ? "<none>" : out.stream_id.c_str();
  const char* buffer_name = out.port_name.empty() ? "<none>" : out.port_name.c_str();
  const char* caps = out.caps_string.empty() ? "<none>" : out.caps_string.c_str();
  const char* media = out.media_type.empty() ? "<none>" : out.media_type.c_str();

  if (bundle_fields >= 0) {
    std::fprintf(stderr,
                 "[FLOW] %s sample size=%" G_GSIZE_FORMAT
                 " pts=%s dts=%s dur=%s flags=0x%x caps=%s media=%s frame_id=%"
                 G_GINT64_FORMAT " stream_id=%s buffer_name=%s output_index=%d bundle_fields=%d\n",
                 where,
                 size,
                 pts_s.c_str(),
                 dts_s.c_str(),
                 dur_s.c_str(),
                 flags,
                 caps,
                 media,
                 static_cast<gint64>(out.frame_id),
                 stream_id,
                 buffer_name,
                 out.output_index,
                 bundle_fields);
  } else {
    std::fprintf(stderr,
                 "[FLOW] %s sample size=%" G_GSIZE_FORMAT
                 " pts=%s dts=%s dur=%s flags=0x%x caps=%s media=%s frame_id=%"
                 G_GINT64_FORMAT " stream_id=%s buffer_name=%s output_index=%d\n",
                 where,
                 size,
                 pts_s.c_str(),
                 dts_s.c_str(),
                 dur_s.c_str(),
                 flags,
                 caps,
                 media,
                 static_cast<gint64>(out.frame_id),
                 stream_id,
                 buffer_name,
                 out.output_index);
  }
}

static RunInputResult output_from_sample_stream_inner(GstSample* sample,
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

  fill_output_meta_from_sample(sample, &out);

  log_sample_flow(out, sample, where);

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
  out.payload_tag = normalize_format(neat_format_name(neat));
  out.format = out.payload_tag;
  if (copy_output) {
    out.neat = neat.clone();
    out.owned = true;
  } else {
    out.neat = neat;
    out.owned = false;
  }
  return out;
}

static RunInputResult output_from_sample_stream(GstSample* sample,
                                                const char* where,
                                                bool copy_output,
                                                bool map_tensor_ref) {
  if (auto bundle = bundle_from_sample_meta(sample, where, copy_output, map_tensor_ref)) {
    return *bundle;
  }
  RunInputResult out =
      output_from_sample_stream_inner(sample, where, copy_output, map_tensor_ref);
  if (pipeline_internal::env_bool("SIMA_SAMPLE_FORCE_BUNDLE", false)) {
    RunInputResult forced;
    forced.kind = RunOutputKind::Bundle;
    forced.owned = out.owned;
    forced.frame_id = out.frame_id;
    forced.stream_id = out.stream_id;
    forced.fields.emplace_back(std::move(out));
    if (sample_debug_enabled()) {
      std::fprintf(stderr, "[SAMPLE] %s: forced bundle with 1 field\n", where);
    }
    return forced;
  }
  return out;
}

static std::optional<RunInputResult> bundle_from_sample_meta(GstSample* sample,
                                                             const char* where,
                                                             bool copy_output,
                                                             bool map_tensor_ref) {
  if (!sample) return std::nullopt;
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) return std::nullopt;
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, kSampleMetaName);
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s) return std::nullopt;

  const GValue* list_val = gst_structure_get_value(s, "fields");
  if (!list_val || !GST_VALUE_HOLDS_LIST(list_val)) return std::nullopt;
  const guint field_count = gst_value_list_get_size(list_val);

  RunInputResult out;
  out.kind = RunOutputKind::Bundle;
  out.owned = copy_output;

  GstCaps* out_caps = gst_sample_get_caps(sample);
  const GstStructure* out_st = out_caps ? gst_caps_get_structure(out_caps, 0) : nullptr;
  const char* media = out_st ? gst_structure_get_name(out_st) : nullptr;
  out.caps_string = pipeline_internal::gst_caps_to_string_safe(out_caps);
  out.media_type = media ? media : "";
  fill_output_meta_from_sample(sample, &out);

  log_sample_flow(out, sample, where, static_cast<int>(field_count));

  if (sample_debug_enabled()) {
    std::fprintf(stderr, "[SAMPLE] %s: bundle meta fields=%u\n", where, field_count);
  }

  out.fields.reserve(field_count);
  for (guint i = 0; i < field_count; ++i) {
    const GValue* entry_val = gst_value_list_get_value(list_val, i);
    if (!entry_val || !GST_VALUE_HOLDS_STRUCTURE(entry_val)) {
      if (sample_debug_enabled()) {
        std::fprintf(stderr, "[SAMPLE] %s: field[%u] invalid entry\n", where, i);
      }
      continue;
    }
    const GstStructure* entry = static_cast<const GstStructure*>(g_value_get_boxed(entry_val));
    if (!entry) continue;

    const char* field_name = gst_structure_get_string(entry, "name");
    const char* buffer_name = gst_structure_get_string(entry, "buffer-name");
    const char* caps_str = gst_structure_get_string(entry, "caps");

    const GValue* buf_val = gst_structure_get_value(entry, "buffer");
    if (!buf_val || !GST_VALUE_HOLDS_BUFFER(buf_val)) {
      if (sample_debug_enabled()) {
        std::fprintf(stderr, "[SAMPLE] %s: field[%u] missing buffer\n", where, i);
      }
      continue;
    }
    GstBuffer* field_buf = gst_value_get_buffer(buf_val);
    if (!field_buf) continue;

    GstCaps* field_caps = nullptr;
    if (caps_str && *caps_str) {
      field_caps = gst_caps_from_string(caps_str);
    }
    if (!field_caps) {
      field_caps = gst_sample_get_caps(sample);
      if (field_caps) gst_caps_ref(field_caps);
    }
    if (!field_caps) {
      if (sample_debug_enabled()) {
        std::fprintf(stderr, "[SAMPLE] %s: field[%u] missing caps\n", where, i);
      }
      continue;
    }

    GstSample* field_sample = gst_sample_new(field_buf, field_caps, nullptr, nullptr);
    gst_caps_unref(field_caps);
    if (!field_sample) continue;

    RunInputResult field_out =
        output_from_sample_stream_inner(field_sample, where, copy_output, map_tensor_ref);
    gst_sample_unref(field_sample);

    if (field_name && *field_name) {
      field_out.port_name = field_name;
    } else if (buffer_name && *buffer_name) {
      field_out.port_name = buffer_name;
    }

    if (out.frame_id < 0 && field_out.frame_id >= 0) {
      out.frame_id = field_out.frame_id;
    }
    if (out.stream_id.empty() && !field_out.stream_id.empty()) {
      out.stream_id = field_out.stream_id;
    }

    if (sample_debug_enabled() || sample_bytes_enabled()) {
      const char* name = field_out.port_name.empty() ? "field" : field_out.port_name.c_str();
      if (sample_debug_enabled()) {
        std::fprintf(stderr,
                     "[SAMPLE] %s: field[%u] name=%s caps=%s\n",
                     where,
                     i,
                     name,
                     field_out.caps_string.empty() ? "<none>" : field_out.caps_string.c_str());
      }
      if (sample_bytes_enabled()) {
        const size_t buf_bytes = static_cast<size_t>(gst_buffer_get_size(field_buf));
        size_t tensor_bytes = 0;
        if (field_out.neat.has_value()) {
          tensor_bytes = neat_tensor_bytes_tight(*field_out.neat);
        }
        std::fprintf(stderr,
                     "[SAMPLE] %s: field[%u] name=%s buffer-bytes=%zu tensor-bytes=%zu copy=%s\n",
                     where,
                     i,
                     name,
                     buf_bytes,
                     tensor_bytes,
                     copy_output ? "true" : "false");
      }
    }

    out.fields.emplace_back(std::move(field_out));
  }

  if (out.fields.empty()) {
    if (sample_debug_enabled()) {
      std::fprintf(stderr, "[SAMPLE] %s: bundle meta had no valid fields\n", where);
    }
    return std::nullopt;
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
                        },
                        std::nullopt,
                        std::nullopt,
                        std::nullopt);
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
                        },
                        std::nullopt,
                        std::nullopt,
                        std::nullopt);
}

void InputStream::push_message(const RunOutput& msg) {
  if (!try_push_message(msg)) {
    throw std::runtime_error("InputStream::push_message: appsrc push failed");
  }
}

bool InputStream::try_push_message(const RunOutput& msg) {
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::try_push_message: stream is closed");
  }
  if (!state_->appsrc) {
    throw std::runtime_error(
        "InputStream::try_push_message: appsrc not available (no InputAppSrc)");
  }

  if (msg.kind == RunOutputKind::Bundle) {
    std::string err;
    auto holder = pipeline_internal::make_sample_holder_from_bundle(msg, &err);
    if (!holder) {
      throw std::runtime_error(err.empty()
                                   ? "InputStream::try_push_message: bundle to sample failed"
                                   : err);
    }
    return try_push_holder(holder);
  }

  if (msg.kind != RunOutputKind::Tensor || !msg.neat.has_value()) {
    throw std::runtime_error("InputStream::try_push_message: missing tensor");
  }

  const NeatTensor& input = *msg.neat;
  if (input.storage && input.storage->holder) {
    GstBuffer* buf = pipeline_internal::buffer_from_tensor_holder(input.storage->holder);
    if (!buf) {
      throw std::runtime_error("InputStream::try_push_message: missing GstBuffer");
    }
    update_simaai_meta_fields(buf,
                              msg.frame_id >= 0 ? std::optional<int64_t>(msg.frame_id) : std::nullopt,
                              msg.stream_id.empty() ? std::nullopt : std::optional<std::string>(msg.stream_id),
                              msg.port_name.empty() ? std::nullopt : std::optional<std::string>(msg.port_name));
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(state_->appsrc), buf);
    if (ret != GST_FLOW_OK) {
      gst_buffer_unref(buf);
      return false;
    }
    return true;
  }

  if (!state_->opt.allow_mismatched_input) {
    validate_input_matches_caps(state_->cfg, state_->src_opt, input, "InputStream::try_push_message");
  } else if (!input.storage) {
    throw std::invalid_argument("InputStream::try_push_message: NeatTensor missing storage");
  }

  const size_t input_bytes = neat_tensor_bytes_tight(input);
  return push_with_fill("InputStream::try_push_message",
                        [&](uint8_t* dst, size_t dst_bytes) -> size_t {
                          const size_t copy_bytes = std::min(input_bytes, dst_bytes);
                          if (!input.storage || copy_bytes == 0) return 0;

                          NeatMapping mapping = input.map(NeatMapMode::Read);
                          if (!mapping.data) {
                            throw std::runtime_error("InputStream::try_push_message: map failed");
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
                                    "InputStream::try_push_message: NeatTensor plane out of range");
                              }
                              std::memcpy(dst + offset, base + plane_offset, take);
                              offset += take;
                            }
                            return offset;
                          }

                          const size_t base_offset = static_cast<size_t>(input.byte_offset);
                          if (base_offset + copy_bytes > mapping.size_bytes) {
                            throw std::runtime_error(
                                "InputStream::try_push_message: NeatTensor buffer out of range");
                          }
                          std::memcpy(dst, base + base_offset, copy_bytes);
                          return copy_bytes;
                        },
                        msg.frame_id >= 0 ? std::optional<int64_t>(msg.frame_id) : std::nullopt,
                        msg.stream_id.empty() ? std::nullopt : std::optional<std::string>(msg.stream_id),
                        msg.port_name.empty() ? std::nullopt : std::optional<std::string>(msg.port_name));
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
