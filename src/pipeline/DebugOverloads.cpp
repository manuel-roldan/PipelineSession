#include "pipeline/debug/DebugOverloads.h"

#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "nodes/common/AppSink.h"
#include "nodes/common/Caps.h"
#include "nodes/io/InputAppSrc.h"
#include "pipeline/PipelineSession.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/SimaaiGuard.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#if __has_include(<simaai/gstsimaaibufferpool.h>)
#include <simaai/gstsimaaibufferpool.h>
#define SIMA_HAS_SIMAAI_POOL 1
#else
#define SIMA_HAS_SIMAAI_POOL 0
#endif

namespace sima::nodes::groups {
namespace {

bool debug_overloads_enabled() {
  const char* env = std::getenv("SIMA_DEBUG_OVERLOADS_LOG");
  return env && env[0] != '\0' && env[0] != '0';
}

void debug_log(const std::string& msg) {
  if (!debug_overloads_enabled()) return;
  std::cerr << "[debug_overloads] " << msg << "\n";
}

std::string spec_brief(const OutputSpec& spec) {
  std::string out;
  out.reserve(128);
  out += "{media=" + (spec.media_type.empty() ? "?" : spec.media_type);
  out += " format=" + (spec.format.empty() ? "?" : spec.format);
  out += " w=" + std::to_string(spec.width);
  out += " h=" + std::to_string(spec.height);
  out += " d=" + std::to_string(spec.depth);
  out += " mem=" + (spec.memory.empty() ? "?" : spec.memory);
  out += " bytes=" + std::to_string(spec.byte_size);
  out += "}";
  return out;
}

std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

OutputSpec spec_from_caps(GstCaps* caps) {
  OutputSpec out;
  if (!caps) return out;

  const GstStructure* st = gst_caps_get_structure(caps, 0);
  if (!st) return out;

  const char* name = gst_structure_get_name(st);
  if (!name) return out;
  out.media_type = name;

  const char* fmt = gst_structure_get_string(st, "format");
  if (fmt) out.format = fmt;

  int w = 0;
  int h = 0;
  int d = 0;
  if (gst_structure_get_int(st, "width", &w)) out.width = w;
  if (gst_structure_get_int(st, "height", &h)) out.height = h;
  if (gst_structure_get_int(st, "depth", &d)) out.depth = d;

  int fps_n = 0;
  int fps_d = 1;
  if (gst_structure_get_fraction(st, "framerate", &fps_n, &fps_d)) {
    out.fps_num = fps_n;
    out.fps_den = (fps_d == 0) ? 1 : fps_d;
  }

  if (out.media_type == "video/x-raw") {
    out.dtype = "UInt8";
    if (out.format == "RGB" || out.format == "BGR") out.layout = "HWC";
    if (out.format == "GRAY8") out.layout = "HW";
    if (out.format == "NV12" || out.format == "I420") out.layout = "Planar";
  } else if (out.media_type == "application/vnd.simaai.tensor") {
    if (out.format == "FP32" || out.format == "DETESSDEQUANT") out.dtype = "Float32";
    if (out.format == "DETESS") out.dtype = "UInt16";
    if (out.width > 0 && out.height > 0 && out.depth > 0) out.layout = "HWC";
  }

  const std::string feat = pipeline_internal::caps_features_string(caps);
  if (feat.find("SystemMemory") != std::string::npos) out.memory = "SystemMemory";
  out.byte_size = expected_byte_size(out);
  out.certainty = SpecCertainty::Derived;
  out.note = "caps";
  return out;
}

OutputSpec spec_from_caps_string(const std::string& caps_string) {
  if (caps_string.empty()) return {};
  GstCaps* caps = gst_caps_from_string(caps_string.c_str());
  if (!caps) return {};
  OutputSpec out = spec_from_caps(caps);
  gst_caps_unref(caps);
  return out;
}

std::size_t tensor_byte_size(const FrameTensor& t) {
  std::size_t total = 0;
  for (const auto& p : t.planes) total += p.size();
  return total;
}

bool is_tensor_media_type(const OutputSpec& spec) {
  return spec.media_type == "application/vnd.simaai.tensor";
}

bool is_video_media_type(const OutputSpec& spec) {
  return spec.media_type == "video/x-raw";
}

FrameTensor tensor_from_packet(const TapPacket& pkt, const OutputSpec& spec) {
  FrameTensor out;
  out.caps_string = pkt.caps_string;
  out.pts_ns = pkt.pts_ns;
  out.dts_ns = pkt.dts_ns;
  out.duration_ns = pkt.duration_ns;
  out.keyframe = pkt.keyframe;

  const int w = pkt.video.width;
  const int h = pkt.video.height;

  if (pkt.format == TapFormat::NV12) {
    if (w <= 0 || h <= 0) throw std::runtime_error("NV12: missing width/height");
    const std::size_t y_sz = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    const std::size_t uv_sz = y_sz / 2;
    if (pkt.bytes.size() != y_sz + uv_sz) {
      throw std::runtime_error("NV12: byte size mismatch");
    }
    out.dtype = TensorDType::UInt8;
    out.layout = TensorLayout::Planar;
    out.format = "NV12";
    out.width = w;
    out.height = h;
    out.shape = {h, w};
    out.strides = {w, 1};
    out.planes.push_back(std::vector<uint8_t>(pkt.bytes.begin(),
                                              pkt.bytes.begin() + static_cast<long>(y_sz)));
    out.planes.push_back(std::vector<uint8_t>(pkt.bytes.begin() + static_cast<long>(y_sz),
                                              pkt.bytes.end()));
    return out;
  }

  if (pkt.format == TapFormat::I420) {
    if (w <= 0 || h <= 0) throw std::runtime_error("I420: missing width/height");
    const std::size_t y_sz = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    const std::size_t u_sz = y_sz / 4;
    const std::size_t v_sz = u_sz;
    if (pkt.bytes.size() != y_sz + u_sz + v_sz) {
      throw std::runtime_error("I420: byte size mismatch");
    }
    out.dtype = TensorDType::UInt8;
    out.layout = TensorLayout::Planar;
    out.format = "I420";
    out.width = w;
    out.height = h;
    out.shape = {h, w};
    out.strides = {w, 1};
    auto it = pkt.bytes.begin();
    out.planes.push_back(std::vector<uint8_t>(it, it + static_cast<long>(y_sz)));
    it += static_cast<long>(y_sz);
    out.planes.push_back(std::vector<uint8_t>(it, it + static_cast<long>(u_sz)));
    it += static_cast<long>(u_sz);
    out.planes.push_back(std::vector<uint8_t>(it, it + static_cast<long>(v_sz)));
    return out;
  }

  if (pkt.format == TapFormat::RGB || pkt.format == TapFormat::BGR) {
    if (w <= 0 || h <= 0) throw std::runtime_error("RGB/BGR: missing width/height");
    const std::size_t sz = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3;
    if (pkt.bytes.size() != sz) throw std::runtime_error("RGB/BGR: byte size mismatch");
    out.dtype = TensorDType::UInt8;
    out.layout = TensorLayout::HWC;
    out.format = (pkt.format == TapFormat::RGB) ? "RGB" : "BGR";
    out.width = w;
    out.height = h;
    out.shape = {h, w, 3};
    out.strides = {w * 3, 3, 1};
    out.planes.push_back(pkt.bytes);
    return out;
  }

  if (pkt.format == TapFormat::GRAY8) {
    if (w <= 0 || h <= 0) throw std::runtime_error("GRAY8: missing width/height");
    const std::size_t sz = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    if (pkt.bytes.size() != sz) throw std::runtime_error("GRAY8: byte size mismatch");
    out.dtype = TensorDType::UInt8;
    out.layout = TensorLayout::HW;
    out.format = "GRAY8";
    out.width = w;
    out.height = h;
    out.shape = {h, w};
    out.strides = {w, 1};
    out.planes.push_back(pkt.bytes);
    return out;
  }

  if (is_tensor_media_type(spec)) {
    out.format = spec.format;
    out.width = spec.width;
    out.height = spec.height;
    out.dtype = (spec.dtype == "Float32") ? TensorDType::Float32 : TensorDType::UInt16;
    if (spec.depth > 0 && spec.width > 0 && spec.height > 0) {
      out.layout = TensorLayout::HWC;
      out.shape = {spec.height, spec.width, spec.depth};
      const int row_stride = spec.width * spec.depth;
      out.strides = {row_stride, spec.depth, 1};
    } else {
      out.layout = TensorLayout::Unknown;
    }
    out.planes.push_back(pkt.bytes);
    return out;
  }

  throw std::runtime_error("Unsupported packet format for tensor conversion");
}

void append_warning(debug::DebugOutput& out, const std::string& msg, bool strict) {
  if (strict) {
    throw std::runtime_error(msg);
  }
  out.warnings.push_back(msg);
}

void validate_expected(const OutputSpec& expected,
                       const OutputSpec& observed,
                       std::size_t actual_bytes,
                       debug::DebugOutput& out,
                       const debug::DebugOptions& dbg) {
  if (!expected.media_type.empty() && !observed.media_type.empty() &&
      expected.media_type != observed.media_type) {
    append_warning(out, "media_type mismatch: expected " + expected.media_type +
                           " got " + observed.media_type, dbg.strict);
  }
  if (expected.width > 0 && observed.width > 0 && expected.width != observed.width) {
    append_warning(out, "width mismatch", dbg.strict);
  }
  if (expected.height > 0 && observed.height > 0 && expected.height != observed.height) {
    append_warning(out, "height mismatch", dbg.strict);
  }
  if (!expected.format.empty() && !observed.format.empty() &&
      expected.format != observed.format) {
    append_warning(out, "format mismatch: expected " + expected.format +
                           " got " + observed.format, dbg.strict);
  }
  const std::size_t exp_bytes = expected_byte_size(expected);
  if (exp_bytes > 0 && actual_bytes > 0 && exp_bytes != actual_bytes) {
    append_warning(out, "byte size mismatch: expected " + std::to_string(exp_bytes) +
                           " got " + std::to_string(actual_bytes), dbg.strict);
  }
}

debug::DebugOutput output_from_packet(const TapPacket& pkt,
                                      const OutputSpec& expected,
                                      const debug::DebugOptions& dbg) {
  debug::DebugOutput out;
  out.expected = expected;
  out.caps_string = pkt.caps_string;
  out.observed = spec_from_caps_string(pkt.caps_string);

  if (out.observed.is_unknown()) {
    if (pkt.format != TapFormat::Unknown) {
      out.observed.media_type = "video/x-raw";
      out.observed.format = pkt.video.format;
      out.observed.width = pkt.video.width;
      out.observed.height = pkt.video.height;
      out.observed.dtype = "UInt8";
      out.observed.certainty = SpecCertainty::Hint;
      out.observed.note = "tap packet";
    }
  }

  try {
    if (!out.observed.is_unknown() &&
        (is_video_media_type(out.observed) || is_tensor_media_type(out.observed))) {
      out.tensor = tensor_from_packet(pkt, out.observed);
      out.tensorizable = true;
    } else {
      out.bytes = pkt.bytes;
      out.tensorizable = false;
    }
  } catch (const std::exception& e) {
    append_warning(out, e.what(), dbg.strict);
    out.bytes = pkt.bytes;
    out.tensorizable = false;
  }

  const std::size_t actual_bytes = out.tensor.has_value()
      ? tensor_byte_size(*out.tensor)
      : out.bytes.size();
  validate_expected(out.expected, out.observed, actual_bytes, out, dbg);
  out.unknown = out.observed.is_unknown() || !out.tensorizable;
  return out;
}

debug::DebugOutput output_from_sample(GstSample* sample,
                                      const OutputSpec& expected,
                                      const debug::DebugOptions& dbg) {
  debug::DebugOutput out;
  out.expected = expected;

  GstCaps* caps = gst_sample_get_caps(sample);
  out.caps_string = pipeline_internal::gst_caps_to_string_safe(caps);
  out.observed = spec_from_caps(caps);

  bool tensor_ok = false;
  try {
    FrameTensorRef ref = pipeline_internal::sample_to_tensor_ref(sample);
    out.tensor = ref.to_copy();
    out.tensorizable = true;
    tensor_ok = true;
  } catch (const std::exception& e) {
    append_warning(out, e.what(), dbg.strict);
    out.tensorizable = false;
  }

  if (!tensor_ok) {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstMapInfo mi{};
    if (buffer && gst_buffer_map(buffer, &mi, GST_MAP_READ)) {
      out.bytes.resize(static_cast<std::size_t>(mi.size));
      if (mi.size) std::memcpy(out.bytes.data(), mi.data, mi.size);
      gst_buffer_unmap(buffer, &mi);
    } else {
      append_warning(out, "failed to map output buffer", dbg.strict);
    }
  }

  const std::size_t actual_bytes = out.tensor.has_value()
      ? tensor_byte_size(*out.tensor)
      : out.bytes.size();
  validate_expected(out.expected, out.observed, actual_bytes, out, dbg);
  out.unknown = out.observed.is_unknown() || !out.tensorizable;
  return out;
}

struct InputBufferPoolGuard {
#if SIMA_HAS_SIMAAI_POOL
  std::unique_ptr<GstBufferPool, decltype(&gst_simaai_free_buffer_pool)> pool{
      nullptr, gst_simaai_free_buffer_pool};
#else
  std::unique_ptr<GstBufferPool, void(*)(GstBufferPool*)> pool{nullptr, +[](GstBufferPool*) {}};
#endif
};

GstBuffer* allocate_input_buffer(std::size_t bytes,
                                 const sima::InputAppSrcOptions& opt,
                                 InputBufferPoolGuard& guard) {
#if SIMA_HAS_SIMAAI_POOL
  if (opt.use_simaai_pool) {
    GstBufferPool* pool = guard.pool.get();
    if (!pool) {
      gst_simaai_segment_memory_init_once();
      GstMemoryFlags flags = static_cast<GstMemoryFlags>(
          GST_SIMAAI_MEMORY_TARGET_EV74 | GST_SIMAAI_MEMORY_FLAG_CACHED);
      GstBufferPool* new_pool = gst_simaai_allocate_buffer_pool(
          /*allocator_user_data=*/nullptr,
          gst_simaai_memory_get_segment_allocator(),
          bytes,
          opt.pool_min_buffers,
          opt.pool_max_buffers,
          flags);
      if (new_pool) {
        guard.pool.reset(new_pool);
        pool = new_pool;
      }
    }

    if (pool) {
      GstBuffer* buf = nullptr;
      if (gst_buffer_pool_acquire_buffer(pool, &buf, nullptr) == GST_FLOW_OK && buf) {
        return buf;
      }
      return nullptr;
    }
    return nullptr;
  }
#else
  (void)opt;
  (void)guard;
#endif

  return gst_buffer_new_allocate(nullptr, bytes, nullptr);
}

void maybe_add_simaai_meta(GstBuffer* buffer,
                           const sima::InputAppSrcOptions& opt) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer || !opt.use_simaai_pool) return;
  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  if (!meta) return;
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) return;
  gint64 phys_addr =
      gst_simaai_segment_memory_get_phys_addr(gst_buffer_peek_memory(buffer, 0));
  gst_structure_set(s,
                    "buffer-id", G_TYPE_INT64, phys_addr,
                    "buffer-name", G_TYPE_STRING, "decoder",
                    "buffer-offset", G_TYPE_INT64, static_cast<gint64>(0),
                    "frame-id", G_TYPE_INT64, static_cast<gint64>(0),
                    "stream-id", G_TYPE_STRING, "0",
                    "timestamp", G_TYPE_UINT64, static_cast<guint64>(0),
                    nullptr);
#else
  (void)buffer;
  (void)opt;
#endif
}

void configure_appsrc(GstElement* appsrc, const sima::InputAppSrcOptions& opt) {
  if (!appsrc) return;
  g_object_set(G_OBJECT(appsrc),
               "is-live", opt.is_live ? TRUE : FALSE,
               "format", GST_FORMAT_TIME,
               "do-timestamp", opt.do_timestamp ? TRUE : FALSE,
               "block", opt.block ? TRUE : FALSE,
               "stream-type", opt.stream_type,
               "max-bytes", static_cast<guint64>(opt.max_bytes),
               nullptr);
}

OutputSpec spec_from_appsrc_options(const InputAppSrcOptions& opt) {
  OutputSpec out;
  out.media_type = opt.media_type.empty() ? "video/x-raw" : opt.media_type;
  out.format = upper_copy(opt.format);
  out.width = opt.width;
  out.height = opt.height;
  out.depth = opt.depth;
  out.certainty = SpecCertainty::Derived;
  out.note = "InputAppSrc options";
  if (out.media_type == "video/x-raw") {
    out.dtype = "UInt8";
  } else if (out.media_type == "application/vnd.simaai.tensor") {
    if (out.format == "FP32") out.dtype = "Float32";
  }
  out.byte_size = expected_byte_size(out);
  return out;
}

GstCaps* build_caps_from_spec(const OutputSpec& spec) {
  if (spec.media_type == "video/x-raw") {
    GstStructure* st = gst_structure_new_empty("video/x-raw");
    if (!spec.format.empty()) {
      gst_structure_set(st, "format", G_TYPE_STRING, spec.format.c_str(), nullptr);
    }
    if (spec.width > 0) {
      gst_structure_set(st, "width", G_TYPE_INT, spec.width, nullptr);
    }
    if (spec.height > 0) {
      gst_structure_set(st, "height", G_TYPE_INT, spec.height, nullptr);
    }
    return gst_caps_new_full(st, nullptr);
  }
  if (spec.media_type == "application/vnd.simaai.tensor") {
    GstStructure* st = gst_structure_new_empty("application/vnd.simaai.tensor");
    if (!spec.format.empty()) {
      gst_structure_set(st, "format", G_TYPE_STRING, spec.format.c_str(), nullptr);
    }
    if (spec.width > 0) {
      gst_structure_set(st, "width", G_TYPE_INT, spec.width, nullptr);
    }
    if (spec.height > 0) {
      gst_structure_set(st, "height", G_TYPE_INT, spec.height, nullptr);
    }
    if (spec.depth > 0) {
      gst_structure_set(st, "depth", G_TYPE_INT, spec.depth, nullptr);
    }
    return gst_caps_new_full(st, nullptr);
  }
  return nullptr;
}

std::vector<uint8_t> flatten_tensor(const FrameTensor& t) {
  std::vector<uint8_t> out;
  if (t.planes.empty()) return out;
  if (t.planes.size() == 1) {
    return t.planes[0];
  }
  for (const auto& p : t.planes) {
    out.insert(out.end(), p.begin(), p.end());
  }
  return out;
}

OutputSpec input_spec_from_debug(const debug::DebugOutput& input) {
  if (input.tensor.has_value()) {
    const auto& t = *input.tensor;
    OutputSpec out;
    out.width = t.width;
    out.height = t.height;
    out.format = t.format;
    out.dtype = (t.dtype == TensorDType::Float32) ? "Float32" : "UInt8";
    if (t.layout == TensorLayout::HWC) out.layout = "HWC";
    if (t.layout == TensorLayout::HW) out.layout = "HW";
    if (t.layout == TensorLayout::Planar) out.layout = "Planar";
    out.media_type = (t.format == "FP32" || t.format == "DETESS" || t.format == "DETESSDEQUANT")
        ? "application/vnd.simaai.tensor"
        : "video/x-raw";
    if (t.shape.size() == 3) out.depth = static_cast<int>(t.shape[2]);
    out.byte_size = tensor_byte_size(t);
    out.certainty = SpecCertainty::Derived;
    out.note = "input tensor";
    return out;
  }

  if (!input.observed.is_unknown()) return input.observed;
  if (!input.caps_string.empty()) return spec_from_caps_string(input.caps_string);
  return {};
}

void check_input_vs_expected(const OutputSpec& input,
                             const OutputSpec& expected,
                             const debug::DebugOptions& dbg) {
  if (!expected.media_type.empty() && !input.media_type.empty() &&
      expected.media_type != input.media_type) {
    if (dbg.strict) throw std::runtime_error("input media_type mismatch");
  }
  if (!expected.format.empty() && !input.format.empty() &&
      expected.format != input.format) {
    if (dbg.strict) throw std::runtime_error("input format mismatch");
  }
  if (expected.width > 0 && input.width > 0 && expected.width != input.width) {
    if (dbg.strict) throw std::runtime_error("input width mismatch");
  }
  if (expected.height > 0 && input.height > 0 && expected.height != input.height) {
    if (dbg.strict) throw std::runtime_error("input height mismatch");
  }
  if (expected.depth > 0 && input.depth > 0 && expected.depth != input.depth) {
    if (dbg.strict) throw std::runtime_error("input depth mismatch");
  }
}

struct PipelineHandle {
  PipelineHandle() = default;
  PipelineHandle(const PipelineHandle&) = delete;
  PipelineHandle& operator=(const PipelineHandle&) = delete;
  PipelineHandle(PipelineHandle&& other) noexcept {
    *this = std::move(other);
  }
  PipelineHandle& operator=(PipelineHandle&& other) noexcept {
    if (this == &other) return *this;
    if (appsrc) gst_object_unref(appsrc);
    if (appsink) gst_object_unref(appsink);
    if (pipeline) pipeline_internal::stop_and_unref(pipeline);
    pipeline = other.pipeline;
    appsrc = other.appsrc;
    appsink = other.appsink;
    guard = std::move(other.guard);
    other.pipeline = nullptr;
    other.appsrc = nullptr;
    other.appsink = nullptr;
    return *this;
  }

  GstElement* pipeline = nullptr;
  GstElement* appsrc = nullptr;
  GstElement* appsink = nullptr;
  std::shared_ptr<void> guard;

  ~PipelineHandle() {
    if (appsrc) gst_object_unref(appsrc);
    if (appsink) gst_object_unref(appsink);
    if (pipeline) pipeline_internal::stop_and_unref(pipeline);
  }
};

PipelineHandle launch_pipeline(const std::string& pipeline_str,
                               bool need_appsrc,
                               const char* where) {
  PipelineHandle h;

  sima::gst_init_once();

  debug_log(std::string(where) + " gst_is_initialized=" +
            (gst_is_initialized() ? "true" : "false"));
  debug_log(std::string(where) + " pipeline=" + pipeline_str);
  const char* plugin_path = std::getenv("GST_PLUGIN_PATH");
  const char* registry = std::getenv("GST_REGISTRY");
  const char* registry_1_0 = std::getenv("GST_REGISTRY_1_0");
  if (plugin_path) debug_log(std::string(where) + " GST_PLUGIN_PATH=" + plugin_path);
  if (registry) debug_log(std::string(where) + " GST_REGISTRY=" + registry);
  if (registry_1_0) debug_log(std::string(where) + " GST_REGISTRY_1_0=" + registry_1_0);

  std::string guard_err;
  h.guard = pipeline_internal::acquire_simaai_guard(where, pipeline_str,
                                                    /*force=*/false, &guard_err);
  if (!guard_err.empty()) {
    throw std::runtime_error(guard_err);
  }

  GError* err = nullptr;
  h.pipeline = gst_parse_launch(pipeline_str.c_str(), &err);
  if (!h.pipeline) {
    std::string msg = err ? err->message : "unknown";
    if (err) g_error_free(err);
    throw std::runtime_error(std::string(where) + ": gst_parse_launch failed: " + msg);
  }
  if (err) g_error_free(err);

  h.appsink = gst_bin_get_by_name(GST_BIN(h.pipeline), "mysink");
  if (!h.appsink) {
    pipeline_internal::stop_and_unref(h.pipeline);
    throw std::runtime_error(std::string(where) + ": appsink 'mysink' not found");
  }

  if (need_appsrc) {
    h.appsrc = gst_bin_get_by_name(GST_BIN(h.pipeline), "mysrc");
    if (!h.appsrc) {
      gst_object_unref(h.appsink);
      pipeline_internal::stop_and_unref(h.pipeline);
      throw std::runtime_error(std::string(where) + ": appsrc 'mysrc' not found");
    }
  }

  GstStateChangeReturn r = gst_element_set_state(h.pipeline, GST_STATE_PLAYING);
  if (r == GST_STATE_CHANGE_FAILURE) {
    throw std::runtime_error(std::string(where) + ": failed to set PLAYING");
  }
  GstState cur = GST_STATE_VOID_PENDING;
  GstState pend = GST_STATE_VOID_PENDING;
  gst_element_get_state(h.pipeline, &cur, &pend, 2 * GST_SECOND);

  return h;
}

debug::DebugOutput run_producer_once(const NodeGroup& group,
                                     const OutputSpec& expected,
                                     const debug::DebugOptions& dbg) {
  debug_log(std::string("producer_once expected=") + spec_brief(expected));
  PipelineSession p;
  p.add(group);
  if (dbg.force_system_memory && expected.media_type == "video/x-raw") {
    p.add(nodes::CapsRaw(expected.format,
                         expected.width,
                         expected.height,
                         expected.fps_num,
                         sima::CapsMemory::SystemMemory));
  }
  p.add(nodes::OutputAppSink());

  const std::string pipeline_str = p.to_gst(false);
  PipelineHandle h = launch_pipeline(pipeline_str, /*need_appsrc=*/false,
                                     "debug::producer");

  auto sample_opt = pipeline_internal::try_pull_sample_sliced(
      h.pipeline, h.appsink, dbg.timeout_ms, nullptr, "debug::producer");
  if (!sample_opt.has_value()) {
    throw std::runtime_error("debug::producer: timeout waiting for sample");
  }

  GstSample* sample = sample_opt.value();
  debug::DebugOutput out = output_from_sample(sample, expected, dbg);
  gst_sample_unref(sample);
  return out;
}

debug::DebugStream run_producer_stream(const NodeGroup& group,
                                       const OutputSpec& expected,
                                       const debug::DebugOptions& dbg) {
  debug_log(std::string("producer_stream expected=") + spec_brief(expected));
  PipelineSession p;
  p.add(group);
  if (dbg.force_system_memory && expected.media_type == "video/x-raw") {
    p.add(nodes::CapsRaw(expected.format,
                         expected.width,
                         expected.height,
                         expected.fps_num,
                         sima::CapsMemory::SystemMemory));
  }
  p.add(nodes::OutputAppSink());

  const std::string pipeline_str = p.to_gst(false);
  auto handle = std::make_shared<PipelineHandle>(
      launch_pipeline(pipeline_str, /*need_appsrc=*/false, "debug::producer_stream"));

  debug::DebugStream out;
  out.expected = expected;
  out.state = handle;
  out.next = [handle, expected, dbg](int timeout_ms) -> std::optional<debug::DebugOutput> {
    auto sample_opt = pipeline_internal::try_pull_sample_sliced(
        handle->pipeline, handle->appsink, timeout_ms, nullptr, "debug::producer_stream");
    if (!sample_opt.has_value()) return std::nullopt;
    GstSample* sample = sample_opt.value();
    debug::DebugOutput out = output_from_sample(sample, expected, dbg);
    gst_sample_unref(sample);
    return out;
  };
  out.close = [handle]() mutable {
    handle.reset();
  };
  return out;
}

debug::DebugOutput run_with_input_once(const NodeGroup& group,
                                       const InputAppSrcOptions& src_opt,
                                       const debug::DebugOutput& input,
                                       const OutputSpec& expected_out,
                                       const debug::DebugOptions& dbg,
                                       const char* where) {
  debug_log(std::string(where) + " expected_out=" + spec_brief(expected_out));
  PipelineSession p;
  p.add(nodes::InputAppSrc(src_opt));
  p.add(group);
  p.add(nodes::OutputAppSink());

  const std::string pipeline_str = p.to_gst(false);
  PipelineHandle h = launch_pipeline(pipeline_str, /*need_appsrc=*/true, where);

  OutputSpec expected_in = spec_from_appsrc_options(src_opt);
  OutputSpec input_spec = input_spec_from_debug(input);
  debug_log(std::string(where) + " expected_in=" + spec_brief(expected_in));
  debug_log(std::string(where) + " input_spec=" + spec_brief(input_spec));
  check_input_vs_expected(input_spec, expected_in, dbg);

  GstCaps* caps = build_caps_from_spec(expected_in);
  if (!caps) {
    throw std::runtime_error(std::string(where) + ": unsupported input caps");
  }
  gst_app_src_set_caps(GST_APP_SRC(h.appsrc), caps);
  gst_caps_unref(caps);
  configure_appsrc(h.appsrc, src_opt);

  std::vector<uint8_t> bytes;
  if (input.tensor.has_value()) {
    bytes = flatten_tensor(*input.tensor);
  } else if (!input.bytes.empty()) {
    bytes = input.bytes;
  }
  if (bytes.empty()) {
    throw std::runtime_error(std::string(where) + ": empty input buffer");
  }

  InputBufferPoolGuard pool_guard;
  GstBuffer* buf = allocate_input_buffer(bytes.size(), src_opt, pool_guard);
  if (!buf) throw std::runtime_error(std::string(where) + ": failed to allocate GstBuffer");

  GstMapInfo mi{};
  if (!gst_buffer_map(buf, &mi, GST_MAP_WRITE)) {
    gst_buffer_unref(buf);
    throw std::runtime_error(std::string(where) + ": failed to map GstBuffer");
  }
  std::memcpy(mi.data, bytes.data(), bytes.size());
  gst_buffer_unmap(buf, &mi);
  maybe_add_simaai_meta(buf, src_opt);

  if (gst_app_src_push_buffer(GST_APP_SRC(h.appsrc), buf) != GST_FLOW_OK) {
    gst_buffer_unref(buf);
    throw std::runtime_error(std::string(where) + ": appsrc push failed");
  }
  gst_app_src_end_of_stream(GST_APP_SRC(h.appsrc));

  auto sample_opt = pipeline_internal::try_pull_sample_sliced(
      h.pipeline, h.appsink, dbg.timeout_ms, nullptr, where);
  if (!sample_opt.has_value()) {
    throw std::runtime_error(std::string(where) + ": timeout waiting for output");
  }

  GstSample* sample = sample_opt.value();
  debug::DebugOutput out = output_from_sample(sample, expected_out, dbg);
  gst_sample_unref(sample);
  return out;
}

struct StreamBridge {
  debug::DebugStream input;
  InputAppSrcOptions src_opt;
  OutputSpec expected_out;
  debug::DebugOptions dbg;
  std::shared_ptr<PipelineHandle> handle;
  bool caps_set = false;

  std::optional<debug::DebugOutput> next(int timeout_ms) {
    if (!input.next) return std::nullopt;
    auto in = input.next(timeout_ms);
    if (!in.has_value()) return std::nullopt;

    OutputSpec expected_in = spec_from_appsrc_options(src_opt);
    OutputSpec input_spec = input_spec_from_debug(*in);
    check_input_vs_expected(input_spec, expected_in, dbg);

    if (!caps_set) {
      GstCaps* caps = build_caps_from_spec(expected_in);
      if (!caps) throw std::runtime_error("debug::stream: unsupported input caps");
      gst_app_src_set_caps(GST_APP_SRC(handle->appsrc), caps);
      gst_caps_unref(caps);
      configure_appsrc(handle->appsrc, src_opt);
      caps_set = true;
    }

    std::vector<uint8_t> bytes;
    if (in->tensor.has_value()) {
      bytes = flatten_tensor(*in->tensor);
    } else if (!in->bytes.empty()) {
      bytes = in->bytes;
    }
    if (bytes.empty()) {
      if (dbg.strict) throw std::runtime_error("debug::stream: empty input buffer");
      return std::nullopt;
    }

    InputBufferPoolGuard pool_guard;
    GstBuffer* buf = allocate_input_buffer(bytes.size(), src_opt, pool_guard);
    if (!buf) throw std::runtime_error("debug::stream: failed to allocate GstBuffer");

    GstMapInfo mi{};
    if (!gst_buffer_map(buf, &mi, GST_MAP_WRITE)) {
      gst_buffer_unref(buf);
      throw std::runtime_error("debug::stream: failed to map GstBuffer");
    }
    std::memcpy(mi.data, bytes.data(), bytes.size());
    gst_buffer_unmap(buf, &mi);
    maybe_add_simaai_meta(buf, src_opt);

    if (gst_app_src_push_buffer(GST_APP_SRC(handle->appsrc), buf) != GST_FLOW_OK) {
      gst_buffer_unref(buf);
      throw std::runtime_error("debug::stream: appsrc push failed");
    }

    auto sample_opt = pipeline_internal::try_pull_sample_sliced(
        handle->pipeline, handle->appsink, timeout_ms, nullptr, "debug::stream");
    if (!sample_opt.has_value()) return std::nullopt;

    GstSample* sample = sample_opt.value();
    debug::DebugOutput out = output_from_sample(sample, expected_out, dbg);
    gst_sample_unref(sample);
    return out;
  }
};

debug::DebugStream run_with_input_stream(const NodeGroup& group,
                                         const InputAppSrcOptions& src_opt,
                                         const debug::DebugStream& input,
                                         const OutputSpec& expected_out,
                                         const debug::DebugOptions& dbg,
                                         const char* where) {
  debug_log(std::string(where) + " expected_out=" + spec_brief(expected_out));
  PipelineSession p;
  p.add(nodes::InputAppSrc(src_opt));
  p.add(group);
  p.add(nodes::OutputAppSink());

  const std::string pipeline_str = p.to_gst(false);
  auto handle = std::make_shared<PipelineHandle>(
      launch_pipeline(pipeline_str, /*need_appsrc=*/true, where));

  auto bridge = std::make_shared<StreamBridge>();
  bridge->input = input;
  bridge->src_opt = src_opt;
  bridge->expected_out = expected_out;
  bridge->dbg = dbg;
  bridge->handle = handle;

  debug::DebugStream out;
  out.expected = expected_out;
  out.state = bridge;
  out.next = [bridge](int timeout_ms) {
    return bridge->next(timeout_ms);
  };
  out.close = [bridge]() mutable {
    bridge->handle.reset();
  };
  return out;
}

} // namespace

debug::DebugOutput ImageInputGroup(const ImageInputGroupOptions& opt,
                                   debug::OutputTag,
                                   const debug::DebugOptions& dbg) {
  return run_producer_once(ImageInputGroup(opt),
                           ImageInputGroupOutputSpec(opt),
                           dbg);
}

debug::DebugStream ImageInputGroup(const ImageInputGroupOptions& opt,
                                   debug::StreamTag,
                                   const debug::DebugOptions& dbg) {
  return run_producer_stream(ImageInputGroup(opt),
                             ImageInputGroupOutputSpec(opt),
                             dbg);
}

debug::DebugOutput RtspInputGroup(const RtspInputGroupOptions& opt,
                                  debug::OutputTag,
                                  const debug::DebugOptions& dbg) {
  return run_producer_once(RtspInputGroup(opt),
                           RtspInputGroupOutputSpec(opt),
                           dbg);
}

debug::DebugStream RtspInputGroup(const RtspInputGroupOptions& opt,
                                  debug::StreamTag,
                                  const debug::DebugOptions& dbg) {
  return run_producer_stream(RtspInputGroup(opt),
                             RtspInputGroupOutputSpec(opt),
                             dbg);
}

debug::DebugOutput VideoInputGroup(const VideoInputGroupOptions& opt,
                                   debug::OutputTag,
                                   const debug::DebugOptions& dbg) {
  return run_producer_once(VideoInputGroup(opt),
                           VideoInputGroupOutputSpec(opt),
                           dbg);
}

debug::DebugStream VideoInputGroup(const VideoInputGroupOptions& opt,
                                   debug::StreamTag,
                                   const debug::DebugOptions& dbg) {
  return run_producer_stream(VideoInputGroup(opt),
                             VideoInputGroupOutputSpec(opt),
                             dbg);
}

debug::DebugOutput Infer(const debug::DebugOutput& input,
                         const sima::mpk::ModelMPK& model,
                         const debug::DebugOptions& dbg) {
  const bool tensor_mode = input.observed.media_type == "application/vnd.simaai.tensor";
  const InputAppSrcOptions src_opt = model.input_appsrc_options(tensor_mode);
  return run_with_input_once(model.to_node_group(sima::mpk::ModelStage::Full),
                             src_opt, input, OutputSpec{}, dbg, "debug::Infer");
}

debug::DebugStream Infer(const debug::DebugStream& input,
                         const sima::mpk::ModelMPK& model,
                         const debug::DebugOptions& dbg) {
  const bool tensor_mode = input.expected.media_type == "application/vnd.simaai.tensor";
  const InputAppSrcOptions src_opt = model.input_appsrc_options(tensor_mode);
  return run_with_input_stream(model.to_node_group(sima::mpk::ModelStage::Full),
                               src_opt, input, OutputSpec{}, dbg, "debug::Infer(stream)");
}

debug::DebugOutput Preprocess(const debug::DebugOutput& input,
                              const sima::mpk::ModelMPK& model,
                              const debug::DebugOptions& dbg) {
  const bool tensor_mode = input.observed.media_type == "application/vnd.simaai.tensor";
  const InputAppSrcOptions src_opt = model.input_appsrc_options(tensor_mode);
  return run_with_input_once(model.to_node_group(sima::mpk::ModelStage::Preprocess),
                             src_opt, input, OutputSpec{}, dbg, "debug::Preprocess");
}

debug::DebugStream Preprocess(const debug::DebugStream& input,
                              const sima::mpk::ModelMPK& model,
                              const debug::DebugOptions& dbg) {
  const bool tensor_mode = input.expected.media_type == "application/vnd.simaai.tensor";
  const InputAppSrcOptions src_opt = model.input_appsrc_options(tensor_mode);
  return run_with_input_stream(model.to_node_group(sima::mpk::ModelStage::Preprocess),
                               src_opt, input, OutputSpec{}, dbg, "debug::Preprocess(stream)");
}

debug::DebugOutput MLA(const debug::DebugOutput& input,
                       const sima::mpk::ModelMPK& model,
                       const debug::DebugOptions& dbg) {
  const bool tensor_mode = input.observed.media_type == "application/vnd.simaai.tensor";
  const InputAppSrcOptions src_opt = model.input_appsrc_options(tensor_mode);
  return run_with_input_once(model.to_node_group(sima::mpk::ModelStage::MlaOnly),
                             src_opt, input, OutputSpec{}, dbg, "debug::MLA");
}

debug::DebugStream MLA(const debug::DebugStream& input,
                       const sima::mpk::ModelMPK& model,
                       const debug::DebugOptions& dbg) {
  const bool tensor_mode = input.expected.media_type == "application/vnd.simaai.tensor";
  const InputAppSrcOptions src_opt = model.input_appsrc_options(tensor_mode);
  return run_with_input_stream(model.to_node_group(sima::mpk::ModelStage::MlaOnly),
                               src_opt, input, OutputSpec{}, dbg, "debug::MLA(stream)");
}

debug::DebugOutput Postprocess(const debug::DebugOutput& input,
                               const sima::mpk::ModelMPK& model,
                               const debug::DebugOptions& dbg) {
  const bool tensor_mode = input.observed.media_type == "application/vnd.simaai.tensor";
  const InputAppSrcOptions src_opt = model.input_appsrc_options(tensor_mode);
  return run_with_input_once(model.to_node_group(sima::mpk::ModelStage::Postprocess),
                             src_opt, input, OutputSpec{}, dbg, "debug::Postprocess");
}

debug::DebugStream Postprocess(const debug::DebugStream& input,
                               const sima::mpk::ModelMPK& model,
                               const debug::DebugOptions& dbg) {
  const bool tensor_mode = input.expected.media_type == "application/vnd.simaai.tensor";
  const InputAppSrcOptions src_opt = model.input_appsrc_options(tensor_mode);
  return run_with_input_stream(model.to_node_group(sima::mpk::ModelStage::Postprocess),
                               src_opt, input, OutputSpec{}, dbg, "debug::Postprocess(stream)");
}

} // namespace sima::nodes::groups

namespace sima::nodes {
namespace {

std::string upper_copy_local(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

OutputSpec spec_from_appsrc_options_local(const InputAppSrcOptions& opt) {
  OutputSpec out;
  out.media_type = opt.media_type.empty() ? "video/x-raw" : opt.media_type;
  out.format = upper_copy_local(opt.format);
  out.width = opt.width;
  out.height = opt.height;
  out.depth = opt.depth;
  out.certainty = SpecCertainty::Derived;
  out.note = "InputAppSrc options";
  if (out.media_type == "video/x-raw") {
    out.dtype = "UInt8";
  } else if (out.media_type == "application/vnd.simaai.tensor") {
    if (out.format == "FP32") out.dtype = "Float32";
  }
  out.byte_size = expected_byte_size(out);
  return out;
}

std::vector<uint8_t> mat_to_bytes(const cv::Mat& mat) {
  if (mat.empty()) return {};
  const std::size_t bytes = mat.total() * mat.elemSize();
  std::vector<uint8_t> out(bytes);
  if (bytes == 0) return out;

  if (mat.isContinuous()) {
    std::memcpy(out.data(), mat.data, bytes);
    return out;
  }

  const std::size_t row_bytes = static_cast<std::size_t>(mat.cols) * mat.elemSize();
  for (int r = 0; r < mat.rows; ++r) {
    const uint8_t* row = mat.ptr<uint8_t>(r);
    std::memcpy(out.data() + r * row_bytes, row, row_bytes);
  }
  return out;
}

} // namespace

debug::DebugOutput InputAppSrc(const cv::Mat& input,
                               const sima::InputAppSrcOptions& opt,
                               const debug::DebugOptions& dbg) {
  if (input.empty()) {
    throw std::runtime_error("debug::InputAppSrc: input is empty");
  }

  debug::DebugOutput out;
  auto warn_or_throw = [&](const std::string& msg) {
    if (dbg.strict) throw std::runtime_error(msg);
    out.warnings.push_back(msg);
  };

  InputAppSrcOptions used = opt;
  used.media_type = used.media_type.empty() ? "video/x-raw" : used.media_type;

  const int in_w = input.cols;
  const int in_h = input.rows;
  const int in_c = input.channels();

  if (used.media_type == "video/x-raw") {
    if (used.format.empty()) {
      used.format = (in_c == 1) ? "GRAY8" : "BGR";
    }
    used.format = upper_copy_local(used.format);
    if (used.format == "GRAY") used.format = "GRAY8";

    if (used.format == "RGB" || used.format == "BGR") {
      if (input.type() != CV_8UC3) {
        throw std::runtime_error("debug::InputAppSrc: expected CV_8UC3 for RGB/BGR");
      }
      used.depth = 3;
    } else if (used.format == "GRAY8") {
      if (input.type() != CV_8UC1) {
        throw std::runtime_error("debug::InputAppSrc: expected CV_8UC1 for GRAY8");
      }
      used.depth = 1;
    } else {
      throw std::runtime_error("debug::InputAppSrc: unsupported video format " + used.format);
    }
  } else if (used.media_type == "application/vnd.simaai.tensor") {
    if (used.format.empty()) used.format = "FP32";
    used.format = upper_copy_local(used.format);
    if (used.format != "FP32") {
      throw std::runtime_error("debug::InputAppSrc: only FP32 tensor input supported");
    }
    if (input.type() != CV_32FC1 && input.type() != CV_32FC3) {
      throw std::runtime_error("debug::InputAppSrc: tensor input must be CV_32FC1 or CV_32FC3");
    }
    if (used.depth <= 0) used.depth = in_c;
  } else {
    throw std::runtime_error("debug::InputAppSrc: unsupported media_type " + used.media_type);
  }

  if (used.width > 0 && used.width != in_w) {
    warn_or_throw("debug::InputAppSrc: width mismatch, overriding to input width");
  }
  if (used.height > 0 && used.height != in_h) {
    warn_or_throw("debug::InputAppSrc: height mismatch, overriding to input height");
  }
  used.width = in_w;
  used.height = in_h;

  std::vector<uint8_t> bytes = mat_to_bytes(input);
  if (bytes.empty()) {
    throw std::runtime_error("debug::InputAppSrc: failed to extract input bytes");
  }

  FrameTensor tensor;
  tensor.width = used.width;
  tensor.height = used.height;
  tensor.format = used.format;

  if (used.media_type == "video/x-raw") {
    tensor.dtype = TensorDType::UInt8;
    if (used.format == "RGB" || used.format == "BGR") {
      tensor.layout = TensorLayout::HWC;
      tensor.shape = {tensor.height, tensor.width, used.depth};
      const int row_stride = tensor.width * used.depth;
      tensor.strides = {row_stride, used.depth, 1};
    } else {
      tensor.layout = TensorLayout::HW;
      tensor.shape = {tensor.height, tensor.width};
      tensor.strides = {tensor.width, 1};
    }
  } else {
    tensor.dtype = TensorDType::Float32;
    tensor.layout = TensorLayout::HWC;
    tensor.shape = {tensor.height, tensor.width, used.depth};
    const int elem = static_cast<int>(sizeof(float));
    const int row_stride = tensor.width * used.depth * elem;
    tensor.strides = {row_stride, used.depth * elem, elem};
  }

  tensor.planes.push_back(std::move(bytes));
  out.tensor = std::move(tensor);
  out.tensorizable = true;
  out.unknown = false;

  out.expected = spec_from_appsrc_options_local(used);
  out.expected.memory = "SystemMemory";
  out.observed = out.expected;
  if (opt.use_simaai_pool) {
    warn_or_throw("debug::InputAppSrc: use_simaai_pool not supported; using SystemMemory");
  }

  return out;
}

} // namespace sima::nodes
