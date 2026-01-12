// src/pipeline/TapStream.cpp
#include "pipeline/TapStream.h"

#include "pipeline/Errors.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <glib.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sima {
namespace {

using sima::pipeline_internal::DiagCtx;
using sima::pipeline_internal::diag_as_ctx;

static TapFormat tap_format_from_caps(GstCaps* caps, TapVideoInfo* out_vi) {
  if (!caps || gst_caps_is_any(caps) || gst_caps_is_empty(caps)) return TapFormat::Unknown;

  const GstStructure* st = gst_caps_get_structure(caps, 0);
  if (!st) return TapFormat::Unknown;

  const char* name = gst_structure_get_name(st);
  if (!name) return TapFormat::Unknown;

  // Fill some best-effort video info (width/height/framerate/format string).
  if (out_vi) {
    int w = 0, h = 0;
    (void)gst_structure_get_int(st, "width", &w);
    (void)gst_structure_get_int(st, "height", &h);
    out_vi->width = w;
    out_vi->height = h;

    int fps_n = 0, fps_d = 1;
    if (gst_structure_get_fraction(st, "framerate", &fps_n, &fps_d)) {
      out_vi->fps_num = fps_n;
      out_vi->fps_den = (fps_d == 0) ? 1 : fps_d;
    }

    const char* fmt = gst_structure_get_string(st, "format");
    out_vi->format = fmt ? fmt : "";
  }

  // Encoded video
  if (std::strcmp(name, "video/x-h264") == 0) return TapFormat::H264;
  if (std::strcmp(name, "video/x-h265") == 0) return TapFormat::H265;

  // Images
  if (std::strcmp(name, "image/jpeg") == 0) return TapFormat::JPEG;
  if (std::strcmp(name, "image/png") == 0) return TapFormat::PNG;

  // Raw video
  if (std::strcmp(name, "video/x-raw") == 0) {
    const char* fmt = gst_structure_get_string(st, "format");
    if (!fmt) return TapFormat::Unknown;
    if (std::strcmp(fmt, "NV12") == 0) return TapFormat::NV12;
    if (std::strcmp(fmt, "I420") == 0) return TapFormat::I420;
    if (std::strcmp(fmt, "RGB") == 0) return TapFormat::RGB;
    if (std::strcmp(fmt, "BGR") == 0) return TapFormat::BGR;
    if (std::strcmp(fmt, "GRAY8") == 0) return TapFormat::GRAY8;
    return TapFormat::Unknown;
  }

  return TapFormat::Unknown;
}

static bool parse_keyframe(GstBuffer* buf) {
  if (!buf) return false;
  // If DELTA_UNIT is set, it's not a keyframe (best-effort heuristic).
  return !GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_DELTA_UNIT);
}

static int64_t ts_or_minus1(GstClockTime t) {
  return GST_CLOCK_TIME_IS_VALID(t) ? static_cast<int64_t>(t) : -1;
}

static size_t env_size_t(const char* key, size_t def_val) {
  const char* v = std::getenv(key);
  if (!v || !*v) return def_val;
  char* end = nullptr;
  unsigned long long x = std::strtoull(v, &end, 10);
  if (end == v) return def_val;
  return static_cast<size_t>(x);
}

static void pack_raw_video_tight(GstSample* sample, std::vector<uint8_t>* out_bytes) {
  // NOTE: This function throws on failure; caller decides strict vs non-strict behavior.
  GstCaps* caps = gst_sample_get_caps(sample);
  if (!caps) throw std::runtime_error("No caps on sample");

  GstVideoInfo info;
  std::memset(&info, 0, sizeof(info));
  if (!gst_video_info_from_caps(&info, caps)) {
    throw std::runtime_error("gst_video_info_from_caps failed");
  }

  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) throw std::runtime_error("No buffer on sample");

  GstVideoFrame frame;
  std::memset(&frame, 0, sizeof(frame));
  if (!gst_video_frame_map(&frame, &info, buffer, GST_MAP_READ)) {
    throw std::runtime_error("gst_video_frame_map failed (non-mappable memory?)");
  }

  auto unmap = [&]() { gst_video_frame_unmap(&frame); };
  try {
    const int w = GST_VIDEO_INFO_WIDTH(&info);
    const int h = GST_VIDEO_INFO_HEIGHT(&info);

    // Compute tight size + copy row-by-row to eliminate stride.
    const GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT(&info);

    // A safety cap against accidental huge allocations from bogus caps.
    const size_t max_bytes = env_size_t("SIMA_TAP_MAX_BYTES", 64ull * 1024ull * 1024ull);

    auto reserve_checked = [&](size_t n) {
      if (n > max_bytes) {
        std::ostringstream ss;
        ss << "Tap payload too large (" << n << " bytes > cap " << max_bytes << ")";
        throw std::runtime_error(ss.str());
      }
      out_bytes->clear();
      out_bytes->resize(n);
    };

    uint8_t* dst = nullptr;

    if (fmt == GST_VIDEO_FORMAT_NV12) {
      const size_t y_sz = static_cast<size_t>(w) * static_cast<size_t>(h);
      const size_t uv_sz = static_cast<size_t>(w) * static_cast<size_t>(h / 2);
      reserve_checked(y_sz + uv_sz);
      dst = out_bytes->data();

      const uint8_t* y = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
      const uint8_t* uv = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 1));
      const int y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
      const int uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 1);

      for (int r = 0; r < h; ++r) {
        std::memcpy(dst + static_cast<size_t>(r) * w, y + static_cast<size_t>(r) * y_stride, w);
      }
      const size_t uv_off = y_sz;
      for (int r = 0; r < h / 2; ++r) {
        std::memcpy(dst + uv_off + static_cast<size_t>(r) * w,
                    uv + static_cast<size_t>(r) * uv_stride,
                    w);
      }
      return;
    }

    if (fmt == GST_VIDEO_FORMAT_I420) {
      const size_t y_sz = static_cast<size_t>(w) * static_cast<size_t>(h);
      const size_t u_sz = static_cast<size_t>(w / 2) * static_cast<size_t>(h / 2);
      const size_t v_sz = u_sz;
      reserve_checked(y_sz + u_sz + v_sz);
      dst = out_bytes->data();

      const uint8_t* y = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
      const uint8_t* u = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 1));
      const uint8_t* v = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 2));
      const int y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
      const int u_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 1);
      const int v_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 2);

      for (int r = 0; r < h; ++r) {
        std::memcpy(dst + static_cast<size_t>(r) * w, y + static_cast<size_t>(r) * y_stride, w);
      }

      size_t off = y_sz;
      const int cw = w / 2;
      const int ch = h / 2;

      for (int r = 0; r < ch; ++r) {
        std::memcpy(dst + off + static_cast<size_t>(r) * cw,
                    u + static_cast<size_t>(r) * u_stride,
                    cw);
      }

      off += u_sz;
      for (int r = 0; r < ch; ++r) {
        std::memcpy(dst + off + static_cast<size_t>(r) * cw,
                    v + static_cast<size_t>(r) * v_stride,
                    cw);
      }
      return;
    }

    if (fmt == GST_VIDEO_FORMAT_RGB || fmt == GST_VIDEO_FORMAT_BGR) {
      const size_t row = static_cast<size_t>(w) * 3;
      const size_t sz = row * static_cast<size_t>(h);
      reserve_checked(sz);
      dst = out_bytes->data();

      const uint8_t* p = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
      const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);

      for (int r = 0; r < h; ++r) {
        std::memcpy(dst + static_cast<size_t>(r) * row,
                    p + static_cast<size_t>(r) * stride,
                    row);
      }
      return;
    }

    if (fmt == GST_VIDEO_FORMAT_GRAY8) {
      const size_t row = static_cast<size_t>(w);
      const size_t sz = row * static_cast<size_t>(h);
      reserve_checked(sz);
      dst = out_bytes->data();

      const uint8_t* p = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
      const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);

      for (int r = 0; r < h; ++r) {
        std::memcpy(dst + static_cast<size_t>(r) * row,
                    p + static_cast<size_t>(r) * stride,
                    row);
      }
      return;
    }

    throw std::runtime_error("Unsupported raw video format for tight packing");
  } catch (...) {
    unmap();
    throw;
  }

  unmap();
}

static void map_buffer_copy(GstBuffer* buffer, std::vector<uint8_t>* out) {
  out->clear();
  if (!buffer) return;

  const size_t max_bytes = env_size_t("SIMA_TAP_MAX_BYTES", 64ull * 1024ull * 1024ull);

  GstMapInfo mi;
  std::memset(&mi, 0, sizeof(mi));
  if (!gst_buffer_map(buffer, &mi, GST_MAP_READ)) {
    throw std::runtime_error("gst_buffer_map failed (non-mappable memory?)");
  }

  const size_t n = static_cast<size_t>(mi.size);
  if (n > max_bytes) {
    gst_buffer_unmap(buffer, &mi);
    std::ostringstream ss;
    ss << "Tap payload too large (" << n << " bytes > cap " << max_bytes << ")";
    throw std::runtime_error(ss.str());
  }

  out->resize(n);
  if (n) std::memcpy(out->data(), mi.data, n);
  gst_buffer_unmap(buffer, &mi);
}

} // namespace

TapStream::TapStream(GstElement* pipeline, GstElement* appsink)
  : pipeline_(pipeline),
    appsink_(appsink) {
  if (pipeline_) gst_object_ref(pipeline_);
  if (appsink_) gst_object_ref(appsink_);
}

TapStream::~TapStream() { close(); }

TapStream::TapStream(TapStream&& o) noexcept {
  *this = std::move(o);
}

TapStream& TapStream::operator=(TapStream&& o) noexcept {
  if (this == &o) return *this;

  close();

  pipeline_ = o.pipeline_;
  appsink_ = o.appsink_;
  debug_pipeline_ = std::move(o.debug_pipeline_);
  diag_ = std::move(o.diag_);
  tap_node_index_ = o.tap_node_index_;
  tap_sink_name_ = std::move(o.tap_sink_name_);

  o.pipeline_ = nullptr;
  o.appsink_ = nullptr;
  o.tap_node_index_ = -1;
  o.tap_sink_name_.clear();

  return *this;
}

void TapStream::close() {
  auto diag = diag_as_ctx(diag_);

  // Best-effort drain before teardown (keeps the last bus messages).
  pipeline_internal::drain_bus(pipeline_, diag);

  pipeline_internal::stop_and_unref(appsink_);
  pipeline_internal::stop_and_unref(pipeline_);
}

std::optional<TapPacket> TapStream::next(int timeout_ms) {
  if (!pipeline_ || !appsink_) return std::nullopt;

  auto diag = diag_as_ctx(diag_);

  // Fast fail if pipeline already reported an error.
  pipeline_internal::throw_if_bus_error(pipeline_, diag, "TapStream::next(pre)");

  auto sample_opt =
    pipeline_internal::try_pull_sample_sliced(pipeline_, appsink_, timeout_ms, diag, "TapStream::next");

  if (!sample_opt.has_value()) {
    // Match FrameStream behavior: optionally treat timeout as "no data".
    const bool timeout_returns_null =
      pipeline_internal::env_bool("SIMA_GST_TIMEOUT_RETURNS_NULL", true);

    if (timeout_returns_null) {
      return std::nullopt;
    }

    pipeline_internal::maybe_dump_dot(pipeline_, "tap_timeout");

    PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
    rep.pipeline_string = rep.pipeline_string.empty() ? debug_pipeline_ : rep.pipeline_string;
    rep.repro_note = "TapStream::next timeout with SIMA_GST_TIMEOUT_RETURNS_NULL=0";
    if (diag) rep.repro_note += "\n" + pipeline_internal::boundary_summary(diag);

    throw PipelineError(rep.repro_note, std::move(rep));
  }

  GstSample* sample = sample_opt.value();
  // Ensure unref no matter what.
  struct SampleUnref {
    GstSample* s;
    ~SampleUnref() { if (s) gst_sample_unref(s); }
  } sample_guard{sample};

  TapPacket pkt;
  pkt.source_node_index = tap_node_index_;
  pkt.source_element = !tap_sink_name_.empty()
                         ? tap_sink_name_
                         : (GST_IS_OBJECT(appsink_) ? GST_OBJECT_NAME(appsink_) : "");

  GstCaps* caps = gst_sample_get_caps(sample);
  pkt.caps_string = pipeline_internal::gst_caps_to_string_safe(caps);
  pkt.memory_features = pipeline_internal::caps_features_string(caps);

  GstBuffer* buf = gst_sample_get_buffer(sample);
  pkt.keyframe = parse_keyframe(buf);
  if (buf) {
    pkt.pts_ns = ts_or_minus1(GST_BUFFER_PTS(buf));
    pkt.dts_ns = ts_or_minus1(GST_BUFFER_DTS(buf));
    pkt.duration_ns = ts_or_minus1(GST_BUFFER_DURATION(buf));
  }

  // Decide format + populate video info fields.
  pkt.format = tap_format_from_caps(caps, &pkt.video);

  const bool strict_mappable = pipeline_internal::env_bool("SIMA_TAP_STRICT_MAPPABLE", false);

  try {
    // Raw video: pack tightly to remove stride.
    if (pkt.format == TapFormat::NV12 || pkt.format == TapFormat::I420 ||
        pkt.format == TapFormat::RGB  || pkt.format == TapFormat::BGR  ||
        pkt.format == TapFormat::GRAY8) {
      pack_raw_video_tight(sample, &pkt.bytes);
      pkt.memory_mappable = true;
      pkt.non_mappable_reason.clear();
      return pkt;
    }

    // Everything else: copy full buffer bytes (best-effort).
    map_buffer_copy(buf, &pkt.bytes);
    pkt.memory_mappable = true;
    pkt.non_mappable_reason.clear();
    return pkt;
  } catch (const std::exception& e) {
    pkt.memory_mappable = false;
    pkt.bytes.clear();
    pkt.non_mappable_reason = e.what();

    if (!strict_mappable) {
      // Non-strict mode: return metadata even if payload is non-mappable.
      return pkt;
    }

    // Strict mode: escalate with a structured report.
    pipeline_internal::maybe_dump_dot(pipeline_, "tap_non_mappable");

    PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
    rep.pipeline_string = rep.pipeline_string.empty() ? debug_pipeline_ : rep.pipeline_string;

    std::ostringstream ss;
    ss << "TapStream: non-mappable tap payload (strict mode)\n"
       << "caps=" << pkt.caps_string << "\n"
       << "features=" << pkt.memory_features << "\n"
       << "reason=" << pkt.non_mappable_reason << "\n";
    if (diag) ss << "\n" << pipeline_internal::boundary_summary(diag);

    rep.repro_note = ss.str();
    throw PipelineError(rep.repro_note, std::move(rep));
  }
}

PipelineReport TapStream::report_snapshot(bool heavy) const {
  auto diag = diag_as_ctx(diag_);
  PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
  if (rep.pipeline_string.empty()) rep.pipeline_string = debug_pipeline_;

  if (heavy && pipeline_) {
    pipeline_internal::maybe_dump_dot(pipeline_, "tap_snapshot");
    // Keep this cheap: boundary summary is the most useful “what is stuck?” signal.
    if (diag) rep.repro_note = pipeline_internal::boundary_summary(diag);
  }

  return rep;
}

} // namespace sima
