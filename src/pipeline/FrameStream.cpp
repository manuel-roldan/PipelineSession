// src/pipeline/FrameStream.cpp
#include "pipeline/FrameStream.h"

#include "pipeline/Errors.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"


#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <cstring>
#include <string>
#include <utility>

namespace sima {

namespace {

using sima::pipeline_internal::diag_as_ctx;

static int64_t buffer_pts_ns(GstSample* sample) {
  if (!sample) return -1;
  GstBuffer* buf = gst_sample_get_buffer(sample);
  if (!buf) return -1;

  const GstClockTime pts = GST_BUFFER_PTS(buf);
  if (pts == GST_CLOCK_TIME_NONE) return -1;
  return static_cast<int64_t>(pts);
}

static int64_t buffer_dts_ns(GstSample* sample) {
  if (!sample) return -1;
  GstBuffer* buf = gst_sample_get_buffer(sample);
  if (!buf) return -1;

  const GstClockTime dts = GST_BUFFER_DTS(buf);
  if (dts == GST_CLOCK_TIME_NONE) return -1;
  return static_cast<int64_t>(dts);
}

static int64_t buffer_duration_ns(GstSample* sample) {
  if (!sample) return -1;
  GstBuffer* buf = gst_sample_get_buffer(sample);
  if (!buf) return -1;

  const GstClockTime dur = GST_BUFFER_DURATION(buf);
  if (dur == GST_CLOCK_TIME_NONE) return -1;
  return static_cast<int64_t>(dur);
}

static bool buffer_is_keyframe(GstSample* sample) {
  if (!sample) return false;
  GstBuffer* buf = gst_sample_get_buffer(sample);
  if (!buf) return false;

  // DELTA_UNIT set => not a keyframe (typically)
  return !GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_DELTA_UNIT);
}

} // namespace

FrameStream::FrameStream(GstElement* pipeline, GstElement* appsink)
    : pipeline_(pipeline), appsink_(appsink) {}

FrameStream::FrameStream(FrameStream&& other) noexcept {
  *this = std::move(other);
}

FrameStream& FrameStream::operator=(FrameStream&& other) noexcept {
  if (this == &other) return *this;

  close();

  pipeline_ = other.pipeline_;
  appsink_ = other.appsink_;
  debug_pipeline_ = std::move(other.debug_pipeline_);
  diag_ = std::move(other.diag_);

  other.pipeline_ = nullptr;
  other.appsink_ = nullptr;

  return *this;
}

FrameStream::~FrameStream() {
  close();
}

void FrameStream::close() {
  auto diag = diag_as_ctx(diag_);

  // Best-effort: drain bus before teardown (useful for capturing late errors).
  if (pipeline_ && diag) {
    pipeline_internal::drain_bus(pipeline_, diag, "FrameStream::close(pre)");
  }

  if (appsink_) {
    pipeline_internal::stop_and_unref(appsink_);
    appsink_ = nullptr;
  }

  if (pipeline_) {
    pipeline_internal::stop_and_unref(pipeline_);
    pipeline_ = nullptr;
  }

  diag_.reset();
  debug_pipeline_.clear();
}

std::optional<FrameNV12Ref> FrameStream::next(int timeout_ms) {
  if (!appsink_) return std::nullopt;

  auto diag = diag_as_ctx(diag_);

  // Catch pipeline errors early.
  if (pipeline_ && diag) {
    pipeline_internal::drain_bus(pipeline_, diag, "FrameStream::next(pre)");
    pipeline_internal::throw_if_bus_error(pipeline_, diag, "FrameStream::next(pre)");
  }

  auto sample_opt =
      pipeline_internal::try_pull_sample_sliced(pipeline_, appsink_, timeout_ms, diag, "FrameStream::next");

  if (!sample_opt.has_value()) {
    // Timeout or EOS. Drain bus once more to surface any errors.
    if (pipeline_ && diag) {
      pipeline_internal::drain_bus(pipeline_, diag, "FrameStream::next(post-null)");
      pipeline_internal::throw_if_bus_error(pipeline_, diag, "FrameStream::next(post-null)");
    }
    return std::nullopt;
  }

  // Keep the sample + mapped GstVideoFrame alive for zero-copy pointers.
  GstSample* sample = *sample_opt;
  auto holder = std::make_shared<pipeline_internal::SampleHolder>(sample);

  std::string map_err;
  if (!pipeline_internal::map_video_frame_read(*holder, map_err)) {
    PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
    rep.repro_note = "FrameStream: failed to map GstVideoFrame: " + map_err;
    throw PipelineError(rep.repro_note, std::move(rep));
  }

  const GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT(&holder->vinfo);
  if (fmt != GST_VIDEO_FORMAT_NV12) {
    PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
    rep.repro_note = "FrameStream: expected NV12 at appsink, got " +
                     std::string(gst_video_format_to_string(fmt));
    throw PipelineError(rep.repro_note, std::move(rep));
  }

  FrameNV12Ref out;
  out.width = static_cast<int>(GST_VIDEO_INFO_WIDTH(&holder->vinfo));
  out.height = static_cast<int>(GST_VIDEO_INFO_HEIGHT(&holder->vinfo));

  out.pts_ns = buffer_pts_ns(holder->sample);
  out.dts_ns = buffer_dts_ns(holder->sample);
  out.duration_ns = buffer_duration_ns(holder->sample);
  out.keyframe = buffer_is_keyframe(holder->sample);

  out.y = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&holder->frame, 0));
  out.uv = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&holder->frame, 1));

  out.y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&holder->frame, 0);
  out.uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&holder->frame, 1);

  GstCaps* caps = gst_sample_get_caps(holder->sample);
  out.caps_string = pipeline_internal::gst_caps_to_string_safe(caps);

  if (!out.y || !out.uv || out.width <= 0 || out.height <= 0) {
    throw PipelineError("FrameStream: invalid NV12 plane pointers or dimensions");
  }

  out.holder = holder;

  // Post-drain to capture warnings/errors produced during pull.
  if (pipeline_ && diag) {
    pipeline_internal::drain_bus(pipeline_, diag, "FrameStream::next(post)");
    pipeline_internal::throw_if_bus_error(pipeline_, diag, "FrameStream::next(post)");
  }

  return out;
}

std::optional<FrameNV12> FrameStream::next_copy(int timeout_ms) {
  auto ref = next(timeout_ms);
  if (!ref.has_value()) return std::nullopt;

  FrameNV12 out;
  out.width = ref->width;
  out.height = ref->height;
  out.pts_ns = ref->pts_ns;
  out.dts_ns = ref->dts_ns;
  out.duration_ns = ref->duration_ns;
  out.keyframe = ref->keyframe;

  const int w = out.width;
  const int h = out.height;

  const size_t y_bytes = static_cast<size_t>(w) * static_cast<size_t>(h);
  const size_t uv_bytes = static_cast<size_t>(w) * static_cast<size_t>(h / 2);

  out.nv12.resize(y_bytes + uv_bytes);

  // Copy Y plane row-by-row to handle stride.
  uint8_t* y_dst = out.nv12.data();
  for (int row = 0; row < h; ++row) {
    const uint8_t* y_src = ref->y + static_cast<size_t>(row) * static_cast<size_t>(ref->y_stride);
    std::memcpy(y_dst + static_cast<size_t>(row) * static_cast<size_t>(w), y_src, static_cast<size_t>(w));
  }

  // Copy UV plane row-by-row to handle stride.
  uint8_t* uv_dst = out.nv12.data() + y_bytes;
  for (int row = 0; row < h / 2; ++row) {
    const uint8_t* uv_src = ref->uv + static_cast<size_t>(row) * static_cast<size_t>(ref->uv_stride);
    std::memcpy(uv_dst + static_cast<size_t>(row) * static_cast<size_t>(w), uv_src, static_cast<size_t>(w));
  }

  return out;
}

PipelineReport FrameStream::report_snapshot(bool /*heavy*/) const {
  auto diag = diag_as_ctx(diag_);
  if (!diag) return PipelineReport{};
  return diag->snapshot_basic();
}

} // namespace sima
