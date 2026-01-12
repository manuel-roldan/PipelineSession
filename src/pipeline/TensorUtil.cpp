#include "pipeline/internal/TensorUtil.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace sima::pipeline_internal {
namespace {

struct MappedSample {
  GstSample* sample = nullptr;
  GstVideoInfo info{};
  GstVideoFrame frame{};
  bool mapped = false;

  ~MappedSample() {
    if (mapped) {
      gst_video_frame_unmap(&frame);
    }
    if (sample) {
      gst_sample_unref(sample);
    }
  }
};

static int64_t ts_or_minus1(GstClockTime t) {
  return GST_CLOCK_TIME_IS_VALID(t) ? static_cast<int64_t>(t) : -1;
}

static bool parse_keyframe(GstBuffer* buf) {
  if (!buf) return false;
  return !GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_DELTA_UNIT);
}

static std::string caps_to_string(GstCaps* caps) {
  if (!caps) return "";
  gchar* s = gst_caps_to_string(caps);
  if (!s) return "";
  std::string out(s);
  g_free(s);
  return out;
}

} // namespace

FrameTensorRef sample_to_tensor_ref(GstSample* sample) {
  if (!sample) {
    throw std::runtime_error("sample_to_tensor_ref: null sample");
  }

  GstCaps* caps = gst_sample_get_caps(sample);
  if (!caps) {
    throw std::runtime_error("sample_to_tensor_ref: missing caps");
  }

  GstVideoInfo info;
  std::memset(&info, 0, sizeof(info));
  if (!gst_video_info_from_caps(&info, caps)) {
    throw std::runtime_error("sample_to_tensor_ref: gst_video_info_from_caps failed");
  }

  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    throw std::runtime_error("sample_to_tensor_ref: missing buffer");
  }

  auto holder = std::make_shared<MappedSample>();
  // Holder preserves the GstVideoFrame mapping so bindings can be zero-copy later.
  holder->sample = gst_sample_ref(sample);
  holder->info = info;

  if (!gst_video_frame_map(&holder->frame, &holder->info, buffer, GST_MAP_READ)) {
    throw std::runtime_error("sample_to_tensor_ref: gst_video_frame_map failed");
  }
  holder->mapped = true;

  FrameTensorRef out;
  out.dtype = TensorDType::UInt8;
  out.width = GST_VIDEO_INFO_WIDTH(&info);
  out.height = GST_VIDEO_INFO_HEIGHT(&info);
  out.caps_string = caps_to_string(caps);
  out.pts_ns = ts_or_minus1(GST_BUFFER_PTS(buffer));
  out.dts_ns = ts_or_minus1(GST_BUFFER_DTS(buffer));
  out.duration_ns = ts_or_minus1(GST_BUFFER_DURATION(buffer));
  out.keyframe = parse_keyframe(buffer);
  out.holder = holder;

  const GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT(&info);

  if (fmt == GST_VIDEO_FORMAT_RGB || fmt == GST_VIDEO_FORMAT_BGR) {
    out.format = (fmt == GST_VIDEO_FORMAT_RGB) ? "RGB" : "BGR";
    out.layout = TensorLayout::HWC;

    TensorPlaneRef p;
    p.data = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&holder->frame, 0));
    p.stride = GST_VIDEO_FRAME_PLANE_STRIDE(&holder->frame, 0);
    p.width = out.width;
    p.height = out.height;
    out.planes.push_back(p);

    out.shape = {out.height, out.width, 3};
    out.strides = {p.stride, 3, 1};
    return out;
  }

  if (fmt == GST_VIDEO_FORMAT_GRAY8) {
    out.format = "GRAY8";
    out.layout = TensorLayout::HW;

    TensorPlaneRef p;
    p.data = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&holder->frame, 0));
    p.stride = GST_VIDEO_FRAME_PLANE_STRIDE(&holder->frame, 0);
    p.width = out.width;
    p.height = out.height;
    out.planes.push_back(p);

    out.shape = {out.height, out.width};
    out.strides = {p.stride, 1};
    return out;
  }

  if (fmt == GST_VIDEO_FORMAT_NV12) {
    out.format = "NV12";
    out.layout = TensorLayout::Planar;

    TensorPlaneRef y;
    y.data = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&holder->frame, 0));
    y.stride = GST_VIDEO_FRAME_PLANE_STRIDE(&holder->frame, 0);
    y.width = out.width;
    y.height = out.height;

    TensorPlaneRef uv;
    uv.data = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&holder->frame, 1));
    uv.stride = GST_VIDEO_FRAME_PLANE_STRIDE(&holder->frame, 1);
    uv.width = out.width;
    uv.height = out.height / 2;

    out.planes.push_back(y);
    out.planes.push_back(uv);
    out.shape = {out.height, out.width};
    out.strides = {y.stride, 1};
    return out;
  }

  if (fmt == GST_VIDEO_FORMAT_I420) {
    out.format = "I420";
    out.layout = TensorLayout::Planar;

    TensorPlaneRef y;
    y.data = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&holder->frame, 0));
    y.stride = GST_VIDEO_FRAME_PLANE_STRIDE(&holder->frame, 0);
    y.width = out.width;
    y.height = out.height;

    TensorPlaneRef u;
    u.data = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&holder->frame, 1));
    u.stride = GST_VIDEO_FRAME_PLANE_STRIDE(&holder->frame, 1);
    u.width = out.width / 2;
    u.height = out.height / 2;

    TensorPlaneRef v;
    v.data = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&holder->frame, 2));
    v.stride = GST_VIDEO_FRAME_PLANE_STRIDE(&holder->frame, 2);
    v.width = out.width / 2;
    v.height = out.height / 2;

    out.planes.push_back(y);
    out.planes.push_back(u);
    out.planes.push_back(v);
    out.shape = {out.height, out.width};
    out.strides = {y.stride, 1};
    return out;
  }

  throw std::runtime_error("sample_to_tensor_ref: unsupported video format");
}

} // namespace sima::pipeline_internal
