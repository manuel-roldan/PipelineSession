// src/pipeline/internal/CapsBridge.cpp
#include "pipeline/internal/CapsBridge.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <cstring>
#include <cctype>
#include <sstream>
#include <string>

namespace sima::pipeline_internal {
namespace {

NeatImageSpec::PixelFormat pixel_format_from_gst(GstVideoFormat fmt) {
  switch (fmt) {
    case GST_VIDEO_FORMAT_RGB: return NeatImageSpec::PixelFormat::RGB;
    case GST_VIDEO_FORMAT_BGR: return NeatImageSpec::PixelFormat::BGR;
    case GST_VIDEO_FORMAT_GRAY8: return NeatImageSpec::PixelFormat::GRAY8;
    case GST_VIDEO_FORMAT_NV12: return NeatImageSpec::PixelFormat::NV12;
    case GST_VIDEO_FORMAT_I420: return NeatImageSpec::PixelFormat::I420;
    default: return NeatImageSpec::PixelFormat::UNKNOWN;
  }
}

TensorDType dtype_from_tensor_format(const std::string& fmt) {
  std::string s;
  s.reserve(fmt.size());
  for (char c : fmt) {
    s.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  if (s == "DETESS") return TensorDType::UInt16;
  if (s == "DETESSDEQUANT" || s == "FP32") return TensorDType::Float32;
  if (s == "EVXX_INT8" || s == "EV74_INT8" || s == "INT8") return TensorDType::Int8;
  if (s == "EVXX_BFLOAT16" || s == "BF16" || s == "BFLOAT16") return TensorDType::BFloat16;
  if (s == "UINT8") return TensorDType::UInt8;
  return TensorDType::UInt8;
}

const char* dtype_name(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::UInt8: return "UInt8";
    case TensorDType::Int8: return "Int8";
    case TensorDType::UInt16: return "UInt16";
    case TensorDType::Int16: return "Int16";
    case TensorDType::Int32: return "Int32";
    case TensorDType::BFloat16: return "BFloat16";
    case TensorDType::Float32: return "Float32";
    case TensorDType::Float64: return "Float64";
  }
  return "Unknown";
}

const char* image_format_name(NeatImageSpec::PixelFormat fmt) {
  switch (fmt) {
    case NeatImageSpec::PixelFormat::RGB: return "RGB";
    case NeatImageSpec::PixelFormat::BGR: return "BGR";
    case NeatImageSpec::PixelFormat::GRAY8: return "GRAY8";
    case NeatImageSpec::PixelFormat::NV12: return "NV12";
    case NeatImageSpec::PixelFormat::I420: return "I420";
    case NeatImageSpec::PixelFormat::UNKNOWN: return "UNKNOWN";
  }
  return "UNKNOWN";
}

} // namespace

NeatTensorConstraint neat_constraint_from_caps(GstCaps* caps) {
  NeatTensorConstraint out;
  if (!caps) return out;

  const GstStructure* st = gst_caps_get_structure(caps, 0);
  const char* media = st ? gst_structure_get_name(st) : nullptr;
  if (!media) return out;

  if (std::string(media) == "application/vnd.simaai.tensor") {
    int w = 0;
    int h = 0;
    int d = 0;
    gst_structure_get_int(st, "width", &w);
    gst_structure_get_int(st, "height", &h);
    gst_structure_get_int(st, "depth", &d);

    const char* fmt = gst_structure_get_string(st, "format");
    const std::string fmt_str = fmt ? fmt : "";
    out.dtypes.push_back(dtype_from_tensor_format(fmt_str));

    if (w > 0 && h > 0 && d > 0) {
      out.rank = 3;
      out.shape = {h, w, d};
    } else if (w > 0 && h > 0) {
      out.rank = 2;
      out.shape = {h, w};
    } else if (w > 0) {
      out.rank = 1;
      out.shape = {w};
    }
    return out;
  }

  if (std::string(media).rfind("video/x-raw", 0) == 0) {
    GstVideoInfo info;
    std::memset(&info, 0, sizeof(info));
    if (!gst_video_info_from_caps(&info, caps)) {
      return out;
    }

    const int w = GST_VIDEO_INFO_WIDTH(&info);
    const int h = GST_VIDEO_INFO_HEIGHT(&info);
    const GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT(&info);
    const NeatImageSpec::PixelFormat pixel = pixel_format_from_gst(fmt);
    out.dtypes.push_back(TensorDType::UInt8);
    if (pixel != NeatImageSpec::PixelFormat::UNKNOWN) {
      out.image_format = pixel;
    }

    if (pixel == NeatImageSpec::PixelFormat::NV12 ||
        pixel == NeatImageSpec::PixelFormat::I420) {
      out.rank = 2;
      out.shape = {h, w};
      out.allow_composite = true;
      return out;
    }

    if (pixel == NeatImageSpec::PixelFormat::GRAY8) {
      out.rank = 3;
      out.shape = {h, w, 1};
      out.allow_composite = false;
      return out;
    }

    if (pixel == NeatImageSpec::PixelFormat::RGB ||
        pixel == NeatImageSpec::PixelFormat::BGR) {
      out.rank = 3;
      out.shape = {h, w, 3};
      out.allow_composite = false;
      return out;
    }

    out.rank = (w > 0 && h > 0) ? 2 : -1;
    if (w > 0 && h > 0) out.shape = {h, w};
    return out;
  }

  return out;
}

std::string neat_constraint_debug_string(const NeatTensorConstraint& constraint) {
  std::ostringstream ss;
  ss << "{rank=" << constraint.rank;
  if (!constraint.shape.empty()) {
    ss << " shape=";
    for (size_t i = 0; i < constraint.shape.size(); ++i) {
      if (i) ss << "x";
      ss << constraint.shape[i];
    }
  }
  if (!constraint.dtypes.empty()) {
    ss << " dtypes=[";
    for (size_t i = 0; i < constraint.dtypes.size(); ++i) {
      if (i) ss << ",";
      ss << dtype_name(constraint.dtypes[i]);
    }
    ss << "]";
  }
  if (constraint.image_format.has_value()) {
    ss << " image=" << image_format_name(*constraint.image_format);
  }
  ss << " composite=" << (constraint.allow_composite ? "true" : "false");
  ss << "}";
  return ss.str();
}

} // namespace sima::pipeline_internal
