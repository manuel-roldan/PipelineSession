#include "pipeline/NeatTensorAdapters.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <cctype>
#include <cstring>
#include <stdexcept>
#include <mutex>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

namespace sima {
namespace {

std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

std::size_t bytes_per_element(TensorDType dtype) {
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

TensorDType dtype_from_cv_depth(int depth) {
  switch (depth) {
    case CV_8U: return TensorDType::UInt8;
    case CV_8S: return TensorDType::Int8;
    case CV_16U: return TensorDType::UInt16;
    case CV_16S: return TensorDType::Int16;
    case CV_32S: return TensorDType::Int32;
    case CV_32F: return TensorDType::Float32;
    case CV_64F: return TensorDType::Float64;
    default:
      throw std::runtime_error("from_cv_mat: unsupported cv::Mat depth");
  }
}

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

NeatPlane make_plane(NeatPlaneRole role, int64_t h, int64_t w,
                     int64_t stride_bytes, int64_t offset_bytes) {
  NeatPlane plane;
  plane.role = role;
  plane.shape = {h, w};
  plane.strides_bytes = {stride_bytes, 1};
  plane.byte_offset = offset_bytes;
  return plane;
}

TensorDType dtype_from_tensor_format(const std::string& fmt) {
  const std::string s = upper_copy(fmt);
  if (s == "DETESS") return TensorDType::UInt16;
  if (s == "DETESSDEQUANT" || s == "FP32") return TensorDType::Float32;
  if (s == "EVXX_INT8" || s == "EV74_INT8" || s == "INT8") return TensorDType::Int8;
  if (s == "EVXX_BFLOAT16" || s == "BF16" || s == "BFLOAT16") return TensorDType::BFloat16;
  if (s == "UINT8") return TensorDType::UInt8;
  return TensorDType::UInt8;
}

bool looks_tessellated(const std::string& fmt) {
  const std::string s = upper_copy(fmt);
  return s.find("TESS") != std::string::npos || s.find("EVXX") != std::string::npos ||
         s.find("EV74") != std::string::npos;
}

std::shared_ptr<NeatStorage> make_gst_sample_storage(GstSample* sample) {
  if (!sample) return {};
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) return {};

  auto holder = std::shared_ptr<void>(gst_sample_ref(sample),
                                      [](void* p) { gst_sample_unref(static_cast<GstSample*>(p)); });
  struct GstMapState {
    GstMapInfo info{};
    bool mapped = false;
    std::mutex mu;
  };
  auto map_state = std::make_shared<GstMapState>();

  auto storage = std::make_shared<NeatStorage>();
  storage->kind = NeatStorageKind::GstSample;
  storage->device = {NeatDeviceType::CPU, 0};
  storage->size_bytes = gst_buffer_get_size(buffer);
  storage->holder = holder;
  storage->data = nullptr;
  storage->map_fn = [holder, map_state](NeatMapMode mode) {
    GstSample* s = static_cast<GstSample*>(holder.get());
    GstBuffer* b = s ? gst_sample_get_buffer(s) : nullptr;
    if (!b) return NeatMapping{};

    std::lock_guard<std::mutex> lock(map_state->mu);
    if (map_state->mapped) {
      return NeatMapping{};
    }
    GstMapFlags flags = GST_MAP_READ;
    if (mode == NeatMapMode::Write) {
      flags = GST_MAP_WRITE;
    } else if (mode == NeatMapMode::ReadWrite) {
      flags = static_cast<GstMapFlags>(GST_MAP_READ | GST_MAP_WRITE);
    }
    if (!gst_buffer_map(b, &map_state->info, flags)) {
      return NeatMapping{};
    }
    map_state->mapped = true;
    GstBuffer* buffer_ref = gst_buffer_ref(b);

    NeatMapping mapping;
    mapping.data = map_state->info.data;
    mapping.size_bytes = map_state->info.size;
    mapping.unmap = [buffer_ref, map_state]() {
      std::lock_guard<std::mutex> lock(map_state->mu);
      if (map_state->mapped) {
        gst_buffer_unmap(buffer_ref, &map_state->info);
        map_state->mapped = false;
      }
      gst_buffer_unref(buffer_ref);
    };
    return mapping;
  };
  return storage;
}

NeatTensor from_gst_tensor_sample(GstSample* sample,
                                  const GstStructure* st,
                                  GstBuffer* buffer) {
  NeatTensor out;
  out.storage = make_gst_sample_storage(sample);
  if (!out.storage) {
    throw std::runtime_error("from_gst_sample: missing tensor storage");
  }

  int w = 0;
  int h = 0;
  int d = 0;
  gst_structure_get_int(st, "width", &w);
  gst_structure_get_int(st, "height", &h);
  gst_structure_get_int(st, "depth", &d);

  const char* fmt = gst_structure_get_string(st, "format");
  const std::string fmt_str = fmt ? fmt : "";
  out.dtype = dtype_from_tensor_format(fmt_str);
  out.device = {NeatDeviceType::CPU, 0};
  out.read_only = true;
  out.layout = TensorLayout::Unknown;

  const std::size_t elem = bytes_per_element(out.dtype);
  if (w > 0 && h > 0 && d > 0) {
    out.shape = {h, w, d};
    out.strides_bytes = {static_cast<int64_t>(w * d * elem),
                         static_cast<int64_t>(d * elem),
                         static_cast<int64_t>(elem)};
  } else if (w > 0 && h > 0) {
    out.shape = {h, w};
    out.strides_bytes = {static_cast<int64_t>(w * elem),
                         static_cast<int64_t>(elem)};
  } else if (w > 0) {
    out.shape = {w};
    out.strides_bytes = {static_cast<int64_t>(elem)};
  } else {
    const std::size_t bytes = buffer ? gst_buffer_get_size(buffer) : 0;
    const std::size_t elems = (elem > 0) ? (bytes / elem) : 0;
    if (elems > 0) {
      out.shape = {static_cast<int64_t>(elems)};
      out.strides_bytes = {static_cast<int64_t>(elem)};
    }
  }

  if (!fmt_str.empty()) {
    NeatTessSpec tess;
    tess.format = fmt_str;
    out.semantic.tess = tess;
  }
  return out;
}

} // namespace

#if defined(SIMA_WITH_OPENCV)
NeatTensor from_cv_mat(const cv::Mat& mat, const std::optional<NeatImageSpec>& image) {
  if (mat.empty() || mat.data == nullptr) {
    throw std::runtime_error("from_cv_mat: empty cv::Mat");
  }

  auto holder = std::make_shared<cv::Mat>(mat);
  const std::size_t bytes = holder->step * static_cast<std::size_t>(holder->rows);

  NeatTensor out;
  out.storage = make_cpu_external_storage(holder->data, bytes, holder, true);
  out.dtype = dtype_from_cv_depth(holder->depth());
  out.device = {NeatDeviceType::CPU, 0};
  out.byte_offset = 0;
  out.read_only = true;

  const int channels = holder->channels();
  out.shape = {holder->rows, holder->cols, channels > 0 ? channels : 1};
  out.strides_bytes = {static_cast<int64_t>(holder->step),
                       static_cast<int64_t>(holder->elemSize()),
                       static_cast<int64_t>(holder->elemSize1())};
  out.layout = TensorLayout::HWC;

  if (image.has_value()) {
    out.semantic.image = image;
  }
  return out;
}
#endif

NeatTensor from_gst_sample(GstSample* sample) {
  if (!sample) {
    throw std::runtime_error("from_gst_sample: null sample");
  }

  GstCaps* caps = gst_sample_get_caps(sample);
  if (!caps) {
    throw std::runtime_error("from_gst_sample: missing caps");
  }

  const GstStructure* st = gst_caps_get_structure(caps, 0);
  const char* media = st ? gst_structure_get_name(st) : nullptr;
  if (media && std::string(media) == "application/vnd.simaai.tensor") {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
      throw std::runtime_error("from_gst_sample: missing buffer");
    }
    return from_gst_tensor_sample(sample, st, buffer);
  }

  GstVideoInfo info;
  std::memset(&info, 0, sizeof(info));
  if (!gst_video_info_from_caps(&info, caps)) {
    throw std::runtime_error("from_gst_sample: gst_video_info_from_caps failed");
  }

  auto storage = make_gst_sample_storage(sample);
  if (!storage) {
    throw std::runtime_error("from_gst_sample: missing buffer");
  }

  const GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT(&info);
  NeatImageSpec::PixelFormat pixel = pixel_format_from_gst(fmt);
  if (pixel == NeatImageSpec::PixelFormat::UNKNOWN) {
    throw std::runtime_error("from_gst_sample: unsupported pixel format");
  }

  NeatTensor out;
  out.storage = std::move(storage);
  out.dtype = TensorDType::UInt8;
  out.device = {NeatDeviceType::CPU, 0};
  out.read_only = true;

  const int w = GST_VIDEO_INFO_WIDTH(&info);
  const int h = GST_VIDEO_INFO_HEIGHT(&info);
  out.shape = {h, w};
  out.strides_bytes = {GST_VIDEO_INFO_PLANE_STRIDE(&info, 0), 1};
  out.layout = TensorLayout::HW;

  NeatImageSpec image;
  image.format = pixel;
  out.semantic.image = image;

  if (fmt == GST_VIDEO_FORMAT_NV12) {
    const int64_t y_stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);
    const int64_t uv_stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 1);
    const int64_t y_offset = GST_VIDEO_INFO_PLANE_OFFSET(&info, 0);
    const int64_t uv_offset = GST_VIDEO_INFO_PLANE_OFFSET(&info, 1);
    out.planes.push_back(make_plane(NeatPlaneRole::Y, h, w, y_stride, y_offset));
    out.planes.push_back(make_plane(NeatPlaneRole::UV, h / 2, w, uv_stride, uv_offset));
    return out;
  }

  if (fmt == GST_VIDEO_FORMAT_I420) {
    const int64_t y_stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);
    const int64_t u_stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 1);
    const int64_t v_stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 2);
    const int64_t y_offset = GST_VIDEO_INFO_PLANE_OFFSET(&info, 0);
    const int64_t u_offset = GST_VIDEO_INFO_PLANE_OFFSET(&info, 1);
    const int64_t v_offset = GST_VIDEO_INFO_PLANE_OFFSET(&info, 2);
    out.planes.push_back(make_plane(NeatPlaneRole::Y, h, w, y_stride, y_offset));
    out.planes.push_back(make_plane(NeatPlaneRole::U, h / 2, w / 2, u_stride, u_offset));
    out.planes.push_back(make_plane(NeatPlaneRole::V, h / 2, w / 2, v_stride, v_offset));
    return out;
  }

  if (fmt == GST_VIDEO_FORMAT_RGB || fmt == GST_VIDEO_FORMAT_BGR ||
      fmt == GST_VIDEO_FORMAT_GRAY8) {
    const int channels = (fmt == GST_VIDEO_FORMAT_GRAY8) ? 1 : 3;
    out.shape = {h, w, channels};
    const int64_t elem = 1;
    const int64_t row_stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);
    out.strides_bytes = {row_stride,
                         static_cast<int64_t>(channels) * elem,
                         elem};
    out.layout = TensorLayout::HWC;
    return out;
  }

  throw std::runtime_error("from_gst_sample: unsupported pixel format");
}

} // namespace sima
