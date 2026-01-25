#include "pipeline/NeatTensorOpenCV.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace sima {
namespace {

int expected_channels(NeatImageSpec::PixelFormat fmt) {
  switch (fmt) {
    case NeatImageSpec::PixelFormat::GRAY8:
      return 1;
    case NeatImageSpec::PixelFormat::RGB:
    case NeatImageSpec::PixelFormat::BGR:
      return 3;
    default:
      break;
  }
  return -1;
}

TensorLayout expected_layout(NeatImageSpec::PixelFormat fmt) {
  switch (fmt) {
    case NeatImageSpec::PixelFormat::GRAY8:
      return TensorLayout::HW;
    case NeatImageSpec::PixelFormat::RGB:
    case NeatImageSpec::PixelFormat::BGR:
      return TensorLayout::HWC;
    default:
      break;
  }
  return TensorLayout::Unknown;
}

int cv_type_for_format(NeatImageSpec::PixelFormat fmt) {
  switch (fmt) {
    case NeatImageSpec::PixelFormat::GRAY8:
      return CV_8UC1;
    case NeatImageSpec::PixelFormat::RGB:
    case NeatImageSpec::PixelFormat::BGR:
      return CV_8UC3;
    default:
      break;
  }
  return -1;
}

void validate_cv_mat(const cv::Mat& mat, NeatImageSpec::PixelFormat fmt) {
  if (mat.empty() || mat.data == nullptr) {
    throw std::runtime_error("from_cv_mat: empty cv::Mat");
  }
  const int channels = expected_channels(fmt);
  if (channels <= 0) {
    throw std::runtime_error("from_cv_mat: unsupported pixel format");
  }
  if (mat.depth() != CV_8U) {
    throw std::runtime_error("from_cv_mat: only CV_8U is supported");
  }
  if (mat.channels() != channels) {
    throw std::runtime_error("from_cv_mat: channel count mismatch");
  }
}

} // namespace

NeatTensor NeatTensor::from_cv_mat(const cv::Mat& mat,
                                   NeatImageSpec::PixelFormat fmt,
                                   bool read_only) {
  return sima::from_cv_mat(mat, fmt, read_only);
}

NeatTensor from_cv_mat(const cv::Mat& mat,
                       NeatImageSpec::PixelFormat fmt,
                       bool read_only) {
  validate_cv_mat(mat, fmt);

  auto holder = std::make_shared<cv::Mat>(mat);
  const std::size_t bytes = holder->step * static_cast<std::size_t>(holder->rows);

  NeatTensor out;
  out.storage = make_cpu_external_storage(holder->data, bytes, holder, read_only);
  out.dtype = TensorDType::UInt8;
  out.device = {NeatDeviceType::CPU, 0};
  out.byte_offset = 0;
  out.read_only = read_only;

  const int rows = holder->rows;
  const int cols = holder->cols;
  const int channels = holder->channels();
  if (fmt == NeatImageSpec::PixelFormat::GRAY8) {
    out.shape = {rows, cols};
    out.strides_bytes = {static_cast<int64_t>(holder->step), 1};
    out.layout = TensorLayout::HW;
  } else {
    out.shape = {rows, cols, channels};
    out.strides_bytes = {static_cast<int64_t>(holder->step),
                         static_cast<int64_t>(holder->elemSize()),
                         static_cast<int64_t>(holder->elemSize1())};
    out.layout = TensorLayout::HWC;
  }

  NeatImageSpec image;
  image.format = fmt;
  out.semantic.image = image;
  return out;
}

std::optional<NeatCvMatView> NeatTensor::map_cv_mat_view(
    NeatImageSpec::PixelFormat desired) const {
  if (!semantic.image.has_value()) return std::nullopt;
  if (semantic.image->format != desired) return std::nullopt;
  if (!is_dense()) return std::nullopt;
  if (dtype != TensorDType::UInt8) return std::nullopt;

  const TensorLayout want_layout = expected_layout(desired);
  if (layout != want_layout) return std::nullopt;
  const int channels = expected_channels(desired);
  if (channels <= 0) return std::nullopt;

  const int h = height();
  const int w = width();
  if (h <= 0 || w <= 0) return std::nullopt;
  if (desired != NeatImageSpec::PixelFormat::GRAY8) {
    if (shape.size() < 3 || shape[2] != channels) return std::nullopt;
  } else {
    if (shape.size() < 2) return std::nullopt;
  }

  NeatMapping mapping = map_read();
  if (!mapping.data) return std::nullopt;

  std::size_t step = 0;
  if (!strides_bytes.empty()) {
    step = static_cast<std::size_t>(strides_bytes[0]);
  } else {
    step = static_cast<std::size_t>(w * channels);
  }

  const int cv_type = cv_type_for_format(desired);
  if (cv_type <= 0) return std::nullopt;

  cv::Mat mat(h, w, cv_type, mapping.data, step);
  NeatCvMatView view;
  view.mapping = std::move(mapping);
  view.mat = mat;
  return view;
}

cv::Mat NeatTensor::to_cv_mat_copy(NeatImageSpec::PixelFormat desired) const {
  if (desired != NeatImageSpec::PixelFormat::BGR &&
      desired != NeatImageSpec::PixelFormat::RGB &&
      desired != NeatImageSpec::PixelFormat::GRAY8) {
    throw std::runtime_error("to_cv_mat_copy: unsupported desired format");
  }

  auto view = map_cv_mat_view(desired);
  if (view.has_value()) {
    return view->mat.clone();
  }

  if (desired == NeatImageSpec::PixelFormat::BGR &&
      (is_nv12() || is_i420())) {
    const int w = width();
    const int h = height();
    if (w <= 0 || h <= 0) {
      throw std::runtime_error("to_cv_mat_copy: invalid NV12/I420 dimensions");
    }
    std::vector<uint8_t> yuv = is_nv12()
        ? copy_nv12_contiguous()
        : copy_i420_contiguous();
    cv::Mat yuv_mat(h + h / 2, w, CV_8UC1, yuv.data());
    cv::Mat bgr;
    const int code = is_nv12()
        ? cv::COLOR_YUV2BGR_NV12
        : cv::COLOR_YUV2BGR_I420;
    cv::cvtColor(yuv_mat, bgr, code);
    return bgr;
  }

  NeatTensor cpu_tensor = cpu().contiguous();
  auto cpu_view = cpu_tensor.map_cv_mat_view(desired);
  if (!cpu_view.has_value()) {
    throw std::runtime_error("to_cv_mat_copy: tensor layout mismatch");
  }
  return cpu_view->mat.clone();
}

} // namespace sima
#endif
