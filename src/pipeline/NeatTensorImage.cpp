#include "pipeline/NeatTensorCore.h"

#include <cstring>
#include <stdexcept>

namespace sima {
namespace {

int pick_width_from_planes(const NeatTensor& t) {
  const NeatPlane* y = t.try_plane(NeatPlaneRole::Y);
  if (y && y->shape.size() >= 2) {
    return static_cast<int>(y->shape[1]);
  }
  return -1;
}

int pick_height_from_planes(const NeatTensor& t) {
  const NeatPlane* y = t.try_plane(NeatPlaneRole::Y);
  if (y && y->shape.size() >= 1) {
    return static_cast<int>(y->shape[0]);
  }
  return -1;
}

} // namespace

int NeatTensor::width() const {
  if (shape.size() >= 2) return static_cast<int>(shape[1]);
  if (shape.size() == 1) return static_cast<int>(shape[0]);
  return pick_width_from_planes(*this);
}

int NeatTensor::height() const {
  if (shape.size() >= 1) return static_cast<int>(shape[0]);
  return pick_height_from_planes(*this);
}

int NeatTensor::channels() const {
  if (layout == TensorLayout::HWC && shape.size() >= 3) {
    return static_cast<int>(shape[2]);
  }
  if (semantic.image.has_value()) {
    switch (semantic.image->format) {
      case NeatImageSpec::PixelFormat::RGB: return 3;
      case NeatImageSpec::PixelFormat::BGR: return 3;
      case NeatImageSpec::PixelFormat::GRAY8: return 1;
      case NeatImageSpec::PixelFormat::NV12:
      case NeatImageSpec::PixelFormat::I420:
      case NeatImageSpec::PixelFormat::UNKNOWN:
        break;
    }
  }
  return -1;
}

std::optional<NeatImageSpec::PixelFormat> NeatTensor::image_format() const {
  if (!semantic.image.has_value()) return std::nullopt;
  return semantic.image->format;
}

bool NeatTensor::is_nv12() const {
  return semantic.image.has_value() &&
         semantic.image->format == NeatImageSpec::PixelFormat::NV12;
}

bool NeatTensor::is_i420() const {
  return semantic.image.has_value() &&
         semantic.image->format == NeatImageSpec::PixelFormat::I420;
}

std::optional<NeatNv12Mapped> NeatTensor::map_nv12_read() const {
  if (!is_nv12()) return std::nullopt;
  if (dtype != TensorDType::UInt8) {
    throw std::runtime_error("map_nv12_read: NV12 requires UInt8 dtype");
  }

  const NeatPlane* y_plane = try_plane(NeatPlaneRole::Y);
  const NeatPlane* uv_plane = try_plane(NeatPlaneRole::UV);
  if (!y_plane || !uv_plane) {
    throw std::runtime_error("map_nv12_read: missing Y/UV planes");
  }

  int w = width();
  int h = height();
  if (w <= 0) w = pick_width_from_planes(*this);
  if (h <= 0) h = pick_height_from_planes(*this);
  if (w <= 0 || h <= 0) {
    throw std::runtime_error("map_nv12_read: invalid dimensions");
  }
  if ((w % 2) != 0 || (h % 2) != 0) {
    throw std::runtime_error("map_nv12_read: NV12 requires even dimensions");
  }

  if (y_plane->shape.size() >= 2) {
    if (y_plane->shape[0] != h || y_plane->shape[1] != w) {
      throw std::runtime_error("map_nv12_read: Y plane shape mismatch");
    }
  }
  if (uv_plane->shape.size() >= 2) {
    if (uv_plane->shape[0] != h / 2 || uv_plane->shape[1] != w) {
      throw std::runtime_error("map_nv12_read: UV plane shape mismatch");
    }
  }

  const int64_t y_stride = !y_plane->strides_bytes.empty()
      ? y_plane->strides_bytes[0]
      : static_cast<int64_t>(w);
  const int64_t uv_stride = !uv_plane->strides_bytes.empty()
      ? uv_plane->strides_bytes[0]
      : static_cast<int64_t>(w);
  if (y_stride < w || uv_stride < w) {
    throw std::runtime_error("map_nv12_read: invalid plane stride");
  }

  NeatMapping mapping = map_read();
  if (!mapping.data) {
    throw std::runtime_error("map_nv12_read: mapping failed");
  }

  const std::size_t total = mapping.size_bytes;
  const std::size_t y_end = static_cast<std::size_t>(y_plane->byte_offset) +
      static_cast<std::size_t>(y_stride) * static_cast<std::size_t>(h - 1) +
      static_cast<std::size_t>(w);
  const std::size_t uv_end = static_cast<std::size_t>(uv_plane->byte_offset) +
      static_cast<std::size_t>(uv_stride) * static_cast<std::size_t>(h / 2 - 1) +
      static_cast<std::size_t>(w);
  if (total > 0 && (y_end > total || uv_end > total)) {
    throw std::runtime_error("map_nv12_read: plane exceeds buffer bounds");
  }

  const uint8_t* base = static_cast<const uint8_t*>(mapping.data);
  NeatNv12Mapped out;
  out.mapping = std::move(mapping);
  out.view.width = w;
  out.view.height = h;
  out.view.y = base + y_plane->byte_offset;
  out.view.y_stride = y_stride;
  out.view.uv = base + uv_plane->byte_offset;
  out.view.uv_stride = uv_stride;
  return out;
}

std::size_t NeatTensor::nv12_required_bytes() const {
  if (!is_nv12()) return 0;
  const int w = width();
  const int h = height();
  if (w <= 0 || h <= 0) return 0;
  return static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3 / 2;
}

bool NeatTensor::copy_nv12_contiguous_to(uint8_t* dst, std::size_t dst_size) const {
  if (!dst) return false;
  const std::size_t required = nv12_required_bytes();
  if (required == 0 || dst_size < required) return false;
  auto mapped = map_nv12_read();
  if (!mapped.has_value()) return false;

  const NeatNv12View& view = mapped->view;
  const int w = view.width;
  const int h = view.height;
  const int uv_h = h / 2;

  uint8_t* dst_y = dst;
  for (int y = 0; y < h; ++y) {
    std::memcpy(dst_y + static_cast<std::size_t>(y) * w,
                view.y + static_cast<std::size_t>(y) * view.y_stride,
                static_cast<std::size_t>(w));
  }

  uint8_t* dst_uv = dst + static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  for (int y = 0; y < uv_h; ++y) {
    std::memcpy(dst_uv + static_cast<std::size_t>(y) * w,
                view.uv + static_cast<std::size_t>(y) * view.uv_stride,
                static_cast<std::size_t>(w));
  }

  return true;
}

std::vector<uint8_t> NeatTensor::copy_nv12_contiguous() const {
  const std::size_t required = nv12_required_bytes();
  if (required == 0) {
    throw std::runtime_error("copy_nv12_contiguous: not an NV12 tensor");
  }
  std::vector<uint8_t> out(required);
  if (!copy_nv12_contiguous_to(out.data(), out.size())) {
    throw std::runtime_error("copy_nv12_contiguous: copy failed");
  }
  return out;
}

} // namespace sima
