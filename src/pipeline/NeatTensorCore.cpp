#include "pipeline/NeatTensorCore.h"

#include "pipeline/internal/NeatTensorTransfer.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace sima {
namespace {

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

const char* device_name(NeatDeviceType type) {
  switch (type) {
    case NeatDeviceType::CPU: return "CPU";
    case NeatDeviceType::SIMA_APU: return "SIMA_APU";
    case NeatDeviceType::SIMA_CVU: return "SIMA_CVU";
    case NeatDeviceType::SIMA_MLA: return "SIMA_MLA";
    case NeatDeviceType::UNKNOWN: return "UNKNOWN";
  }
  return "UNKNOWN";
}

const char* layout_name(TensorLayout layout) {
  switch (layout) {
    case TensorLayout::Unknown: return "Unknown";
    case TensorLayout::HWC: return "HWC";
    case TensorLayout::CHW: return "CHW";
    case TensorLayout::HW: return "HW";
    case TensorLayout::Planar: return "Planar";
  }
  return "Unknown";
}

std::size_t dtype_bytes(TensorDType dtype) {
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
  return 0;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape,
                                        std::size_t elem_bytes) {
  if (shape.empty()) return {};
  std::vector<int64_t> strides(shape.size(), 0);
  int64_t stride = static_cast<int64_t>(elem_bytes);
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return strides;
}

std::size_t dense_size_bytes(const std::vector<int64_t>& shape,
                             std::size_t elem_bytes) {
  if (shape.empty()) return 0;
  std::size_t total = elem_bytes;
  for (auto dim : shape) {
    if (dim <= 0) return 0;
    total *= static_cast<std::size_t>(dim);
  }
  return total;
}

bool copy_dense_strided(const uint8_t* src,
                        std::size_t src_bytes,
                        const std::vector<int64_t>& shape,
                        const std::vector<int64_t>& src_strides,
                        std::size_t elem_bytes,
                        uint8_t* dst) {
  if (!src || !dst) return false;
  if (shape.empty()) return true;
  if (src_strides.size() != shape.size()) return false;

  std::size_t max_offset = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    const int64_t dim = shape[i];
    const int64_t stride = src_strides[i];
    if (dim <= 0 || stride <= 0) return false;
    max_offset += static_cast<std::size_t>(dim - 1) *
                  static_cast<std::size_t>(stride);
  }
  if (src_bytes > 0 && max_offset + elem_bytes > src_bytes) {
    return false;
  }

  const std::vector<int64_t> dst_strides = contiguous_strides(shape, elem_bytes);
  std::vector<int64_t> idx(shape.size(), 0);
  std::size_t total_elems = 1;
  for (const auto dim : shape) {
    if (dim <= 0) return false;
    total_elems *= static_cast<std::size_t>(dim);
  }

  for (std::size_t n = 0; n < total_elems; ++n) {
    std::size_t src_offset = 0;
    std::size_t dst_offset = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
      src_offset += static_cast<std::size_t>(idx[i]) *
                    static_cast<std::size_t>(src_strides[i]);
      dst_offset += static_cast<std::size_t>(idx[i]) *
                    static_cast<std::size_t>(dst_strides[i]);
    }
    std::memcpy(dst + dst_offset, src + src_offset, elem_bytes);
    for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
      idx[static_cast<size_t>(i)]++;
      if (idx[static_cast<size_t>(i)] < shape[static_cast<size_t>(i)]) {
        break;
      }
      idx[static_cast<size_t>(i)] = 0;
    }
  }
  return true;
}

} // namespace

std::shared_ptr<NeatStorage> make_cpu_owned_storage(std::size_t size_bytes) {
  void* ptr = nullptr;
  if (size_bytes > 0) {
    if (posix_memalign(&ptr, 64, size_bytes) != 0) {
      ptr = std::malloc(size_bytes);
    }
  }
  if (size_bytes > 0 && !ptr) {
    throw std::bad_alloc();
  }
  auto buf = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(ptr),
                                      [](uint8_t* p) { std::free(p); });
  auto holder = std::shared_ptr<void>(buf, buf.get());

  auto storage = std::make_shared<NeatStorage>();
  storage->kind = NeatStorageKind::CpuOwned;
  storage->device = {NeatDeviceType::CPU, 0};
  storage->size_bytes = size_bytes;
  storage->holder = holder;
  storage->data = buf.get();
  storage->map_fn = [holder, size_bytes](NeatMapMode) {
    NeatMapping mapping;
    mapping.data = holder.get();
    mapping.size_bytes = size_bytes;
    return mapping;
  };
  return storage;
}

std::shared_ptr<NeatStorage> make_cpu_external_storage(void* data,
                                                       std::size_t size_bytes,
                                                       std::shared_ptr<void> holder,
                                                       bool /*read_only*/) {
  auto storage = std::make_shared<NeatStorage>();
  storage->kind = NeatStorageKind::CpuExternal;
  storage->device = {NeatDeviceType::CPU, 0};
  storage->size_bytes = size_bytes;
  storage->holder = std::move(holder);
  storage->data = data;
  storage->map_fn = [data, size_bytes](NeatMapMode) {
    NeatMapping mapping;
    mapping.data = data;
    mapping.size_bytes = size_bytes;
    return mapping;
  };
  return storage;
}

NeatTensor NeatTensor::clone() const {
  const std::size_t elem = dtype_bytes(dtype);
  if (elem == 0) {
    throw std::runtime_error("NeatTensor::clone: unknown element size");
  }

  if (is_dense()) {
    const std::size_t bytes = dense_size_bytes(shape, elem);
    auto storage_copy = make_cpu_owned_storage(bytes);
    NeatMapping src_map = map(NeatMapMode::Read);
    if (!src_map.data || bytes == 0) {
      throw std::runtime_error("NeatTensor::clone: failed to map source");
    }
    NeatMapping dst_map = storage_copy->map(NeatMapMode::Write);
    if (is_contiguous() || strides_bytes.empty()) {
      std::memcpy(dst_map.data, src_map.data, bytes);
    } else {
      std::vector<int64_t> src_strides = strides_bytes;
      if (src_strides.size() != shape.size()) {
        src_strides = contiguous_strides(shape, elem);
      }
      const std::vector<int64_t> dst_strides = contiguous_strides(shape, elem);
      std::vector<int64_t> idx(shape.size(), 0);
      std::size_t total_elems = 1;
      for (const auto dim : shape) {
        if (dim <= 0) {
          throw std::runtime_error("NeatTensor::clone: invalid shape");
        }
        total_elems *= static_cast<std::size_t>(dim);
      }
      const uint8_t* src = static_cast<const uint8_t*>(src_map.data);
      uint8_t* dst = static_cast<uint8_t*>(dst_map.data);
      for (std::size_t n = 0; n < total_elems; ++n) {
        std::size_t src_offset = 0;
        std::size_t dst_offset = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
          src_offset += static_cast<std::size_t>(idx[i]) *
                        static_cast<std::size_t>(src_strides[i]);
          dst_offset += static_cast<std::size_t>(idx[i]) *
                        static_cast<std::size_t>(dst_strides[i]);
        }
        std::memcpy(dst + dst_offset, src + src_offset, elem);
        for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
          idx[static_cast<size_t>(i)]++;
          if (idx[static_cast<size_t>(i)] < shape[static_cast<size_t>(i)]) {
            break;
          }
          idx[static_cast<size_t>(i)] = 0;
        }
      }
    }

    NeatTensor out;
    out.storage = std::move(storage_copy);
    out.dtype = dtype;
    out.layout = layout;
    out.shape = shape;
    out.strides_bytes = contiguous_strides(shape, elem);
    out.byte_offset = 0;
    out.device = {NeatDeviceType::CPU, 0};
    out.semantic = semantic;
    out.read_only = false;
    return out;
  }

  std::size_t total_bytes = 0;
  std::vector<std::size_t> plane_row_bytes;
  plane_row_bytes.reserve(planes.size());
  for (const auto& plane : planes) {
    if (plane.shape.size() < 2) {
      throw std::runtime_error("NeatTensor::clone: invalid plane shape");
    }
    const int64_t h = plane.shape[0];
    const int64_t w = plane.shape[1];
    if (h <= 0 || w <= 0) {
      throw std::runtime_error("NeatTensor::clone: invalid plane dimensions");
    }
    const std::size_t row_bytes = static_cast<std::size_t>(w) * elem;
    plane_row_bytes.push_back(row_bytes);
    total_bytes += row_bytes * static_cast<std::size_t>(h);
  }

  auto storage_copy = make_cpu_owned_storage(total_bytes);
  NeatMapping src_map = map(NeatMapMode::Read);
  if (!src_map.data || total_bytes == 0) {
    throw std::runtime_error("NeatTensor::clone: failed to map source planes");
  }
  NeatMapping dst_map = storage_copy->map(NeatMapMode::Write);
  uint8_t* dst = static_cast<uint8_t*>(dst_map.data);
  const uint8_t* src = static_cast<const uint8_t*>(src_map.data);

  std::vector<NeatPlane> out_planes;
  out_planes.reserve(planes.size());
  std::size_t offset = 0;
  for (size_t i = 0; i < planes.size(); ++i) {
    const auto& plane = planes[i];
    const std::size_t row_bytes = plane_row_bytes[i];
    const int64_t h = plane.shape[0];
    const int64_t src_stride = !plane.strides_bytes.empty()
        ? plane.strides_bytes[0]
        : static_cast<int64_t>(row_bytes);
    for (int64_t y = 0; y < h; ++y) {
      const uint8_t* src_row = src + plane.byte_offset +
          static_cast<std::size_t>(y) * static_cast<std::size_t>(src_stride);
      uint8_t* dst_row = dst + offset +
          static_cast<std::size_t>(y) * row_bytes;
      std::memcpy(dst_row, src_row, row_bytes);
    }
    NeatPlane out_plane = plane;
    out_plane.byte_offset = static_cast<int64_t>(offset);
    out_plane.strides_bytes = {static_cast<int64_t>(row_bytes), 1};
    out_planes.push_back(std::move(out_plane));
    offset += row_bytes * static_cast<std::size_t>(h);
  }

  NeatTensor out;
  out.storage = std::move(storage_copy);
  out.dtype = dtype;
  out.layout = layout;
  out.shape = shape;
  out.strides_bytes = contiguous_strides(shape, elem);
  out.byte_offset = 0;
  out.device = {NeatDeviceType::CPU, 0};
  out.semantic = semantic;
  out.planes = std::move(out_planes);
  out.read_only = false;
  return out;
}

NeatTensor NeatTensor::contiguous() const {
  if (is_contiguous() && is_dense()) return *this;
  return clone();
}

NeatTensor NeatTensor::to(NeatDevice target) const {
  if (device.type == target.type && device.id == target.id) return *this;
  switch (target.type) {
    case NeatDeviceType::CPU:
      return cpu();
    case NeatDeviceType::SIMA_CVU:
      return cvu();
    case NeatDeviceType::SIMA_MLA:
      return pipeline_internal::transfer_to_device(*this, target, nullptr, nullptr);
    case NeatDeviceType::SIMA_APU:
    case NeatDeviceType::UNKNOWN:
      break;
  }
  throw std::runtime_error("NeatTensor::to: unsupported target device");
}

NeatTensor NeatTensor::cpu() const {
  if (device.type == NeatDeviceType::CPU) {
    if (!storage) {
      throw std::runtime_error("NeatTensor::cpu: missing storage");
    }
    NeatMapping mapping = map_read();
    if (mapping.data) {
      return *this;
    }
  }
  return pipeline_internal::transfer_to_cpu(*this);
}

NeatTensor NeatTensor::cvu() const {
  if (device.type == NeatDeviceType::SIMA_CVU) {
    return *this;
  }
  return pipeline_internal::transfer_to_device(
      *this, {NeatDeviceType::SIMA_CVU, 0}, nullptr, nullptr);
}

NeatTensor NeatTensor::mla(bool force) const {
  if (device.type == NeatDeviceType::SIMA_MLA) {
    return *this;
  }
  if (!force && device.type == NeatDeviceType::SIMA_CVU) {
    return *this;
  }

  if (!force) {
    try {
      return pipeline_internal::transfer_to_device(
          *this, {NeatDeviceType::SIMA_CVU, 0}, nullptr, nullptr);
    } catch (const std::exception&) {
      // Fall back to DMS0.
    }
  }

  return pipeline_internal::transfer_to_device(
      *this, {NeatDeviceType::SIMA_MLA, 0}, nullptr, nullptr);
}

NeatTensor NeatTensor::to_cpu_if_needed() const {
  return cpu();
}

std::size_t NeatTensor::dense_bytes_tight() const {
  if (!is_dense()) return 0;
  const std::size_t elem = dtype_bytes(dtype);
  if (elem == 0) return 0;
  return dense_size_bytes(shape, elem);
}

bool NeatTensor::copy_dense_bytes_tight_to(uint8_t* dst, std::size_t dst_size) const {
  if (!dst) return false;
  if (!is_dense()) return false;
  const std::size_t bytes = dense_bytes_tight();
  if (bytes == 0 || dst_size < bytes) return false;

  NeatMapping mapping = map_read();
  if (!mapping.data || mapping.size_bytes == 0) return false;
  if (mapping.size_bytes < bytes) return false;

  const uint8_t* src = static_cast<const uint8_t*>(mapping.data);
  if (is_contiguous() || strides_bytes.empty()) {
    std::memcpy(dst, src, bytes);
    return true;
  }

  std::vector<int64_t> src_strides = strides_bytes;
  if (src_strides.size() != shape.size()) {
    src_strides = contiguous_strides(shape, dtype_bytes(dtype));
  }
  return copy_dense_strided(src, mapping.size_bytes, shape, src_strides, dtype_bytes(dtype), dst);
}

std::vector<uint8_t> NeatTensor::copy_dense_bytes_tight() const {
  const std::size_t bytes = dense_bytes_tight();
  if (bytes == 0) {
    throw std::runtime_error("copy_dense_bytes_tight: unknown dense size");
  }
  std::vector<uint8_t> out(bytes);
  if (!copy_dense_bytes_tight_to(out.data(), out.size())) {
    throw std::runtime_error("copy_dense_bytes_tight: copy failed");
  }
  return out;
}

bool NeatTensor::copy_payload_bytes_to(uint8_t* dst, std::size_t dst_size) const {
  if (!dst) return false;
  if (is_dense()) {
    const std::size_t bytes = dense_bytes_tight();
    if (bytes > 0) {
      return copy_dense_bytes_tight_to(dst, dst_size);
    }
  }

  NeatMapping mapping = map_read();
  if (!mapping.data || mapping.size_bytes == 0) return false;
  if (dst_size < mapping.size_bytes) return false;
  std::memcpy(dst, mapping.data, mapping.size_bytes);
  return true;
}

std::vector<uint8_t> NeatTensor::copy_payload_bytes() const {
  const std::size_t dense_bytes = dense_bytes_tight();
  if (dense_bytes > 0) {
    std::vector<uint8_t> out(dense_bytes);
    if (!copy_dense_bytes_tight_to(out.data(), out.size())) {
      throw std::runtime_error("copy_payload_bytes: dense copy failed");
    }
    return out;
  }

  NeatMapping mapping = map_read();
  if (!mapping.data || mapping.size_bytes == 0) {
    throw std::runtime_error("copy_payload_bytes: mapping failed");
  }
  std::vector<uint8_t> out(mapping.size_bytes);
  std::memcpy(out.data(), mapping.data, mapping.size_bytes);
  return out;
}

bool NeatTensor::validate(std::string* err) const {
  auto fail = [&](const std::string& msg) {
    if (err) *err = msg;
    return false;
  };

  if (is_composite() && byte_offset != 0) {
    return fail("composite tensor must have byte_offset == 0");
  }

  const std::size_t elem = dtype_bytes(dtype);
  const std::size_t storage_bytes = storage ? storage->size_bytes : 0;

  auto plane_bytes = [&](const NeatPlane& plane) -> std::size_t {
    if (!plane.strides_bytes.empty() && plane.shape.size() >= 1) {
      const int64_t h = plane.shape[0];
      const int64_t stride0 = plane.strides_bytes[0];
      if (h > 0 && stride0 > 0) {
        return static_cast<std::size_t>(stride0) * static_cast<std::size_t>(h);
      }
    }
    if (plane.shape.size() >= 2 && elem > 0) {
      const int64_t h = plane.shape[0];
      const int64_t w = plane.shape[1];
      if (h > 0 && w > 0) {
        return static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * elem;
      }
    }
    if (plane.shape.size() >= 1 && elem > 0) {
      const int64_t n = plane.shape[0];
      if (n > 0) {
        return static_cast<std::size_t>(n) * elem;
      }
    }
    return 0;
  };

  for (const auto& plane : planes) {
    if (plane.byte_offset < 0) {
      return fail("plane byte_offset is negative");
    }
    if (storage_bytes > 0) {
      const std::size_t bytes = plane_bytes(plane);
      if (bytes == 0) {
        return fail("plane bytes could not be computed");
      }
      const std::size_t end =
          static_cast<std::size_t>(plane.byte_offset) + bytes;
      if (end > storage_bytes) {
        return fail("plane exceeds storage bounds");
      }
    }
  }

  if (semantic.image.has_value()) {
    const NeatImageSpec::PixelFormat fmt = semantic.image->format;
    if (fmt == NeatImageSpec::PixelFormat::NV12 ||
        fmt == NeatImageSpec::PixelFormat::I420) {
      if (dtype != TensorDType::UInt8) {
        return fail("NV12/I420 tensors must use UInt8 dtype");
      }
      const int w = width();
      const int h = height();
      if (w > 0 && h > 0) {
        if ((w % 2) != 0 || (h % 2) != 0) {
          return fail("NV12/I420 tensors require even width/height");
        }
      }
    }
    auto has_role = [&](NeatPlaneRole role) {
      for (const auto& plane : planes) {
        if (plane.role == role) return true;
      }
      return false;
    };
    if (fmt == NeatImageSpec::PixelFormat::NV12) {
      if (!has_role(NeatPlaneRole::Y) || !has_role(NeatPlaneRole::UV)) {
        return fail("NV12 tensor missing Y/UV planes");
      }
    }
    if (fmt == NeatImageSpec::PixelFormat::I420) {
      if (!has_role(NeatPlaneRole::Y) ||
          !has_role(NeatPlaneRole::U) ||
          !has_role(NeatPlaneRole::V)) {
        return fail("I420 tensor missing Y/U/V planes");
      }
    }
  }

  return true;
}

std::string NeatTensor::debug_string() const {
  std::string out = "NeatTensor{dtype=";
  out += dtype_name(dtype);
  out += " device=";
  out += device_name(device.type);
  out += ":" + std::to_string(device.id);
  out += " layout=";
  out += layout_name(layout);
  out += " shape=";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) out.push_back('x');
    out += std::to_string(shape[i]);
  }
  out += " strides_bytes=";
  for (size_t i = 0; i < strides_bytes.size(); ++i) {
    if (i) out.push_back(',');
    out += std::to_string(strides_bytes[i]);
  }
  out += " byte_offset=" + std::to_string(byte_offset);
  out += " planes=" + std::to_string(planes.size());
  out += "}";
  return out;
}

} // namespace sima
