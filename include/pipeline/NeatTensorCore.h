#pragma once

#include "pipeline/TensorTypes.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

namespace sima {

enum class NeatDeviceType {
  CPU = 0,
  SIMA_APU,
  SIMA_CVU,
  SIMA_MLA,
  UNKNOWN,
};

struct NeatDevice {
  NeatDeviceType type = NeatDeviceType::CPU;
  int id = 0;
};

enum class NeatStorageKind {
  CpuOwned = 0,
  CpuExternal,
  GstSample,
  DeviceHandle,
  Unknown,
};

enum class NeatPlaneRole {
  Unknown = 0,
  Y,
  U,
  V,
  UV,
};

enum class NeatMapMode {
  Read = 0,
  Write,
  ReadWrite,
};

struct NeatImageSpec {
  enum class PixelFormat {
    RGB = 0,
    BGR,
    GRAY8,
    NV12,
    I420,
    UNKNOWN,
  };

  PixelFormat format = PixelFormat::UNKNOWN;
  std::string color_space;
};

struct NeatAudioSpec {
  int sample_rate = 0;
  int channels = 0;
  bool interleaved = true;
};

struct NeatTokensSpec {
  int vocab_size = 0;
};

struct NeatQuantSpec {
  float scale = 1.0f;
  int32_t zero_point = 0;
  int axis = -1;
  std::vector<float> scales;
  std::vector<int32_t> zero_points;
};

struct NeatTessSpec {
  int tile_width = 0;
  int tile_height = 0;
  int tile_channels = 0;
  std::string format;
};

struct NeatSemantic {
  std::optional<NeatImageSpec> image;
  std::optional<NeatAudioSpec> audio;
  std::optional<NeatTokensSpec> tokens;
  std::optional<NeatTessSpec> tess;
  std::optional<NeatQuantSpec> quant;
};

struct NeatMapping {
  void* data = nullptr;
  std::size_t size_bytes = 0;
  std::function<void()> unmap;
  std::shared_ptr<void> keepalive;

  NeatMapping() = default;
  NeatMapping(const NeatMapping&) = delete;
  NeatMapping& operator=(const NeatMapping&) = delete;
  NeatMapping(NeatMapping&& other) noexcept { *this = std::move(other); }
  NeatMapping& operator=(NeatMapping&& other) noexcept {
    if (this != &other) {
      if (unmap) unmap();
      data = other.data;
      size_bytes = other.size_bytes;
      unmap = std::move(other.unmap);
      keepalive = std::move(other.keepalive);
      other.data = nullptr;
      other.size_bytes = 0;
      other.unmap = nullptr;
      other.keepalive.reset();
    }
    return *this;
  }
  ~NeatMapping() {
    if (unmap) unmap();
  }
};

#if defined(SIMA_WITH_OPENCV)
struct NeatCvMatView {
  NeatMapping mapping;
  cv::Mat mat;
};
#endif

struct NeatSegment {
  std::string name;
  std::size_t size_bytes = 0;
};

struct NeatStorage {
  NeatStorageKind kind = NeatStorageKind::Unknown;
  NeatDevice device{};
  std::size_t size_bytes = 0;
  std::shared_ptr<void> holder;
  void* data = nullptr;
  std::function<NeatMapping(NeatMapMode)> map_fn;
  std::uint64_t sima_mem_target_flags = 0;
  std::uint64_t sima_mem_flags = 0;
  std::vector<NeatSegment> sima_segments;

  NeatMapping map(NeatMapMode mode) const {
    NeatMapping mapping;
    if (map_fn) {
      mapping = map_fn(mode);
    } else {
      mapping.data = data;
      mapping.size_bytes = size_bytes;
    }
    if (!mapping.keepalive) {
      mapping.keepalive = holder;
    }
    return mapping;
  }
};

std::shared_ptr<NeatStorage> make_cpu_owned_storage(std::size_t size_bytes);
std::shared_ptr<NeatStorage> make_cpu_external_storage(void* data,
                                                       std::size_t size_bytes,
                                                       std::shared_ptr<void> holder = {},
                                                       bool read_only = true);

struct NeatPlane {
  NeatPlaneRole role = NeatPlaneRole::Unknown;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides_bytes;
  int64_t byte_offset = 0;
};

struct NeatNv12View {
  int width = 0;
  int height = 0;
  const uint8_t* y = nullptr;
  int64_t y_stride = 0;
  const uint8_t* uv = nullptr;
  int64_t uv_stride = 0;
};

struct NeatNv12Mapped {
  NeatMapping mapping;
  NeatNv12View view;
};

struct NeatI420View {
  int width = 0;
  int height = 0;
  const uint8_t* y = nullptr;
  int64_t y_stride = 0;
  const uint8_t* u = nullptr;
  int64_t u_stride = 0;
  const uint8_t* v = nullptr;
  int64_t v_stride = 0;
};

struct NeatI420Mapped {
  NeatMapping mapping;
  NeatI420View view;
};

struct NeatTensor {
  std::shared_ptr<NeatStorage> storage;
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides_bytes;
  int64_t byte_offset = 0;
  NeatDevice device{};
  NeatSemantic semantic{};
  std::vector<NeatPlane> planes;
  bool read_only = true;

  bool is_dense() const { return planes.empty(); }
  bool is_composite() const { return !planes.empty(); }

  bool is_contiguous() const {
    if (shape.empty()) return true;
    if (strides_bytes.empty()) return true;
    std::size_t elem = dtype_bytes(dtype);
    if (elem == 0) return false;
    std::int64_t expected = static_cast<std::int64_t>(elem);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      if (strides_bytes[static_cast<size_t>(i)] != expected) return false;
      expected *= shape[static_cast<size_t>(i)];
    }
    return true;
  }

  const NeatPlane* try_plane(NeatPlaneRole role) const noexcept {
    for (const auto& plane : planes) {
      if (plane.role == role) return &plane;
    }
    return nullptr;
  }

  bool has_plane(NeatPlaneRole role) const noexcept {
    return try_plane(role) != nullptr;
  }

  const NeatPlane& plane(NeatPlaneRole role) const {
    const NeatPlane* found = try_plane(role);
    if (!found) throw std::runtime_error("NeatTensor::plane: plane not found");
    return *found;
  }

  NeatMapping map(NeatMapMode mode) const {
    if (read_only && mode != NeatMapMode::Read) {
      throw std::runtime_error("NeatTensor::map: tensor is read-only");
    }
    if (!storage) return {};
    NeatMapping base = storage->map(mode);
    if (!base.data) return base;
    NeatMapping out = std::move(base);
    if (!out.keepalive && storage) {
      out.keepalive = std::static_pointer_cast<void>(storage);
    }
    if (byte_offset != 0) {
      out.data = static_cast<uint8_t*>(out.data) + byte_offset;
      if (out.size_bytes > static_cast<std::size_t>(byte_offset)) {
        out.size_bytes = out.size_bytes - static_cast<std::size_t>(byte_offset);
      }
    }
#if defined(NEAT_VALIDATE_ON_MAP)
    std::string err;
    if (!validate(&err)) {
      throw std::runtime_error("NeatTensor::map: " + err);
    }
#endif
    return out;
  }

  NeatMapping map_read() const { return map(NeatMapMode::Read); }
  NeatMapping map_write() const { return map(NeatMapMode::Write); }

  template <typename T>
  T* data_ptr() {
    if (read_only) {
      throw std::runtime_error("NeatTensor::data_ptr: tensor is read-only");
    }
    return const_cast<T*>(const_data_ptr<T>());
  }

  template <typename T>
  const T* data_ptr() const {
    return const_data_ptr<T>();
  }

  NeatTensor contiguous() const;
  NeatTensor clone() const;
  NeatTensor to(NeatDevice target) const;
  NeatTensor cpu() const;
  NeatTensor cvu() const;
  NeatTensor mla(bool force = false) const;
  NeatTensor to_cpu_if_needed() const;
  bool validate(std::string* err) const;

  std::optional<NeatNv12Mapped> map_nv12_read() const;
  std::size_t nv12_required_bytes() const;
  bool copy_nv12_contiguous_to(uint8_t* dst, std::size_t dst_size) const;
  std::vector<uint8_t> copy_nv12_contiguous() const;

  std::optional<NeatI420Mapped> map_i420_read() const;
  std::size_t i420_required_bytes() const;
  bool copy_i420_contiguous_to(uint8_t* dst, std::size_t dst_size) const;
  std::vector<uint8_t> copy_i420_contiguous() const;

  std::size_t dense_bytes_tight() const;
  bool copy_dense_bytes_tight_to(uint8_t* dst, std::size_t dst_size) const;
  std::vector<uint8_t> copy_dense_bytes_tight() const;

  bool copy_payload_bytes_to(uint8_t* dst, std::size_t dst_size) const;
  std::vector<uint8_t> copy_payload_bytes() const;

  int width() const;
  int height() const;
  int channels() const;
  std::optional<NeatImageSpec::PixelFormat> image_format() const;
  bool is_nv12() const;
  bool is_i420() const;

  std::string debug_string() const;

#if defined(SIMA_WITH_OPENCV)
  static NeatTensor from_cv_mat(
      const cv::Mat& mat,
      NeatImageSpec::PixelFormat fmt = NeatImageSpec::PixelFormat::BGR,
      bool read_only = true);
  std::optional<NeatCvMatView> map_cv_mat_view(NeatImageSpec::PixelFormat desired) const;
  cv::Mat to_cv_mat_copy(NeatImageSpec::PixelFormat desired) const;
#endif

private:
  template <typename T>
  const T* const_data_ptr() const {
    if (device.type != NeatDeviceType::CPU) {
      throw std::runtime_error("NeatTensor::data_ptr: tensor is not on CPU");
    }
    if (!is_dense()) {
      throw std::runtime_error("NeatTensor::data_ptr: tensor is composite");
    }
    if (!is_contiguous()) {
      throw std::runtime_error("NeatTensor::data_ptr: call cpu().contiguous() first");
    }
    if (!storage || !storage->data) {
      throw std::runtime_error("NeatTensor::data_ptr: tensor storage is not mappable");
    }
    return reinterpret_cast<const T*>(
        static_cast<const uint8_t*>(storage->data) + byte_offset);
  }

  static std::size_t dtype_bytes(TensorDType dtype) {
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
};

} // namespace sima
