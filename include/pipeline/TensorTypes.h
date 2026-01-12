#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sima {

// Minimal DLPack-like structs (CPU only). This keeps a zero-copy bridge path
// for future bindings without adding a hard dependency today.
namespace dlpack {

enum class DLDeviceType : int {
  kDLCPU = 1,
};

struct DLDevice {
  DLDeviceType device_type;
  int device_id;
};

enum DLDataTypeCode : uint8_t {
  kDLInt = 0,
  kDLUInt = 1,
  kDLFloat = 2,
};

struct DLDataType {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

struct DLTensor {
  void* data;
  DLDevice device;
  int ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
};

struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(DLManagedTensor* self);
};

} // namespace dlpack

enum class TensorDType {
  UInt8,
  Int8,
  UInt16,
  Int16,
  Int32,
  Float32,
  Float64,
};

enum class TensorLayout {
  Unknown = 0,
  HWC,
  CHW,
  HW,
  Planar,
};

struct TensorPlaneRef {
  const uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  int stride = 0;
};

struct FrameTensor {
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::string format;
  int width = 0;
  int height = 0;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::vector<std::vector<uint8_t>> planes;

  int64_t pts_ns = -1;
  int64_t dts_ns = -1;
  int64_t duration_ns = -1;
  bool keyframe = false;
  std::string caps_string;
};

struct FrameTensorRef {
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::string format;
  int width = 0;
  int height = 0;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::vector<TensorPlaneRef> planes;

  int64_t pts_ns = -1;
  int64_t dts_ns = -1;
  int64_t duration_ns = -1;
  bool keyframe = false;
  std::string caps_string;

  // Keeps the mapped sample alive for zero-copy bindings (future Python layer).
  std::shared_ptr<void> holder;

  // Zero-copy DLPack export (CPU only). The caller must invoke the deleter
  // on the returned object when done; the holder keeps the sample alive.
  dlpack::DLManagedTensor* to_dlpack() const;

  FrameTensor to_copy() const;
};

} // namespace sima
