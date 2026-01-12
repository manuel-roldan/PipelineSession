#include "pipeline/TensorTypes.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace sima {
namespace {

struct DlpackHolder {
  dlpack::DLManagedTensor managed;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::shared_ptr<void> holder;
};

static dlpack::DLDataType dtype_to_dlpack(TensorDType dtype) {
  dlpack::DLDataType out{};
  out.lanes = 1;
  switch (dtype) {
    case TensorDType::UInt8: out.code = dlpack::kDLUInt; out.bits = 8; break;
    case TensorDType::Int8: out.code = dlpack::kDLInt; out.bits = 8; break;
    case TensorDType::UInt16: out.code = dlpack::kDLUInt; out.bits = 16; break;
    case TensorDType::Int16: out.code = dlpack::kDLInt; out.bits = 16; break;
    case TensorDType::Int32: out.code = dlpack::kDLInt; out.bits = 32; break;
    case TensorDType::Float32: out.code = dlpack::kDLFloat; out.bits = 32; break;
    case TensorDType::Float64: out.code = dlpack::kDLFloat; out.bits = 64; break;
  }
  return out;
}

static size_t bytes_per_element(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::UInt8: return 1;
    case TensorDType::Int8: return 1;
    case TensorDType::UInt16: return 2;
    case TensorDType::Int16: return 2;
    case TensorDType::Int32: return 4;
    case TensorDType::Float32: return 4;
    case TensorDType::Float64: return 8;
  }
  return 1;
}

} // namespace

dlpack::DLManagedTensor* FrameTensorRef::to_dlpack() const {
  if (planes.size() != 1) {
    throw std::runtime_error("to_dlpack: only single-plane tensors are supported");
  }
  if (shape.empty() || strides.empty()) {
    throw std::runtime_error("to_dlpack: missing shape/strides");
  }

  auto* holder = new DlpackHolder();
  holder->shape = shape;
  holder->strides = strides;
  holder->holder = this->holder;

  dlpack::DLManagedTensor* mt = &holder->managed;
  std::memset(mt, 0, sizeof(*mt));
  mt->dl_tensor.data = const_cast<uint8_t*>(planes[0].data);
  mt->dl_tensor.device = {dlpack::DLDeviceType::kDLCPU, 0};
  mt->dl_tensor.ndim = static_cast<int>(holder->shape.size());
  mt->dl_tensor.dtype = dtype_to_dlpack(dtype);
  mt->dl_tensor.shape = holder->shape.data();
  mt->dl_tensor.strides = holder->strides.data();
  mt->dl_tensor.byte_offset = 0;
  mt->manager_ctx = holder;
  mt->deleter = [](dlpack::DLManagedTensor* self) {
    if (!self || !self->manager_ctx) return;
    auto* h = static_cast<DlpackHolder*>(self->manager_ctx);
    delete h;
  };

  // Hand over ownership to the DLManagedTensor. It keeps the sample alive via holder.
  return mt;
}

FrameTensor FrameTensorRef::to_copy() const {
  FrameTensor out;
  out.dtype = dtype;
  out.layout = layout;
  out.format = format;
  out.width = width;
  out.height = height;
  out.shape = shape;
  out.strides = strides;
  out.pts_ns = pts_ns;
  out.dts_ns = dts_ns;
  out.duration_ns = duration_ns;
  out.keyframe = keyframe;
  out.caps_string = caps_string;

  const size_t elem = bytes_per_element(dtype);
  out.planes.reserve(planes.size());
  for (const auto& p : planes) {
    const int row_bytes = p.width * static_cast<int>(elem);
    std::vector<uint8_t> buf;
    buf.resize(static_cast<size_t>(row_bytes) * static_cast<size_t>(p.height));
    for (int r = 0; r < p.height; ++r) {
      const uint8_t* src = p.data + static_cast<size_t>(r) * static_cast<size_t>(p.stride);
      uint8_t* dst = buf.data() + static_cast<size_t>(r) * static_cast<size_t>(row_bytes);
      std::memcpy(dst, src, static_cast<size_t>(row_bytes));
    }
    out.planes.push_back(std::move(buf));
  }

  return out;
}

} // namespace sima
