// src/pipeline/TensorUtil.cpp
#include "pipeline/internal/TensorUtil.h"

#include "pipeline/internal/GstDiagnosticsUtil.h"

#include <gst/gst.h>

#include <cstring>
#include <memory>
#include <stdexcept>

namespace sima::pipeline_internal {
namespace {

std::size_t neat_dtype_bytes(TensorDType dtype) {
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

GstSample* sample_from_neat_tensor(const NeatTensor& tensor) {
  if (!tensor.storage) return nullptr;
  if (tensor.storage->kind != NeatStorageKind::GstSample) return nullptr;
  if (!tensor.storage->holder) return nullptr;
  return static_cast<GstSample*>(tensor.storage->holder.get());
}

} // namespace

GstBuffer* buffer_from_tensor_holder(const std::shared_ptr<void>& holder) {
  if (!holder) return nullptr;
  auto* sample = static_cast<GstSample*>(holder.get());
  if (!sample) return nullptr;
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) return nullptr;
  return gst_buffer_ref(buffer);
}

NeatTensor copy_neat_tensor_from_sample_memory(const NeatTensor& ref, int memory_index) {
  GstSample* sample = sample_from_neat_tensor(ref);
  if (!sample) {
    throw std::runtime_error("copy_neat_tensor_from_sample_memory: missing sample holder");
  }

  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    throw std::runtime_error("copy_neat_tensor_from_sample_memory: missing buffer");
  }

  const guint n_mems = gst_buffer_n_memory(buffer);
  if (n_mems == 0) {
    throw std::runtime_error("copy_neat_tensor_from_sample_memory: buffer has no memories");
  }

  guint index = static_cast<guint>(memory_index);
  if (memory_index < 0 || index >= n_mems) {
    index = 0;
  }

  GstMemory* mem = gst_buffer_peek_memory(buffer, index);
  if (!mem) {
    throw std::runtime_error("copy_neat_tensor_from_sample_memory: missing buffer memory");
  }

  GstMapInfo map{};
  if (!gst_memory_map(mem, &map, GST_MAP_READ)) {
    throw std::runtime_error("copy_neat_tensor_from_sample_memory: gst_memory_map failed");
  }

  auto storage = make_cpu_owned_storage(map.size);
  NeatMapping dst_map = storage->map(NeatMapMode::Write);
  if (map.size > 0 && dst_map.data) {
    std::memcpy(dst_map.data, map.data, map.size);
  }
  gst_memory_unmap(mem, &map);

  NeatTensor out;
  out.storage = std::move(storage);
  if (ref.storage && ref.storage->holder) {
    out.storage->holder = ref.storage->holder;
  }
  out.dtype = ref.dtype;
  out.device = {NeatDeviceType::CPU, 0};
  out.read_only = false;
  out.semantic = ref.semantic;
  out.byte_offset = 0;

  const std::size_t elem = neat_dtype_bytes(out.dtype);
  if (elem > 0 && (map.size % elem) == 0) {
    const std::size_t elems = map.size / elem;
    out.shape = {static_cast<int64_t>(elems)};
    out.strides_bytes = {static_cast<int64_t>(elem)};
  }

  return out;
}

std::shared_ptr<void> holder_from_neat_tensor(const NeatTensor& tensor) {
  if (!tensor.storage) return nullptr;
  return tensor.storage->holder;
}

} // namespace sima::pipeline_internal
