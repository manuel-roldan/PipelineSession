// src/pipeline/TensorUtil.cpp
#include "pipeline/internal/TensorUtil.h"

#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/SimaaiMemory.h"

#include <gst/gst.h>

#include <cstring>
#include <memory>
#include <mutex>
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

struct SimaaiMemoryInfo {
  std::uint64_t target_flags = 0;
  std::uint64_t mem_flags = 0;
  NeatDevice device{NeatDeviceType::CPU, 0};
};

void update_mem_flags(GstMemory* mem, SimaaiMemoryInfo* info) {
  if (!mem || !info) return;
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_EV74)) {
    info->target_flags |= GST_SIMAAI_MEMORY_TARGET_EV74;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS0)) {
    info->target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS0;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS1)) {
    info->target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS1;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS2)) {
    info->target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS2;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_TARGET_DMS3)) {
    info->target_flags |= GST_SIMAAI_MEMORY_TARGET_DMS3;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_FLAG_CACHED)) {
    info->mem_flags |= GST_SIMAAI_MEMORY_FLAG_CACHED;
  }
  if (GST_MEMORY_FLAG_IS_SET(mem, GST_SIMAAI_MEMORY_FLAG_RDONLY)) {
    info->mem_flags |= GST_SIMAAI_MEMORY_FLAG_RDONLY;
  }
}

SimaaiMemoryInfo infer_simaai_memory_info(GstBuffer* buffer) {
  SimaaiMemoryInfo info;
  if (!buffer) return info;

  const guint n_mems = gst_buffer_n_memory(buffer);
  for (guint i = 0; i < n_mems; ++i) {
    GstMemory* mem = gst_buffer_peek_memory(buffer, i);
    update_mem_flags(mem, &info);
  }

  if (info.target_flags & GST_SIMAAI_MEMORY_TARGET_EV74) {
    info.device = {NeatDeviceType::SIMA_CVU, 0};
  } else if (info.target_flags & GST_SIMAAI_MEMORY_TARGET_DMS0) {
    info.device = {NeatDeviceType::SIMA_MLA, 0};
  } else if (info.target_flags & GST_SIMAAI_MEMORY_TARGET_DMS1) {
    info.device = {NeatDeviceType::SIMA_MLA, 1};
  } else if (info.target_flags & GST_SIMAAI_MEMORY_TARGET_DMS2) {
    info.device = {NeatDeviceType::SIMA_MLA, 2};
  } else if (info.target_flags & GST_SIMAAI_MEMORY_TARGET_DMS3) {
    info.device = {NeatDeviceType::SIMA_MLA, 3};
  }

  return info;
}

} // namespace

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
  const SimaaiMemoryInfo mem_info = infer_simaai_memory_info(buffer);
  storage->device = mem_info.device;
  storage->size_bytes = gst_buffer_get_size(buffer);
  storage->holder = holder;
  storage->data = nullptr;
  storage->sima_mem_target_flags = mem_info.target_flags;
  storage->sima_mem_flags = mem_info.mem_flags;
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
