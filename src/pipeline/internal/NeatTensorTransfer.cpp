#include "pipeline/internal/NeatTensorTransfer.h"

#include "pipeline/internal/SimaaiMemory.h"
#include "pipeline/internal/TensorUtil.h"

#include <gst/gst.h>
#include <gst/gstbufferpool.h>

#if __has_include(<simaai/gstsimaaibufferpool.h>)
#include <simaai/gstsimaaibufferpool.h>
#include <simaai/gstsimaaisegmentallocator.h>
#define SIMA_HAS_SIMAAI_POOL 1
#else
#define SIMA_HAS_SIMAAI_POOL 0
#endif

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace sima::pipeline_internal {
namespace {

struct PoolEntry {
  std::shared_ptr<GstBufferPool> pool;
  std::vector<std::string> segment_names;
};

struct PoolCache {
  std::mutex mu;
  std::unordered_map<std::string, PoolEntry> pools;
  std::size_t hits = 0;
  std::size_t misses = 0;
};

PoolCache& pool_cache() {
  static PoolCache cache;
  return cache;
}

std::string pool_key(std::uint64_t target_flags,
                     std::uint64_t mem_flags,
                     const std::vector<NeatSegment>& segments) {
  std::ostringstream oss;
  oss << target_flags << ":" << mem_flags << ":" << segments.size();
  for (const auto& seg : segments) {
    oss << "|" << seg.name << ":" << seg.size_bytes;
  }
  return oss.str();
}

std::optional<PoolEntry> create_pool_entry(std::uint64_t target_flags,
                                           std::uint64_t mem_flags,
                                           const std::vector<NeatSegment>& segments) {
#if SIMA_HAS_SIMAAI_POOL
  if (segments.empty()) return {};
  gst_simaai_segment_memory_init_once();

  std::vector<gsize> sizes;
  sizes.reserve(segments.size());
  PoolEntry entry;
  entry.segment_names.reserve(segments.size());
  for (const auto& seg : segments) {
    sizes.push_back(static_cast<gsize>(seg.size_bytes));
    entry.segment_names.push_back(seg.name);
  }

  std::vector<const char*> names;
  names.reserve(entry.segment_names.size());
  for (const auto& name : entry.segment_names) {
    names.push_back(name.c_str());
  }

  GstMemoryFlags flags = static_cast<GstMemoryFlags>(target_flags | mem_flags);
  GstBufferPool* pool = gst_simaai_allocate_buffer_pool2(
      /*object=*/nullptr,
      gst_simaai_memory_get_segment_allocator(),
      /*min_buffers=*/2,
      /*max_buffers=*/0,
      flags,
      static_cast<gsize>(segments.size()),
      sizes.data(),
      names.data());
  if (!pool) return {};
  entry.pool = std::shared_ptr<GstBufferPool>(pool, [](GstBufferPool* p) {
    gst_simaai_free_buffer_pool(p);
  });
  return entry;
#else
  (void)target_flags;
  (void)mem_flags;
  (void)segments;
  return {};
#endif
}

std::shared_ptr<GstBufferPool> get_pool(std::uint64_t target_flags,
                                        std::uint64_t mem_flags,
                                        const std::vector<NeatSegment>& segments) {
  const std::string key = pool_key(target_flags, mem_flags, segments);
  PoolCache& cache = pool_cache();
  {
    std::lock_guard<std::mutex> lock(cache.mu);
    auto it = cache.pools.find(key);
    if (it != cache.pools.end() && it->second.pool) {
      cache.hits++;
      return it->second.pool;
    }
    cache.misses++;
  }

  std::optional<PoolEntry> created = create_pool_entry(target_flags, mem_flags, segments);
  if (!created.has_value() || !created->pool) return {};

  std::lock_guard<std::mutex> lock(cache.mu);
  auto [it, inserted] = cache.pools.emplace(key, std::move(*created));
  if (!inserted && it->second.pool) {
    return it->second.pool;
  }
  return it->second.pool;
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

std::vector<NeatPlane> contiguous_planes(const NeatTensor& src, std::size_t elem) {
  if (src.planes.empty()) return {};
  std::vector<std::size_t> plane_row_bytes;
  plane_row_bytes.reserve(src.planes.size());
  for (const auto& plane : src.planes) {
    if (plane.shape.size() < 2) {
      throw std::runtime_error("transfer: invalid plane shape");
    }
    const int64_t h = plane.shape[0];
    const int64_t w = plane.shape[1];
    if (h <= 0 || w <= 0) {
      throw std::runtime_error("transfer: invalid plane dimensions");
    }
    plane_row_bytes.push_back(static_cast<std::size_t>(w) * elem);
  }

  std::vector<NeatPlane> out_planes;
  out_planes.reserve(src.planes.size());
  std::size_t offset = 0;
  for (size_t i = 0; i < src.planes.size(); ++i) {
    NeatPlane out_plane = src.planes[i];
    const std::size_t row_bytes = plane_row_bytes[i];
    const int64_t h = out_plane.shape[0];
    out_plane.byte_offset = static_cast<int64_t>(offset);
    out_plane.strides_bytes = {static_cast<int64_t>(row_bytes), 1};
    out_planes.push_back(std::move(out_plane));
    offset += row_bytes * static_cast<std::size_t>(h);
  }
  return out_planes;
}

bool copy_tensor_payload(const NeatTensor& src, uint8_t* dst, std::size_t dst_size) {
  if (!dst) return false;
  if (src.is_nv12()) {
    return src.copy_nv12_contiguous_to(dst, dst_size);
  }
  if (src.is_i420()) {
    return src.copy_i420_contiguous_to(dst, dst_size);
  }
  const std::size_t dense_bytes = src.dense_bytes_tight();
  if (src.is_dense() && dense_bytes > 0) {
    return src.copy_dense_bytes_tight_to(dst, dst_size);
  }
  return src.copy_payload_bytes_to(dst, dst_size);
}

std::size_t compute_payload_bytes(const NeatTensor& src) {
  if (src.is_nv12()) {
    const std::size_t bytes = src.nv12_required_bytes();
    if (bytes > 0) return bytes;
  }
  if (src.is_i420()) {
    const std::size_t bytes = src.i420_required_bytes();
    if (bytes > 0) return bytes;
  }
  if (src.is_dense()) {
    const std::size_t bytes = src.dense_bytes_tight();
    if (bytes > 0) return bytes;
  }
  return 0;
}

std::size_t segment_total_bytes(const std::vector<NeatSegment>& segments) {
  std::size_t total = 0;
  for (const auto& seg : segments) {
    total += seg.size_bytes;
  }
  return total;
}

bool segments_contain_names(const std::vector<NeatSegment>& segments,
                            const std::vector<std::string>& names) {
  for (const auto& required : names) {
    bool found = false;
    for (const auto& seg : segments) {
      if (seg.name == required) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

std::vector<NeatSegment> resolve_segments(const NeatTensor& src,
                                          const std::vector<NeatSegment>* required_segments,
                                          const std::vector<std::string>* required_names,
                                          std::size_t payload_bytes) {
  std::vector<NeatSegment> segments;

  if (required_segments && !required_segments->empty()) {
    segments = *required_segments;
  } else if (required_names && !required_names->empty()) {
    if (!src.storage || src.storage->sima_segments.empty()) {
      throw std::runtime_error("transfer: required segment names but no segment layout");
    }
    segments = src.storage->sima_segments;
  } else if (src.storage && !src.storage->sima_segments.empty()) {
    segments = src.storage->sima_segments;
  }

  if (segments.empty()) {
    NeatSegment seg;
    seg.name = "tensor";
    seg.size_bytes = payload_bytes;
    segments.push_back(std::move(seg));
  }

  if (required_names && !required_names->empty()) {
    if (!segments_contain_names(segments, *required_names)) {
      throw std::runtime_error("transfer: missing required segments");
    }
  }

  const std::size_t total = segment_total_bytes(segments);
  if (payload_bytes == 0 || total < payload_bytes) {
    throw std::runtime_error("transfer: segment layout too small for payload");
  }

  return segments;
}

std::uint64_t target_flags_from_device(const NeatDevice& target) {
  if (target.type == NeatDeviceType::SIMA_CVU) {
    return static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_EV74);
  }
  if (target.type == NeatDeviceType::SIMA_MLA) {
    switch (target.id) {
      case 0: return static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_DMS0);
      case 1: return static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_DMS1);
      case 2: return static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_DMS2);
      case 3: return static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_TARGET_DMS3);
      default: break;
    }
  }
  return 0;
}

void copy_gst_metadata(GstBuffer* dst, GstBuffer* src) {
  if (!dst || !src) return;
  const GstBufferCopyFlags flags = static_cast<GstBufferCopyFlags>(
      GST_BUFFER_COPY_METADATA | GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_FLAGS);
  gst_buffer_copy_into(dst, src, flags, 0, -1);
}

NeatTensor finalize_transfer_tensor(const NeatTensor& src,
                                    const std::shared_ptr<NeatStorage>& storage,
                                    const std::vector<NeatSegment>& segments) {
  if (!storage) {
    throw std::runtime_error("transfer: missing destination storage");
  }

  const std::size_t elem = dtype_bytes(src.dtype);
  if (elem == 0) {
    throw std::runtime_error("transfer: unknown element size");
  }

  NeatTensor out = src;
  out.storage = storage;
  out.device = storage->device;
  out.read_only = false;
  out.byte_offset = 0;
  out.strides_bytes = contiguous_strides(out.shape, elem);
  if (out.is_composite()) {
    out.planes = contiguous_planes(src, elem);
  }

  out.storage->sima_segments = segments;
  return out;
}

} // namespace

NeatTransferPoolStats neat_transfer_pool_stats() {
  PoolCache& cache = pool_cache();
  std::lock_guard<std::mutex> lock(cache.mu);
  NeatTransferPoolStats stats;
  stats.hits = cache.hits;
  stats.misses = cache.misses;
  stats.entries = cache.pools.size();
  return stats;
}

NeatTensor transfer_to_device(const NeatTensor& src,
                              const NeatDevice& target,
                              const std::vector<NeatSegment>* required_segments,
                              const std::vector<std::string>* required_segment_names) {
  const std::uint64_t target_flags = target_flags_from_device(target);
  if (target_flags == 0) {
    throw std::runtime_error("transfer: unsupported target device");
  }

#if !SIMA_HAS_SIMAAI_POOL
  throw std::runtime_error("transfer: simaai buffer pool unavailable");
#else
  std::uint64_t mem_flags = 0;
  if (src.storage) {
    mem_flags = src.storage->sima_mem_flags;
  }
  if (mem_flags == 0) {
    mem_flags = static_cast<std::uint64_t>(GST_SIMAAI_MEMORY_FLAG_CACHED);
  }

  std::size_t payload_bytes = compute_payload_bytes(src);
  if (payload_bytes == 0) {
    NeatMapping mapping = src.map_read();
    if (!mapping.data) {
      throw std::runtime_error("transfer: source mapping failed");
    }
    payload_bytes = mapping.size_bytes;
  }
  if (payload_bytes == 0) {
    throw std::runtime_error("transfer: unknown payload size");
  }

  const std::vector<NeatSegment> segments =
      resolve_segments(src, required_segments, required_segment_names, payload_bytes);

  std::shared_ptr<GstBufferPool> pool = get_pool(target_flags, mem_flags, segments);
  if (!pool) {
    throw std::runtime_error("transfer: buffer pool allocation failed");
  }

  GstBuffer* dst = nullptr;
  if (gst_buffer_pool_acquire_buffer(pool.get(), &dst, nullptr) != GST_FLOW_OK || !dst) {
    throw std::runtime_error("transfer: buffer pool acquire failed");
  }

  GstBuffer* src_buf = nullptr;
  if (src.storage && src.storage->kind == NeatStorageKind::GstSample) {
    src_buf = buffer_from_tensor_holder(src.storage->holder);
  }
  if (src_buf) {
    copy_gst_metadata(dst, src_buf);
    gst_buffer_unref(src_buf);
  }

  GstMapInfo map{};
  if (!gst_buffer_map(dst, &map, GST_MAP_WRITE)) {
    gst_buffer_unref(dst);
    throw std::runtime_error("transfer: destination map failed");
  }

  bool ok = false;
  try {
    ok = copy_tensor_payload(src, static_cast<uint8_t*>(map.data), map.size);
  } catch (const std::exception&) {
    gst_buffer_unmap(dst, &map);
    gst_buffer_unref(dst);
    throw;
  }
  gst_buffer_unmap(dst, &map);
  if (!ok) {
    gst_buffer_unref(dst);
    throw std::runtime_error("transfer: payload copy failed");
  }

  GstSample* sample = gst_sample_new(dst, nullptr, nullptr, nullptr);
  if (!sample) {
    gst_buffer_unref(dst);
    throw std::runtime_error("transfer: failed to wrap GstSample");
  }

  auto storage = make_gst_sample_storage(sample);
  gst_sample_unref(sample);
  if (!storage) {
    throw std::runtime_error("transfer: failed to create storage");
  }

  return finalize_transfer_tensor(src, storage, segments);
#endif
}

NeatTensor transfer_to_cpu(const NeatTensor& src) {
  std::size_t payload_bytes = compute_payload_bytes(src);
  if (payload_bytes == 0) {
    NeatMapping mapping = src.map_read();
    if (!mapping.data) {
      throw std::runtime_error("transfer: source mapping failed");
    }
    payload_bytes = mapping.size_bytes;
  }
  if (payload_bytes == 0) {
    throw std::runtime_error("transfer: unknown payload size");
  }

  auto storage = make_cpu_owned_storage(payload_bytes);
  NeatMapping dst_map = storage->map(NeatMapMode::Write);
  if (!dst_map.data) {
    throw std::runtime_error("transfer: destination map failed");
  }
  if (!copy_tensor_payload(src, static_cast<uint8_t*>(dst_map.data), dst_map.size_bytes)) {
    throw std::runtime_error("transfer: payload copy failed");
  }

  return finalize_transfer_tensor(src, storage, {});
}

} // namespace sima::pipeline_internal
