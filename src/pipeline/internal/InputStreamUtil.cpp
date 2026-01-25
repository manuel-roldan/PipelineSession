// src/pipeline/internal/InputStreamUtil.cpp
#include "InputStreamUtil.h"

#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "sima/nodes/io/InputAppSrc.h"

#include <gst/gst.h>

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

#if __has_include(<simaai/gstsimaaibufferpool.h>)
#include <simaai/gstsimaaibufferpool.h>
#define SIMA_HAS_SIMAAI_POOL 1
#else
#define SIMA_HAS_SIMAAI_POOL 0
#endif

namespace sima {
namespace {

std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

size_t dtype_bytes(TensorDType dtype) {
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

int shape_dim(const std::vector<int64_t>& shape, size_t idx) {
  if (shape.size() <= idx) return -1;
  const int64_t v = shape[idx];
  return (v > 0) ? static_cast<int>(v) : -1;
}

std::string fmt_from_neat_image(const NeatTensor& input) {
  if (!input.semantic.image.has_value()) return {};
  switch (input.semantic.image->format) {
    case NeatImageSpec::PixelFormat::RGB: return "RGB";
    case NeatImageSpec::PixelFormat::BGR: return "BGR";
    case NeatImageSpec::PixelFormat::GRAY8: return "GRAY8";
    case NeatImageSpec::PixelFormat::NV12: return "NV12";
    case NeatImageSpec::PixelFormat::I420: return "I420";
    case NeatImageSpec::PixelFormat::UNKNOWN: return {};
  }
  return {};
}

std::string fmt_from_dtype(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::UInt8: return "UINT8";
    case TensorDType::Int8: return "INT8";
    case TensorDType::UInt16: return "UINT16";
    case TensorDType::Int16: return "INT16";
    case TensorDType::Int32: return "INT32";
    case TensorDType::BFloat16: return "BF16";
    case TensorDType::Float32: return "FP32";
    case TensorDType::Float64: return "FP64";
  }
  return {};
}

bool neat_plane_is_tight(const NeatPlane& plane, TensorDType dtype) {
  if (plane.shape.size() < 2 || plane.strides_bytes.size() < 2) return false;
  const int64_t h = plane.shape[0];
  const int64_t w = plane.shape[1];
  if (h <= 0 || w <= 0) return false;
  const size_t elem = dtype_bytes(dtype);
  const int64_t expected_stride = static_cast<int64_t>(w * elem);
  if (plane.strides_bytes[0] != expected_stride) return false;
  if (plane.strides_bytes[1] != static_cast<int64_t>(elem)) return false;
  return true;
}

size_t neat_plane_bytes_tight(const NeatPlane& plane, TensorDType dtype) {
  if (plane.shape.size() < 2) return 0;
  const int64_t h = plane.shape[0];
  const int64_t w = plane.shape[1];
  if (h <= 0 || w <= 0) return 0;
  const size_t elem = dtype_bytes(dtype);
  return static_cast<size_t>(w) * static_cast<size_t>(h) * elem;
}

size_t neat_dense_bytes_tight(const NeatTensor& input) {
  if (!input.is_dense() || input.shape.empty()) return 0;
  size_t total = dtype_bytes(input.dtype);
  for (const auto dim : input.shape) {
    if (dim <= 0) return 0;
    total *= static_cast<size_t>(dim);
  }
  return total;
}

size_t neat_tensor_bytes_tight(const NeatTensor& input) {
  if (input.is_composite()) {
    size_t total = 0;
    for (const auto& plane : input.planes) {
      total += neat_plane_bytes_tight(plane, input.dtype);
    }
    return total;
  }
  return neat_dense_bytes_tight(input);
}

void debug_pool_log(const char* msg) {
  if (pipeline_internal::env_bool("SIMA_DEBUG_INPUT_POOL", false)) {
    std::fprintf(stderr, "%s\n", msg);
  }
}

void debug_pool_timing(const char* stage,
                       const InputAppSrcOptions& opt,
                       size_t bytes,
                       const std::chrono::steady_clock::time_point& start,
                       bool ok,
                       bool used_pool) {
  if (!pipeline_internal::env_bool("SIMA_INPUTSTREAM_ALLOC_DEBUG", false)) return;
  const auto end = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(end - start).count();
  std::fprintf(stderr,
               "[DBG] input_buffer %s bytes=%zu pool=%s ok=%d min=%d max=%d ms=%.3f\n",
               stage,
               bytes,
               used_pool ? "true" : "false",
               ok ? 1 : 0,
               opt.pool_min_buffers,
               opt.pool_max_buffers,
               ms);
}

void free_simaai_pool(GstBufferPool* pool) {
#if SIMA_HAS_SIMAAI_POOL
  if (!pool) return;
  gst_simaai_free_buffer_pool(pool);
#else
  (void)pool;
#endif
}

} // namespace

InputCapsConfig infer_input_caps(const InputAppSrcOptions& opt,
                                 const cv::Mat& input) {
  if (input.empty()) {
    throw std::invalid_argument("run(input): input frame is empty");
  }

  InputCapsConfig out;
  std::string media = opt.media_type;
  if (media.empty()) {
    if (input.type() == CV_32FC1 || input.type() == CV_32FC3) {
      media = "application/vnd.simaai.tensor";
    } else {
      media = "video/x-raw";
    }
  }
  out.media_type = media;

  const int in_w = input.cols;
  const int in_h = input.rows;
  const int in_c = input.channels();

  const bool is_video = (out.media_type == "video/x-raw");
  const bool is_tensor = (out.media_type == "application/vnd.simaai.tensor");

  if (!is_video && !is_tensor) {
    throw std::invalid_argument("run(input): unsupported media_type: " + out.media_type);
  }

  std::string fmt = upper_copy(opt.format);
  if (is_video) {
    if (fmt.empty()) {
      fmt = (in_c == 1) ? "GRAY8" : "BGR";
    }
    if (fmt == "GRAY") fmt = "GRAY8";

    if (fmt != "RGB" && fmt != "BGR" && fmt != "GRAY8") {
      throw std::invalid_argument("run(input): unsupported video format: " + fmt);
    }

    if ((fmt == "RGB" || fmt == "BGR") && input.type() != CV_8UC3) {
      throw std::invalid_argument("run(input): expected CV_8UC3 for video/x-raw " + fmt);
    }
    if (fmt == "GRAY8" && input.type() != CV_8UC1) {
      throw std::invalid_argument("run(input): expected CV_8UC1 for video/x-raw GRAY8");
    }

    out.width = (opt.width > 0) ? opt.width : in_w;
    out.height = (opt.height > 0) ? opt.height : in_h;
    out.depth = (opt.depth > 0) ? opt.depth : in_c;
    if (opt.width > 0 && opt.width != in_w) {
      throw std::invalid_argument("run(input): input width does not match InputAppSrcOptions");
    }
    if (opt.height > 0 && opt.height != in_h) {
      throw std::invalid_argument("run(input): input height does not match InputAppSrcOptions");
    }
    if (opt.depth > 0 && opt.depth != in_c) {
      throw std::invalid_argument("run(input): input depth does not match InputAppSrcOptions");
    }
  } else {
    if (fmt.empty()) fmt = "FP32";
    if (fmt != "FP32") {
      throw std::invalid_argument("run(input): only FP32 tensor input is supported");
    }
    if (input.type() != CV_32FC1 && input.type() != CV_32FC3) {
      throw std::invalid_argument("run(input): tensor input must be CV_32FC1 or CV_32FC3");
    }
    out.width = (opt.width > 0) ? opt.width : in_w;
    out.height = (opt.height > 0) ? opt.height : in_h;
    out.depth = (opt.depth > 0) ? opt.depth : in_c;
    if (opt.width > 0 && opt.width != in_w) {
      throw std::invalid_argument("run(input): tensor input width does not match InputAppSrcOptions");
    }
    if (opt.height > 0 && opt.height != in_h) {
      throw std::invalid_argument("run(input): tensor input height does not match InputAppSrcOptions");
    }
    if (opt.depth > 0 && opt.depth != in_c) {
      throw std::invalid_argument("run(input): tensor input depth does not match InputAppSrcOptions");
    }
  }

  out.format = fmt;
  out.bytes = input.total() * input.elemSize();
  return out;
}

InputCapsConfig infer_input_caps(const InputAppSrcOptions& opt,
                                 const NeatTensor& input) {
  if (!input.storage) {
    throw std::invalid_argument("run(input): NeatTensor missing storage");
  }
  if (input.device.type != NeatDeviceType::CPU) {
    throw std::invalid_argument("run(input): NeatTensor must be on CPU");
  }

  InputCapsConfig out;
  std::string media = opt.media_type;
  if (media.empty()) {
    media = input.semantic.image.has_value()
        ? "video/x-raw"
        : "application/vnd.simaai.tensor";
  }
  out.media_type = media;

  if (out.media_type == "video/x-raw") {
    std::string fmt = upper_copy(opt.format);
    if (fmt.empty()) fmt = upper_copy(fmt_from_neat_image(input));
    if (fmt == "GRAY") fmt = "GRAY8";
    if (fmt.empty()) {
      throw std::invalid_argument("run(input): NeatTensor video input missing format");
    }
    if (fmt != "RGB" && fmt != "BGR" && fmt != "GRAY8" &&
        fmt != "NV12" && fmt != "I420") {
      throw std::invalid_argument("run(input): unsupported video format: " + fmt);
    }
    if (!input.semantic.image.has_value()) {
      throw std::invalid_argument("run(input): NeatTensor video input missing ImageSpec");
    }
    if (input.dtype != TensorDType::UInt8) {
      throw std::invalid_argument("run(input): NeatTensor video input must be UInt8");
    }

    int h = shape_dim(input.shape, 0);
    int w = shape_dim(input.shape, 1);
    int d = shape_dim(input.shape, 2);
    if (input.is_composite() && !input.planes.empty()) {
      const NeatPlane& y = input.planes.front();
      if (y.shape.size() >= 2) {
        h = (h > 0) ? h : static_cast<int>(y.shape[0]);
        w = (w > 0) ? w : static_cast<int>(y.shape[1]);
      }
    }

    out.width = (opt.width > 0) ? opt.width : w;
    out.height = (opt.height > 0) ? opt.height : h;
    out.depth = (opt.depth > 0) ? opt.depth : d;

    if (out.width <= 0 || out.height <= 0) {
      throw std::invalid_argument("run(input): NeatTensor video input missing width/height");
    }
    if (opt.width > 0 && w > 0 && opt.width != w) {
      throw std::invalid_argument("run(input): NeatTensor width does not match InputAppSrcOptions");
    }
    if (opt.height > 0 && h > 0 && opt.height != h) {
      throw std::invalid_argument("run(input): NeatTensor height does not match InputAppSrcOptions");
    }
    if (fmt == "RGB" || fmt == "BGR") {
      if (out.depth <= 0) out.depth = 3;
      if (opt.depth > 0 && opt.depth != 3) {
        throw std::invalid_argument("run(input): NeatTensor depth does not match RGB/BGR");
      }
    } else if (fmt == "GRAY8") {
      if (out.depth <= 0) out.depth = 1;
      if (opt.depth > 0 && opt.depth != 1) {
        throw std::invalid_argument("run(input): NeatTensor depth does not match GRAY8");
      }
    }

    out.format = fmt;
    out.bytes = neat_tensor_bytes_tight(input);
    if (out.bytes == 0) {
      throw std::invalid_argument("run(input): NeatTensor has invalid byte size");
    }
    return out;
  }

  if (out.media_type == "application/x-simaai.sample") {
    out.format = "SAMPLE";
    out.bytes = neat_tensor_bytes_tight(input);
    if (out.bytes == 0) {
      out.bytes = 1;
    }
    return out;
  }

  if (out.media_type != "application/vnd.simaai.tensor") {
    throw std::invalid_argument("run(input): unsupported media_type: " + out.media_type);
  }
  if (!input.is_dense()) {
    throw std::invalid_argument("run(input): NeatTensor tensor input must be dense");
  }

  std::string fmt = upper_copy(opt.format);
  if (fmt.empty()) fmt = upper_copy(fmt_from_dtype(input.dtype));
  if (fmt.empty()) {
    throw std::invalid_argument("run(input): NeatTensor tensor input missing format");
  }

  const int shape_h = shape_dim(input.shape, 0);
  const int shape_w = shape_dim(input.shape, 1);
  const int shape_d = shape_dim(input.shape, 2);

  out.width = (opt.width > 0) ? opt.width : shape_w;
  out.height = (opt.height > 0) ? opt.height : shape_h;
  out.depth = (opt.depth > 0) ? opt.depth : shape_d;

  if (out.width <= 0 || out.height <= 0) {
    throw std::invalid_argument("run(input): NeatTensor tensor input missing width/height");
  }

  out.format = fmt;
  out.bytes = neat_dense_bytes_tight(input);
  if (out.bytes == 0) {
    throw std::invalid_argument("run(input): NeatTensor tensor input missing byte size");
  }
  return out;
}

void validate_input_matches_caps(const InputCapsConfig& cfg,
                                 const InputAppSrcOptions& opt,
                                 const cv::Mat& input,
                                 const char* where) {
  if (input.empty()) {
    throw std::invalid_argument(std::string(where) + ": input frame is empty");
  }
  if (input.cols != cfg.width || input.rows != cfg.height) {
    throw std::invalid_argument(std::string(where) + ": input size does not match appsrc caps");
  }

  const bool is_video = (cfg.media_type == "video/x-raw");
  const bool is_tensor = (cfg.media_type == "application/vnd.simaai.tensor");
  if (!is_video && !is_tensor) {
    throw std::invalid_argument(std::string(where) + ": unsupported media_type");
  }

  const std::string fmt = upper_copy(cfg.format);
  if (is_video) {
    if ((fmt == "RGB" || fmt == "BGR") && input.type() != CV_8UC3) {
      throw std::invalid_argument(std::string(where) + ": expected CV_8UC3");
    }
    if (fmt == "GRAY8" && input.type() != CV_8UC1) {
      throw std::invalid_argument(std::string(where) + ": expected CV_8UC1");
    }
  } else {
    if (fmt != "FP32") {
      throw std::invalid_argument(std::string(where) + ": only FP32 tensor input supported");
    }
    if (input.type() != CV_32FC1 && input.type() != CV_32FC3) {
      throw std::invalid_argument(std::string(where) + ": tensor input must be CV_32FC1/CV_32FC3");
    }
  }

  if (opt.depth > 0 && input.channels() != opt.depth) {
    throw std::invalid_argument(std::string(where) + ": input channels do not match appsrc caps");
  }

  const size_t bytes = input.total() * input.elemSize();
  if (bytes != cfg.bytes) {
    throw std::invalid_argument(std::string(where) + ": input byte size mismatch");
  }
}

void validate_input_matches_caps(const InputCapsConfig& cfg,
                                 const InputAppSrcOptions& opt,
                                 const NeatTensor& input,
                                 const char* where) {
  if (!input.storage) {
    throw std::invalid_argument(std::string(where) + ": NeatTensor missing storage");
  }
  if (input.device.type != NeatDeviceType::CPU) {
    throw std::invalid_argument(std::string(where) + ": NeatTensor must be on CPU");
  }

  if (cfg.media_type == "video/x-raw") {
    const std::string fmt = upper_copy(cfg.format);
    if (fmt != "RGB" && fmt != "BGR" && fmt != "GRAY8" &&
        fmt != "NV12" && fmt != "I420") {
      throw std::invalid_argument(std::string(where) + ": unsupported video format");
    }
    if (!input.semantic.image.has_value()) {
      throw std::invalid_argument(std::string(where) + ": missing ImageSpec");
    }
    if (input.dtype != TensorDType::UInt8) {
      throw std::invalid_argument(std::string(where) + ": video input must be UInt8");
    }
    if (cfg.width > 0 && input.shape.size() > 1 &&
        cfg.width != static_cast<int>(input.shape[1])) {
      throw std::invalid_argument(std::string(where) + ": input width does not match caps");
    }
    if (cfg.height > 0 && input.shape.size() > 0 &&
        cfg.height != static_cast<int>(input.shape[0])) {
      throw std::invalid_argument(std::string(where) + ": input height does not match caps");
    }
    if ((fmt == "RGB" || fmt == "BGR") && cfg.depth > 0 &&
        input.shape.size() > 2 && cfg.depth != static_cast<int>(input.shape[2])) {
      throw std::invalid_argument(std::string(where) + ": input depth does not match caps");
    }

    if (input.is_composite()) {
      if ((fmt == "NV12" && input.planes.size() != 2) ||
          (fmt == "I420" && input.planes.size() != 3)) {
        throw std::invalid_argument(std::string(where) + ": invalid plane count");
      }
      for (const auto& plane : input.planes) {
        if (!neat_plane_is_tight(plane, input.dtype)) {
          throw std::invalid_argument(std::string(where) + ": non-tight plane layout");
        }
      }
      size_t expected_offset = 0;
      for (const auto& plane : input.planes) {
        if (plane.byte_offset != static_cast<int64_t>(expected_offset)) {
          throw std::invalid_argument(std::string(where) + ": non-contiguous plane offsets");
        }
        expected_offset += neat_plane_bytes_tight(plane, input.dtype);
      }
      if (input.storage->size_bytes > 0 &&
          input.storage->size_bytes < expected_offset) {
        throw std::invalid_argument(std::string(where) + ": storage too small");
      }
      if (expected_offset != cfg.bytes) {
        throw std::invalid_argument(std::string(where) + ": input byte size mismatch");
      }
      return;
    }

    if (!input.is_contiguous()) {
      throw std::invalid_argument(std::string(where) + ": non-contiguous NeatTensor input");
    }
    const size_t bytes = neat_dense_bytes_tight(input);
    if (bytes != cfg.bytes) {
      throw std::invalid_argument(std::string(where) + ": input byte size mismatch");
    }
    return;
  }

  if (cfg.media_type != "application/vnd.simaai.tensor") {
    throw std::invalid_argument(std::string(where) + ": unsupported media_type");
  }
  if (!input.is_dense()) {
    throw std::invalid_argument(std::string(where) + ": tensor input must be dense");
  }
  if (!input.is_contiguous()) {
    throw std::invalid_argument(std::string(where) + ": non-contiguous tensor input");
  }
  const size_t bytes = neat_dense_bytes_tight(input);
  if (bytes != cfg.bytes) {
    throw std::invalid_argument(std::string(where) + ": input byte size mismatch");
  }
}

GstCaps* build_input_caps(const InputCapsConfig& cfg,
                          const InputAppSrcOptions& opt) {
  return build_caps_with_override("build_input_caps",
                                  cfg.media_type,
                                  cfg.format,
                                  cfg.width,
                                  cfg.height,
                                  cfg.depth,
                                  opt.caps_override);
}

GstCaps* build_caps_with_override(const char* where,
                                  const std::string& media_type,
                                  const std::string& format,
                                  int width,
                                  int height,
                                  int depth,
                                  const std::string& caps_override) {
  const char* tag = where ? where : "build_caps_with_override";
  // Prefer caps_override so appsrc caps match plugin expectations even when
  // plugin-reported caps are incomplete or misleading.
  if (!caps_override.empty()) {
    GstCaps* caps = gst_caps_from_string(caps_override.c_str());
    if (!caps) {
      throw std::runtime_error(std::string(tag) + ": invalid caps_override: " +
                               caps_override);
    }
    return caps;
  }

  const std::string media = media_type.empty() ? "video/x-raw" : media_type;
  if (media == "video/x-raw") {
    if (depth > 0) {
      return gst_caps_new_simple(
          "video/x-raw",
          "format", G_TYPE_STRING, format.c_str(),
          "width", G_TYPE_INT, width,
          "height", G_TYPE_INT, height,
          "depth", G_TYPE_INT, depth,
          nullptr);
    }
    return gst_caps_new_simple(
        "video/x-raw",
        "format", G_TYPE_STRING, format.c_str(),
        "width", G_TYPE_INT, width,
        "height", G_TYPE_INT, height,
        nullptr);
  }

  return gst_caps_new_simple(
      "application/vnd.simaai.tensor",
      "format", G_TYPE_STRING, format.c_str(),
      "width", G_TYPE_INT, width,
      "height", G_TYPE_INT, height,
      "depth", G_TYPE_INT, depth,
      nullptr);
}

GstBuffer* allocate_input_buffer(size_t bytes,
                                 const InputAppSrcOptions& opt,
                                 InputBufferPoolGuard& guard) {
#if SIMA_HAS_SIMAAI_POOL
  if (opt.use_simaai_pool) {
    GstBufferPool* pool = guard.pool.get();
    if (!pool) {
      const auto t_create_start = std::chrono::steady_clock::now();
      gst_simaai_segment_memory_init_once();
      GstMemoryFlags flags = static_cast<GstMemoryFlags>(
          GST_SIMAAI_MEMORY_TARGET_EV74 | GST_SIMAAI_MEMORY_FLAG_CACHED);
      GstBufferPool* new_pool = gst_simaai_allocate_buffer_pool(
          /*allocator_user_data=*/nullptr,
          gst_simaai_memory_get_segment_allocator(),
          bytes,
          opt.pool_min_buffers,
          opt.pool_max_buffers,
          flags);
      if (new_pool) {
        guard.pool = std::unique_ptr<GstBufferPool, void(*)(GstBufferPool*)>(
            new_pool, free_simaai_pool);
        pool = new_pool;
      }
      debug_pool_timing("pool_create", opt, bytes, t_create_start, pool != nullptr, true);
    }

    if (pool) {
      const auto t_acquire_start = std::chrono::steady_clock::now();
      GstBuffer* buf = nullptr;
      if (gst_buffer_pool_acquire_buffer(pool, &buf, nullptr) == GST_FLOW_OK && buf) {
        debug_pool_timing("pool_acquire", opt, bytes, t_acquire_start, true, true);
        return buf;
      }
      debug_pool_timing("pool_acquire", opt, bytes, t_acquire_start, false, true);
      debug_pool_log("InputAppSrc: simaai pool acquired but buffer allocation failed.");
      return nullptr;
    }
    debug_pool_log("InputAppSrc: simaai pool allocation failed.");
    return nullptr;
  }
#else
  (void)opt;
  (void)guard;
#endif

  const auto t_alloc_start = std::chrono::steady_clock::now();
  GstBuffer* buf = gst_buffer_new_allocate(nullptr, bytes, nullptr);
  debug_pool_timing("system_alloc", opt, bytes, t_alloc_start, buf != nullptr, false);
  return buf;
}

int64_t next_input_frame_id() {
  static std::atomic<int64_t> next_id{0};
  return next_id.fetch_add(1);
}

bool maybe_add_simaai_meta(GstBuffer* buffer,
                           int64_t frame_id,
                           const InputAppSrcOptions& opt) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer || !opt.use_simaai_pool) return false;
  dump_sima_meta(buffer, "maybe_add_simaai_meta(before)");
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (meta) {
    GstStructure* s = gst_custom_meta_get_structure(meta);
#if defined(GST_STRUCTURE_IS_WRITABLE)
    if (!s || !GST_STRUCTURE_IS_WRITABLE(s)) {
      gst_buffer_remove_meta(buffer, &meta->meta);
      meta = nullptr;
    }
#else
    if (!s) meta = nullptr;
#endif
  }
  if (!meta) {
    meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  }
  if (!meta) {
    if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false)) {
      std::fprintf(stderr, "[DBG] GstSimaMeta add failed (buffer=%p)\n", buffer);
    }
    return false;
  }
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) {
    if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false)) {
      std::fprintf(stderr, "[DBG] GstSimaMeta missing structure (buffer=%p)\n", buffer);
    }
    return false;
  }
  const std::string name = opt.buffer_name.empty() ? "decoder" : opt.buffer_name;
  gint64 phys_addr =
      gst_simaai_segment_memory_get_phys_addr(gst_buffer_peek_memory(buffer, 0));
  gst_structure_set(s,
                    "buffer-id", G_TYPE_INT64, phys_addr,
                    "buffer-name", G_TYPE_STRING, name.c_str(),
                    "buffer-offset", G_TYPE_INT64, static_cast<gint64>(0),
                    "frame-id", G_TYPE_INT64, static_cast<gint64>(frame_id),
                    "stream-id", G_TYPE_STRING, "0",
                    "timestamp", G_TYPE_UINT64, static_cast<guint64>(0),
                    nullptr);
  if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false)) {
    std::fprintf(stderr,
                 "[DBG] GstSimaMeta set name=%s phys=%lld frame=%lld\n",
                 name.c_str(),
                 static_cast<long long>(phys_addr),
                 static_cast<long long>(frame_id));
  }
  dump_sima_meta(buffer, "maybe_add_simaai_meta(after)");
  return true;
#else
  (void)buffer;
  (void)frame_id;
  (void)opt;
  return false;
#endif
}

void maybe_update_simaai_meta_name(GstBuffer* buffer, const std::string& name) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer || name.empty()) return;
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (!meta) return;
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) return;
#if defined(GST_STRUCTURE_IS_WRITABLE)
  if (!GST_STRUCTURE_IS_WRITABLE(s)) return;
#else
  return;
#endif
  gst_structure_set(s,
                    "buffer-name", G_TYPE_STRING, name.c_str(),
                    nullptr);
#else
  (void)buffer;
  (void)name;
#endif
}

void dump_buffer_memories(GstBuffer* buffer, const char* label) {
  if (!pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false)) return;
  const char* tag = label ? label : "buffer";
  if (!buffer) {
    std::fprintf(stderr, "[DBG] %s mem_count=0 (null buffer)\n", tag);
    return;
  }
  const guint n_mems = gst_buffer_n_memory(buffer);
  std::fprintf(stderr, "[DBG] %s mem_count=%u\n", tag, n_mems);
  for (guint i = 0; i < n_mems; ++i) {
    GstMemory* mem = gst_buffer_peek_memory(buffer, i);
    if (!mem) {
      std::fprintf(stderr, "[DBG] %s mem[%u]=null\n", tag, i);
      continue;
    }
    gsize offset = 0;
    gsize maxsize = 0;
    const gsize size = gst_memory_get_sizes(mem, &offset, &maxsize);
    std::fprintf(stderr,
                 "[DBG] %s mem[%u] size=%zu offset=%zu max=%zu\n",
                 tag,
                 i,
                 static_cast<size_t>(size),
                 static_cast<size_t>(offset),
                 static_cast<size_t>(maxsize));
  }
}

void dump_sima_meta(GstBuffer* buffer, const char* label) {
  if (!pipeline_internal::env_bool("SIMA_INPUTSTREAM_META_DEBUG", false)) return;
#if SIMA_HAS_SIMAAI_POOL
  const char* tag = label ? label : "buffer";
  if (!buffer) {
    std::fprintf(stderr, "[DBG] %s GstSimaMeta=missing (null buffer)\n", tag);
    return;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (!meta) {
    std::fprintf(stderr, "[DBG] %s GstSimaMeta=missing\n", tag);
    return;
  }
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) {
    std::fprintf(stderr, "[DBG] %s GstSimaMeta=missing-structure\n", tag);
    return;
  }
  bool writable = true;
#if defined(GST_STRUCTURE_IS_WRITABLE)
  writable = GST_STRUCTURE_IS_WRITABLE(s);
#endif
  const char* name = gst_structure_get_string(s, "buffer-name");
  gint64 frame_id = 0;
  gint64 phys_addr = 0;
  gst_structure_get_int64(s, "frame-id", &frame_id);
  gst_structure_get_int64(s, "buffer-id", &phys_addr);
  std::fprintf(stderr,
               "[DBG] %s GstSimaMeta present=1 writable=%d name=%s frame=%lld phys=%lld\n",
               tag,
               writable ? 1 : 0,
               name ? name : "",
               static_cast<long long>(frame_id),
               static_cast<long long>(phys_addr));
#else
  (void)buffer;
  (void)label;
#endif
}

bool update_simaai_meta_fields(GstBuffer* buffer,
                               const std::optional<int64_t>& frame_id_override,
                               const std::optional<std::string>& stream_id_override,
                               const std::optional<std::string>& buffer_name_override) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer) return false;
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (!meta) return false;
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) return false;
  bool writable = true;
#if defined(GST_STRUCTURE_IS_WRITABLE)
  writable = GST_STRUCTURE_IS_WRITABLE(s);
#else
  writable = false;
#endif
  GstStructure* snapshot = nullptr;
  if (!writable) {
    if (!gst_buffer_is_writable(buffer)) {
      return false;
    }
    snapshot = gst_structure_copy(s);
    gst_buffer_remove_meta(buffer, &meta->meta);
    meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
    s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
    if (!s) {
      if (snapshot) gst_structure_free(snapshot);
      return false;
    }
    if (snapshot) {
      const gint n_fields = gst_structure_n_fields(snapshot);
      for (gint i = 0; i < n_fields; ++i) {
        const char* fname = gst_structure_nth_field_name(snapshot, i);
        if (!fname) continue;
        const GValue* val = gst_structure_get_value(snapshot, fname);
        if (!val) continue;
        gst_structure_set_value(s, fname, val);
      }
      gst_structure_free(snapshot);
    }
  }
  if (frame_id_override.has_value()) {
    gst_structure_set(s, "frame-id", G_TYPE_INT64,
                      static_cast<gint64>(*frame_id_override), nullptr);
  }
  if (stream_id_override.has_value()) {
    gst_structure_set(s, "stream-id", G_TYPE_STRING,
                      stream_id_override->c_str(), nullptr);
  }
  if (buffer_name_override.has_value()) {
    gst_structure_set(s, "buffer-name", G_TYPE_STRING,
                      buffer_name_override->c_str(), nullptr);
  }
  return true;
#else
  (void)buffer;
  (void)frame_id_override;
  (void)stream_id_override;
  (void)buffer_name_override;
  return false;
#endif
}

GstBuffer* attach_simaai_meta_inplace(GstBuffer* buffer,
                                      const InputAppSrcOptions& opt,
                                      InputBufferPoolGuard& guard,
                                      const char* label,
                                      const std::optional<int64_t>& frame_id_override,
                                      const std::optional<std::string>& stream_id_override,
                                      const std::optional<std::string>& buffer_name_override) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer) return nullptr;
  const std::string name = buffer_name_override.value_or(
      opt.buffer_name.empty() ? "decoder" : opt.buffer_name);
  dump_sima_meta(buffer, label);

  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  bool writable = true;
#if defined(GST_STRUCTURE_IS_WRITABLE)
  writable = s && GST_STRUCTURE_IS_WRITABLE(s);
#else
  writable = s != nullptr;
#endif
  if (meta && s && writable) {
    gst_structure_set(s, "buffer-name", G_TYPE_STRING, name.c_str(), nullptr);
    if (frame_id_override.has_value()) {
      gst_structure_set(s, "frame-id", G_TYPE_INT64,
                        static_cast<gint64>(*frame_id_override), nullptr);
    }
    if (stream_id_override.has_value()) {
      gst_structure_set(s, "stream-id", G_TYPE_STRING,
                        stream_id_override->c_str(), nullptr);
    }
    dump_sima_meta(buffer, label);
    return buffer;
  }
  if (meta && s && !writable) {
    gst_buffer_remove_meta(buffer, &meta->meta);
    meta = nullptr;
    s = nullptr;
  }
  if (!meta) {
    meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
    s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  }
  if (!s) {
    std::fprintf(stderr, "[DBG] %s GstSimaMeta add failed\n", label ? label : "buffer");
    return buffer;
  }
  gint64 phys_addr = 0;
  if (gst_buffer_n_memory(buffer) > 0) {
    phys_addr = gst_simaai_segment_memory_get_phys_addr(gst_buffer_peek_memory(buffer, 0));
  }
  const gint64 frame_id = frame_id_override.has_value()
      ? static_cast<gint64>(*frame_id_override)
      : static_cast<gint64>(next_input_frame_id());
  const std::string stream_id = stream_id_override.value_or("0");
  gst_structure_set(s,
                    "buffer-id", G_TYPE_INT64, phys_addr,
                    "buffer-name", G_TYPE_STRING, name.c_str(),
                    "buffer-offset", G_TYPE_INT64, static_cast<gint64>(0),
                    "frame-id", G_TYPE_INT64, frame_id,
                    "stream-id", G_TYPE_STRING, stream_id.c_str(),
                    "timestamp", G_TYPE_UINT64, static_cast<guint64>(0),
                    nullptr);
  dump_sima_meta(buffer, label);
  return buffer;
#else
  (void)buffer;
  (void)opt;
  (void)guard;
  (void)label;
  return buffer;
#endif
}

} // namespace sima
