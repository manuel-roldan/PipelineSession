#pragma once

#include "sima/pipeline/PipelineReport.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct _GstElement;
using GstElement = _GstElement;

namespace sima {

// =============================
// Debug payload types
// =============================
enum class TapFormat {
  Unknown = 0,
  NV12,
  I420,
  RGB,
  BGR,
  GRAY8,
  H264,
  H265,
  JPEG,
  PNG,
};

struct TapVideoInfo {
  int width = 0;
  int height = 0;
  int fps_num = 0;
  int fps_den = 1;
  std::string format;
};

struct TapPacket {
  TapFormat format = TapFormat::Unknown;

  // Best-effort metadata
  bool keyframe = false;
  TapVideoInfo video;
  std::string caps_string;

  // Timing (nanoseconds). -1 means unknown/not present.
  int64_t pts_ns = -1;
  int64_t dts_ns = -1;
  int64_t duration_ns = -1;

  // Memory contract / mapping
  bool memory_mappable = true;     // if false, bytes will be empty (unless strict mode)
  std::string memory_features;     // e.g. "memory:SystemMemory" / "memory:DMABuf" / "<none>"
  std::string non_mappable_reason; // filled when memory_mappable=false
  int source_node_index = -1;      // node index for this tap point (best-effort)
  std::string source_element;      // element name where tap sample came from (appsink)

  // Raw bytes (filled when we can map/pack; may be empty for non-mappable memory)
  std::vector<uint8_t> bytes;
};

// =============================
// Data types (legacy typed frame)
// =============================
struct FrameNV12 {
  int width = 0;
  int height = 0;
  bool keyframe = false;

  int64_t pts_ns = -1;
  int64_t dts_ns = -1;
  int64_t duration_ns = -1;

  std::vector<uint8_t> nv12; // tightly packed: w*h Y + w*h/2 UV
};

// =============================
// Zero-copy view (fast path default)
// =============================
struct FrameNV12Ref {
  int width = 0;
  int height = 0;
  bool keyframe = false;

  int64_t pts_ns = -1;
  int64_t dts_ns = -1;
  int64_t duration_ns = -1;

  // NV12 planes (strided)
  const uint8_t* y = nullptr;
  const uint8_t* uv = nullptr;
  int y_stride = 0;
  int uv_stride = 0;

  std::string caps_string;
  std::shared_ptr<void> holder; // opaque RAII (unmaps + unrefs sample)
};

// =============================
// FrameStream (appsink wrapper)
// =============================
class FrameStream {
public:
  FrameStream() = default;
  FrameStream(GstElement* pipeline, GstElement* appsink);
  ~FrameStream();

  FrameStream(const FrameStream&) = delete;
  FrameStream& operator=(const FrameStream&) = delete;

  FrameStream(FrameStream&&) noexcept;
  FrameStream& operator=(FrameStream&&) noexcept;

  // Fast path (default): returns a strided NV12 view with ref-counted lifetime.
  std::optional<FrameNV12Ref> next(int timeout_ms = 1000);

  // Convenience: tight packed copy (legacy).
  std::optional<FrameNV12> next_copy(int timeout_ms = 1000);

  void close();

  const std::string& debug_pipeline() const { return debug_pipeline_; }

  // Snapshot report at any time (cheap by default; heavy adds caps/dot if enabled).
  PipelineReport report_snapshot(bool heavy = false) const;

private:
  friend class PipelineSession;
  void set_debug_pipeline(std::string s) { debug_pipeline_ = std::move(s); }
  void set_diag(std::shared_ptr<void> diag) { diag_ = std::move(diag); }

  GstElement* pipeline_ = nullptr; // owned ref
  GstElement* appsink_ = nullptr;  // owned ref
  std::string debug_pipeline_;

  std::shared_ptr<void> diag_; // opaque diagnostics context
};

// =============================
// TapStream (generic appsink wrapper)
// =============================
class TapStream {
public:
  TapStream() = default;
  TapStream(GstElement* pipeline, GstElement* appsink);
  ~TapStream();

  TapStream(const TapStream&) = delete;
  TapStream& operator=(const TapStream&) = delete;

  TapStream(TapStream&&) noexcept;
  TapStream& operator=(TapStream&&) noexcept;

  std::optional<TapPacket> next(int timeout_ms = 1000);

  void close();

  const std::string& debug_pipeline() const { return debug_pipeline_; }
  PipelineReport report_snapshot(bool heavy = false) const;

private:
  friend class PipelineSession;
  void set_debug_pipeline(std::string s) { debug_pipeline_ = std::move(s); }
  void set_diag(std::shared_ptr<void> diag) { diag_ = std::move(diag); }
  void set_tap_meta(int node_index, std::string appsink_name) {
    tap_node_index_ = node_index;
    tap_sink_name_ = std::move(appsink_name);
  }

  GstElement* pipeline_ = nullptr; // owned ref
  GstElement* appsink_ = nullptr;  // owned ref
  std::string debug_pipeline_;

  std::shared_ptr<void> diag_;
  int tap_node_index_ = -1;
  std::string tap_sink_name_;
};

} // namespace sima
