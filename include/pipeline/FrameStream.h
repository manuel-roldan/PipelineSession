// include/pipeline/FrameStream.h
#pragma once

#include "sima/pipeline/PipelineReport.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

typedef struct _GstElement GstElement;

namespace sima {

// Owned contiguous NV12 (tightly packed: Y then UV)
struct FrameNV12 {
  int width = 0;
  int height = 0;

  int64_t pts_ns = -1;
  int64_t dts_ns = -1;
  int64_t duration_ns = -1;
  bool keyframe = false;

  // size = width * height * 3/2 (tightly packed)
  std::vector<uint8_t> nv12;
};

// Zero-copy view into a mapped GstVideoFrame.
// Pointers remain valid as long as `holder` is kept alive.
struct FrameNV12Ref {
  int width = 0;
  int height = 0;

  int64_t pts_ns = -1;
  int64_t dts_ns = -1;
  int64_t duration_ns = -1;
  bool keyframe = false;

  const uint8_t* y = nullptr;
  const uint8_t* uv = nullptr;

  int y_stride = 0;
  int uv_stride = 0;

  std::string caps_string;

  // Opaque keepalive (typically a shared_ptr<SampleHolder>).
  std::shared_ptr<void> holder;
};

class FrameStream {
 public:
  FrameStream() = default;
  FrameStream(GstElement* pipeline, GstElement* appsink);

  FrameStream(const FrameStream&) = delete;
  FrameStream& operator=(const FrameStream&) = delete;

  FrameStream(FrameStream&& other) noexcept;
  FrameStream& operator=(FrameStream&& other) noexcept;

  ~FrameStream();

  explicit operator bool() const noexcept { return pipeline_ != nullptr && appsink_ != nullptr; }

  // Zero-copy: returns pointers into the underlying mapped GstVideoFrame.
  // The returned FrameNV12Ref keeps the sample alive via `holder`.
  std::optional<FrameNV12Ref> next(int timeout_ms);

  // Copy: returns tightly packed NV12 (Y then UV), stride-correct.
  std::optional<FrameNV12> next_copy(int timeout_ms);

  // Snapshot pipeline diagnostics (the `heavy` flag is kept for API compatibility).
  PipelineReport report_snapshot(bool heavy) const;

  void close();
  void kill() { close(); }

  // Debug helpers used by PipelineSession.
  void set_debug_pipeline(std::string name) { debug_pipeline_ = std::move(name); }
  void set_diag(std::shared_ptr<void> diag) { diag_ = std::move(diag); }
  void set_guard(std::shared_ptr<void> guard) { guard_ = std::move(guard); }

 private:
  GstElement* pipeline_ = nullptr;
  GstElement* appsink_ = nullptr;

  std::string debug_pipeline_;
  std::shared_ptr<void> diag_;
  std::shared_ptr<void> guard_;
};

} // namespace sima
