// PipelineSession.h
#pragma once

#include <gst/gst.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sima {

// =============================
// Memory contract
// =============================
enum class MemoryContract {
  // Must be CPU-mappable (typically SystemMemory); violations are hard errors.
  RequireSystemMemoryMappable = 0,

  // Prefer device/zero-copy (runner may avoid forcing SystemMemory); still report contract mismatches.
  PreferDeviceZeroCopy,

  // Allow either; if non-mappable, return empty payload with explicit reason + location.
  AllowEitherButReport,
};

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
// Lifetime rules:
// - y/uv pointers are valid as long as `holder` is alive.
// - holder is ref-counted and keeps the underlying GstSample mapped.
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
// Structured diagnostics
// =============================
struct BusMessage {
  std::string type;         // e.g. "ERROR", "WARNING", "STATE_CHANGED"
  std::string src;          // element/object name
  std::string detail;       // formatted message (incl debug if present)
  int64_t wall_time_us = 0; // monotonic
};

struct BoundaryFlowStats {
  std::string boundary_name;  // sima_b<N>
  int after_node_index = -1;  // upstream node index
  int before_node_index = -1; // downstream node index (may be -1 for terminal tap boundary)

  uint64_t in_buffers = 0;  // observed on identity:sink
  uint64_t out_buffers = 0; // observed on identity:src

  int64_t last_in_pts_ns = -1;
  int64_t last_out_pts_ns = -1;

  int64_t last_in_wall_us = 0;
  int64_t last_out_wall_us = 0;
};

struct NodeReport {
  int index = -1;
  std::string kind;                  // "FileSrc", "H264Decode", ...
  std::string user_label;            // e.g. DebugPoint name
  std::string gst_fragment;          // fragment with names
  std::vector<std::string> elements; // element names owned by this node
};

struct PipelineReport {
  std::string pipeline_string;

  std::vector<NodeReport> nodes;
  std::vector<BusMessage> bus;
  std::vector<BoundaryFlowStats> boundaries;

  // Heavy-on-failure add-ons
  std::string caps_dump;
  std::vector<std::string> dot_paths;

  // Repro helpers
  std::string repro_gst_launch; // gst-launch-1.0 -v '...'
  std::string repro_env;        // env vars suggestion (GST_DEBUG, DOT dir)
  std::string repro_note;       // human help text

  // Optional: JSON serialization for CI / support bundling
  std::string to_json() const;
};

// Exception that carries a structured report.
class PipelineError : public std::runtime_error {
public:
  PipelineError(std::string msg, PipelineReport report);
  const PipelineReport& report() const { return report_; }

private:
  PipelineReport report_;
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

// =============================
// Builder Node API
// =============================
class Node {
public:
  virtual ~Node() = default;

  // Deterministic type label (used in reports).
  virtual std::string kind() const = 0;

  // Optional human label (e.g., DebugPoint name).
  virtual std::string user_label() const { return ""; }

  // Node fragment with deterministic element names (namespace = n<idx>_...).
  virtual std::string gst_fragment(int node_index) const = 0;

  // Deterministic list of element names this node creates.
  virtual std::vector<std::string> element_names(int node_index) const = 0;

  // Optional memory contract for this node (runner may still override).
  virtual MemoryContract memory_contract() const { return MemoryContract::AllowEitherButReport; }
};

// =============================
// Raw gst-launch fragment node (escape hatch)
// =============================
class GstNode final : public Node {
public:
  explicit GstNode(std::string fragment);
  std::string kind() const override { return "GstNode"; }
  std::string user_label() const override { return fragment_; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const std::string& fragment() const { return fragment_; }

private:
  std::string fragment_;
};

// =============================
// DebugPoint node
// =============================
class DebugPoint final : public Node {
public:
  explicit DebugPoint(std::string name = "dbg");
  std::string kind() const override { return "DebugPoint"; }
  std::string user_label() const override { return name_; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const std::string& name() const { return name_; }

private:
  std::string name_ = "dbg";
};

// ---- File input nodes ----
class FileSrc final : public Node {
public:
  explicit FileSrc(std::string path);
  std::string kind() const override { return "FileSrc"; }
  std::string user_label() const override { return path_; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const std::string& path() const { return path_; }

private:
  std::string path_;
};

class JpegDec final : public Node {
public:
  std::string kind() const override { return "JpegDec"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
};

class QtDemuxVideoPad final : public Node {
public:
  explicit QtDemuxVideoPad(int video_pad_index = 0);
  std::string kind() const override { return "QtDemuxVideoPad"; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  int video_pad_index() const { return idx_; }

private:
  int idx_ = 0;
};

class Queue final : public Node {
public:
  std::string kind() const override { return "Queue"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
};

class H264ParseAu final : public Node {
public:
  std::string kind() const override { return "H264ParseAu"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
};

class VideoConvert final : public Node {
public:
  std::string kind() const override { return "VideoConvert"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
};

class VideoScale final : public Node {
public:
  std::string kind() const override { return "VideoScale"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
};

// ---- RTSP client nodes ----
class RTSPInput final : public Node {
public:
  RTSPInput(std::string url, int latency_ms = 200, bool tcp = true);
  std::string kind() const override { return "RTSPInput"; }
  std::string user_label() const override { return url_; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const std::string& url() const { return url_; }
  int latency_ms() const { return latency_ms_; }
  bool tcp() const { return tcp_; }

private:
  std::string url_;
  int latency_ms_ = 200;
  bool tcp_ = true;
};

class H264DepayParse final : public Node {
public:
  std::string kind() const override { return "H264DepayParse"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
};

class H264Decode final : public Node {
public:
  H264Decode(int sima_allocator_type = 2, std::string out_format = "NV12");
  std::string kind() const override { return "H264Decode"; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  int sima_allocator_type() const { return sima_allocator_type_; }
  const std::string& out_format() const { return out_format_; }

private:
  int sima_allocator_type_ = 2;
  std::string out_format_ = "NV12";
};

class OutputAppSink final : public Node {
public:
  std::string kind() const override { return "OutputAppSink"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
};

// ---- Server-side nodes ----
class AppSrcImage final : public Node {
public:
  AppSrcImage(std::string image_path,
              int content_w,
              int content_h,
              int enc_w,
              int enc_h,
              int fps);

  std::string kind() const override { return "AppSrcImage"; }
  std::string user_label() const override { return image_path_; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const std::string& image_path() const { return image_path_; }
  int content_w() const { return content_w_; }
  int content_h() const { return content_h_; }
  int enc_w() const { return enc_w_; }
  int enc_h() const { return enc_h_; }
  int fps() const { return fps_; }
  const std::shared_ptr<std::vector<uint8_t>>& nv12_enc() const { return nv12_enc_; }

private:
  std::string image_path_;
  int content_w_ = 0;
  int content_h_ = 0;
  int enc_w_ = 0;
  int enc_h_ = 0;
  int fps_ = 30;

  std::shared_ptr<std::vector<uint8_t>> nv12_enc_;
};

// =============================
// Typed, parameterized helpers (end-user surface)
// =============================
enum class CapsMemory {
  Any = 0,
  SystemMemory,
};

// H264 encoder node (HW). End-users should not build strings.
class H264EncodeSima final : public Node {
public:
  H264EncodeSima(int w, int h, int fps,
                 int bitrate_kbps = 400,
                 std::string profile = "baseline",
                 std::string level = "4.0");

  std::string kind() const override { return "H264EncodeSima"; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  int width() const { return w_; }
  int height() const { return h_; }
  int fps() const { return fps_; }
  int bitrate_kbps() const { return bitrate_kbps_; }
  const std::string& profile() const { return profile_; }
  const std::string& level() const { return level_; }

private:
  int w_ = 0;
  int h_ = 0;
  int fps_ = 30;

  int bitrate_kbps_ = 400;
  std::string profile_ = "baseline";
  std::string level_ = "4.0";
};

class H264Parse final : public Node {
public:
  explicit H264Parse(int config_interval = 1);
  std::string kind() const override { return "H264Parse"; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  int config_interval() const { return config_interval_; }

private:
  int config_interval_ = 1;
};

class RtpH264Pay final : public Node {
public:
  RtpH264Pay(int pt = 96, int config_interval = 1);
  std::string kind() const override { return "RtpH264Pay"; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  int pt() const { return pt_; }
  int config_interval() const { return config_interval_; }

private:
  int pt_ = 96;
  int config_interval_ = 1;
};

// =============================
// RTSP Server handle
// =============================
struct RtspServerOptions {
  std::string mount = "image";
  int port = 8554;
};

class RtspServerHandle {
public:
  RtspServerHandle() = default;
  ~RtspServerHandle();

  RtspServerHandle(const RtspServerHandle&) = delete;
  RtspServerHandle& operator=(const RtspServerHandle&) = delete;

  RtspServerHandle(RtspServerHandle&&) noexcept;
  RtspServerHandle& operator=(RtspServerHandle&&) noexcept;

  const std::string& url() const { return url_; }
  void stop();
  bool running() const;

private:
  friend class PipelineSession;

  std::string url_;
  void* impl_ = nullptr;
};

// =============================
// PipelineSession
// =============================
struct ValidateOptions {
  bool parse_launch = true;  // build gst pipeline and verify element naming contract
  bool enforce_names = true; // ensure no unnamed/foreign elements exist
};

struct RunDebugOptions {
  int timeout_ms = 1000;
};

struct RunDebugResult {
  std::optional<FrameNV12> first_frame; // tight packed copy
  PipelineReport report;
};

class PipelineSession {
public:
  // Core: add a node (factory functions return std::shared_ptr<Node>)
  PipelineSession& add(std::shared_ptr<Node> node);

  // Explicit raw-string escape hatch (keeps "power user" obvious)
  PipelineSession& gst(std::string fragment);

  // Typed runner: last node must be OutputAppSink() and negotiated into NV12.
  FrameStream run();

  // Split at DebugPoint(point_name), ignore nodes after it, attach appsink and return TapStream.
  TapStream run_tap(const std::string& point_name);

  // Server-style run
  RtspServerHandle run_rtsp(const RtspServerOptions& opt);

  // Build + validate pipeline (no PLAYING). Returns machine-readable report.
  PipelineReport validate(const ValidateOptions& opt = {}) const;

  // Run once, pull 1 frame (copy) with full structured report.
  RunDebugResult run_debug(const RunDebugOptions& opt = {});

  const std::string& last_pipeline() const { return last_pipeline_; }

private:
  std::vector<std::shared_ptr<Node>> nodes_;
  std::string last_pipeline_;
};

// =============================
// Customer-facing "function" API
// =============================
// Rule: end users should only use sima::nodes::* in docs/examples.
// Raw strings are only via nodes::Gst(...) or p.gst("...").
namespace nodes {

// Escape hatch
std::shared_ptr<Node> Gst(std::string fragment);

// Source / basic blocks
std::shared_ptr<Node> FileSrc(std::string path);
std::shared_ptr<Node> JpegDec();
std::shared_ptr<Node> ImageFreeze(int num_buffers = -1);
std::shared_ptr<Node> VideoConvert();
std::shared_ptr<Node> VideoScale();
std::shared_ptr<Node> VideoRate();
std::shared_ptr<Node> Queue();
std::shared_ptr<Node> DebugPoint(std::string name);

// Caps helpers (typed; always produce a capsfilter)
std::shared_ptr<Node> CapsRaw(std::string format,
                              int width = -1,
                              int height = -1,
                              int fps = -1,
                              CapsMemory memory = CapsMemory::Any);

std::shared_ptr<Node> CapsNV12SysMem(int w, int h, int fps);
std::shared_ptr<Node> CapsI420(int w, int h, int fps, CapsMemory memory = CapsMemory::Any);

// Codec helpers (typed)
std::shared_ptr<Node> H264EncodeSima(int w, int h, int fps,
                                     int bitrate_kbps = 400,
                                     std::string profile = "baseline",
                                     std::string level = "4.0");

std::shared_ptr<Node> H264EncodeSW(int bitrate_kbps = 400); // picks x264enc/openh264enc/avenc_h264
std::shared_ptr<Node> H264Parse(int config_interval = 1);
std::shared_ptr<Node> QtDemuxVideoPad(int video_pad_index);
std::shared_ptr<Node> H264ParseAu();
std::shared_ptr<Node> RTSPInput(std::string url, int latency_ms, bool tcp);
std::shared_ptr<Node> H264DepayParse();
std::shared_ptr<Node> AppSrcImage(std::string image_path, int content_w, int content_h,
                                  int enc_w, int enc_h, int fps);
std::shared_ptr<Node> RtpH264Pay(int pt, int config_interval);


// Decode
std::shared_ptr<Node> H264Decode(int sima_allocator_type = 2,
                                 std::string out_format = "NV12");

// Terminal
std::shared_ptr<Node> OutputAppSink();

} // namespace nodes

} // namespace sima
