// src/pipeline/PipelineSession.cpp
#include "pipeline/PipelineSession.h"

#include "gst/GstInit.h"
#include "gst/GstParseLaunch.h"
#include "gst/GstBusWatch.h"
#include "gst/GstHelpers.h"

#include "pipeline/Errors.h"
#include "pipeline/PipelineReport.h"
#include "pipeline/TapStream.h"
#include "pipeline/TensorStream.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/TensorUtil.h"
#include "builder/Node.h"
#include "builder/NodeGroup.h"
#include "builder/GraphPrinter.h"
#include "contracts/ContractRegistry.h"
#include "contracts/Validators.h"

#include <gst/gst.h>
#include <gst/gstdebugutils.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/video/video.h>
#include <glib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sima {

using sima::pipeline_internal::DiagCtx;

// =====================================================================================
// Small helpers (env, name sanitization, DOT dumps, etc.)
// =====================================================================================

static bool env_bool(const char* key, bool def_val) {
  const char* v = std::getenv(key);
  if (!v || !*v) return def_val;
  if (!std::strcmp(v, "1") || !std::strcmp(v, "true") || !std::strcmp(v, "TRUE") ||
      !std::strcmp(v, "yes") || !std::strcmp(v, "YES") ||
      !std::strcmp(v, "on")  || !std::strcmp(v, "ON")) {
    return true;
  }
  if (!std::strcmp(v, "0") || !std::strcmp(v, "false") || !std::strcmp(v, "FALSE") ||
      !std::strcmp(v, "no") || !std::strcmp(v, "NO") ||
      !std::strcmp(v, "off") || !std::strcmp(v, "OFF")) {
    return false;
  }
  return def_val;
}

static std::string env_str(const char* key, const std::string& def_val = "") {
  const char* v = std::getenv(key);
  if (!v) return def_val;
  return std::string(v);
}

static std::string sanitize_name(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (char c : in) {
    const bool ok =
      (c >= 'a' && c <= 'z') ||
      (c >= 'A' && c <= 'Z') ||
      (c >= '0' && c <= '9') ||
      (c == '_' || c == '-');
    out.push_back(ok ? c : '_');
  }
  if (out.empty()) out = "dbg";
  if (!out.empty() && (out[0] >= '0' && out[0] <= '9')) out = "_" + out;
  return out;
}

static void maybe_dump_dot(GstElement* pipeline, const std::string& tag) {
  if (!pipeline) return;
  const std::string dir = env_str("SIMA_GST_DOT_DIR", "");
  if (dir.empty()) return;

  // Tell GStreamer where to dump dot graphs.
  g_setenv("GST_DEBUG_DUMP_DOT_DIR", dir.c_str(), TRUE);

  const std::string t = "sima_" + sanitize_name(tag);
  gst_debug_bin_to_dot_file_with_ts(
      GST_BIN(pipeline),
      GST_DEBUG_GRAPH_SHOW_ALL,
      t.c_str());
}

static void stop_and_unref(GstElement*& e) {
  if (!e) return;

  GstElement* local = e;
  e = nullptr;

  auto done = std::make_shared<std::atomic<bool>>(false);

  std::thread([local, done]() {
    gst_element_send_event(local, gst_event_new_eos());
    gst_element_set_state(local, GST_STATE_NULL);
    gst_object_unref(local);
    done->store(true);
  }).detach();

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (done->load()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  g_printerr("[WARN] stop_and_unref(): teardown timed out. "
             "Leaking pipeline to avoid hanging.\n");
}

// =====================================================================================
// Simple JSON helpers for pipeline save/load (schema is controlled by us).
// =====================================================================================

namespace {

struct JsonValue {
  enum class Type {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object,
  };

  Type type = Type::Null;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::vector<JsonValue> arr;
  std::unordered_map<std::string, JsonValue> obj;
};

class JsonParser {
public:
  explicit JsonParser(const std::string& s) : s_(s) {}

  JsonValue parse() {
    skip_ws();
    JsonValue v = parse_value();
    skip_ws();
    if (pos_ != s_.size()) {
      throw std::runtime_error("JSON: trailing characters");
    }
    return v;
  }

private:
  const std::string& s_;
  size_t pos_ = 0;

  void skip_ws() {
    while (pos_ < s_.size() &&
           (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t')) {
      ++pos_;
    }
  }

  char peek() const {
    if (pos_ >= s_.size()) return '\0';
    return s_[pos_];
  }

  char get() {
    if (pos_ >= s_.size()) throw std::runtime_error("JSON: unexpected end");
    return s_[pos_++];
  }

  void expect(char c) {
    if (get() != c) throw std::runtime_error("JSON: unexpected character");
  }

  JsonValue parse_value() {
    skip_ws();
    const char c = peek();
    if (c == '"') return parse_string();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == 't' || c == 'f') return parse_bool();
    if (c == 'n') return parse_null();
    if ((c >= '0' && c <= '9') || c == '-') return parse_number();
    throw std::runtime_error("JSON: invalid value");
  }

  JsonValue parse_string() {
    JsonValue v;
    v.type = JsonValue::Type::String;
    expect('"');
    std::string out;
    while (pos_ < s_.size()) {
      char c = get();
      if (c == '"') break;
      if (c == '\\') {
        char esc = get();
        switch (esc) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          default: throw std::runtime_error("JSON: unsupported escape");
        }
      } else {
        out.push_back(c);
      }
    }
    v.str = std::move(out);
    return v;
  }

  JsonValue parse_number() {
    JsonValue v;
    v.type = JsonValue::Type::Number;
    size_t start = pos_;
    if (peek() == '-') ++pos_;
    while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    if (pos_ < s_.size() && s_[pos_] == '.') {
      ++pos_;
      while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }
    const std::string num = s_.substr(start, pos_ - start);
    v.num = std::stod(num);
    return v;
  }

  JsonValue parse_bool() {
    JsonValue v;
    v.type = JsonValue::Type::Bool;
    if (s_.compare(pos_, 4, "true") == 0) {
      v.b = true;
      pos_ += 4;
      return v;
    }
    if (s_.compare(pos_, 5, "false") == 0) {
      v.b = false;
      pos_ += 5;
      return v;
    }
    throw std::runtime_error("JSON: invalid bool");
  }

  JsonValue parse_null() {
    if (s_.compare(pos_, 4, "null") != 0) {
      throw std::runtime_error("JSON: invalid null");
    }
    pos_ += 4;
    JsonValue v;
    v.type = JsonValue::Type::Null;
    return v;
  }

  JsonValue parse_array() {
    JsonValue v;
    v.type = JsonValue::Type::Array;
    expect('[');
    skip_ws();
    if (peek() == ']') { get(); return v; }
    while (true) {
      v.arr.push_back(parse_value());
      skip_ws();
      char c = get();
      if (c == ']') break;
      if (c != ',') throw std::runtime_error("JSON: expected ',' or ']'");
      skip_ws();
    }
    return v;
  }

  JsonValue parse_object() {
    JsonValue v;
    v.type = JsonValue::Type::Object;
    expect('{');
    skip_ws();
    if (peek() == '}') { get(); return v; }
    while (true) {
      JsonValue key = parse_string();
      skip_ws();
      expect(':');
      JsonValue val = parse_value();
      v.obj.emplace(key.str, std::move(val));
      skip_ws();
      char c = get();
      if (c == '}') break;
      if (c != ',') throw std::runtime_error("JSON: expected ',' or '}'");
      skip_ws();
    }
    return v;
  }
};

static std::string json_escape(const std::string& in) {
  std::ostringstream oss;
  for (char c : in) {
    switch (c) {
      case '"': oss << "\\\""; break;
      case '\\': oss << "\\\\"; break;
      case '\b': oss << "\\b"; break;
      case '\f': oss << "\\f"; break;
      case '\n': oss << "\\n"; break;
      case '\r': oss << "\\r"; break;
      case '\t': oss << "\\t"; break;
      default: oss << c; break;
    }
  }
  return oss.str();
}

// Node wrapper used for load() when only the fragment is available.
class ConfiguredNode final : public Node {
public:
  ConfiguredNode(std::string kind,
                 std::string label,
                 std::string fragment,
                 std::vector<std::string> elements)
      : kind_(std::move(kind)),
        label_(std::move(label)),
        fragment_(std::move(fragment)),
        elements_(std::move(elements)) {}

  std::string kind() const override { return kind_; }
  std::string user_label() const override { return label_; }
  std::string gst_fragment(int /*node_index*/) const override { return fragment_; }
  std::vector<std::string> element_names(int /*node_index*/) const override {
    return elements_;
  }

private:
  std::string kind_;
  std::string label_;
  std::string fragment_;
  std::vector<std::string> elements_;
};

} // namespace

// =====================================================================================
// Debug helpers (TapPacket conversion for run_debug)
// =====================================================================================

static TapFormat tap_format_from_caps(GstCaps* caps, TapVideoInfo* out_vi) {
  if (!caps || gst_caps_is_any(caps) || gst_caps_is_empty(caps)) return TapFormat::Unknown;

  const GstStructure* st = gst_caps_get_structure(caps, 0);
  if (!st) return TapFormat::Unknown;

  const char* name = gst_structure_get_name(st);
  if (!name) return TapFormat::Unknown;

  if (out_vi) {
    int w = 0, h = 0;
    (void)gst_structure_get_int(st, "width", &w);
    (void)gst_structure_get_int(st, "height", &h);
    out_vi->width = w;
    out_vi->height = h;

    int fps_n = 0, fps_d = 1;
    if (gst_structure_get_fraction(st, "framerate", &fps_n, &fps_d)) {
      out_vi->fps_num = fps_n;
      out_vi->fps_den = (fps_d == 0) ? 1 : fps_d;
    }

    const char* fmt = gst_structure_get_string(st, "format");
    out_vi->format = fmt ? fmt : "";
  }

  if (std::strcmp(name, "video/x-h264") == 0) return TapFormat::H264;
  if (std::strcmp(name, "video/x-h265") == 0) return TapFormat::H265;
  if (std::strcmp(name, "image/jpeg") == 0) return TapFormat::JPEG;
  if (std::strcmp(name, "image/png") == 0) return TapFormat::PNG;

  if (std::strcmp(name, "video/x-raw") == 0) {
    const char* fmt = gst_structure_get_string(st, "format");
    if (!fmt) return TapFormat::Unknown;
    if (std::strcmp(fmt, "NV12") == 0) return TapFormat::NV12;
    if (std::strcmp(fmt, "I420") == 0) return TapFormat::I420;
    if (std::strcmp(fmt, "RGB") == 0) return TapFormat::RGB;
    if (std::strcmp(fmt, "BGR") == 0) return TapFormat::BGR;
    if (std::strcmp(fmt, "GRAY8") == 0) return TapFormat::GRAY8;
  }

  return TapFormat::Unknown;
}

static bool parse_keyframe(GstBuffer* buf) {
  if (!buf) return false;
  return !GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_DELTA_UNIT);
}

static int64_t ts_or_minus1(GstClockTime t) {
  return GST_CLOCK_TIME_IS_VALID(t) ? static_cast<int64_t>(t) : -1;
}

static size_t env_size_t(const char* key, size_t def_val) {
  const char* v = std::getenv(key);
  if (!v || !*v) return def_val;
  char* end = nullptr;
  unsigned long long x = std::strtoull(v, &end, 10);
  if (end == v) return def_val;
  return static_cast<size_t>(x);
}

static void pack_raw_video_tight(GstSample* sample, std::vector<uint8_t>* out_bytes) {
  GstCaps* caps = gst_sample_get_caps(sample);
  if (!caps) throw std::runtime_error("No caps on sample");

  GstVideoInfo info;
  std::memset(&info, 0, sizeof(info));
  if (!gst_video_info_from_caps(&info, caps)) {
    throw std::runtime_error("gst_video_info_from_caps failed");
  }

  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) throw std::runtime_error("No buffer on sample");

  GstVideoFrame frame;
  std::memset(&frame, 0, sizeof(frame));
  if (!gst_video_frame_map(&frame, &info, buffer, GST_MAP_READ)) {
    throw std::runtime_error("gst_video_frame_map failed (non-mappable memory?)");
  }

  auto unmap = [&]() { gst_video_frame_unmap(&frame); };
  try {
    const int w = GST_VIDEO_INFO_WIDTH(&info);
    const int h = GST_VIDEO_INFO_HEIGHT(&info);

    const GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT(&info);

    const size_t max_bytes = env_size_t("SIMA_TAP_MAX_BYTES", 64ull * 1024ull * 1024ull);

    auto reserve_checked = [&](size_t n) {
      if (n > max_bytes) {
        std::ostringstream ss;
        ss << "Tap payload too large (" << n << " bytes > cap " << max_bytes << ")";
        throw std::runtime_error(ss.str());
      }
      out_bytes->clear();
      out_bytes->resize(n);
    };

    uint8_t* dst = nullptr;

    if (fmt == GST_VIDEO_FORMAT_NV12) {
      const size_t y_sz = static_cast<size_t>(w) * static_cast<size_t>(h);
      const size_t uv_sz = y_sz / 2;
      reserve_checked(y_sz + uv_sz);
      dst = out_bytes->data();

      const uint8_t* y = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
      const uint8_t* uv = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 1));
      const int y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
      const int uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 1);

      for (int r = 0; r < h; ++r) {
        std::memcpy(dst + static_cast<size_t>(r) * w,
                    y + static_cast<size_t>(r) * y_stride,
                    w);
      }
      const size_t uv_off = y_sz;
      for (int r = 0; r < h / 2; ++r) {
        std::memcpy(dst + uv_off + static_cast<size_t>(r) * w,
                    uv + static_cast<size_t>(r) * uv_stride,
                    w);
      }
      unmap();
      return;
    }

    if (fmt == GST_VIDEO_FORMAT_I420) {
      const size_t y_sz = static_cast<size_t>(w) * static_cast<size_t>(h);
      const size_t u_sz = static_cast<size_t>(w / 2) * static_cast<size_t>(h / 2);
      const size_t v_sz = u_sz;
      reserve_checked(y_sz + u_sz + v_sz);
      dst = out_bytes->data();

      const uint8_t* y = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
      const uint8_t* u = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 1));
      const uint8_t* v = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 2));
      const int y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
      const int u_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 1);
      const int v_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 2);

      for (int r = 0; r < h; ++r) {
        std::memcpy(dst + static_cast<size_t>(r) * w,
                    y + static_cast<size_t>(r) * y_stride,
                    w);
      }

      size_t off = y_sz;
      const int cw = w / 2;
      const int ch = h / 2;

      for (int r = 0; r < ch; ++r) {
        std::memcpy(dst + off + static_cast<size_t>(r) * cw,
                    u + static_cast<size_t>(r) * u_stride,
                    cw);
      }

      off += u_sz;
      for (int r = 0; r < ch; ++r) {
        std::memcpy(dst + off + static_cast<size_t>(r) * cw,
                    v + static_cast<size_t>(r) * v_stride,
                    cw);
      }
      unmap();
      return;
    }

    if (fmt == GST_VIDEO_FORMAT_RGB || fmt == GST_VIDEO_FORMAT_BGR) {
      const size_t row = static_cast<size_t>(w) * 3;
      const size_t sz = row * static_cast<size_t>(h);
      reserve_checked(sz);
      dst = out_bytes->data();

      const uint8_t* p = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
      const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);

      for (int r = 0; r < h; ++r) {
        std::memcpy(dst + static_cast<size_t>(r) * row,
                    p + static_cast<size_t>(r) * stride,
                    row);
      }
      unmap();
      return;
    }

    if (fmt == GST_VIDEO_FORMAT_GRAY8) {
      const size_t row = static_cast<size_t>(w);
      const size_t sz = row * static_cast<size_t>(h);
      reserve_checked(sz);
      dst = out_bytes->data();

      const uint8_t* p = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
      const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);

      for (int r = 0; r < h; ++r) {
        std::memcpy(dst + static_cast<size_t>(r) * row,
                    p + static_cast<size_t>(r) * stride,
                    row);
      }
      unmap();
      return;
    }

    throw std::runtime_error("Unsupported raw video format for tight packing");
  } catch (...) {
    unmap();
    throw;
  }
}

static void map_buffer_copy(GstBuffer* buffer, std::vector<uint8_t>* out) {
  out->clear();
  if (!buffer) return;

  const size_t max_bytes = env_size_t("SIMA_TAP_MAX_BYTES", 64ull * 1024ull * 1024ull);

  GstMapInfo mi;
  std::memset(&mi, 0, sizeof(mi));
  if (!gst_buffer_map(buffer, &mi, GST_MAP_READ)) {
    throw std::runtime_error("gst_buffer_map failed (non-mappable memory?)");
  }

  const size_t n = static_cast<size_t>(mi.size);
  if (n > max_bytes) {
    gst_buffer_unmap(buffer, &mi);
    std::ostringstream ss;
    ss << "Tap payload too large (" << n << " bytes > cap " << max_bytes << ")";
    throw std::runtime_error(ss.str());
  }

  out->resize(n);
  if (n) std::memcpy(out->data(), mi.data, n);
  gst_buffer_unmap(buffer, &mi);
}

static TapPacket sample_to_tap_packet(GstSample* sample,
                                      int node_index,
                                      const std::string& sink_name) {
  TapPacket pkt;
  pkt.source_node_index = node_index;
  pkt.source_element = sink_name;

  GstCaps* caps = gst_sample_get_caps(sample);
  pkt.caps_string = pipeline_internal::gst_caps_to_string_safe(caps);
  pkt.memory_features = pipeline_internal::caps_features_string(caps);

  GstBuffer* buf = gst_sample_get_buffer(sample);
  pkt.keyframe = parse_keyframe(buf);
  if (buf) {
    pkt.pts_ns = ts_or_minus1(GST_BUFFER_PTS(buf));
    pkt.dts_ns = ts_or_minus1(GST_BUFFER_DTS(buf));
    pkt.duration_ns = ts_or_minus1(GST_BUFFER_DURATION(buf));
  }

  pkt.format = tap_format_from_caps(caps, &pkt.video);

  try {
    if (pkt.format == TapFormat::NV12 || pkt.format == TapFormat::I420 ||
        pkt.format == TapFormat::RGB  || pkt.format == TapFormat::BGR  ||
        pkt.format == TapFormat::GRAY8) {
      pack_raw_video_tight(sample, &pkt.bytes);
      pkt.memory_mappable = true;
      pkt.non_mappable_reason.clear();
      return pkt;
    }

    map_buffer_copy(buf, &pkt.bytes);
    pkt.memory_mappable = true;
    pkt.non_mappable_reason.clear();
    return pkt;
  } catch (const std::exception& e) {
    pkt.memory_mappable = false;
    pkt.bytes.clear();
    pkt.non_mappable_reason = e.what();
    return pkt;
  }
}

static std::string build_debug_segment(
    const std::vector<std::shared_ptr<Node>>& nodes,
    int start_idx,
    int end_idx,
    bool with_appsrc,
    const std::string& appsrc_name,
    const std::string& appsink_name) {
  std::ostringstream ss;

  if (with_appsrc) {
    ss << "appsrc name=" << appsrc_name
       << " is-live=true format=time do-timestamp=false";
    if (start_idx <= end_idx) ss << " ! ";
  }

  for (int i = start_idx; i <= end_idx; ++i) {
    if (i > start_idx) ss << " ! ";
    ss << nodes[static_cast<size_t>(i)]->gst_fragment(i);
  }

  ss << " ! appsink name=" << appsink_name
     << " emit-signals=false sync=false max-buffers=1 drop=true";

  return ss.str();
}

// =====================================================================================
// Diag context + boundary stats (compatible with PipelineReport / TapStream)
// =====================================================================================

static std::string boundary_summary_local(const std::shared_ptr<DiagCtx>& diag) {
  if (!diag || diag->boundaries.empty()) return "";

  const int64_t now = (int64_t)g_get_monotonic_time();
  int best_idx = -1;
  int64_t best_t = 0;
  bool best_out = false;

  // Take consistent snapshots (atomics -> plain stats) to avoid races and to keep
  // this function simple.
  std::vector<BoundaryFlowStats> snaps;
  snaps.reserve(diag->boundaries.size());

  for (size_t i = 0; i < diag->boundaries.size(); ++i) {
    const auto* b = diag->boundaries[i].get();
    if (!b) continue;

    BoundaryFlowStats s = b->snapshot();
    snaps.push_back(s);

    if (s.last_out_wall_us > best_t) {
      best_t = s.last_out_wall_us;
      best_idx = (int)i;
      best_out = true;
    }
    if (s.last_in_wall_us > best_t) {
      best_t = s.last_in_wall_us;
      best_idx = (int)i;
      best_out = false;
    }
  }

  std::ostringstream ss;
  ss << "BoundaryFlow:\n";
  for (const auto& s : snaps) {
    ss << "  - " << s.boundary_name
       << " after=" << s.after_node_index
       << " before=" << s.before_node_index
       << " in=" << s.in_buffers
       << " out=" << s.out_buffers
       << " last_in_age_ms="
       << (s.last_in_wall_us ? ((now - s.last_in_wall_us) / 1000) : 0)
       << " last_out_age_ms="
       << (s.last_out_wall_us ? ((now - s.last_out_wall_us) / 1000) : 0)
       << "\n";
  }

  if (best_idx >= 0 && best_t > 0) {
    const auto* b = diag->boundaries[(size_t)best_idx].get();
    if (b) {
      BoundaryFlowStats s = b->snapshot();
      ss << "LikelyStall: last activity "
         << (best_out ? "leaving " : "entering ")
         << s.boundary_name
         << " age_ms=" << ((now - best_t) / 1000)
         << " (after node " << s.after_node_index
         << ", before node " << s.before_node_index << ")\n";
    }
  }

  return ss.str();
}

// =====================================================================================
// Boundary probes
// =====================================================================================

struct BoundaryProbeCtx {
  sima::pipeline_internal::BoundaryFlowCounters* counters = nullptr; // atomics
  bool is_in = false;
};

static GstPadProbeReturn boundary_probe_cb(GstPad*,
                                          GstPadProbeInfo* info,
                                          gpointer user_data) {
  auto* ctx = reinterpret_cast<BoundaryProbeCtx*>(user_data);
  if (!ctx || !ctx->counters) return GST_PAD_PROBE_OK;

  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0)
    return GST_PAD_PROBE_OK;

  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf) return GST_PAD_PROBE_OK;

  const int64_t now = (int64_t)g_get_monotonic_time();
  const GstClockTime pts = GST_BUFFER_PTS(buf);
  const int64_t pts_ns = (pts == GST_CLOCK_TIME_NONE) ? -1 : (int64_t)pts;

  using std::memory_order_relaxed;

  if (ctx->is_in) {
    ctx->counters->in_buffers.fetch_add(1, memory_order_relaxed);
    ctx->counters->last_in_wall_us.store(now, memory_order_relaxed);
    if (pts_ns >= 0) ctx->counters->last_in_pts_ns.store(pts_ns, memory_order_relaxed);
  } else {
    ctx->counters->out_buffers.fetch_add(1, memory_order_relaxed);
    ctx->counters->last_out_wall_us.store(now, memory_order_relaxed);
    if (pts_ns >= 0) ctx->counters->last_out_pts_ns.store(pts_ns, memory_order_relaxed);
  }

  return GST_PAD_PROBE_OK;
}

static void attach_boundary_probes(GstElement* pipeline,
                                   const std::shared_ptr<DiagCtx>& diag) {
  if (!pipeline || !diag) return;
  if (!env_bool("SIMA_GST_BOUNDARY_PROBES", false)) return;

  for (auto& bptr : diag->boundaries) {
    auto* b = bptr.get();
    if (!b) continue;

    GstElement* ident =
        gst_bin_get_by_name(GST_BIN(pipeline), b->boundary_name.c_str());
    if (!ident) continue;

    GstPad* sink = gst_element_get_static_pad(ident, "sink");
    GstPad* src  = gst_element_get_static_pad(ident, "src");

    if (sink) {
      auto* ctx = new BoundaryProbeCtx();
      ctx->counters = b;
      ctx->is_in = true;
      gst_pad_add_probe(
          sink,
          GST_PAD_PROBE_TYPE_BUFFER,
          boundary_probe_cb,
          ctx,
          +[](gpointer p) { delete reinterpret_cast<BoundaryProbeCtx*>(p); });
      gst_object_unref(sink);
    }

    if (src) {
      auto* ctx = new BoundaryProbeCtx();
      ctx->counters = b;
      ctx->is_in = false;
      gst_pad_add_probe(
          src,
          GST_PAD_PROBE_TYPE_BUFFER,
          boundary_probe_cb,
          ctx,
          +[](gpointer p) { delete reinterpret_cast<BoundaryProbeCtx*>(p); });
      gst_object_unref(src);
    }

    gst_object_unref(ident);
  }
}

// =====================================================================================
// Bus/meta plumbing + improved error diagnostics (parse_error + DOT dumps)
// =====================================================================================

static void drain_bus_into_diag(GstElement* pipeline,
                                const std::shared_ptr<DiagCtx>& diag) {
  if (!pipeline || !diag) return;
  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus) return;

  while (GstMessage* msg = gst_bus_pop(bus)) {
    std::string line = gst_message_to_string(msg);
    const char* src =
        (GST_MESSAGE_SRC(msg) && GST_IS_OBJECT(GST_MESSAGE_SRC(msg)))
            ? GST_OBJECT_NAME(GST_MESSAGE_SRC(msg))
            : "<unknown>";
    diag->push_bus(gst_message_type_get_name(GST_MESSAGE_TYPE(msg)),
                   src ? src : "<unknown>",
                   line);
    gst_message_unref(msg);
  }
  gst_object_unref(bus);
}

static void throw_if_bus_error_local(GstElement* pipeline,
                                     const std::shared_ptr<DiagCtx>& diag,
                                     const char* where) {
  if (!pipeline) return;
  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus) return;

  while (GstMessage* msg = gst_bus_pop(bus)) {
    // Always record something helpful in the bus log.
    std::string line = gst_message_to_string(msg);
    const char* src =
        (GST_MESSAGE_SRC(msg) && GST_IS_OBJECT(GST_MESSAGE_SRC(msg)))
            ? GST_OBJECT_NAME(GST_MESSAGE_SRC(msg))
            : "<unknown>";

    if (diag) {
      diag->push_bus(gst_message_type_get_name(GST_MESSAGE_TYPE(msg)),
                     src ? src : "<unknown>",
                     line);
    }

    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
      // Prefer structured error parsing (better than generic stringify).
      GError* e = nullptr;
      gchar* dbg = nullptr;
      gst_message_parse_error(msg, &e, &dbg);

      const std::string err_msg = (e && e->message) ? e->message : "unknown";
      const std::string dbg_msg = (dbg) ? dbg : "";

      if (e) g_error_free(e);
      if (dbg) g_free(dbg);

      gst_message_unref(msg);
      gst_object_unref(bus);

      maybe_dump_dot(pipeline, std::string(where) + "_error");

      PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
      rep.repro_note =
          std::string(where) + ": GST ERROR: " + err_msg +
          (dbg_msg.empty() ? "" : ("\nGST_DEBUG: " + dbg_msg)) +
          "\n" + boundary_summary_local(diag);
      throw PipelineError(rep.repro_note, std::move(rep));
    }

    gst_message_unref(msg);
  }

  gst_object_unref(bus);
}

// =====================================================================================
// BuildResult: from Node list → pipeline string + diag + sink name
// =====================================================================================

struct BuildResult {
  std::string pipeline_string;
  std::shared_ptr<DiagCtx> diag;
  std::string appsink_name; // mysink or tap_*
  int tap_node_index = -1;
};

static bool should_insert_boundaries_for_mode(const char* mode_key,
                                              bool def_val) {
  return env_bool(mode_key, def_val);
}

static BuildResult build_pipeline_full(
    const std::vector<std::shared_ptr<Node>>& nodes,
    bool insert_boundaries,
    const std::string& appsink_name) {
  if (nodes.empty()) {
    throw std::runtime_error("InvalidPipeline: no nodes");
  }

  BuildResult br;
  br.diag = std::make_shared<DiagCtx>();
  br.appsink_name = appsink_name;

  std::ostringstream ss;

  br.diag->node_reports.reserve(nodes.size());
  if (insert_boundaries) {
    br.diag->boundaries.reserve(nodes.size() ? nodes.size() - 1 : 0);
  }

  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!nodes[i]) {
      throw std::runtime_error("InvalidPipeline: null node");
    }

    if (i) ss << " ! ";

    NodeReport nr;
    nr.index = (int)i;
    nr.kind = nodes[i]->kind();
    nr.user_label = nodes[i]->user_label();
    nr.gst_fragment = nodes[i]->gst_fragment((int)i);
    nr.elements = nodes[i]->element_names((int)i);
    br.diag->node_reports.push_back(nr);

    ss << nr.gst_fragment;

    if (insert_boundaries && i + 1 < nodes.size()) {
      const std::string bname = "sima_b" + std::to_string(i);
      ss << " ! identity name=" << bname << " silent=true";

      // IMPORTANT: boundaries store atomic counters (BoundaryFlowCounters),
      // not plain stats. Reports use snapshot() to convert atomics -> stats.
      auto ctr = std::make_unique<pipeline_internal::BoundaryFlowCounters>();
      ctr->boundary_name = bname;
      ctr->after_node_index = (int)i;
      ctr->before_node_index = (int)(i + 1);
      br.diag->boundaries.push_back(std::move(ctr));
    }
  }

  br.diag->pipeline_string = ss.str();
  br.pipeline_string = br.diag->pipeline_string;
  return br;
}

static BuildResult build_pipeline_tap(
    const std::vector<std::shared_ptr<Node>>& nodes,
    const std::string& debug_point_name,
    bool insert_boundaries) {
  if (nodes.empty()) {
    throw std::runtime_error("InvalidPipeline: no nodes");
  }

  const std::string want = debug_point_name.empty() ? "dbg" : debug_point_name;
  const std::string tap_name = "tap_" + sanitize_name(want);

  int cut = -1;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!nodes[i]) continue;
    if (nodes[i]->kind() == "DebugPoint" &&
        nodes[i]->user_label() == want) {
      cut = (int)i;
      break;
    }
  }
  if (cut < 0) {
    throw std::runtime_error("run_tap: DebugPoint not found: " + want);
  }

  std::vector<std::shared_ptr<Node>> trunc(
      nodes.begin(), nodes.begin() + (size_t)cut + 1);

  BuildResult br = build_pipeline_full(trunc, insert_boundaries, tap_name);
  br.tap_node_index = cut;
  br.appsink_name = tap_name;

  std::ostringstream ss;
  ss << br.pipeline_string
     << " ! appsink name=" << tap_name
     << " emit-signals=false sync=false max-buffers=1 drop=true";

  br.pipeline_string = ss.str();
  br.diag->pipeline_string = br.pipeline_string;
  return br;
}

static void enforce_sink_last(const std::vector<std::shared_ptr<Node>>& nodes) {
  if (nodes.empty()) {
    throw std::runtime_error("InvalidPipeline: no nodes");
  }
  if (!nodes.back() || nodes.back()->kind() != "OutputAppSink") {
    throw std::runtime_error(
        "InvalidPipeline: last node must be OutputAppSink() for run()");
  }
}

// Best-effort naming contract enforcement; optional via env.
static void enforce_names_contract(GstElement* pipeline,
                                   const BuildResult& br) {
  if (!pipeline || !GST_IS_BIN(pipeline) || !br.diag) return;

  std::unordered_set<std::string> allowed;

  for (const auto& n : br.diag->node_reports) {
    for (const auto& e : n.elements) {
      allowed.insert(e);
    }
  }
  for (const auto& b : br.diag->boundaries) {
    if (b) allowed.insert(b->boundary_name);
  }
  if (!br.appsink_name.empty()) {
    allowed.insert(br.appsink_name);
  }

  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it) return;

  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* el = GST_ELEMENT(g_value_get_object(&item));
    g_value_reset(&item);
    if (!el) continue;

    const char* name = GST_ELEMENT_NAME(el);
    if (name && *name) {
      const std::string n(name);
      const bool internal_ok =
        (n.rfind("queue", 0) == 0) ||
        (n.rfind("typefind", 0) == 0) ||
        (n.rfind("rtpbin", 0) == 0) ||
        (n.rfind("decodebin", 0) == 0);

      if (!internal_ok && allowed.find(n) == allowed.end()) {
        gst_iterator_free(it);
        PipelineReport rep = br.diag->snapshot_basic();
        rep.repro_note =
          "NamingContractViolation: element '" + n +
          "' is not owned by any node.\n"
          "Fix: ensure every fragment uses deterministic names and "
          "element_names() matches.\n";
        throw PipelineError(rep.repro_note, std::move(rep));
      }
    }
  }

  g_value_unset(&item);
  gst_iterator_free(it);
}

// =====================================================================================
// State helper
// =====================================================================================

static void set_state_or_throw(GstElement* pipeline,
                               GstState target,
                               const char* where,
                               const std::shared_ptr<DiagCtx>& diag) {
  if (!pipeline) {
    throw std::runtime_error(std::string(where) + ": pipeline is null");
  }

  GstStateChangeReturn r = gst_element_set_state(pipeline, target);
  if (r == GST_STATE_CHANGE_FAILURE) {
    maybe_dump_dot(pipeline, std::string(where) + "_set_state_failure");
    PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
    rep.repro_note =
        std::string(where) + ": failed to set state.\n" +
        boundary_summary_local(diag);
    throw PipelineError(rep.repro_note, std::move(rep));
  }

  GstState cur = GST_STATE_VOID_PENDING;
  GstState pend = GST_STATE_VOID_PENDING;
  gst_element_get_state(pipeline, &cur, &pend, 2 * GST_SECOND);

  drain_bus_into_diag(pipeline, diag);
  throw_if_bus_error_local(pipeline, diag, where);
}

// =====================================================================================
// RtspServer internal impl (behaviorally similar)
// =====================================================================================

struct RtspServerImpl {
  std::string url;
  std::string mount_path;
  int port = 8554;

  std::thread th;
  std::atomic<bool> running{false};
  GMainLoop* loop = nullptr;

  int enc_w = 0;
  int enc_h = 0;
  int fps = 30;
  std::shared_ptr<std::vector<uint8_t>> nv12_enc;
};

struct PushCtx {
  GstElement* appsrc = nullptr;
  guint timer_id = 0;

  int w = 0, h = 0, fps = 30;
  guint64 frame_count = 0;
  guint64 frame_duration_ns = 0;

  std::shared_ptr<std::vector<uint8_t>> nv12;
  std::atomic<bool> stopped{false};
};

static std::string ensure_mount_path(const std::string& mount) {
  if (mount.empty()) return "/image";
  if (!mount.empty() && mount[0] == '/') return mount;
  return "/" + mount;
}

static std::string make_rtsp_url(int port, const std::string& mount) {
  return "rtsp://127.0.0.1:" + std::to_string(port) + ensure_mount_path(mount);
}

static gboolean push_frame_cb(gpointer user_data) {
  auto* pc = reinterpret_cast<PushCtx*>(user_data);
  if (!pc || !pc->appsrc || pc->stopped.load()) return G_SOURCE_REMOVE;
  if (!pc->nv12 || pc->nv12->empty()) return G_SOURCE_REMOVE;

  const size_t y_sz = (size_t)pc->w * (size_t)pc->h;
  const size_t uv_sz = y_sz / 2;
  const size_t total = y_sz + uv_sz;
  if (pc->nv12->size() != total) return G_SOURCE_REMOVE;

  GstBuffer* buf = gst_buffer_new_allocate(nullptr, total, nullptr);
  if (!buf) return G_SOURCE_REMOVE;

  const guint64 pts = pc->frame_count * pc->frame_duration_ns;
  GST_BUFFER_PTS(buf) = pts;
  GST_BUFFER_DTS(buf) = pts;
  GST_BUFFER_DURATION(buf) = pc->frame_duration_ns;

  GstMapInfo map{};
  if (!gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
    gst_buffer_unref(buf);
    return G_SOURCE_REMOVE;
  }

  std::memcpy(map.data, pc->nv12->data(), total);
  gst_buffer_unmap(buf, &map);

  GstFlowReturn fr = gst_app_src_push_buffer(GST_APP_SRC(pc->appsrc), buf);
  if (fr == GST_FLOW_FLUSHING || fr == GST_FLOW_EOS || fr != GST_FLOW_OK) {
    pc->stopped.store(true);
    return G_SOURCE_REMOVE;
  }

  pc->frame_count++;
  return G_SOURCE_CONTINUE;
}

static void media_unprepared_cb(GstRTSPMedia*, gpointer user_data) {
  auto* pc = reinterpret_cast<PushCtx*>(user_data);
  if (!pc) return;

  pc->stopped.store(true);
  if (pc->timer_id) {
    g_source_remove(pc->timer_id);
    pc->timer_id = 0;
  }

  if (pc->appsrc) {
    gst_object_unref(pc->appsrc);
    pc->appsrc = nullptr;
  }

  delete pc;
}

// RtspServerHandle implementation
RtspServerHandle::~RtspServerHandle() { stop(); }

RtspServerHandle::RtspServerHandle(RtspServerHandle&& o) noexcept {
  url_ = std::move(o.url_);
  impl_ = o.impl_;
  o.impl_ = nullptr;
}

RtspServerHandle& RtspServerHandle::operator=(RtspServerHandle&& o) noexcept {
  if (this != &o) {
    stop();
    url_ = std::move(o.url_);
    impl_ = o.impl_;
    o.impl_ = nullptr;
  }
  return *this;
}

bool RtspServerHandle::running() const {
  auto* impl = reinterpret_cast<RtspServerImpl*>(impl_);
  return impl && impl->running.load();
}

void RtspServerHandle::stop() {
  auto* impl = reinterpret_cast<RtspServerImpl*>(impl_);
  if (!impl) return;

  if (impl->loop) g_main_loop_quit(impl->loop);
  if (impl->th.joinable()) impl->th.join();

  delete impl;
  impl_ = nullptr;

  std::this_thread::sleep_for(std::chrono::milliseconds(250));
}

// =====================================================================================
// PipelineSession – public methods
// =====================================================================================

PipelineSession& PipelineSession::add(std::shared_ptr<Node> node) {
  nodes_.push_back(std::move(node));
  return *this;
}

PipelineSession& PipelineSession::add(const NodeGroup& group) {
  const auto& gnodes = group.nodes();
  nodes_.insert(nodes_.end(), gnodes.begin(), gnodes.end());
  return *this;
}

PipelineSession& PipelineSession::add(NodeGroup&& group) {
  auto& gnodes = group.nodes_mut();
  nodes_.insert(nodes_.end(),
                std::make_move_iterator(gnodes.begin()),
                std::make_move_iterator(gnodes.end()));
  gnodes.clear();
  return *this;
}

PipelineSession& PipelineSession::gst(std::string fragment) {
  return add(nodes::Gst(std::move(fragment)));
}

FrameStream PipelineSession::run() {
  gst_init_once();

  enforce_sink_last(nodes_);

  require_element("appsink", "PipelineSession::run");
  require_element("identity", "PipelineSession::run");

  const bool insert_boundaries =
      should_insert_boundaries_for_mode("SIMA_GST_RUN_INSERT_BOUNDARIES", false);

  BuildResult br = build_pipeline_full(nodes_, insert_boundaries, "mysink");
  last_pipeline_ = br.pipeline_string;

  GError* err = nullptr;
  GstElement* pipeline = gst_parse_launch(last_pipeline_.c_str(), &err);
  if (!pipeline) {
    std::string msg = err ? err->message : "unknown";
    if (err) g_error_free(err);
    throw std::runtime_error(
        "PipelineSession::run: gst_parse_launch failed: " + msg +
        "\nPipeline:\n" + last_pipeline_);
  }
  if (err) g_error_free(err);

  if (env_bool("SIMA_GST_ENFORCE_NAMES", false)) {
    enforce_names_contract(pipeline, br);
  }

  attach_boundary_probes(pipeline, br.diag);

  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
  if (!sink) {
    maybe_dump_dot(pipeline, "run_missing_mysink");
    stop_and_unref(pipeline);
    throw std::runtime_error(
        "PipelineSession::run: appsink 'mysink' not found.\n"
        "Fix: add OutputAppSink() as the last node.\nPipeline:\n" +
        last_pipeline_);
  }

  try {
    set_state_or_throw(pipeline, GST_STATE_PLAYING, "PipelineSession::run", br.diag);
  } catch (...) {
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    throw;
  }

  FrameStream fs(pipeline, sink);
  fs.set_debug_pipeline(last_pipeline_);
  fs.set_diag(br.diag);   // implicit shared_ptr<DiagCtx> -> shared_ptr<void>
  return fs;
}

TensorStream PipelineSession::run_tensor() {
  gst_init_once();

  enforce_sink_last(nodes_);

  require_element("appsink", "PipelineSession::run_tensor");
  require_element("identity", "PipelineSession::run_tensor");

  const bool insert_boundaries =
      should_insert_boundaries_for_mode("SIMA_GST_RUN_INSERT_BOUNDARIES", false);

  BuildResult br = build_pipeline_full(nodes_, insert_boundaries, "mysink");
  last_pipeline_ = br.pipeline_string;

  GError* err = nullptr;
  GstElement* pipeline = gst_parse_launch(last_pipeline_.c_str(), &err);
  if (!pipeline) {
    std::string msg = err ? err->message : "unknown";
    if (err) g_error_free(err);
    throw std::runtime_error(
        "PipelineSession::run_tensor: gst_parse_launch failed: " + msg +
        "\nPipeline:\n" + last_pipeline_);
  }
  if (err) g_error_free(err);

  if (env_bool("SIMA_GST_ENFORCE_NAMES", false)) {
    enforce_names_contract(pipeline, br);
  }

  attach_boundary_probes(pipeline, br.diag);

  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
  if (!sink) {
    maybe_dump_dot(pipeline, "run_tensor_missing_mysink");
    stop_and_unref(pipeline);
    throw std::runtime_error(
        "PipelineSession::run_tensor: appsink 'mysink' not found.\n"
        "Fix: add OutputAppSink() as the last node.\nPipeline:\n" +
        last_pipeline_);
  }

  try {
    set_state_or_throw(pipeline, GST_STATE_PLAYING, "PipelineSession::run_tensor", br.diag);
  } catch (...) {
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    throw;
  }

  TensorStream ts(pipeline, sink);
  ts.set_debug_pipeline(last_pipeline_);
  ts.set_diag(br.diag);
  return ts;
}

TapStream PipelineSession::run_tap(const std::string& point_name) {
  gst_init_once();

  require_element("appsink", "PipelineSession::run_tap");
  require_element("identity", "PipelineSession::run_tap");

  const bool insert_boundaries =
      should_insert_boundaries_for_mode("SIMA_GST_TAP_INSERT_BOUNDARIES", true);

  BuildResult br = build_pipeline_tap(nodes_, point_name, insert_boundaries);
  last_pipeline_ = br.pipeline_string;

  GError* err = nullptr;
  GstElement* pipeline = gst_parse_launch(last_pipeline_.c_str(), &err);
  if (!pipeline) {
    std::string msg = err ? err->message : "unknown";
    if (err) g_error_free(err);
    throw std::runtime_error(
        "PipelineSession::run_tap: gst_parse_launch failed: " + msg +
        "\nPipeline:\n" + last_pipeline_);
  }
  if (err) g_error_free(err);

  if (env_bool("SIMA_GST_ENFORCE_NAMES", false)) {
    enforce_names_contract(pipeline, br);
  }

  attach_boundary_probes(pipeline, br.diag);

  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), br.appsink_name.c_str());
  if (!sink) {
    maybe_dump_dot(pipeline, "run_tap_missing_" + br.appsink_name);
    stop_and_unref(pipeline);
    throw std::runtime_error(
        "PipelineSession::run_tap: appsink '" + br.appsink_name +
        "' not found.\nPipeline:\n" + last_pipeline_);
  }

  try {
    set_state_or_throw(pipeline, GST_STATE_PLAYING, "PipelineSession::run_tap", br.diag);
  } catch (...) {
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    throw;
  }

  TapStream ts(pipeline, sink);
  ts.set_debug_pipeline(last_pipeline_);
  ts.set_diag(br.diag);
  ts.set_tap_meta(br.tap_node_index, br.appsink_name);
  return ts;
}

PipelineReport PipelineSession::validate(const ValidateOptions& opt) const {
  gst_init_once();

  PipelineReport rep;

  {
    NodeGroup group(nodes_);
    ValidationContext ctx;
    ctx.mode = ValidationContext::Mode::Validate;
    ctx.strict = true;

    ContractRegistry reg;
    reg.add(validators::NonEmptyPipeline());
    reg.add(validators::NoNullNodes());
    reg.add(validators::UniqueDebugPointLabels());

    ValidationReport vrep = reg.validate(group, ctx);
    if (vrep.has_errors()) {
      rep.pipeline_string = "<builder-validation failed>";
      rep.repro_note = "validate: contract checks failed.\n" + vrep.to_string();
      return rep;
    }
  }

  if (!opt.parse_launch) {
    rep.pipeline_string = "<parse_launch disabled>";
    rep.repro_note = "validate(parse_launch=false): skipped.";
    return rep;
  }

  const bool insert_boundaries =
      should_insert_boundaries_for_mode("SIMA_GST_VALIDATE_INSERT_BOUNDARIES", true);

  BuildResult br = build_pipeline_full(nodes_, insert_boundaries, "mysink");
  rep.pipeline_string = br.pipeline_string;

  GError* err = nullptr;
  GstElement* pipeline = gst_parse_launch(rep.pipeline_string.c_str(), &err);
  if (!pipeline) {
    std::string msg = err ? err->message : "unknown";
    if (err) g_error_free(err);
    rep.repro_note = "validate: gst_parse_launch failed: " + msg;
    rep.repro_gst_launch = "gst-launch-1.0 -v '" + rep.pipeline_string + "'";
    return rep;
  }
  if (err) g_error_free(err);

  attach_boundary_probes(pipeline, br.diag);

  if (opt.enforce_names) {
    enforce_names_contract(pipeline, br);
  }

  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
  if (!sink) {
    maybe_dump_dot(pipeline, "validate_missing_mysink");
    rep.repro_note =
        "validate: appsink 'mysink' not found (ensure OutputAppSink is last).";
    stop_and_unref(pipeline);
    return rep;
  }

  try {
    set_state_or_throw(pipeline, GST_STATE_PAUSED, "PipelineSession::validate", br.diag);
  } catch (const PipelineError& e) {
    rep = e.report();
    rep.repro_note = std::string("validate: failed to PAUSE/preroll.\n") + rep.repro_note;
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    return rep;
  }

  const int timeout_ms =
      std::max(10, std::atoi(env_str("SIMA_GST_VALIDATE_TIMEOUT_MS", "2000").c_str()));

  GstSample* sample = nullptr;
#if GST_CHECK_VERSION(1,10,0)
  sample = gst_app_sink_try_pull_preroll(GST_APP_SINK(sink),
                                        (guint64)timeout_ms * GST_MSECOND);
#else
  sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink),
                                        (guint64)timeout_ms * GST_MSECOND);
#endif
  if (sample) gst_sample_unref(sample);

  // snapshot_basic already snapshots boundaries atomics -> BoundaryFlowStats.
  rep = br.diag->snapshot_basic();
  rep.pipeline_string = br.pipeline_string;

  rep.repro_note = sample
      ? "validate: preroll OK (PAUSED)."
      : "validate: preroll timed out in PAUSED (live source or negotiation stall).";
  rep.repro_note += "\n" + boundary_summary_local(br.diag);

  gst_object_unref(sink);
  stop_and_unref(pipeline);
  return rep;
}

RtspServerHandle PipelineSession::run_rtsp(const RtspServerOptions& opt) {
  gst_init_once();

  require_element("appsrc", "PipelineSession::run_rtsp");
  require_element("rtph264pay", "PipelineSession::run_rtsp");
  require_element("h264parse", "PipelineSession::run_rtsp");

  AppSrcImage* src_img = nullptr;
  for (auto& n : nodes_) {
    if (!src_img) {
      src_img = dynamic_cast<AppSrcImage*>(n.get());
    }
  }
  if (!src_img) {
    throw std::runtime_error("PipelineSession::run_rtsp: missing AppSrcImage node");
  }

  std::ostringstream ss;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (i) ss << " ! ";
    ss << nodes_[i]->gst_fragment((int)i);
  }
  last_pipeline_ = "( " + ss.str() + " )";

  auto* impl = new RtspServerImpl();
  impl->port = opt.port;
  impl->mount_path = ensure_mount_path(opt.mount);
  impl->url = make_rtsp_url(opt.port, opt.mount);
  impl->enc_w = src_img->enc_w();
  impl->enc_h = src_img->enc_h();
  impl->fps = src_img->fps();
  impl->nv12_enc = src_img->nv12_enc();

  RtspServerHandle handle;
  handle.url_ = impl->url;
  handle.impl_ = impl;

  impl->th = std::thread([impl, launch = last_pipeline_]() {
    GstRTSPServer* server = gst_rtsp_server_new();
    gst_rtsp_server_set_service(server, std::to_string(impl->port).c_str());

    GstRTSPMountPoints* mounts = gst_rtsp_server_get_mount_points(server);
    GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, launch.c_str());
    gst_rtsp_media_factory_set_shared(factory, FALSE);

    g_signal_connect(
        factory,
        "media-configure",
        G_CALLBACK(+[](GstRTSPMediaFactory*, GstRTSPMedia* media, gpointer user_data) {
          auto* impl = reinterpret_cast<RtspServerImpl*>(user_data);
          if (!impl) return;

          GstElement* top = gst_rtsp_media_get_element(media);
          if (!top) return;

          GstElement* src = gst_bin_get_by_name_recurse_up(GST_BIN(top), "mysrc");
          if (!src) {
            gst_object_unref(top);
            return;
          }

          GstCaps* caps = gst_caps_new_simple(
              "video/x-raw",
              "format", G_TYPE_STRING, "NV12",
              "width", G_TYPE_INT, impl->enc_w,
              "height", G_TYPE_INT, impl->enc_h,
              "framerate", GST_TYPE_FRACTION, impl->fps, 1,
              nullptr);
          gst_app_src_set_caps(GST_APP_SRC(src), caps);
          gst_caps_unref(caps);

          g_object_set(G_OBJECT(src),
                       "is-live", TRUE,
                       "format", GST_FORMAT_TIME,
                       "do-timestamp", FALSE,
                       nullptr);

          auto* pc = new PushCtx();
          pc->appsrc = (GstElement*)gst_object_ref(src);
          pc->w = impl->enc_w;
          pc->h = impl->enc_h;
          pc->fps = impl->fps;
          pc->nv12 = impl->nv12_enc;
          pc->frame_duration_ns = gst_util_uint64_scale_int(GST_SECOND, 1, pc->fps);

          const int period_ms = std::max(1, 1000 / pc->fps);
          g_signal_connect(media, "unprepared", G_CALLBACK(media_unprepared_cb), pc);
          pc->timer_id = g_timeout_add(period_ms, push_frame_cb, pc);

          gst_object_unref(src);
          gst_object_unref(top);
        }),
        impl);

    gst_rtsp_mount_points_add_factory(mounts, impl->mount_path.c_str(), factory);
    g_object_unref(mounts);

    if (gst_rtsp_server_attach(server, nullptr) == 0) {
      g_object_unref(server);
      return;
    }

    impl->loop = g_main_loop_new(nullptr, FALSE);
    impl->running.store(true);
    g_main_loop_run(impl->loop);

    impl->running.store(false);
    g_main_loop_unref(impl->loop);
    impl->loop = nullptr;

    g_object_unref(server);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  return handle;
}

RunDebugResult PipelineSession::run_debug(const RunDebugOptions& opt) {
  RunDebugResult out;

  gst_init_once();

  std::vector<int> dbg_indices;
  std::vector<std::string> dbg_labels;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i] && nodes_[i]->kind() == "DebugPoint") {
      std::string label = nodes_[i]->user_label();
      if (label.empty()) label = "dbg";
      dbg_indices.push_back(static_cast<int>(i));
      dbg_labels.push_back(std::move(label));
    }
  }

  if (dbg_indices.empty()) {
    try {
      FrameStream fs = run();
      out.first_frame = fs.next_copy(opt.timeout_ms);
      out.report = fs.report_snapshot(/*heavy=*/true);
      fs.close();
      if (out.report.repro_note.empty()) {
        out.report.repro_note = "run_debug: OK (no DebugPoint nodes)";
      }
    } catch (const PipelineError& e) {
      out.report = e.report();
      if (out.report.repro_note.empty()) {
        out.report.repro_note = e.what();
      }
    } catch (const std::exception& e) {
      out.report.pipeline_string = last_pipeline_;
      out.report.repro_note = std::string("run_debug: exception: ") + e.what();
    }
    return out;
  }

  require_element("appsink", "PipelineSession::run_debug");
  require_element("identity", "PipelineSession::run_debug");
  if (dbg_indices.size() > 1) {
    require_element("appsrc", "PipelineSession::run_debug");
  }

  std::vector<std::string> segment_strings;
  std::vector<std::string> sink_names;
  std::vector<std::string> src_names;
  segment_strings.reserve(dbg_indices.size());
  sink_names.reserve(dbg_indices.size());
  src_names.reserve(dbg_indices.size());

  for (size_t i = 0; i < dbg_indices.size(); ++i) {
    const std::string label = sanitize_name(dbg_labels[i]);
    sink_names.push_back("dbg_sink_" + label + "_" + std::to_string(i));
    src_names.push_back("dbg_src_" + std::to_string(i));
  }

  int prev = -1;
  for (size_t i = 0; i < dbg_indices.size(); ++i) {
    const int start_idx = prev + 1;
    const int end_idx = dbg_indices[i];
    const bool with_appsrc = (i > 0);
    const std::string seg = build_debug_segment(
        nodes_, start_idx, end_idx, with_appsrc, src_names[i], sink_names[i]);
    segment_strings.push_back(seg);
    prev = end_idx;
  }

  std::ostringstream combined;
  for (size_t i = 0; i < segment_strings.size(); ++i) {
    if (i) combined << "\n";
    combined << "SEG" << i << ": " << segment_strings[i];
  }
  last_pipeline_ = combined.str();

  auto diag = std::make_shared<DiagCtx>();
  diag->pipeline_string = last_pipeline_;
  diag->node_reports.reserve(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (!nodes_[i]) continue;
    NodeReport nr;
    nr.index = static_cast<int>(i);
    nr.kind = nodes_[i]->kind();
    nr.user_label = nodes_[i]->user_label();
    nr.gst_fragment = nodes_[i]->gst_fragment(static_cast<int>(i));
    nr.elements = nodes_[i]->element_names(static_cast<int>(i));
    diag->node_reports.push_back(std::move(nr));
  }

  std::vector<GstElement*> pipelines(segment_strings.size(), nullptr);
  std::vector<GstElement*> sinks(segment_strings.size(), nullptr);
  std::vector<GstElement*> srcs(segment_strings.size(), nullptr);
  std::vector<std::string> seg_errors(segment_strings.size());

  auto cleanup = [&]() {
    for (auto* s : sinks) {
      if (s) gst_object_unref(s);
    }
    for (auto* s : srcs) {
      if (s) gst_object_unref(s);
    }
    for (auto*& p : pipelines) {
      if (p) stop_and_unref(p);
    }
  };

  try {
    for (size_t i = 0; i < segment_strings.size(); ++i) {
      GError* err = nullptr;
      pipelines[i] = gst_parse_launch(segment_strings[i].c_str(), &err);
      if (!pipelines[i]) {
        std::string msg = err ? err->message : "unknown";
        if (err) g_error_free(err);
        seg_errors[i] = "gst_parse_launch failed: " + msg;
        continue;
      }
      if (err) g_error_free(err);

      sinks[i] = gst_bin_get_by_name(GST_BIN(pipelines[i]), sink_names[i].c_str());
      if (!sinks[i]) {
        seg_errors[i] = "appsink '" + sink_names[i] + "' not found";
      }

      if (i > 0) {
        srcs[i] = gst_bin_get_by_name(GST_BIN(pipelines[i]), src_names[i].c_str());
        if (!srcs[i]) {
          if (!seg_errors[i].empty()) seg_errors[i] += "; ";
          seg_errors[i] += "appsrc '" + src_names[i] + "' not found";
        }
      }
    }

    for (size_t i = 0; i < pipelines.size(); ++i) {
      if (!pipelines[i] || !seg_errors[i].empty()) continue;
      GstStateChangeReturn r = gst_element_set_state(pipelines[i], GST_STATE_PLAYING);
      if (r == GST_STATE_CHANGE_FAILURE) {
        seg_errors[i] = "failed to set PLAYING";
        continue;
      }
      GstState cur = GST_STATE_VOID_PENDING;
      GstState pend = GST_STATE_VOID_PENDING;
      GstStateChangeReturn r2 =
          gst_element_get_state(pipelines[i], &cur, &pend, 2 * GST_SECOND);
      if (r2 == GST_STATE_CHANGE_FAILURE) {
        seg_errors[i] = "failed to reach PLAYING";
      }
    }

    out.taps.clear();
    out.taps.reserve(dbg_indices.size());

    for (size_t i = 0; i < dbg_indices.size(); ++i) {
      RunDebugTap tap;
      tap.name = dbg_labels[i];

      if (!seg_errors[i].empty()) {
        tap.error = seg_errors[i];
        out.taps.push_back(std::move(tap));
        continue;
      }

      try {
        auto sample_opt = pipeline_internal::try_pull_sample_sliced(
            pipelines[i],
            sinks[i],
            opt.timeout_ms,
            diag,
            "PipelineSession::run_debug");

        if (!sample_opt.has_value()) {
          tap.error = "timeout or EOS";
          out.taps.push_back(std::move(tap));
          continue;
        }

        GstSample* sample = sample_opt.value();
        tap.packet = sample_to_tap_packet(sample, dbg_indices[i], sink_names[i]);

        try {
          tap.tensor = pipeline_internal::sample_to_tensor_ref(sample);
          tap.last_good_tensor = tap.tensor;
          out.tensors.emplace(tap.name, *tap.tensor);
        } catch (...) {
          // Non-raw taps are allowed; tensor conversion is best-effort.
        }

        if (i + 1 < dbg_indices.size() &&
            pipelines[i + 1] && srcs[i + 1] && seg_errors[i + 1].empty()) {
          GstCaps* caps = gst_sample_get_caps(sample);
          if (caps) {
            gst_app_src_set_caps(GST_APP_SRC(srcs[i + 1]), caps);
          }

          GstBuffer* buf = gst_sample_get_buffer(sample);
          if (buf) {
            GstBuffer* b = gst_buffer_ref(buf);
            gst_app_src_push_buffer(GST_APP_SRC(srcs[i + 1]), b);
            gst_app_src_end_of_stream(GST_APP_SRC(srcs[i + 1]));
          }
        }

        gst_sample_unref(sample);
        out.taps.push_back(std::move(tap));
      } catch (const PipelineError& e) {
        tap.error = e.what();
        out.taps.push_back(std::move(tap));
      } catch (const std::exception& e) {
        tap.error = e.what();
        out.taps.push_back(std::move(tap));
      }
    }

    for (size_t i = 0; i < pipelines.size(); ++i) {
      pipeline_internal::drain_bus(pipelines[i], diag, "PipelineSession::run_debug");
    }

    out.report = diag->snapshot_basic();
    out.report.pipeline_string = last_pipeline_;

    bool any_error = false;
    for (const auto& t : out.taps) {
      if (!t.error.empty()) {
        any_error = true;
        break;
      }
    }
    if (out.report.repro_note.empty()) {
      out.report.repro_note = any_error
          ? "run_debug: partial results (see per-tap errors)"
          : "run_debug: OK";
    }

    cleanup();
    return out;
  } catch (const PipelineError& e) {
    out.report = e.report();
    if (out.report.repro_note.empty()) {
      out.report.repro_note = e.what();
    }
  } catch (const std::exception& e) {
    out.report.pipeline_string = last_pipeline_;
    out.report.repro_note = std::string("run_debug: exception: ") + e.what();
  }

  cleanup();
  return out;
}

PipelineSession& PipelineSession::add_output_tensor(const OutputTensorOptions& opt) {
  OutputTensorOptions o = opt;
  if (o.format.empty()) o.format = "RGB";
  if (o.dtype != TensorDType::UInt8) {
    throw std::runtime_error("add_output_tensor: only UInt8 is supported for now");
  }

  if (o.use_videoconvert) add(nodes::VideoConvert());
  if (o.use_videoscale) add(nodes::VideoScale());

  // Force SystemMemory to keep CPU-accessible tensors for future bindings.
  add(nodes::CapsRaw(o.format,
                     o.target_width,
                     o.target_height,
                     o.target_fps,
                     sima::CapsMemory::SystemMemory));
  add(nodes::OutputAppSink());
  return *this;
}

PipelineSession& PipelineSession::add_output_torch(const OutputTensorOptions& opt) {
  OutputTensorOptions o = opt;
  o.layout = TensorLayout::CHW;
  return add_output_tensor(o);
}

PipelineSession& PipelineSession::add_output_numpy(const OutputTensorOptions& opt) {
  OutputTensorOptions o = opt;
  o.layout = TensorLayout::HWC;
  return add_output_tensor(o);
}

PipelineSession& PipelineSession::add_output_tensorflow(const OutputTensorOptions& opt) {
  OutputTensorOptions o = opt;
  o.layout = TensorLayout::HWC;
  return add_output_tensor(o);
}

std::string PipelineSession::describe(const GraphPrinter::Options& opt) const {
  NodeGroup group(nodes_);
  return GraphPrinter::to_text(group, opt);
}

std::string PipelineSession::to_gst(bool insert_boundaries) const {
  BuildResult br = build_pipeline_full(nodes_, insert_boundaries, "mysink");
  return br.pipeline_string;
}

void PipelineSession::save(const std::string& path) const {
  std::ostringstream oss;
  oss << "{\n  \"version\": 1,\n  \"nodes\": [\n";

  for (size_t i = 0; i < nodes_.size(); ++i) {
    const auto& n = nodes_[i];
    if (!n) continue;
    if (i) oss << ",\n";

    const std::string kind = n->kind();
    const std::string label = n->user_label();
    const std::string fragment = n->gst_fragment(static_cast<int>(i));
    const auto elements = n->element_names(static_cast<int>(i));

    oss << "    {\"kind\":\"" << json_escape(kind) << "\","
        << "\"label\":\"" << json_escape(label) << "\","
        << "\"fragment\":\"" << json_escape(fragment) << "\","
        << "\"elements\":[";

    for (size_t e = 0; e < elements.size(); ++e) {
      if (e) oss << ",";
      oss << "\"" << json_escape(elements[e]) << "\"";
    }
    oss << "]}";
  }

  oss << "\n  ]\n}\n";

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("PipelineSession::save: failed to open file");
  }
  out << oss.str();
}

PipelineSession PipelineSession::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("PipelineSession::load: failed to open file");
  }
  std::ostringstream buf;
  buf << in.rdbuf();

  std::string content = buf.str();
  JsonParser parser(content);
  JsonValue root = parser.parse();
  if (root.type != JsonValue::Type::Object) {
    throw std::runtime_error("PipelineSession::load: invalid JSON root");
  }

  auto it_nodes = root.obj.find("nodes");
  if (it_nodes == root.obj.end() || it_nodes->second.type != JsonValue::Type::Array) {
    throw std::runtime_error("PipelineSession::load: missing nodes array");
  }

  PipelineSession sess;
  const auto& arr = it_nodes->second.arr;
  for (const auto& n : arr) {
    if (n.type != JsonValue::Type::Object) {
      throw std::runtime_error("PipelineSession::load: node entry must be object");
    }

    const auto it_kind = n.obj.find("kind");
    const auto it_label = n.obj.find("label");
    const auto it_frag = n.obj.find("fragment");
    const auto it_elems = n.obj.find("elements");

    std::string kind = (it_kind != n.obj.end() && it_kind->second.type == JsonValue::Type::String)
        ? it_kind->second.str : "Gst";
    std::string label = (it_label != n.obj.end() && it_label->second.type == JsonValue::Type::String)
        ? it_label->second.str : "";
    std::string fragment = (it_frag != n.obj.end() && it_frag->second.type == JsonValue::Type::String)
        ? it_frag->second.str : "";

    std::vector<std::string> elements;
    if (it_elems != n.obj.end() && it_elems->second.type == JsonValue::Type::Array) {
      for (const auto& e : it_elems->second.arr) {
        if (e.type == JsonValue::Type::String) {
          elements.push_back(e.str);
        }
      }
    }

    if (fragment.empty()) {
      throw std::runtime_error("PipelineSession::load: node fragment is empty");
    }

    sess.add(std::make_shared<ConfiguredNode>(kind, label, fragment, elements));
  }

  return sess;
}

} // namespace sima
