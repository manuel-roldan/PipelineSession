// PipelineSession.cpp
#include "PipelineSession.h"

#include <gst/gst.h>
#include <gst/gstdebugutils.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/video/video.h>
#include <glib.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <condition_variable>

namespace fs = std::filesystem;

namespace sima {

// =====================================================================================
// Helpers
// =====================================================================================

static void gst_init_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    int argc = 0;
    char** argv = nullptr;
    gst_init(&argc, &argv);
  });
}

static bool env_bool(const char* key, bool def_val) {
  const char* v = std::getenv(key);
  if (!v || !*v) return def_val;
  if (!std::strcmp(v, "1") || !std::strcmp(v, "true") || !std::strcmp(v, "TRUE") ||
      !std::strcmp(v, "yes") || !std::strcmp(v, "YES") || !std::strcmp(v, "on")  || !std::strcmp(v, "ON")) {
    return true;
  }
  if (!std::strcmp(v, "0") || !std::strcmp(v, "false") || !std::strcmp(v, "FALSE") ||
      !std::strcmp(v, "no") || !std::strcmp(v, "NO") || !std::strcmp(v, "off") || !std::strcmp(v, "OFF")) {
    return false;
  }
  return def_val;
}

static std::string env_str(const char* key, const std::string& def_val = "") {
  const char* v = std::getenv(key);
  if (!v) return def_val;
  return std::string(v);
}

static uint64_t now_mono_us() {
  return (uint64_t)g_get_monotonic_time();
}

static const char* state_name(GstState s) {
  switch (s) {
    case GST_STATE_VOID_PENDING: return "VOID_PENDING";
    case GST_STATE_NULL:         return "NULL";
    case GST_STATE_READY:        return "READY";
    case GST_STATE_PAUSED:       return "PAUSED";
    case GST_STATE_PLAYING:      return "PLAYING";
    default:                     return "UNKNOWN";
  }
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

static std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\\': o += "\\\\"; break;
      case '"':  o += "\\\""; break;
      case '\n': o += "\\n";  break;
      case '\r': o += "\\r";  break;
      case '\t': o += "\\t";  break;
      default:
        if ((unsigned char)c < 0x20) o += " ";
        else o += c;
    }
  }
  return o;
}

static std::string gst_caps_to_string_safe(GstCaps* caps) {
  if (!caps) return "<null caps>";
  gchar* s = gst_caps_to_string(caps);
  if (!s) return "<caps_to_string failed>";
  std::string out = s;
  g_free(s);
  return out;
}

static std::string gst_structure_to_string_safe(const GstStructure* st) {
  if (!st) return "<null structure>";
  gchar* s = gst_structure_to_string(st);
  if (!s) return "<structure_to_string failed>";
  std::string out = s;
  g_free(s);
  return out;
}

static std::string gst_message_to_string(GstMessage* msg) {
  if (!msg) return "<null message>";
  std::ostringstream ss;
  const GstMessageType t = GST_MESSAGE_TYPE(msg);
  ss << gst_message_type_get_name(t);

  if (t == GST_MESSAGE_ERROR) {
    GError* e = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_error(msg, &e, &dbg);
    ss << ": " << (e ? e->message : "unknown");
    if (dbg && *dbg) ss << " | " << dbg;
    if (e) g_error_free(e);
    if (dbg) g_free(dbg);
    return ss.str();
  }
  if (t == GST_MESSAGE_WARNING) {
    GError* e = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_warning(msg, &e, &dbg);
    ss << ": " << (e ? e->message : "unknown");
    if (dbg && *dbg) ss << " | " << dbg;
    if (e) g_error_free(e);
    if (dbg) g_free(dbg);
    return ss.str();
  }
  if (t == GST_MESSAGE_INFO) {
    GError* e = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_info(msg, &e, &dbg);
    ss << ": " << (e ? e->message : "unknown");
    if (dbg && *dbg) ss << " | " << dbg;
    if (e) g_error_free(e);
    if (dbg) g_free(dbg);
    return ss.str();
  }
  if (t == GST_MESSAGE_STATE_CHANGED) {
    if (GST_MESSAGE_SRC(msg) && GST_IS_OBJECT(GST_MESSAGE_SRC(msg))) {
      ss << " src=" << GST_OBJECT_NAME(GST_MESSAGE_SRC(msg));
    }
    GstState old_s, new_s, pend_s;
    gst_message_parse_state_changed(msg, &old_s, &new_s, &pend_s);
    ss << " " << state_name(old_s) << " -> " << state_name(new_s)
       << " (pending " << state_name(pend_s) << ")";
    return ss.str();
  }
  if (t == GST_MESSAGE_EOS) {
    ss << " (EOS)";
    return ss.str();
  }
  if (t == GST_MESSAGE_ASYNC_DONE) {
    ss << " (ASYNC_DONE)";
    return ss.str();
  }
  if (t == GST_MESSAGE_STREAM_START) {
    ss << " (STREAM_START)";
    return ss.str();
  }

  const GstStructure* st = gst_message_get_structure(msg);
  if (st) ss << " " << gst_structure_to_string_safe(st);
  return ss.str();
}

static bool element_exists(const char* factory) {
  gst_init_once();
  GstElementFactory* f = gst_element_factory_find(factory);
  if (f) { gst_object_unref(f); return true; }
  return false;
}

static void require_element(const char* factory, const char* context) {
  if (!element_exists(factory)) {
    std::ostringstream ss;
    ss << context << ": required GStreamer element not found: " << factory;
    throw std::runtime_error(ss.str());
  }
}

static void maybe_dump_dot(GstElement* pipeline, const std::string& tag) {
  if (!pipeline) return;
  const std::string dir = env_str("SIMA_GST_DOT_DIR", "");
  if (dir.empty()) return;
  g_setenv("GST_DEBUG_DUMP_DOT_DIR", dir.c_str(), TRUE);
  const std::string t = "sima_" + sanitize_name(tag);
  gst_debug_bin_to_dot_file_with_ts(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, t.c_str());
}

// =====================================================================================
// PipelineReport JSON
// =====================================================================================

std::string PipelineReport::to_json() const {
  std::ostringstream ss;
  ss << "{";
  ss << "\"pipeline_string\":\"" << json_escape(pipeline_string) << "\",";
  ss << "\"nodes\":[";
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (i) ss << ",";
    const auto& n = nodes[i];
    ss << "{";
    ss << "\"index\":" << n.index << ",";
    ss << "\"kind\":\"" << json_escape(n.kind) << "\",";
    ss << "\"user_label\":\"" << json_escape(n.user_label) << "\",";
    ss << "\"gst_fragment\":\"" << json_escape(n.gst_fragment) << "\",";
    ss << "\"elements\":[";
    for (size_t j = 0; j < n.elements.size(); ++j) {
      if (j) ss << ",";
      ss << "\"" << json_escape(n.elements[j]) << "\"";
    }
    ss << "]";
    ss << "}";
  }
  ss << "],";

  ss << "\"bus\":[";
  for (size_t i = 0; i < bus.size(); ++i) {
    if (i) ss << ",";
    const auto& m = bus[i];
    ss << "{"
       << "\"type\":\"" << json_escape(m.type) << "\","
       << "\"src\":\"" << json_escape(m.src) << "\","
       << "\"detail\":\"" << json_escape(m.detail) << "\","
       << "\"wall_time_us\":" << m.wall_time_us
       << "}";
  }
  ss << "],";

  ss << "\"boundaries\":[";
  for (size_t i = 0; i < boundaries.size(); ++i) {
    if (i) ss << ",";
    const auto& b = boundaries[i];
    ss << "{"
       << "\"boundary_name\":\"" << json_escape(b.boundary_name) << "\","
       << "\"after_node_index\":" << b.after_node_index << ","
       << "\"before_node_index\":" << b.before_node_index << ","
       << "\"in_buffers\":" << b.in_buffers << ","
       << "\"out_buffers\":" << b.out_buffers << ","
       << "\"last_in_pts_ns\":" << b.last_in_pts_ns << ","
       << "\"last_out_pts_ns\":" << b.last_out_pts_ns << ","
       << "\"last_in_wall_us\":" << b.last_in_wall_us << ","
       << "\"last_out_wall_us\":" << b.last_out_wall_us
       << "}";
  }
  ss << "],";

  ss << "\"caps_dump\":\"" << json_escape(caps_dump) << "\",";
  ss << "\"dot_paths\":[";
  for (size_t i = 0; i < dot_paths.size(); ++i) {
    if (i) ss << ",";
    ss << "\"" << json_escape(dot_paths[i]) << "\"";
  }
  ss << "],";

  ss << "\"repro_gst_launch\":\"" << json_escape(repro_gst_launch) << "\",";
  ss << "\"repro_env\":\"" << json_escape(repro_env) << "\",";
  ss << "\"repro_note\":\"" << json_escape(repro_note) << "\"";
  ss << "}";
  return ss.str();
}

// =====================================================================================
// PipelineError
// =====================================================================================

PipelineError::PipelineError(std::string msg, PipelineReport report)
  : std::runtime_error(std::move(msg)), report_(std::move(report)) {}

// =====================================================================================
// Diagnostics context (cheap always-on; heavy on demand)
// =====================================================================================

struct DiagCtx {
  std::string pipeline_string;
  std::vector<NodeReport> node_reports;

  // Boundary stats (owned pointers so probes keep stable addresses)
  std::vector<std::unique_ptr<BoundaryFlowStats>> boundaries;

  // bus collection
  mutable std::mutex bus_mu;
  std::vector<BusMessage> bus;

  static int64_t now_us() { return (int64_t)g_get_monotonic_time(); }

  void push_bus(const std::string& type, const std::string& src, const std::string& detail) {
    std::lock_guard<std::mutex> lk(bus_mu);
    bus.push_back(BusMessage{type, src, detail, now_us()});
  }

  PipelineReport snapshot_basic() const {
    PipelineReport rep;
    rep.pipeline_string = pipeline_string;
    rep.nodes = node_reports;

    {
      std::lock_guard<std::mutex> lk(bus_mu);
      rep.bus = bus;
    }

    rep.boundaries.reserve(boundaries.size());
    for (const auto& b : boundaries) {
      if (!b) continue;
      rep.boundaries.push_back(*b);
    }

    // Fill repro helpers (always cheap)
    rep.repro_gst_launch = "gst-launch-1.0 -v '" + pipeline_string + "'";
    rep.repro_env =
      "Suggested env vars:\n"
      "  export GST_DEBUG=3,*sima*:6\n"
      "  export GST_DEBUG_NO_COLOR=1\n"
      "  export SIMA_GST_DOT_DIR=/tmp/sima_dot\n"
      "  export GST_DEBUG_DUMP_DOT_DIR=$SIMA_GST_DOT_DIR\n";
    return rep;
  }
};

// =====================================================================================
// Boundary probes (deterministic “where stuck”)
// =====================================================================================

struct BoundaryProbeCtx {
  BoundaryFlowStats* stats = nullptr;
  bool is_in = false; // true => identity sink pad, false => identity src pad
};

static GstPadProbeReturn boundary_probe_cb(GstPad* /*pad*/, GstPadProbeInfo* info, gpointer user_data) {
  auto* ctx = reinterpret_cast<BoundaryProbeCtx*>(user_data);
  if (!ctx || !ctx->stats) return GST_PAD_PROBE_OK;

  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0) return GST_PAD_PROBE_OK;

  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf) return GST_PAD_PROBE_OK;

  const int64_t now = (int64_t)g_get_monotonic_time();

  const GstClockTime pts = GST_BUFFER_PTS(buf);
  const int64_t pts_ns = (pts == GST_CLOCK_TIME_NONE) ? -1 : (int64_t)pts;

  if (ctx->is_in) {
    ctx->stats->in_buffers++;
    ctx->stats->last_in_wall_us = now;
    if (pts_ns >= 0) ctx->stats->last_in_pts_ns = pts_ns;
  } else {
    ctx->stats->out_buffers++;
    ctx->stats->last_out_wall_us = now;
    if (pts_ns >= 0) ctx->stats->last_out_pts_ns = pts_ns;
  }
  return GST_PAD_PROBE_OK;
}

static void attach_boundary_probes(GstElement* pipeline, std::shared_ptr<DiagCtx>& diag) {
  if (!pipeline || !diag) return;
  if (!env_bool("SIMA_GST_BOUNDARY_PROBES", false)) return; // keep run() hot path clean by default

  for (auto& bptr : diag->boundaries) {
    auto* b = bptr.get();
    if (!b) continue;

    GstElement* ident = gst_bin_get_by_name(GST_BIN(pipeline), b->boundary_name.c_str());
    if (!ident) continue;

    GstPad* sink = gst_element_get_static_pad(ident, "sink");
    GstPad* src  = gst_element_get_static_pad(ident, "src");

    if (sink) {
      auto* ctx = new BoundaryProbeCtx();
      ctx->stats = b;
      ctx->is_in = true;
      gst_pad_add_probe(sink, GST_PAD_PROBE_TYPE_BUFFER, boundary_probe_cb, ctx,
                        +[](gpointer p) { delete reinterpret_cast<BoundaryProbeCtx*>(p); });
      gst_object_unref(sink);
    }
    if (src) {
      auto* ctx = new BoundaryProbeCtx();
      ctx->stats = b;
      ctx->is_in = false;
      gst_pad_add_probe(src, GST_PAD_PROBE_TYPE_BUFFER, boundary_probe_cb, ctx,
                        +[](gpointer p) { delete reinterpret_cast<BoundaryProbeCtx*>(p); });
      gst_object_unref(src);
    }

    gst_object_unref(ident);
  }
}

static std::string boundary_summary(const std::shared_ptr<DiagCtx>& diag) {
  if (!diag || diag->boundaries.empty()) return "";

  const int64_t now = (int64_t)g_get_monotonic_time();
  int best_idx = -1;
  int64_t best_t = 0;
  bool best_out = false;

  for (size_t i = 0; i < diag->boundaries.size(); ++i) {
    auto* b = diag->boundaries[i].get();
    if (!b) continue;
    if (b->last_out_wall_us > best_t) { best_t = b->last_out_wall_us; best_idx = (int)i; best_out = true; }
    if (b->last_in_wall_us  > best_t) { best_t = b->last_in_wall_us;  best_idx = (int)i; best_out = false; }
  }

  std::ostringstream ss;
  ss << "BoundaryFlow:\n";
  for (auto& bp : diag->boundaries) {
    if (!bp) continue;
    ss << "  - " << bp->boundary_name
       << " after=" << bp->after_node_index
       << " before=" << bp->before_node_index
       << " in=" << bp->in_buffers
       << " out=" << bp->out_buffers
       << " last_in_age_ms=" << (bp->last_in_wall_us ? ((now - bp->last_in_wall_us) / 1000) : 0)
       << " last_out_age_ms=" << (bp->last_out_wall_us ? ((now - bp->last_out_wall_us) / 1000) : 0)
       << "\n";
  }
  if (best_idx >= 0 && best_t > 0) {
    auto* b = diag->boundaries[(size_t)best_idx].get();
    ss << "LikelyStall: last activity "
       << (best_out ? "leaving " : "entering ")
       << b->boundary_name
       << " age_ms=" << ((now - best_t) / 1000)
       << " (after node " << b->after_node_index
       << ", before node " << b->before_node_index << ")\n";
  }
  return ss.str();
}

// =====================================================================================
// Bus draining (cheap always-on)
// =====================================================================================

static void drain_bus(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag) {
  if (!pipeline) return;
  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus) return;

  while (GstMessage* msg = gst_bus_pop(bus)) {
    const GstMessageType t = GST_MESSAGE_TYPE(msg);
    const char* src = (GST_MESSAGE_SRC(msg) && GST_IS_OBJECT(GST_MESSAGE_SRC(msg)))
                        ? GST_OBJECT_NAME(GST_MESSAGE_SRC(msg))
                        : "<unknown>";
    std::string line = gst_message_to_string(msg);
    if (diag) diag->push_bus(gst_message_type_get_name(t), src ? src : "<unknown>", line);
    gst_message_unref(msg);
  }
  gst_object_unref(bus);
}

static void throw_if_bus_error(GstElement* pipeline, const std::shared_ptr<DiagCtx>& diag, const char* where) {
  if (!pipeline) return;

  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus) return;

  while (GstMessage* msg = gst_bus_pop(bus)) {
    const GstMessageType t = GST_MESSAGE_TYPE(msg);
    const char* src = (GST_MESSAGE_SRC(msg) && GST_IS_OBJECT(GST_MESSAGE_SRC(msg)))
                        ? GST_OBJECT_NAME(GST_MESSAGE_SRC(msg))
                        : "<unknown>";
    std::string line = gst_message_to_string(msg);
    if (diag) diag->push_bus(gst_message_type_get_name(t), src ? src : "<unknown>", line);

    if (t == GST_MESSAGE_ERROR) {
      gst_message_unref(msg);
      gst_object_unref(bus);
      if (pipeline) maybe_dump_dot(pipeline, std::string(where) + "_error");

      PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
      rep.repro_note = std::string(where) + ": GST ERROR: " + line;
      if (diag) rep.repro_note += "\n" + boundary_summary(diag);
      throw PipelineError(rep.repro_note, std::move(rep));
    }

    gst_message_unref(msg);
  }
  gst_object_unref(bus);
}

// =====================================================================================
// Caps / memory contract helpers
// =====================================================================================

static std::string caps_features_string(GstCaps* caps) {
  if (!caps) return "<none>";
  GstCapsFeatures* f = gst_caps_get_features(caps, 0);
  if (!f) return "<none>";
#if GST_CHECK_VERSION(1,16,0)
  gchar* s = gst_caps_features_to_string(f);
  if (!s) return "<none>";
  std::string out = s;
  g_free(s);
  return out;
#else
  // Best-effort on older versions.
  if (gst_caps_features_is_any(f)) return "ANY";
  if (gst_caps_features_contains(f, GST_CAPS_FEATURE_MEMORY_SYSTEM_MEMORY)) return "memory:SystemMemory";
  return "<features>";
#endif
}

static void require_system_memory_or_throw(GstCaps* caps, const std::string& where, const std::shared_ptr<DiagCtx>& diag) {
  if (!caps) {
    PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
    rep.repro_note = where + ": missing caps";
    throw PipelineError(rep.repro_note, std::move(rep));
  }
  GstCapsFeatures* f = gst_caps_get_features(caps, 0);
  if (!f) return; // permissive if absent
  if (!gst_caps_features_is_any(f) &&
      !gst_caps_features_contains(f, GST_CAPS_FEATURE_MEMORY_SYSTEM_MEMORY)) {
    PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
    rep.repro_note =
      where + ": illegal buffer memory type (not SystemMemory).\nCaps: " + gst_caps_to_string_safe(caps) + "\n";
    throw PipelineError(rep.repro_note, std::move(rep));
  }
}

// =====================================================================================
// Stop/unref pipeline
// =====================================================================================

static void stop_and_unref(GstElement*& e) {
  if (!e) return;

  // Move ownership out so callers can't double-close.
  GstElement* local = e;
  e = nullptr;

  // Teardown can deadlock inside plugin change_state.
  // Do it async and bail out if it takes too long.
  auto done = std::make_shared<std::atomic<bool>>(false);

  std::thread([local, done]() {
    // Best-effort: nudge EOS (non-blocking)
    gst_element_send_event(local, gst_event_new_eos());

    // This is where you currently hang sometimes.
    gst_element_set_state(local, GST_STATE_NULL);

    // Unref pipeline
    gst_object_unref(local);

    done->store(true);
  }).detach();

  // Watchdog: don't block forever.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (done->load()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  std::cerr << "[WARN] stop_and_unref(): teardown timed out (likely simaaidecoder state-change deadlock). "
               "Leaking pipeline to avoid hanging.\n";
  // Intentionally leak if still stuck; process/test can continue.
}

// =====================================================================================
// Raw tight packer for TapStream
// =====================================================================================

static TapFormat infer_tap_format_and_meta(GstCaps* caps, TapPacket& out) {
  out.format = TapFormat::Unknown;
  out.video = TapVideoInfo{};
  if (!caps) return out.format;

  out.caps_string = gst_caps_to_string_safe(caps);
  if (gst_caps_is_empty(caps) || gst_caps_is_any(caps)) return out.format;

  const GstStructure* st = gst_caps_get_structure(caps, 0);
  if (!st) return out.format;
  const char* media = gst_structure_get_name(st);
  if (!media) return out.format;

  if (std::strcmp(media, "video/x-raw") == 0) {
    const gchar* fmt = gst_structure_get_string(st, "format");
    if (fmt) out.video.format = fmt;
    gst_structure_get_int(st, "width", &out.video.width);
    gst_structure_get_int(st, "height", &out.video.height);
    int n = 0, d = 1;
    if (gst_structure_get_fraction(st, "framerate", &n, &d)) {
      out.video.fps_num = n;
      out.video.fps_den = d;
    }
    if (fmt) {
      if (std::strcmp(fmt, "NV12") == 0) out.format = TapFormat::NV12;
      else if (std::strcmp(fmt, "I420") == 0) out.format = TapFormat::I420;
      else if (std::strcmp(fmt, "RGB") == 0) out.format = TapFormat::RGB;
      else if (std::strcmp(fmt, "BGR") == 0) out.format = TapFormat::BGR;
      else if (std::strcmp(fmt, "GRAY8") == 0) out.format = TapFormat::GRAY8;
      else out.format = TapFormat::Unknown;
    }
    return out.format;
  }

  if (std::strcmp(media, "video/x-h264") == 0) { out.format = TapFormat::H264; return out.format; }
  if (std::strcmp(media, "video/x-h265") == 0) { out.format = TapFormat::H265; return out.format; }
  if (std::strcmp(media, "image/jpeg") == 0) { out.format = TapFormat::JPEG; return out.format; }
  if (std::strcmp(media, "image/png") == 0)  { out.format = TapFormat::PNG;  return out.format; }

  return out.format;
}

static bool pack_raw_video_tight(GstCaps* caps, GstBuffer* buf, TapPacket& out) {
  if (!caps || !buf) return false;

  GstVideoInfo vinfo;
  if (!gst_video_info_from_caps(&vinfo, caps)) return false;

  const int w = GST_VIDEO_INFO_WIDTH(&vinfo);
  const int h = GST_VIDEO_INFO_HEIGHT(&vinfo);
  if (w <= 0 || h <= 0) return false;

  GstVideoFrame vframe;
  std::memset(&vframe, 0, sizeof(vframe));
  if (!gst_video_frame_map(&vframe, &vinfo, buf, GST_MAP_READ)) return false;

  auto unmap = [&]() { gst_video_frame_unmap(&vframe); };

  try {
    if (out.format == TapFormat::NV12) {
      const size_t y_sz = (size_t)w * (size_t)h;
      const size_t uv_sz = y_sz / 2;
      out.bytes.resize(y_sz + uv_sz);

      const int y_stride  = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 0);
      const int uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 1);

      const uint8_t* y_src  = (const uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 0);
      const uint8_t* uv_src = (const uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 1);

      uint8_t* dst_y  = out.bytes.data();
      uint8_t* dst_uv = out.bytes.data() + y_sz;

      for (int row = 0; row < h; ++row) {
        std::memcpy(dst_y + (size_t)row * (size_t)w,
                    y_src + (size_t)row * (size_t)y_stride,
                    (size_t)w);
      }
      for (int row = 0; row < h / 2; ++row) {
        std::memcpy(dst_uv + (size_t)row * (size_t)w,
                    uv_src + (size_t)row * (size_t)uv_stride,
                    (size_t)w);
      }

      unmap();
      return true;
    }

    if (out.format == TapFormat::I420) {
      const size_t y_sz = (size_t)w * (size_t)h;
      const size_t u_sz = y_sz / 4;
      const size_t v_sz = u_sz;
      out.bytes.resize(y_sz + u_sz + v_sz);

      const int y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 0);
      const int u_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 1);
      const int v_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 2);

      const uint8_t* y_src = (const uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 0);
      const uint8_t* u_src = (const uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 1);
      const uint8_t* v_src = (const uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 2);

      uint8_t* y_dst = out.bytes.data();
      uint8_t* u_dst = out.bytes.data() + y_sz;
      uint8_t* v_dst = out.bytes.data() + y_sz + u_sz;

      for (int row = 0; row < h; ++row) {
        std::memcpy(y_dst + (size_t)row * (size_t)w,
                    y_src + (size_t)row * (size_t)y_stride,
                    (size_t)w);
      }

      const int cw = w / 2;
      const int ch = h / 2;

      for (int row = 0; row < ch; ++row) {
        std::memcpy(u_dst + (size_t)row * (size_t)cw,
                    u_src + (size_t)row * (size_t)u_stride,
                    (size_t)cw);
        std::memcpy(v_dst + (size_t)row * (size_t)cw,
                    v_src + (size_t)row * (size_t)v_stride,
                    (size_t)cw);
      }

      unmap();
      return true;
    }

    if (out.format == TapFormat::RGB || out.format == TapFormat::BGR) {
      const size_t row_bytes = (size_t)w * 3;
      out.bytes.resize(row_bytes * (size_t)h);

      const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 0);
      const uint8_t* src = (const uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 0);
      uint8_t* dst = out.bytes.data();

      for (int row = 0; row < h; ++row) {
        std::memcpy(dst + (size_t)row * row_bytes,
                    src + (size_t)row * (size_t)stride,
                    row_bytes);
      }

      unmap();
      return true;
    }

    if (out.format == TapFormat::GRAY8) {
      const size_t row_bytes = (size_t)w;
      out.bytes.resize(row_bytes * (size_t)h);

      const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 0);
      const uint8_t* src = (const uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 0);
      uint8_t* dst = out.bytes.data();

      for (int row = 0; row < h; ++row) {
        std::memcpy(dst + (size_t)row * row_bytes,
                    src + (size_t)row * (size_t)stride,
                    row_bytes);
      }

      unmap();
      return true;
    }
  } catch (...) {
    unmap();
    throw;
  }

  unmap();
  return false;
}

// =====================================================================================
// Builder rendering (node ↔ element mapping, optional boundary markers)
// =====================================================================================

struct BuildResult {
  std::string pipeline_string;
  std::shared_ptr<DiagCtx> diag;
  std::string appsink_name; // mysink or tap_*
  int tap_node_index = -1;  // for run_tap
};

static bool should_insert_boundaries_for_mode(const char* mode_key, bool def_val) {
  // Examples:
  //   SIMA_GST_RUN_INSERT_BOUNDARIES=0 (fast hot path)
  //   SIMA_GST_TAP_INSERT_BOUNDARIES=1
  //   SIMA_GST_VALIDATE_INSERT_BOUNDARIES=1
  return env_bool(mode_key, def_val);
}

static BuildResult build_pipeline_full(const std::vector<std::shared_ptr<Node>>& nodes,
                                      bool insert_boundaries,
                                      const std::string& appsink_name /*mysink*/) {
  if (nodes.empty()) throw std::runtime_error("InvalidPipeline: no nodes");

  BuildResult br;
  br.diag = std::make_shared<DiagCtx>();
  br.appsink_name = appsink_name;

  std::ostringstream ss;

  br.diag->node_reports.reserve(nodes.size());
  if (insert_boundaries) br.diag->boundaries.reserve(nodes.size() ? nodes.size() - 1 : 0);

  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!nodes[i]) throw std::runtime_error("InvalidPipeline: null node");

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

      auto st = std::make_unique<BoundaryFlowStats>();
      st->boundary_name = bname;
      st->after_node_index = (int)i;
      st->before_node_index = (int)(i + 1);
      br.diag->boundaries.push_back(std::move(st));
    }
  }

  br.diag->pipeline_string = ss.str();
  br.pipeline_string = br.diag->pipeline_string;
  return br;
}

static BuildResult build_pipeline_tap(const std::vector<std::shared_ptr<Node>>& nodes,
                                     const std::string& debug_point_name,
                                     bool insert_boundaries) {
  if (nodes.empty()) throw std::runtime_error("InvalidPipeline: no nodes");

  const std::string want = debug_point_name.empty() ? "dbg" : debug_point_name;
  const std::string tap_name = "tap_" + sanitize_name(want);

  // Find DebugPoint index (no dynamic_cast => easier future contributions)
  int cut = -1;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!nodes[i]) continue;
    if (nodes[i]->kind() == "DebugPoint" && nodes[i]->user_label() == want) {
      cut = (int)i;
      break;
    }
  }
  if (cut < 0) throw std::runtime_error("run_tap: DebugPoint not found: " + want);

  // Build truncated node list [0..cut] and append appsink
  std::vector<std::shared_ptr<Node>> trunc(nodes.begin(), nodes.begin() + (size_t)cut + 1);

  BuildResult br = build_pipeline_full(trunc, insert_boundaries, tap_name);

  // Append appsink
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
  if (nodes.empty()) throw std::runtime_error("InvalidPipeline: no nodes");
  if (!nodes.back() || nodes.back()->kind() != "OutputAppSink") {
    throw std::runtime_error("InvalidPipeline: last node must be OutputAppSink() for run()");
  }
}


static void enforce_names_contract(GstElement* pipeline, const BuildResult& br) {
  if (!pipeline || !GST_IS_BIN(pipeline) || !br.diag) return;

  // Allowed element names: node-owned + boundaries + sink
  std::unordered_set<std::string> allowed;

  for (const auto& n : br.diag->node_reports) {
    for (const auto& e : n.elements) allowed.insert(e);
  }
  for (const auto& b : br.diag->boundaries) {
    if (b) allowed.insert(b->boundary_name);
  }
  if (!br.appsink_name.empty()) allowed.insert(br.appsink_name);

  // Iterate elements in the bin
  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it) return;

  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* el = GST_ELEMENT(g_value_get_object(&item));
    g_value_reset(&item);
    if (!el) continue;

    const char* name = GST_ELEMENT_NAME(el);
    if (name && *name) {
      // Allow internal-ish names if any appear (be permissive to avoid false positives).
      const std::string n(name);
      const bool internal_ok =
        (n.rfind("queue", 0) == 0) || (n.rfind("typefind", 0) == 0) ||
        (n.rfind("rtpbin", 0) == 0) || (n.rfind("decodebin", 0) == 0);

      if (!internal_ok && allowed.find(n) == allowed.end()) {
        gst_iterator_free(it);
        PipelineReport rep = br.diag->snapshot_basic();
        rep.repro_note = "NamingContractViolation: element '" + n + "' is not owned by any node.\n"
                         "Fix: ensure every fragment uses deterministic names and element_names() matches.\n";
        throw PipelineError(rep.repro_note, std::move(rep));
      }
    }
  }

  g_value_unset(&item);
  gst_iterator_free(it);
}

// =====================================================================================
// FrameStream (fast path: FrameNV12Ref)
// =====================================================================================

struct SampleHolder {
  GstSample* sample = nullptr;
  GstVideoFrame frame{};
  GstVideoInfo vinfo{};
  bool mapped = false;

  ~SampleHolder() {
    if (mapped) gst_video_frame_unmap(&frame);
    if (sample) gst_sample_unref(sample);
    sample = nullptr;
  }
};

FrameStream::FrameStream(GstElement* pipeline, GstElement* appsink)
  : pipeline_(pipeline), appsink_(appsink) {}

FrameStream::~FrameStream() { close(); }

FrameStream::FrameStream(FrameStream&& o) noexcept {
  pipeline_ = o.pipeline_;
  appsink_  = o.appsink_;
  debug_pipeline_ = std::move(o.debug_pipeline_);
  diag_ = std::move(o.diag_);
  o.pipeline_ = nullptr;
  o.appsink_ = nullptr;
}

FrameStream& FrameStream::operator=(FrameStream&& o) noexcept {
  if (this != &o) {
    close();
    pipeline_ = o.pipeline_;
    appsink_  = o.appsink_;
    debug_pipeline_ = std::move(o.debug_pipeline_);
    diag_ = std::move(o.diag_);
    o.pipeline_ = nullptr;
    o.appsink_ = nullptr;
  }
  return *this;
}

void FrameStream::close() {
  if (appsink_) {
    gst_object_unref(appsink_);
    appsink_ = nullptr;
  }
  if (pipeline_) {
    stop_and_unref(pipeline_);
  }
  diag_.reset();
}

static std::optional<GstSample*> try_pull_sample_sliced(GstElement* pipeline,
                                                        GstElement* appsink,
                                                        int timeout_ms,
                                                        const std::shared_ptr<DiagCtx>& diag,
                                                        const char* where) {
  if (!pipeline || !appsink) return std::nullopt;

  const int slice_ms = std::max(10, std::atoi(env_str("SIMA_GST_POLL_SLICE_MS", "200").c_str()));
  const bool infinite = (timeout_ms < 0);
  int remaining = timeout_ms;

  while (true) {
    int this_ms = 0;
    if (timeout_ms == 0) this_ms = 0;
    else if (infinite) this_ms = slice_ms;
    else this_ms = std::min(slice_ms, remaining);

    GstSample* s = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), (guint64)this_ms * GST_MSECOND);
    if (s) return s;

    drain_bus(pipeline, diag);
    throw_if_bus_error(pipeline, diag, where);

    if (timeout_ms == 0) return std::nullopt;
    if (!infinite) {
      remaining -= this_ms;
      if (remaining <= 0) return std::nullopt;
    }
  }
}

std::optional<FrameNV12Ref> FrameStream::next(int timeout_ms) {
  if (!pipeline_ || !appsink_) return std::nullopt;

  auto diag = std::static_pointer_cast<DiagCtx>(diag_);

  auto sopt = try_pull_sample_sliced(pipeline_, appsink_, timeout_ms, diag, "FrameStream::next");
  if (!sopt.has_value()) {
    // timeout => return null (or throw if configured)
    if (!env_bool("SIMA_GST_TIMEOUT_RETURNS_NULL", true) && timeout_ms != 0) {
      PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
      rep.repro_note = "FrameStream::next: timeout waiting for sample.\n";
      if (diag) rep.repro_note += boundary_summary(diag);
      maybe_dump_dot(pipeline_, "FrameStream_next_timeout");
      throw PipelineError(rep.repro_note, std::move(rep));
    }
    return std::nullopt;
  }

  GstSample* sample = *sopt;
  GstBuffer* buf = gst_sample_get_buffer(sample);
  GstCaps* caps = gst_sample_get_caps(sample);
  if (!buf || !caps) {
    gst_sample_unref(sample);
    return std::nullopt;
  }

  // Contract: NV12 + SystemMemory for run()
  GstVideoInfo vinfo;
  if (!gst_video_info_from_caps(&vinfo, caps)) {
    gst_sample_unref(sample);
    return std::nullopt;
  }
  if (GST_VIDEO_INFO_FORMAT(&vinfo) != GST_VIDEO_FORMAT_NV12) {
    PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
    rep.repro_note = "FrameStream::next: expected NV12, got caps: " + gst_caps_to_string_safe(caps);
    gst_sample_unref(sample);
    throw PipelineError(rep.repro_note, std::move(rep));
  }
  require_system_memory_or_throw(caps, "FrameStream::next", diag);

  // Map as ref-view (zero-copy)
  auto holder = std::make_shared<SampleHolder>();
  holder->sample = sample;
  holder->vinfo = vinfo;

  GstVideoFrame vframe;
  std::memset(&vframe, 0, sizeof(vframe));
  if (!gst_video_frame_map(&vframe, &holder->vinfo, buf, GST_MAP_READ)) {
    gst_sample_unref(sample);
    return std::nullopt;
  }

  holder->frame = vframe;
  holder->mapped = true;

  FrameNV12Ref out;
  out.width = GST_VIDEO_INFO_WIDTH(&holder->vinfo);
  out.height = GST_VIDEO_INFO_HEIGHT(&holder->vinfo);
  out.keyframe = !GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_DELTA_UNIT);

  // Timing
  out.pts_ns = (GST_BUFFER_PTS(buf) == GST_CLOCK_TIME_NONE) ? -1 : (int64_t)GST_BUFFER_PTS(buf);
  out.dts_ns = (GST_BUFFER_DTS(buf) == GST_CLOCK_TIME_NONE) ? -1 : (int64_t)GST_BUFFER_DTS(buf);
  out.duration_ns = (GST_BUFFER_DURATION(buf) == GST_CLOCK_TIME_NONE) ? -1 : (int64_t)GST_BUFFER_DURATION(buf);

  out.caps_string = gst_caps_to_string_safe(caps);

  out.y = (const uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&holder->frame, 0);
  out.uv = (const uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&holder->frame, 1);
  out.y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&holder->frame, 0);
  out.uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&holder->frame, 1);

  out.holder = holder;

  // Keep draining bus cheaply
  drain_bus(pipeline_, diag);
  throw_if_bus_error(pipeline_, diag, "FrameStream::next");

  return out;
}

std::optional<FrameNV12> FrameStream::next_copy(int timeout_ms) {
  auto ref = next(timeout_ms);
  if (!ref.has_value()) return std::nullopt;

  const auto& r = *ref;
  FrameNV12 out;
  out.width = r.width;
  out.height = r.height;
  out.keyframe = r.keyframe;
  out.pts_ns = r.pts_ns;
  out.dts_ns = r.dts_ns;
  out.duration_ns = r.duration_ns;

  const int w = r.width;
  const int h = r.height;
  out.nv12.resize((size_t)w * (size_t)h * 3 / 2);

  uint8_t* dst_y = out.nv12.data();
  uint8_t* dst_uv = out.nv12.data() + (size_t)w * (size_t)h;

  for (int row = 0; row < h; ++row) {
    std::memcpy(dst_y + (size_t)row * (size_t)w,
                r.y + (size_t)row * (size_t)r.y_stride,
                (size_t)w);
  }
  for (int row = 0; row < h / 2; ++row) {
    std::memcpy(dst_uv + (size_t)row * (size_t)w,
                r.uv + (size_t)row * (size_t)r.uv_stride,
                (size_t)w);
  }
  return out;
}

PipelineReport FrameStream::report_snapshot(bool heavy) const {
  auto diag = std::static_pointer_cast<DiagCtx>(diag_);
  PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};

  if (!heavy || !pipeline_) return rep;

  // Heavy-on-demand: dot dump + caps dump (simple best-effort)
  maybe_dump_dot(pipeline_, "report_snapshot");

  // Quick caps dump
  std::ostringstream ss;
  ss << "CapsDump:\n";
  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline_));
  if (it) {
    GValue item = G_VALUE_INIT;
    int lines = 0;
    const int max_lines = std::max(50, std::atoi(env_str("SIMA_GST_CAPS_MAX_LINES", "600").c_str()));
    while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
      GstElement* el = GST_ELEMENT(g_value_get_object(&item));
      g_value_reset(&item);
      if (!el) continue;

      const char* en = GST_ELEMENT_NAME(el);
      ss << "- " << (en ? en : "<noname>") << "\n";
      lines++;
      if (lines >= max_lines) break;
    }
    g_value_unset(&item);
    gst_iterator_free(it);
  }
  rep.caps_dump = ss.str();

  const std::string dot_dir = env_str("SIMA_GST_DOT_DIR", "");
  if (!dot_dir.empty()) rep.dot_paths.push_back(dot_dir);

  rep.repro_note += boundary_summary(diag);
  return rep;
}

// =====================================================================================
// TapStream
// =====================================================================================

TapStream::TapStream(GstElement* pipeline, GstElement* appsink)
  : pipeline_(pipeline), appsink_(appsink) {}

TapStream::~TapStream() { close(); }

TapStream::TapStream(TapStream&& o) noexcept {
  pipeline_ = o.pipeline_;
  appsink_  = o.appsink_;
  debug_pipeline_ = std::move(o.debug_pipeline_);
  diag_ = std::move(o.diag_);
  tap_node_index_ = o.tap_node_index_;
  tap_sink_name_ = std::move(o.tap_sink_name_);
  o.pipeline_ = nullptr;
  o.appsink_ = nullptr;
  o.tap_node_index_ = -1;
}

TapStream& TapStream::operator=(TapStream&& o) noexcept {
  if (this != &o) {
    close();
    pipeline_ = o.pipeline_;
    appsink_  = o.appsink_;
    debug_pipeline_ = std::move(o.debug_pipeline_);
    diag_ = std::move(o.diag_);
    tap_node_index_ = o.tap_node_index_;
    tap_sink_name_ = std::move(o.tap_sink_name_);
    o.pipeline_ = nullptr;
    o.appsink_ = nullptr;
    o.tap_node_index_ = -1;
  }
  return *this;
}

void TapStream::close() {
  if (appsink_) {
    gst_object_unref(appsink_);
    appsink_ = nullptr;
  }
  if (pipeline_) {
    stop_and_unref(pipeline_);
  }
  diag_.reset();
}

std::optional<TapPacket> TapStream::next(int timeout_ms) {
  if (!pipeline_ || !appsink_) return std::nullopt;

  auto diag = std::static_pointer_cast<DiagCtx>(diag_);

  auto sopt = try_pull_sample_sliced(pipeline_, appsink_, timeout_ms, diag, "TapStream::next");
  if (!sopt.has_value()) {
    if (!env_bool("SIMA_GST_TIMEOUT_RETURNS_NULL", true) && timeout_ms != 0) {
      PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
      rep.repro_note = "TapStream::next: timeout waiting for sample.\n";
      if (diag) rep.repro_note += boundary_summary(diag);
      maybe_dump_dot(pipeline_, "TapStream_next_timeout");
      throw PipelineError(rep.repro_note, std::move(rep));
    }
    return std::nullopt;
  }

  GstSample* sample = *sopt;
  GstBuffer* buf = gst_sample_get_buffer(sample);
  GstCaps* caps = gst_sample_get_caps(sample);

  if (!buf) {
    gst_sample_unref(sample);
    return std::nullopt;
  }

  TapPacket pkt;
  pkt.source_node_index = tap_node_index_;
  pkt.source_element = tap_sink_name_.empty() ? "appsink" : tap_sink_name_;

  if (caps) {
    infer_tap_format_and_meta(caps, pkt);
    pkt.memory_features = caps_features_string(caps);
  } else {
    pkt.caps_string = "<no caps>";
    pkt.memory_features = "<none>";
  }

  pkt.keyframe = !GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_DELTA_UNIT);
  pkt.pts_ns = (GST_BUFFER_PTS(buf) == GST_CLOCK_TIME_NONE) ? -1 : (int64_t)GST_BUFFER_PTS(buf);
  pkt.dts_ns = (GST_BUFFER_DTS(buf) == GST_CLOCK_TIME_NONE) ? -1 : (int64_t)GST_BUFFER_DTS(buf);
  pkt.duration_ns = (GST_BUFFER_DURATION(buf) == GST_CLOCK_TIME_NONE) ? -1 : (int64_t)GST_BUFFER_DURATION(buf);

  // AllowEitherButReport (default): try pack, else map, else report non-mappable.
  bool packed = false;
  if (caps) packed = pack_raw_video_tight(caps, buf, pkt);

  if (!packed) {
    GstMapInfo map{};
    if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
      pkt.bytes.resize(map.size);
      if (map.size) std::memcpy(pkt.bytes.data(), map.data, map.size);
      gst_buffer_unmap(buf, &map);
      pkt.memory_mappable = true;
    } else {
      pkt.bytes.clear();
      pkt.memory_mappable = false;
      pkt.non_mappable_reason =
        "Non-mappable buffer memory (likely device/DMA/NVMM). "
        "If you need bytes, force SystemMemory before the DebugPoint.";
    }
  }

  gst_sample_unref(sample);

  drain_bus(pipeline_, diag);
  throw_if_bus_error(pipeline_, diag, "TapStream::next");

  return pkt;
}

PipelineReport TapStream::report_snapshot(bool heavy) const {
  auto diag = std::static_pointer_cast<DiagCtx>(diag_);
  PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
  if (!heavy || !pipeline_) return rep;
  maybe_dump_dot(pipeline_, "tap_report_snapshot");
  rep.repro_note += boundary_summary(diag);
  const std::string dot_dir = env_str("SIMA_GST_DOT_DIR", "");
  if (!dot_dir.empty()) rep.dot_paths.push_back(dot_dir);
  return rep;
}

// =====================================================================================
// Node implementations
// =====================================================================================

// =====================================================================================
// Function-based nodes (typed helpers)
// =====================================================================================

class ImageFreezeNode final : public Node {
public:
  explicit ImageFreezeNode(int num_buffers) : num_buffers_(num_buffers) {}
  std::string kind() const override { return "ImageFreeze"; }

  std::string gst_fragment(int node_index) const override {
    std::ostringstream ss;
    ss << "imagefreeze name=n" << node_index << "_imagefreeze";
    if (num_buffers_ > 0) ss << " num-buffers=" << num_buffers_;
    return ss.str();
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_imagefreeze"};
  }

private:
  int num_buffers_ = -1;
};

class VideoRateNode final : public Node {
public:
  std::string kind() const override { return "VideoRate"; }

  std::string gst_fragment(int node_index) const override {
    return "videorate name=n" + std::to_string(node_index) + "_videorate";
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_videorate"};
  }
};

class CapsRawNode final : public Node {
public:
  CapsRawNode(std::string format, int w, int h, int fps, CapsMemory mem)
    : format_(std::move(format)), w_(w), h_(h), fps_(fps), mem_(mem) {}

  std::string kind() const override { return "CapsRaw"; }

  std::string gst_fragment(int node_index) const override {
    const std::string name = "n" + std::to_string(node_index) + "_caps";

    std::ostringstream caps;
    caps << "video/x-raw";
    if (mem_ == CapsMemory::SystemMemory) caps << "(memory:SystemMemory)";
    if (!format_.empty()) caps << ",format=" << format_;
    if (w_ > 0) caps << ",width=" << w_;
    if (h_ > 0) caps << ",height=" << h_;
    if (fps_ > 0) caps << ",framerate=" << fps_ << "/1";

    std::ostringstream ss;
    ss << "capsfilter name=" << name << " caps=\"" << caps.str() << "\"";
    return ss.str();
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_caps"};
  }

private:
  std::string format_;
  int w_ = -1;
  int h_ = -1;
  int fps_ = -1;
  CapsMemory mem_ = CapsMemory::Any;
};

// Replace/upgrade your H264EncodeSima to include w/h/fps (no legacy)
H264EncodeSima::H264EncodeSima(int w, int h, int fps,
                               int bitrate_kbps,
                               std::string profile,
                               std::string level)
  : w_(w), h_(h), fps_(fps),
    bitrate_kbps_(bitrate_kbps),
    profile_(std::move(profile)),
    level_(std::move(level)) {}

std::string H264EncodeSima::gst_fragment(int node_index) const {
  std::ostringstream ss;
  ss << "neatencoder name=n" << node_index << "_encoder "
     << "enc-type=h264 "
     << "enc-profile=" << profile_ << " "
     << "enc-level=" << level_ << " "
     << "enc-fmt=NV12 "
     << "enc-width=" << w_ << " "
     << "enc-height=" << h_ << " "
     << "enc-frame-rate=" << fps_ << " "
     << "enc-bitrate=" << bitrate_kbps_ << " "
     << "enc-ip-mode=async "
     << "ip-rate-ctrl=false";
  return ss.str();
}

std::vector<std::string> H264EncodeSima::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_encoder"};
}

// Simple SW encoder node (portable fallback)
class H264EncodeSWNode final : public Node {
public:
  explicit H264EncodeSWNode(int bitrate_kbps) : bitrate_kbps_(bitrate_kbps) {}
  std::string kind() const override { return "H264EncodeSW"; }

  std::string gst_fragment(int node_index) const override {
    std::string factory;
    std::string props;

    if (element_exists("x264enc")) {
      factory = "x264enc";
      props =
        "tune=zerolatency speed-preset=ultrafast "
        "key-int-max=1 bframes=0 "
        "bitrate=" + std::to_string(bitrate_kbps_) + " "
        "byte-stream=true";
    } else if (element_exists("openh264enc")) {
      factory = "openh264enc";
      props = "";
    } else if (element_exists("avenc_h264")) {
      factory = "avenc_h264";
      props = "";
    } else {
      throw std::runtime_error(
        "H264EncodeSW: no software H264 encoder found. Install one of: "
        "x264enc (gst-plugins-ugly), openh264enc (gst-plugins-bad), avenc_h264 (gst-libav).");
    }

    std::ostringstream ss;
    ss << factory << " name=n" << node_index << "_swenc";
    if (!props.empty()) ss << " " << props;
    return ss.str();
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_swenc"};
  }

private:
  int bitrate_kbps_ = 400;
};

// =====================================================================================
// sima::nodes factory functions (what customers call)
// =====================================================================================

namespace nodes {

std::shared_ptr<Node> Gst(std::string fragment) {
  return std::make_shared<sima::GstNode>(std::move(fragment));
}

std::shared_ptr<Node> FileSrc(std::string path) {
  return std::make_shared<sima::FileSrc>(std::move(path));
}

std::shared_ptr<Node> JpegDec() {
  return std::make_shared<sima::JpegDec>();
}

std::shared_ptr<Node> ImageFreeze(int num_buffers) {
  return std::make_shared<ImageFreezeNode>(num_buffers);
}

std::shared_ptr<Node> VideoConvert() {
  return std::make_shared<sima::VideoConvert>();
}

std::shared_ptr<Node> VideoScale() {
  return std::make_shared<sima::VideoScale>();
}

std::shared_ptr<Node> VideoRate() {
  return std::make_shared<VideoRateNode>();
}

std::shared_ptr<Node> Queue() {
  return std::make_shared<sima::Queue>();
}

std::shared_ptr<Node> DebugPoint(std::string name) {
  return std::make_shared<sima::DebugPoint>(std::move(name));
}

std::shared_ptr<Node> CapsRaw(std::string format, int w, int h, int fps, CapsMemory memory) {
  return std::make_shared<CapsRawNode>(std::move(format), w, h, fps, memory);
}

std::shared_ptr<Node> CapsNV12SysMem(int w, int h, int fps) {
  return CapsRaw("NV12", w, h, fps, CapsMemory::SystemMemory);
}

std::shared_ptr<Node> CapsI420(int w, int h, int fps, CapsMemory memory) {
  return CapsRaw("I420", w, h, fps, memory);
}

std::shared_ptr<Node> H264EncodeSima(int w, int h, int fps,
                                     int bitrate_kbps,
                                     std::string profile,
                                     std::string level) {
  return std::make_shared<sima::H264EncodeSima>(w, h, fps, bitrate_kbps,
                                                std::move(profile), std::move(level));
}

std::shared_ptr<Node> H264EncodeSW(int bitrate_kbps) {
  return std::make_shared<H264EncodeSWNode>(bitrate_kbps);
}

std::shared_ptr<Node> H264Parse(int config_interval) {
  return std::make_shared<sima::H264Parse>(config_interval);
}

std::shared_ptr<Node> H264Decode(int sima_allocator_type, std::string out_format) {
  return std::make_shared<sima::H264Decode>(sima_allocator_type, std::move(out_format));
}

std::shared_ptr<Node> QtDemuxVideoPad(int video_pad_index) {
  return std::make_shared<sima::QtDemuxVideoPad>(video_pad_index);
}

std::shared_ptr<Node> H264ParseAu() {
  return std::make_shared<sima::H264ParseAu>();
}

std::shared_ptr<Node> RTSPInput(std::string url, int latency_ms, bool tcp) {
  return std::make_shared<sima::RTSPInput>(std::move(url), latency_ms, tcp);
}

std::shared_ptr<Node> H264DepayParse() {
  return std::make_shared<sima::H264DepayParse>();
}

std::shared_ptr<Node> AppSrcImage(std::string image_path, int content_w, int content_h,
                                  int enc_w, int enc_h, int fps) {
  return std::make_shared<sima::AppSrcImage>(std::move(image_path),
                                             content_w, content_h, enc_w, enc_h, fps);
}

std::shared_ptr<Node> RtpH264Pay(int pt, int config_interval) {
  return std::make_shared<sima::RtpH264Pay>(pt, config_interval);
}

std::shared_ptr<Node> OutputAppSink() {
  return std::make_shared<sima::OutputAppSink>();
}

} // namespace nodes


// -----------------------------
// GstNode (raw fragment) node
// -----------------------------
GstNode::GstNode(std::string fragment) : fragment_(std::move(fragment)) {
  // Keep it permissive. Empty is an error-prone footgun, so make it explicit.
  if (fragment_.empty()) {
    fragment_ = "identity silent=true";
  }
}

std::string GstNode::gst_fragment(int node_index) const {
  (void)node_index;

  // Trim leading/trailing spaces (cheap).
  auto trim = [](const std::string& s) -> std::string {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) b++;
    while (e > b && std::isspace((unsigned char)s[e - 1])) e--;
    return s.substr(b, e - b);
  };

  const std::string frag = trim(fragment_);

  // If the user already provided a name=, or this looks like a complex mini-pipeline,
  // don't try to "help" (we can't safely rename multiple elements).
  const bool has_name = (frag.find("name=") != std::string::npos);
  const bool looks_complex =
    (frag.find('!') != std::string::npos) ||
    (frag.find('(') != std::string::npos) ||
    (frag.find(')') != std::string::npos);

  if (has_name || looks_complex) {
    return frag;
  }

  // Simple single-element case: inject deterministic name so naming contract works.
  // factory token = first word
  std::string factory;
  {
    size_t i = 0;
    while (i < frag.size() && std::isspace((unsigned char)frag[i])) i++;
    size_t j = i;
    while (j < frag.size() && !std::isspace((unsigned char)frag[j])) j++;
    factory = (j > i) ? frag.substr(i, j - i) : "gst";
  }

  const std::string elname = "n" + std::to_string(node_index) + "_" + sanitize_name(factory);

  // Append " name=<elname>" at the end.
  return frag + " name=" + elname;
}

std::vector<std::string> GstNode::element_names(int node_index) const {
  // Best-effort:
  // - If fragment contains name=..., return that name
  // - Else if it's a simple single element, return the deterministic injected name
  // - Else return empty (and naming enforcement will complain, as intended for power-user fragments)

  auto trim = [](const std::string& s) -> std::string {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) b++;
    while (e > b && std::isspace((unsigned char)s[e - 1])) e--;
    return s.substr(b, e - b);
  };

  const std::string frag = trim(fragment_);

  // Try extract user-provided name=
  const size_t pos = frag.find("name=");
  if (pos != std::string::npos) {
    size_t i = pos + 5;
    while (i < frag.size() && std::isspace((unsigned char)frag[i])) i++;

    if (i < frag.size() && frag[i] == '"') {
      // name="foo"
      i++;
      size_t j = i;
      while (j < frag.size() && frag[j] != '"') j++;
      if (j > i) return { frag.substr(i, j - i) };
      return {};
    } else {
      // name=foo
      size_t j = i;
      while (j < frag.size() && !std::isspace((unsigned char)frag[j])) j++;
      if (j > i) return { frag.substr(i, j - i) };
      return {};
    }
  }

  const bool looks_complex =
    (frag.find('!') != std::string::npos) ||
    (frag.find('(') != std::string::npos) ||
    (frag.find(')') != std::string::npos);

  if (looks_complex) return {};

  // Derive factory token and deterministic name
  std::string factory;
  {
    size_t i = 0;
    while (i < frag.size() && std::isspace((unsigned char)frag[i])) i++;
    size_t j = i;
    while (j < frag.size() && !std::isspace((unsigned char)frag[j])) j++;
    factory = (j > i) ? frag.substr(i, j - i) : "gst";
  }

  return { "n" + std::to_string(node_index) + "_" + sanitize_name(factory) };
}

DebugPoint::DebugPoint(std::string name) : name_(std::move(name)) {
  if (name_.empty()) name_ = "dbg";
}

std::string DebugPoint::gst_fragment(int node_index) const {
  (void)node_index;
  const std::string elname = "dbg_" + sanitize_name(name_);
  std::ostringstream ss;
  ss << "identity name=" << elname << " silent=true";
  return ss.str();
}

std::vector<std::string> DebugPoint::element_names(int /*node_index*/) const {
  return {"dbg_" + sanitize_name(name_)};
}

FileSrc::FileSrc(std::string path) : path_(std::move(path)) {}

std::string FileSrc::gst_fragment(int node_index) const {
  const std::string el = "n" + std::to_string(node_index) + "_filesrc";
  std::ostringstream ss;
  ss << "filesrc name=" << el << " location=\"" << path_ << "\"";
  return ss.str();
}

std::vector<std::string> FileSrc::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_filesrc"};
}

std::string JpegDec::gst_fragment(int node_index) const {
  return "jpegdec name=n" + std::to_string(node_index) + "_jpegdec";
}

std::vector<std::string> JpegDec::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_jpegdec"};
}

QtDemuxVideoPad::QtDemuxVideoPad(int video_pad_index) : idx_(video_pad_index) {}

std::string QtDemuxVideoPad::gst_fragment(int node_index) const {
  const std::string base = "n" + std::to_string(node_index) + "_demux";
  std::ostringstream ss;
  ss << "qtdemux name=" << base << " " << base << ".video_" << idx_;
  return ss.str();
}

std::vector<std::string> QtDemuxVideoPad::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_demux"};
}

std::string Queue::gst_fragment(int node_index) const {
  return "queue name=n" + std::to_string(node_index) + "_queue";
}

std::vector<std::string> Queue::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_queue"};
}

std::string H264ParseAu::gst_fragment(int node_index) const {
  const std::string p = "n" + std::to_string(node_index) + "_h264parse";
  const std::string c = "n" + std::to_string(node_index) + "_h264_caps";
  std::ostringstream ss;
  ss << "h264parse name=" << p << " disable-passthrough=true "
     << "! capsfilter name=" << c
     << " caps=\"video/x-h264,stream-format=(string)byte-stream,alignment=(string)au\"";
  return ss.str();
}

std::vector<std::string> H264ParseAu::element_names(int node_index) const {
  return {
    "n" + std::to_string(node_index) + "_h264parse",
    "n" + std::to_string(node_index) + "_h264_caps"
  };
}

std::string VideoConvert::gst_fragment(int node_index) const {
  return "videoconvert name=n" + std::to_string(node_index) + "_videoconvert";
}

std::vector<std::string> VideoConvert::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_videoconvert"};
}

std::string VideoScale::gst_fragment(int node_index) const {
  return "videoscale name=n" + std::to_string(node_index) + "_videoscale";
}

std::vector<std::string> VideoScale::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_videoscale"};
}

RTSPInput::RTSPInput(std::string url, int latency_ms, bool tcp)
  : url_(std::move(url)), latency_ms_(latency_ms), tcp_(tcp) {}

std::string RTSPInput::gst_fragment(int node_index) const {
  const std::string el = "n" + std::to_string(node_index) + "_rtspsrc";
  std::ostringstream ss;
  ss << "rtspsrc name=" << el
     << " location=\"" << url_ << "\" "
     << "latency=" << latency_ms_ << " "
     << "protocols=" << (tcp_ ? "tcp" : "udp");
  return ss.str();
}

std::vector<std::string> RTSPInput::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_rtspsrc"};
}

std::string H264DepayParse::gst_fragment(int node_index) const {
  const std::string rtp = "n" + std::to_string(node_index) + "_rtp_caps";
  const std::string dep = "n" + std::to_string(node_index) + "_depay";
  const std::string par = "n" + std::to_string(node_index) + "_h264parse";
  const std::string hcc = "n" + std::to_string(node_index) + "_h264_caps";

  std::ostringstream ss;
  ss << "capsfilter name=" << rtp
     << " caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\" "
     << "! rtph264depay name=" << dep
     << " ! h264parse name=" << par << " disable-passthrough=true "
     << "! capsfilter name=" << hcc
     << " caps=\"video/x-h264,stream-format=(string)byte-stream,alignment=(string)au\"";
  return ss.str();
}

std::vector<std::string> H264DepayParse::element_names(int node_index) const {
  return {
    "n" + std::to_string(node_index) + "_rtp_caps",
    "n" + std::to_string(node_index) + "_depay",
    "n" + std::to_string(node_index) + "_h264parse",
    "n" + std::to_string(node_index) + "_h264_caps"
  };
}

H264Decode::H264Decode(int sima_allocator_type, std::string out_format)
  : sima_allocator_type_(sima_allocator_type), out_format_(std::move(out_format)) {}

std::string H264Decode::gst_fragment(int node_index) const {
  const std::string dec = "n" + std::to_string(node_index) + "_decoder";
  const std::string vc  = "n" + std::to_string(node_index) + "_videoconvert";
  const std::string cap = "n" + std::to_string(node_index) + "_raw_caps";

  std::ostringstream caps;
  caps << "video/x-raw(memory:SystemMemory),format=" << out_format_;

  std::ostringstream ss;
  ss << "simaaidecoder name=" << dec << " sima-allocator-type=" << sima_allocator_type_
     << " ! videoconvert name=" << vc
     << " ! capsfilter name=" << cap << " caps=\"" << caps.str() << "\"";
  return ss.str();
}

std::vector<std::string> H264Decode::element_names(int node_index) const {
  return {
    "n" + std::to_string(node_index) + "_decoder",
    "n" + std::to_string(node_index) + "_videoconvert",
    "n" + std::to_string(node_index) + "_raw_caps"
  };
}

std::string OutputAppSink::gst_fragment(int /*node_index*/) const {
  return "appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true";
}

std::vector<std::string> OutputAppSink::element_names(int /*node_index*/) const {
  return {"mysink"};
}

// --- Server-side nodes (AppSrcImage + encoder/parse/pay) ---

static std::vector<uint8_t> bgr_to_nv12_tight(const cv::Mat& bgr_in, int w, int h) {
  cv::Mat bgr;
  if (bgr_in.cols != w || bgr_in.rows != h) {
    cv::resize(bgr_in, bgr, cv::Size(w, h), 0, 0, cv::INTER_AREA);
  } else {
    bgr = bgr_in;
  }

  cv::Mat yuv_i420;
  cv::cvtColor(bgr, yuv_i420, cv::COLOR_BGR2YUV_I420);

  const size_t y_sz = (size_t)w * (size_t)h;
  const size_t u_sz = y_sz / 4;
  const size_t v_sz = u_sz;

  if ((size_t)yuv_i420.total() != y_sz + u_sz + v_sz) {
    throw std::runtime_error("bgr_to_nv12_tight: unexpected I420 size from OpenCV");
  }

  const uint8_t* base = yuv_i420.data;
  const uint8_t* y = base;
  const uint8_t* u = base + y_sz;
  const uint8_t* v = base + y_sz + u_sz;

  std::vector<uint8_t> nv12;
  nv12.resize(y_sz + 2 * u_sz);
  std::memcpy(nv12.data(), y, y_sz);

  uint8_t* uv = nv12.data() + y_sz;
  for (size_t i = 0; i < u_sz; ++i) {
    uv[2 * i + 0] = u[i];
    uv[2 * i + 1] = v[i];
  }
  return nv12;
}

static std::vector<uint8_t> nv12_pad_center(const std::vector<uint8_t>& src_nv12,
                                            int src_w, int src_h,
                                            int dst_w, int dst_h) {
  if (dst_w < src_w || dst_h < src_h) throw std::runtime_error("nv12_pad_center: dst must be >= src");
  if ((src_w & 1) || (src_h & 1) || (dst_w & 1) || (dst_h & 1))
    throw std::runtime_error("nv12_pad_center: NV12 requires even dimensions");

  const size_t src_y_sz = (size_t)src_w * (size_t)src_h;
  const size_t src_uv_sz = src_y_sz / 2;
  if (src_nv12.size() != src_y_sz + src_uv_sz)
    throw std::runtime_error("nv12_pad_center: src size mismatch");

  const size_t dst_y_sz = (size_t)dst_w * (size_t)dst_h;
  const size_t dst_uv_sz = dst_y_sz / 2;

  std::vector<uint8_t> dst(dst_y_sz + dst_uv_sz);
  std::memset(dst.data(), 16, dst_y_sz);
  std::memset(dst.data() + dst_y_sz, 128, dst_uv_sz);

  int off_x = (dst_w - src_w) / 2;
  int off_y = (dst_h - src_h) / 2;
  off_x &= ~1;
  off_y &= ~1;

  const uint8_t* src_y  = src_nv12.data();
  const uint8_t* src_uv = src_nv12.data() + src_y_sz;

  uint8_t* dst_y  = dst.data();
  uint8_t* dst_uv = dst.data() + dst_y_sz;

  for (int r = 0; r < src_h; ++r) {
    std::memcpy(dst_y + (size_t)(off_y + r) * (size_t)dst_w + (size_t)off_x,
                src_y + (size_t)r * (size_t)src_w,
                (size_t)src_w);
  }

  const int src_uv_h = src_h / 2;
  const int dst_uv_off_y = off_y / 2;

  for (int r = 0; r < src_uv_h; ++r) {
    std::memcpy(dst_uv + (size_t)(dst_uv_off_y + r) * (size_t)dst_w + (size_t)off_x,
                src_uv + (size_t)r * (size_t)src_w,
                (size_t)src_w);
  }

  return dst;
}

AppSrcImage::AppSrcImage(std::string image_path,
                         int content_w,
                         int content_h,
                         int enc_w,
                         int enc_h,
                         int fps)
  : image_path_(std::move(image_path)),
    content_w_(content_w), content_h_(content_h),
    enc_w_(enc_w), enc_h_(enc_h),
    fps_(fps) {
  gst_init_once();

  if (!fs::exists(image_path_)) {
    throw std::runtime_error("AppSrcImage: file not found: " + image_path_);
  }
  if ((content_w_ & 1) || (content_h_ & 1) || (enc_w_ & 1) || (enc_h_ & 1)) {
    throw std::runtime_error("AppSrcImage: widths/heights must be even for NV12");
  }
  if (enc_w_ < content_w_ || enc_h_ < content_h_) {
    throw std::runtime_error("AppSrcImage: enc dims must be >= content dims");
  }
  if (fps_ <= 0) {
    throw std::runtime_error("AppSrcImage: fps must be > 0");
  }

  cv::Mat bgr = cv::imread(image_path_, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("AppSrcImage: OpenCV failed to read: " + image_path_);
  }

  auto nv12_content = bgr_to_nv12_tight(bgr, content_w_, content_h_);
  std::vector<uint8_t> nv12_enc =
    (enc_w_ == content_w_ && enc_h_ == content_h_)
      ? std::move(nv12_content)
      : nv12_pad_center(nv12_content, content_w_, content_h_, enc_w_, enc_h_);

  nv12_enc_ = std::make_shared<std::vector<uint8_t>>(std::move(nv12_enc));
}

std::string AppSrcImage::gst_fragment(int node_index) const {
  (void)node_index;
  std::ostringstream ss;
  ss << "appsrc name=mysrc is-live=true format=time "
     << "! queue name=n" << node_index << "_queue "
     << "! video/x-raw,format=NV12,width=" << enc_w_ << ",height=" << enc_h_
     << ",framerate=" << fps_ << "/1";
  return ss.str();
}

std::vector<std::string> AppSrcImage::element_names(int node_index) const {
  return {"mysrc", "n" + std::to_string(node_index) + "_queue"};
}

H264Parse::H264Parse(int config_interval) : config_interval_(config_interval) {}

std::string H264Parse::gst_fragment(int node_index) const {
  std::ostringstream ss;
  ss << "h264parse name=n" << node_index << "_h264parse disable-passthrough=true "
     << "config-interval=" << config_interval_;
  return ss.str();
}

std::vector<std::string> H264Parse::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_h264parse"};
}

RtpH264Pay::RtpH264Pay(int pt, int config_interval)
  : pt_(pt), config_interval_(config_interval) {}

std::string RtpH264Pay::gst_fragment(int /*node_index*/) const {
  std::ostringstream ss;
  ss << "rtph264pay name=pay0 pt=" << pt_ << " config-interval=" << config_interval_;
  return ss.str();
}

std::vector<std::string> RtpH264Pay::element_names(int /*node_index*/) const {
  return {"pay0"};
}

// =====================================================================================
// RTSP server backend
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
  GstElement* appsrc = nullptr; // ref'd
  guint timer_id = 0;

  int w = 0, h = 0, fps = 30;
  guint64 frame_count = 0;
  guint64 frame_duration_ns = 0;

  std::shared_ptr<std::vector<uint8_t>> nv12;
  std::atomic<bool> stopped{false};
};

static gboolean push_frame_cb(gpointer user_data) {
  auto* pc = reinterpret_cast<PushCtx*>(user_data);
  if (!pc || !pc->appsrc || pc->stopped.load()) return G_SOURCE_REMOVE;
  if (!pc->nv12 || pc->nv12->empty()) return G_SOURCE_REMOVE;

  const size_t y_sz  = (size_t)pc->w * (size_t)pc->h;
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

static void media_unprepared_cb(GstRTSPMedia* /*media*/, gpointer user_data) {
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

static std::string ensure_mount_path(const std::string& mount) {
  if (mount.empty()) return "/image";
  if (!mount.empty() && mount[0] == '/') return mount;
  return "/" + mount;
}

static std::string make_rtsp_url(int port, const std::string& mount) {
  return "rtsp://127.0.0.1:" + std::to_string(port) + ensure_mount_path(mount);
}

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
// PipelineSession
// =====================================================================================

PipelineSession& PipelineSession::add(std::shared_ptr<Node> node) {
  nodes_.push_back(std::move(node));
  return *this;
}

PipelineSession& PipelineSession::gst(std::string fragment) {
  return add(nodes::Gst(std::move(fragment)));
}


static void set_state_or_throw(GstElement* pipeline,
                               GstState target,
                               const char* where,
                               const std::shared_ptr<DiagCtx>& diag) {
  if (!pipeline) throw std::runtime_error(std::string(where) + ": pipeline is null");

  GstStateChangeReturn r = gst_element_set_state(pipeline, target);
  if (r == GST_STATE_CHANGE_FAILURE) {
    maybe_dump_dot(pipeline, std::string(where) + "_set_state_failure");
    PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
    rep.repro_note = std::string(where) + ": failed to set state.\n";
    throw PipelineError(rep.repro_note, std::move(rep));
  }

  GstState cur = GST_STATE_VOID_PENDING, pend = GST_STATE_VOID_PENDING;
  gst_element_get_state(pipeline, &cur, &pend, 2 * GST_SECOND);
  drain_bus(pipeline, diag);
  throw_if_bus_error(pipeline, diag, where);
}

FrameStream PipelineSession::run() {
  gst_init_once();

  enforce_sink_last(nodes_);

  require_element("appsink", "PipelineSession::run");
  require_element("identity", "PipelineSession::run");

  const bool insert_boundaries = should_insert_boundaries_for_mode("SIMA_GST_RUN_INSERT_BOUNDARIES", false);

  BuildResult br = build_pipeline_full(nodes_, insert_boundaries, "mysink");
  last_pipeline_ = br.pipeline_string;

  GError* err = nullptr;
  GstElement* pipeline = gst_parse_launch(last_pipeline_.c_str(), &err);
  if (!pipeline) {
    std::string msg = err ? err->message : "unknown";
    if (err) g_error_free(err);
    throw std::runtime_error("PipelineSession::run: gst_parse_launch failed: " + msg +
                             "\nPipeline:\n" + last_pipeline_);
  }
  if (err) g_error_free(err);

  if (env_bool("SIMA_GST_ENFORCE_NAMES", false)) {
    enforce_names_contract(pipeline, br);
  }

  // Probes are opt-in (keep hot path fast)
  attach_boundary_probes(pipeline, br.diag);

  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
  if (!sink) {
    maybe_dump_dot(pipeline, "run_missing_mysink");
    stop_and_unref(pipeline);
    throw std::runtime_error("PipelineSession::run: appsink 'mysink' not found.\n"
                             "Fix: add OutputAppSink() as the last node.\nPipeline:\n" + last_pipeline_);
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
  fs.set_diag(br.diag);
  return fs;
}

TapStream PipelineSession::run_tap(const std::string& point_name) {
  gst_init_once();

  require_element("appsink", "PipelineSession::run_tap");
  require_element("identity", "PipelineSession::run_tap");

  const bool insert_boundaries = should_insert_boundaries_for_mode("SIMA_GST_TAP_INSERT_BOUNDARIES", true);

  BuildResult br = build_pipeline_tap(nodes_, point_name, insert_boundaries);
  last_pipeline_ = br.pipeline_string;

  GError* err = nullptr;
  GstElement* pipeline = gst_parse_launch(last_pipeline_.c_str(), &err);
  if (!pipeline) {
    std::string msg = err ? err->message : "unknown";
    if (err) g_error_free(err);
    throw std::runtime_error("PipelineSession::run_tap: gst_parse_launch failed: " + msg +
                             "\nPipeline:\n" + last_pipeline_);
  }
  if (err) g_error_free(err);

  if (env_bool("SIMA_GST_ENFORCE_NAMES", false)) {
    enforce_names_contract(pipeline, br);
  }

  attach_boundary_probes(pipeline, br.diag);

  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), br.appsink_name.c_str());
  if (!sink) {
    maybe_dump_dot(pipeline, "run_tap_missing_sink");
    stop_and_unref(pipeline);
    throw std::runtime_error("PipelineSession::run_tap: appsink '" + br.appsink_name + "' not found.\nPipeline:\n" + last_pipeline_);
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

  if (!opt.parse_launch) {
    rep.pipeline_string = "<parse_launch disabled>";
    rep.repro_note = "validate(parse_launch=false): skipped.";
    return rep;
  }

  // Validate builds pipeline string and (best-effort) prerolls without going PLAYING.
  const bool insert_boundaries = should_insert_boundaries_for_mode("SIMA_GST_VALIDATE_INSERT_BOUNDARIES", true);

  BuildResult br = build_pipeline_full(nodes_, insert_boundaries, "mysink");
  rep.pipeline_string = br.pipeline_string;

  // If user wants naming enforcement, validate fragments exist in the parsed bin.
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

  // Attach probes in validate by default? Still opt-in via env.
  attach_boundary_probes(pipeline, br.diag);

  if (opt.enforce_names) {
    enforce_names_contract(pipeline, br);
  }

  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
  if (!sink) {
    rep.repro_note = "validate: appsink 'mysink' not found (ensure OutputAppSink is last).";
    stop_and_unref(pipeline);
    return rep;
  }

  // Set PAUSED to preroll (no PLAYING)
  try {
    set_state_or_throw(pipeline, GST_STATE_PAUSED, "PipelineSession::validate", br.diag);
  } catch (const PipelineError& e) {
    rep = e.report();
    rep.repro_note = std::string("validate: failed to PAUSE/preroll.\n") + rep.repro_note;
    gst_object_unref(sink);
    stop_and_unref(pipeline);
    return rep;
  }

  // Try pull preroll sample (best effort)
  const int timeout_ms = std::max(10, std::atoi(env_str("SIMA_GST_VALIDATE_TIMEOUT_MS", "2000").c_str()));
  GstSample* sample = nullptr;

#if GST_CHECK_VERSION(1,10,0)
  sample = gst_app_sink_try_pull_preroll(GST_APP_SINK(sink), (guint64)timeout_ms * GST_MSECOND);
#else
  sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), (guint64)timeout_ms * GST_MSECOND);
#endif

  if (sample) gst_sample_unref(sample);

  // Build report from diag snapshot
  rep = br.diag->snapshot_basic();
  rep.pipeline_string = br.pipeline_string;
  rep.nodes = br.diag->node_reports;
  rep.boundaries.reserve(br.diag->boundaries.size());
  for (auto& b : br.diag->boundaries) if (b) rep.boundaries.push_back(*b);

  rep.repro_note =
    sample ? "validate: preroll OK (PAUSED)." : "validate: preroll timed out in PAUSED (live source or negotiation stall).";
  rep.repro_note += "\n" + boundary_summary(br.diag);

  gst_object_unref(sink);
  stop_and_unref(pipeline);
  return rep;
}

RtspServerHandle PipelineSession::run_rtsp(const RtspServerOptions& opt) {
  gst_init_once();

  require_element("appsrc", "PipelineSession::run_rtsp");
  require_element("rtph264pay", "PipelineSession::run_rtsp");
  require_element("h264parse", "PipelineSession::run_rtsp");

  // Find AppSrcImage (for nv12 + dims)
  AppSrcImage* src_img = nullptr;
  for (auto& n : nodes_) {
    if (!src_img) src_img = dynamic_cast<AppSrcImage*>(n.get());
  }
  if (!src_img) throw std::runtime_error("PipelineSession::run_rtsp: missing AppSrcImage node");

  // Build RTSP factory launch string
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

    g_signal_connect(factory, "media-configure",
      G_CALLBACK(+[](GstRTSPMediaFactory*, GstRTSPMedia* media, gpointer user_data) {
        auto* impl = reinterpret_cast<RtspServerImpl*>(user_data);
        if (!impl) return;

        GstElement* top = gst_rtsp_media_get_element(media);
        if (!top) return;

        GstElement* src = gst_bin_get_by_name_recurse_up(GST_BIN(top), "mysrc");
        if (!src) { gst_object_unref(top); return; }

        GstCaps* caps = gst_caps_new_simple(
          "video/x-raw",
          "format", G_TYPE_STRING, "NV12",
          "width", G_TYPE_INT, impl->enc_w,
          "height", G_TYPE_INT, impl->enc_h,
          "framerate", GST_TYPE_FRACTION, impl->fps, 1,
          nullptr
        );
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
      impl
    );

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

  try {
    // Build a normal run() stream and pull 1 frame copy
    FrameStream fs = run();
    out.first_frame = fs.next_copy(opt.timeout_ms);
    out.report = fs.report_snapshot(/*heavy=*/true);
    fs.close();
    if (out.report.repro_note.empty()) out.report.repro_note = "run_debug: OK";
  } catch (const PipelineError& e) {
    out.report = e.report();
    if (out.report.repro_note.empty()) out.report.repro_note = e.what();
  } catch (const std::exception& e) {
    out.report.pipeline_string = last_pipeline_;
    out.report.repro_note = std::string("run_debug: exception: ") + e.what();
  }

  return out;
}

} // namespace sima
