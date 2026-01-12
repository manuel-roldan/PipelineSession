// src/pipeline/internal/GstDiagnosticsUtil.cpp
#include "pipeline/internal/GstDiagnosticsUtil.h"

#include "sima/pipeline/Errors.h"

#include <gst/app/gstappsink.h>
#include <gst/gstdebugutils.h>
#include <glib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <utility>

namespace sima::pipeline_internal {
namespace {

const char* state_name(GstState s) {
  switch (s) {
    case GST_STATE_VOID_PENDING: return "VOID_PENDING";
    case GST_STATE_NULL:         return "NULL";
    case GST_STATE_READY:        return "READY";
    case GST_STATE_PAUSED:       return "PAUSED";
    case GST_STATE_PLAYING:      return "PLAYING";
    default:                     return "UNKNOWN";
  }
}

} // namespace

bool map_video_frame_read(SampleHolder& h, std::string& err) {
  err.clear();
  if (!h.sample) {
    err = "missing sample";
    return false;
  }

  GstCaps* caps = gst_sample_get_caps(h.sample);
  if (!caps) {
    err = "missing caps";
    return false;
  }

  GstBuffer* buf = gst_sample_get_buffer(h.sample);
  if (!buf) {
    err = "missing buffer";
    return false;
  }

  if (!gst_video_info_from_caps(&h.vinfo, caps)) {
    err = "gst_video_info_from_caps failed";
    return false;
  }

  if (!gst_video_frame_map(&h.frame, &h.vinfo, buf, GST_MAP_READ)) {
    err = "gst_video_frame_map failed (non-mappable memory?)";
    return false;
  }

  h.mapped = true;
  return true;
}

bool env_bool(const char* key, bool def_val) {
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

std::string env_str(const char* key, const std::string& def_val) {
  const char* v = std::getenv(key);
  if (!v) return def_val;
  return std::string(v);
}

std::string sanitize_name(const std::string& in) {
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

std::string gst_caps_to_string_safe(GstCaps* caps) {
  if (!caps) return "<null caps>";
  gchar* s = gst_caps_to_string(caps);
  if (!s) return "<caps_to_string failed>";
  std::string out = s;
  g_free(s);
  return out;
}

std::string gst_structure_to_string_safe(const GstStructure* st) {
  if (!st) return "<null structure>";
  gchar* s = gst_structure_to_string(st);
  if (!s) return "<structure_to_string failed>";
  std::string out = s;
  g_free(s);
  return out;
}

std::string gst_message_to_string(GstMessage* msg) {
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

std::string caps_features_string(GstCaps* caps) {
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
  if (gst_caps_features_is_any(f)) return "ANY";
  if (gst_caps_features_contains(f, GST_CAPS_FEATURE_MEMORY_SYSTEM_MEMORY)) return "memory:SystemMemory";
  return "<features>";
#endif
}

void maybe_dump_dot(GstElement* pipeline, const std::string& tag) {
  if (!pipeline) return;

  const std::string dir = env_str("SIMA_GST_DOT_DIR", "");
  if (dir.empty()) return;

  // Tell GStreamer where to dump dot graphs.
  g_setenv("GST_DEBUG_DUMP_DOT_DIR", dir.c_str(), TRUE);

  const std::string t = "sima_" + sanitize_name(tag);
  gst_debug_bin_to_dot_file_with_ts(GST_BIN(pipeline),
                                   GST_DEBUG_GRAPH_SHOW_ALL,
                                   t.c_str());
}

std::string boundary_summary(const std::shared_ptr<DiagCtx>& diag) {
  if (!diag || diag->boundaries.empty()) return "";

  const int64_t now = (int64_t)g_get_monotonic_time();

  int best_idx = -1;
  int64_t best_t = 0;
  bool best_out = false;

  // Use snapshots to keep this readable and consistent.
  std::vector<BoundaryFlowStats> snaps;
  snaps.reserve(diag->boundaries.size());
  for (const auto& bp : diag->boundaries) {
    if (!bp) continue;
    snaps.push_back(bp->snapshot());
  }
  if (snaps.empty()) return "";

  for (size_t i = 0; i < snaps.size(); ++i) {
    const auto& b = snaps[i];
    if (b.last_out_wall_us > best_t) { best_t = b.last_out_wall_us; best_idx = (int)i; best_out = true; }
    if (b.last_in_wall_us  > best_t) { best_t = b.last_in_wall_us;  best_idx = (int)i; best_out = false; }
  }

  std::ostringstream ss;
  ss << "BoundaryFlow:\n";
  for (const auto& b : snaps) {
    ss << "  - " << b.boundary_name
       << " after=" << b.after_node_index
       << " before=" << b.before_node_index
       << " in=" << b.in_buffers
       << " out=" << b.out_buffers
       << " last_in_age_ms=" << (b.last_in_wall_us ? ((now - b.last_in_wall_us) / 1000) : 0)
       << " last_out_age_ms=" << (b.last_out_wall_us ? ((now - b.last_out_wall_us) / 1000) : 0)
       << "\n";
  }

  if (best_idx >= 0 && best_t > 0) {
    const auto& b = snaps[(size_t)best_idx];
    ss << "LikelyStall: last activity "
       << (best_out ? "leaving " : "entering ")
       << b.boundary_name
       << " age_ms=" << ((now - best_t) / 1000)
       << " (after node " << b.after_node_index
       << ", before node " << b.before_node_index << ")\n";
  }

  return ss.str();
}

void drain_bus(GstElement* pipeline,
               const std::shared_ptr<DiagCtx>& diag,
               const char* /*where*/) {
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

void throw_if_bus_error(GstElement* pipeline,
                        const std::shared_ptr<DiagCtx>& diag,
                        const char* where) {
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

      maybe_dump_dot(pipeline, std::string(where) + "_error");

      PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
      rep.repro_note = std::string(where) + ": GST ERROR: " + line;
      if (diag) rep.repro_note += "\n" + boundary_summary(diag);
      throw PipelineError(rep.repro_note, std::move(rep));
    }

    gst_message_unref(msg);
  }

  gst_object_unref(bus);
}

std::optional<GstSample*> try_pull_sample_sliced(GstElement* pipeline,
                                                 GstElement* appsink,
                                                 int timeout_ms,
                                                 const std::shared_ptr<DiagCtx>& diag,
                                                 const char* where) {
  if (!pipeline || !appsink) return std::nullopt;

  const int slice_ms =
      std::max(10, std::atoi(env_str("SIMA_GST_POLL_SLICE_MS", "200").c_str()));

  const bool infinite = (timeout_ms < 0);
  int remaining = timeout_ms;

  while (true) {
    int this_ms = 0;
    if (timeout_ms == 0) this_ms = 0;
    else if (infinite) this_ms = slice_ms;
    else this_ms = std::min(slice_ms, remaining);

    GstSample* s = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink),
                                                (guint64)this_ms * GST_MSECOND);
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

void stop_and_unref(GstElement*& e) {
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

  // Intentionally leak if teardown is stuck to avoid deadlock in callers.
  std::cerr << "[WARN] stop_and_unref(): teardown timed out; leaking pipeline to avoid hanging.\n";
}

} // namespace sima::pipeline_internal
