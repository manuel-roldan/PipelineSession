// include/pipeline/internal/GstDiagnosticsUtil.h
#pragma once

#include "pipeline/internal/Diagnostics.h" // provides sima::pipeline_internal::DiagCtx
#include "pipeline/PipelineReport.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <optional>
#include <string>

namespace sima::pipeline_internal {

struct SampleHolder {
  GstSample* sample = nullptr;
  GstVideoInfo vinfo{};
  GstVideoFrame frame{};
  bool mapped = false;

  explicit SampleHolder(GstSample* s) : sample(s) {}
  ~SampleHolder() {
    if (mapped) gst_video_frame_unmap(&frame);
    if (sample) gst_sample_unref(sample);
  }

  SampleHolder(const SampleHolder&) = delete;
  SampleHolder& operator=(const SampleHolder&) = delete;
};

bool map_video_frame_read(SampleHolder& h, std::string& err);



// -----------------------------
// Small env/string helpers
// -----------------------------
bool env_bool(const char* key, bool def_val);
std::string env_str(const char* key, const std::string& def_val = "");
std::string sanitize_name(const std::string& in);

// -----------------------------
// GStreamer stringify helpers
// -----------------------------
std::string gst_caps_to_string_safe(GstCaps* caps);
std::string gst_structure_to_string_safe(const GstStructure* st);
std::string gst_message_to_string(GstMessage* msg);

// Optional convenience used by TapStream
std::string caps_features_string(GstCaps* caps);

// -----------------------------
// DOT dump helper
// -----------------------------
void maybe_dump_dot(GstElement* pipeline, const std::string& tag);

// -----------------------------
// Diagnostics helpers
// -----------------------------
std::string boundary_summary(const std::shared_ptr<DiagCtx>& diag);

void drain_bus(GstElement* pipeline,
               const std::shared_ptr<DiagCtx>& diag,
               const char* where = nullptr);

void throw_if_bus_error(GstElement* pipeline,
                        const std::shared_ptr<DiagCtx>& diag,
                        const char* where);

// -----------------------------
// Appsink polling helper
// Returns a GstSample* that the caller must gst_sample_unref().
// -----------------------------
std::optional<GstSample*> try_pull_sample_sliced(GstElement* pipeline,
                                                 GstElement* appsink,
                                                 int timeout_ms,
                                                 const std::shared_ptr<DiagCtx>& diag,
                                                 const char* where);

// -----------------------------
// Teardown helper (best-effort, avoids deadlocks)
// -----------------------------
void stop_and_unref(GstElement*& e);

} // namespace sima::pipeline_internal
