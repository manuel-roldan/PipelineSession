#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sima {

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

} // namespace sima
