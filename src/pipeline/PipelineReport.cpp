// src/pipeline/PipelineReport.cpp
#include "sima/pipeline/PipelineReport.h"

#include <cstdint>
#include <sstream>
#include <string>

namespace sima {

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
        // Replace other control chars with a space to keep output readable/stable.
        if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
  }
  return o;
}

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

} // namespace sima
