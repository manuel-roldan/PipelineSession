#include "pipeline/internal/StageConfig.h"

#include "builder/ConfigJsonProvider.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace sima::stages {
namespace {

const nlohmann::json* config_json_from_group(const NodeGroup& group) {
  for (const auto& node : group.nodes()) {
    if (!node) continue;
    auto* provider = dynamic_cast<ConfigJsonProvider*>(node.get());
    if (!provider) continue;
    const nlohmann::json* cfg = provider->config_json();
    if (cfg) return cfg;
  }
  return nullptr;
}

int elem_size_from_string(const std::string& fmt) {
  std::string up;
  up.reserve(fmt.size());
  for (char c : fmt) {
    up.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  if (up.find("INT8") != std::string::npos || up.find("UINT8") != std::string::npos) return 1;
  if (up.find("BF16") != std::string::npos || up.find("BFLOAT16") != std::string::npos) return 2;
  if (up.find("INT16") != std::string::npos || up.find("UINT16") != std::string::npos) return 2;
  if (up.find("INT32") != std::string::npos || up.find("FP32") != std::string::npos) return 4;
  if (up.find("FP64") != std::string::npos) return 8;
  return 1;
}

std::vector<int64_t> read_int_array(const nlohmann::json& v) {
  std::vector<int64_t> out;
  if (v.is_number_integer()) {
    out.push_back(v.get<int64_t>());
    return out;
  }
  if (v.is_number()) {
    out.push_back(static_cast<int64_t>(v.get<double>()));
    return out;
  }
  if (v.is_array()) {
    for (const auto& entry : v) {
      if (entry.is_number_integer()) {
        out.push_back(entry.get<int64_t>());
      } else if (entry.is_number()) {
        out.push_back(static_cast<int64_t>(entry.get<double>()));
      }
    }
  }
  return out;
}

std::vector<std::string> read_string_array(const nlohmann::json& v) {
  std::vector<std::string> out;
  if (v.is_string()) {
    out.push_back(v.get<std::string>());
    return out;
  }
  if (v.is_array()) {
    for (const auto& entry : v) {
      if (entry.is_string()) {
        out.push_back(entry.get<std::string>());
      }
    }
  }
  return out;
}

int read_first_int(const nlohmann::json& v) {
  if (v.is_number_integer()) return v.get<int>();
  if (v.is_number()) return static_cast<int>(v.get<double>());
  if (v.is_array()) {
    for (const auto& entry : v) {
      if (entry.is_number_integer()) return entry.get<int>();
      if (entry.is_number()) return static_cast<int>(entry.get<double>());
    }
  }
  return 0;
}

int read_int_field(const nlohmann::json& j, const char* key) {
  if (!j.contains(key)) return 0;
  return read_first_int(j.at(key));
}

} // namespace

PreprocOutputInfo read_preproc_output_info(const NodeGroup& group) {
  PreprocOutputInfo info;
  const nlohmann::json* cfg = config_json_from_group(group);
  if (!cfg) return info;
  if (cfg->contains("tessellate")) {
    const auto& t = (*cfg)["tessellate"];
    if (t.is_boolean()) info.tessellate = t.get<bool>();
    if (t.is_number_integer()) info.tessellate = t.get<int>() != 0;
  }
  if (cfg->contains("output_dtype") && (*cfg)["output_dtype"].is_string()) {
    info.output_dtype = (*cfg)["output_dtype"].get<std::string>();
  }
  info.dims.width = read_int_field(*cfg, "output_width");
  info.dims.height = read_int_field(*cfg, "output_height");
  info.dims.depth = read_int_field(*cfg, "output_channels");
  if (info.dims.depth <= 0) {
    info.dims.depth = read_int_field(*cfg, "tile_channels");
  }
  if (cfg->contains("output_memory_order") && (*cfg)["output_memory_order"].is_array()) {
    for (const auto& entry : (*cfg)["output_memory_order"]) {
      if (entry.is_string()) {
        info.output_memory_order.push_back(entry.get<std::string>());
      }
    }
  }
  return info;
}

MlaOutputInfo read_mla_output_info(const NodeGroup& group) {
  MlaOutputInfo info;
  const nlohmann::json* cfg = config_json_from_group(group);
  if (!cfg) return info;
  if (cfg->contains("data_type")) {
    const auto& dt = (*cfg)["data_type"];
    if (dt.is_array() && !dt.empty() && dt[0].is_string()) {
      info.data_type = dt[0].get<std::string>();
    } else if (dt.is_string()) {
      info.data_type = dt.get<std::string>();
    }
  }
  info.dims.width = read_int_field(*cfg, "output_width");
  info.dims.height = read_int_field(*cfg, "output_height");
  info.dims.depth = read_int_field(*cfg, "output_depth");
  if (info.dims.width <= 0) info.dims.width = read_int_field(*cfg, "slice_width");
  if (info.dims.height <= 0) info.dims.height = read_int_field(*cfg, "slice_height");
  if (info.dims.depth <= 0) info.dims.depth = read_int_field(*cfg, "slice_depth");
  if (cfg->contains("simaai__params") && (*cfg)["simaai__params"].is_object()) {
    const auto& params = (*cfg)["simaai__params"];
    if (params.contains("outputs") && params["outputs"].is_array() &&
        !params["outputs"].empty() && params["outputs"][0].is_object()) {
      const auto& out = params["outputs"][0];
      if (out.contains("size") && out["size"].is_number_integer()) {
        info.size_bytes = out["size"].get<int64_t>();
      } else if (out.contains("size") && out["size"].is_number()) {
        info.size_bytes = static_cast<int64_t>(out["size"].get<double>());
      }
    }
  }
  return info;
}

BoxDecodeInputInfo read_boxdecode_input_info(const NodeGroup& group) {
  BoxDecodeInputInfo info;
  const nlohmann::json* cfg = config_json_from_group(group);
  if (!cfg) return info;
  info.dims.width = read_int_field(*cfg, "input_width");
  info.dims.height = read_int_field(*cfg, "input_height");
  info.dims.depth = read_int_field(*cfg, "input_depth");
  return info;
}

BoxDecodeExpectedInfo read_boxdecode_expected_info(const NodeGroup& group) {
  BoxDecodeExpectedInfo info;
  const nlohmann::json* cfg = config_json_from_group(group);
  if (!cfg) return info;
  if (cfg->contains("buffers") && (*cfg)["buffers"].is_object()) {
    const auto& buffers = (*cfg)["buffers"];
    if (buffers.contains("input") && buffers["input"].is_array() &&
        !buffers["input"].empty() && buffers["input"][0].is_object()) {
      const auto& in = buffers["input"][0];
      if (in.contains("size") && in["size"].is_number_integer()) {
        info.buffer_size = in["size"].get<int64_t>();
      } else if (in.contains("size") && in["size"].is_number()) {
        info.buffer_size = static_cast<int64_t>(in["size"].get<double>());
      }
    }
  }
  std::vector<int64_t> widths;
  std::vector<int64_t> heights;
  std::vector<int64_t> depths;
  if (cfg->contains("input_width")) widths = read_int_array((*cfg)["input_width"]);
  if (cfg->contains("input_height")) heights = read_int_array((*cfg)["input_height"]);
  if (cfg->contains("input_depth")) depths = read_int_array((*cfg)["input_depth"]);
  const size_t n = std::min(widths.size(), std::min(heights.size(), depths.size()));
  for (size_t i = 0; i < n; ++i) {
    info.total_elems += widths[i] * heights[i] * depths[i];
  }
  std::string dtype;
  if (cfg->contains("data_type")) {
    const auto& dt = (*cfg)["data_type"];
    if (dt.is_array() && !dt.empty() && dt[0].is_string()) {
      dtype = dt[0].get<std::string>();
    } else if (dt.is_string()) {
      dtype = dt.get<std::string>();
    }
  }
  info.elem_size = elem_size_from_string(dtype);
  info.total_bytes = info.total_elems * static_cast<int64_t>(info.elem_size);
  return info;
}

std::string build_boxdecode_caps_override(const NodeGroup& group) {
  const nlohmann::json* cfg = config_json_from_group(group);
  if (!cfg) return {};

  const std::vector<int64_t> widths = cfg->contains("input_width")
      ? read_int_array((*cfg)["input_width"])
      : std::vector<int64_t>{};
  const std::vector<int64_t> heights = cfg->contains("input_height")
      ? read_int_array((*cfg)["input_height"])
      : std::vector<int64_t>{};
  const std::vector<int64_t> depths = cfg->contains("input_depth")
      ? read_int_array((*cfg)["input_depth"])
      : std::vector<int64_t>{};
  if (widths.empty() || heights.empty() || depths.empty()) return {};

  const std::vector<std::string> dtypes = cfg->contains("data_type")
      ? read_string_array((*cfg)["data_type"])
      : std::vector<std::string>{};
  const std::vector<int64_t> slice_w = cfg->contains("slice_width")
      ? read_int_array((*cfg)["slice_width"])
      : std::vector<int64_t>{};
  const std::vector<int64_t> slice_h = cfg->contains("slice_height")
      ? read_int_array((*cfg)["slice_height"])
      : std::vector<int64_t>{};
  const std::vector<int64_t> slice_d = cfg->contains("slice_depth")
      ? read_int_array((*cfg)["slice_depth"])
      : std::vector<int64_t>{};

  const size_t n = std::min(widths.size(), std::min(heights.size(), depths.size()));
  if (n == 0) return {};

  std::ostringstream caps;
  caps << "application/vnd.simaai.tensor,format=MLA";
  if (widths[0] > 0) caps << ",width=" << widths[0];
  if (heights[0] > 0) caps << ",height=" << heights[0];
  if (depths[0] > 0) caps << ",depth=" << depths[0];

  for (size_t i = 0; i < n; ++i) {
    if (i < dtypes.size() && !dtypes[i].empty()) {
      caps << ",data_type__" << i << "=" << dtypes[i];
    }
    caps << ",width__" << i << "=" << widths[i];
    caps << ",height__" << i << "=" << heights[i];
    caps << ",depth__" << i << "=" << depths[i];
    if (i < slice_w.size() && slice_w[i] > 0) {
      caps << ",slice_width__" << i << "=" << slice_w[i];
    }
    if (i < slice_h.size() && slice_h[i] > 0) {
      caps << ",slice_height__" << i << "=" << slice_h[i];
    }
    if (i < slice_d.size() && slice_d[i] > 0) {
      caps << ",slice_depth__" << i << "=" << slice_d[i];
    }
  }

  return caps.str();
}

} // namespace sima::stages
