#include "nodes/sima/Preproc.h"

#include "ModelSession.hpp"
#include "mpk/ModelMPK.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace sima {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct Preproc::PreprocConfigHolder {
  std::string path;
  bool keep = false;
  json config;
  bool has_config = false;
  ~PreprocConfigHolder() {
    if (!keep && !path.empty()) {
      std::remove(path.c_str());
    }
  }
};

namespace {

std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

int channels_from_format(const std::string& fmt, int fallback) {
  if (!fmt.empty()) {
    const std::string up = upper_copy(fmt);
    if (up == "GRAY" || up == "GRAY8") return 1;
    if (up == "RGB" || up == "BGR" || up == "NV12" || up == "I420") return 3;
  }
  return (fallback > 0) ? fallback : 3;
}

std::vector<float> ensure_three(const std::vector<float>& v, float def_val) {
  if (v.empty()) return {def_val, def_val, def_val};
  if (v.size() == 1) return {v[0], v[0], v[0]};
  if (v.size() >= 3) return {v[0], v[1], v[2]};
  std::vector<float> out = v;
  while (out.size() < 3) out.push_back(def_val);
  return out;
}

std::vector<std::string> output_memory_order_default(const PreprocOptions& opt) {
  if (!opt.output_memory_order.empty()) return opt.output_memory_order;
  const std::string next = upper_copy(opt.next_cpu);
  if (next == "APU") {
    return {"output_rgb_image", "output_tessellated_image"};
  }
  return {"output_tessellated_image", "output_rgb_image"};
}

bool has_next_cpu_manual(const json& j) {
  if (!j.contains("simaai__params") || !j["simaai__params"].is_object()) return false;
  const json& params = j["simaai__params"];
  if (!params.contains("next_cpu_manual")) return false;
  const json& val = params["next_cpu_manual"];
  if (val.is_boolean()) return val.get<bool>();
  if (val.is_number_integer()) return val.get<int>() != 0;
  if (val.is_string()) return val.get<std::string>() != "0";
  return false;
}

void set_next_cpu_manual(json& j) {
  if (!j.contains("simaai__params") || !j["simaai__params"].is_object()) {
    j["simaai__params"] = json::object();
  }
  j["simaai__params"]["next_cpu_manual"] = 1;
}

std::string find_preproc_config_path(const sima::mpk::ModelMPK& model) {
  std::string path = model.find_config_path_by_plugin("process_cvu");
  if (path.empty()) path = model.find_config_path_by_plugin("preproc");
  if (path.empty()) path = model.find_config_path_by_processor("CVU");
  return path;
}

std::string make_temp_json_path(const std::string& dir) {
  std::string root = dir.empty() ? "/tmp" : dir;
  std::error_code ec;
  fs::create_directories(root, ec);
  if (ec) {
    throw std::runtime_error("Preproc: failed to create config dir: " + root);
  }

  std::string templ = (fs::path(root) / "sima_preproc_XXXXXX.json").string();
  std::vector<char> buf(templ.begin(), templ.end());
  buf.push_back('\0');

  int fd = mkstemps(buf.data(), 5);
  if (fd < 0) {
    throw std::runtime_error(std::string("Preproc: mkstemps failed: ") + std::strerror(errno));
  }
  close(fd);
  return std::string(buf.data());
}

json load_json_file(const std::string& path, const char* label) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error(std::string(label) + ": failed to open config: " + path);
  }
  json j;
  in >> j;
  return j;
}

void write_json_file(const json& j, const std::string& path, const char* label) {
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error(std::string(label) + ": failed to open config: " + path);
  }
  out << j.dump(4);
}

std::string write_json_temp(const json& j, const std::string& dir) {
  const std::string path = make_temp_json_path(dir);
  write_json_file(j, path, "Preproc");
  return path;
}

} // namespace

PreprocOptions::PreprocOptions(const sima::mpk::ModelMPK& model) {
  const std::string path = find_preproc_config_path(model);
  if (path.empty()) {
    throw std::runtime_error("PreprocOptions: failed to locate preproc config");
  }
  config_json = load_json_file(path, "PreprocOptions");
  next_cpu.clear();
}

PreprocOptions::PreprocOptions(const simaai::ModelSession& model)
    : PreprocOptions(model.model()) {}

namespace {

json build_preproc_json(const PreprocOptions& opt,
                        int in_w,
                        int in_h,
                        int out_w,
                        int out_h,
                        int scaled_w,
                        int scaled_h) {
  const std::string in_fmt = upper_copy(opt.input_img_type);
  const std::string out_fmt = upper_copy(opt.output_img_type);
  const std::vector<float> mean3 = ensure_three(opt.channel_mean, 0.0f);
  const std::vector<float> std3 = ensure_three(opt.channel_stddev, 1.0f);

  json j;
  j["graph_name"] = opt.graph_name;
  j["node_name"] = opt.node_name;
  j["cpu"] = opt.cpu;
  if (!opt.next_cpu.empty()) {
    j["next_cpu"] = opt.next_cpu;
    set_next_cpu_manual(j);
  }

  j["input_buffers"] = json::array({
      {
          {"name", opt.upstream_name},
          {"memories", json::array({
              {
                  {"segment_name", "parent"},
                  {"graph_input_name", opt.graph_input_name},
              },
          })},
      },
  });

  j["output_memory_order"] = output_memory_order_default(opt);
  j["debug"] = opt.debug;

  j["input_width"] = in_w;
  j["input_height"] = in_h;
  j["input_offset"] = opt.input_offset;

  j["output_width"] = out_w;
  j["output_height"] = out_h;
  j["scaled_width"] = scaled_w;
  j["scaled_height"] = scaled_h;

  j["batch_size"] = opt.batch_size;
  j["normalize"] = opt.normalize;
  j["aspect_ratio"] = opt.aspect_ratio;

  j["tile_width"] = opt.tile_width;
  j["tile_height"] = opt.tile_height;
  j["tile_channels"] = opt.tile_channels;
  j["tessellate"] = opt.tessellate;
  j["input_channels"] = opt.input_channels;
  j["output_channels"] = opt.output_channels;

  j["q_zp"] = opt.q_zp;
  j["q_scale"] = opt.q_scale;
  j["channel_mean"] = json::array({mean3[0], mean3[1], mean3[2]});
  j["channel_stddev"] = json::array({std3[0], std3[1], std3[2]});

  j["input_img_type"] = in_fmt;
  j["output_img_type"] = out_fmt;
  j["scaling_type"] = opt.scaling_type;
  j["padding_type"] = opt.padding_type;
  j["input_stride"] = opt.input_stride;
  j["output_stride"] = opt.output_stride;
  j["output_dtype"] = opt.output_dtype;

  j["caps"] = {
      {"sink_pads", json::array({
          {
              {"media_type", "video/x-raw"},
              {"params", json::array({
                  {
                      {"name", "format"},
                      {"type", "string"},
                      {"values", "GRAY, RGB, BGR, I420, NV12"},
                      {"json_field", "input_img_type"},
                  },
                  {
                      {"name", "width"},
                      {"type", "int"},
                      {"values", "1 - 4096"},
                      {"json_field", "input_width"},
                  },
                  {
                      {"name", "height"},
                      {"type", "int"},
                      {"values", "1 - 4096"},
                      {"json_field", "input_height"},
                  },
              })},
          },
      })},
      {"src_pads", json::array({
          {
              {"media_type", "video/x-raw"},
              {"params", json::array({
                  {
                      {"name", "format"},
                      {"type", "string"},
                      {"values", "RGB, BGR"},
                      {"json_field", "output_img_type"},
                  },
                  {
                      {"name", "width"},
                      {"type", "int"},
                      {"values", "1 - 4096"},
                      {"json_field", "output_width"},
                  },
                  {
                      {"name", "height"},
                      {"type", "int"},
                      {"values", "1 - 4096"},
                      {"json_field", "output_height"},
                  },
              })},
          },
      })},
  };

  return j;
}

} // namespace

Preproc::Preproc(PreprocOptions opt) : opt_(std::move(opt)) {
  const int in_w = (opt_.input_width > 0) ? opt_.input_width : opt_.output_width;
  const int in_h = (opt_.input_height > 0) ? opt_.input_height : opt_.output_height;
  const int out_w = (opt_.output_width > 0) ? opt_.output_width : in_w;
  const int out_h = (opt_.output_height > 0) ? opt_.output_height : in_h;
  const int scaled_w = (opt_.scaled_width > 0) ? opt_.scaled_width : out_w;
  const int scaled_h = (opt_.scaled_height > 0) ? opt_.scaled_height : out_h;

  if (in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0) {
    throw std::runtime_error("Preproc: invalid input/output dimensions");
  }

  if (opt_.input_channels <= 0) {
    opt_.input_channels = channels_from_format(opt_.input_img_type, 3);
  }
  if (opt_.output_channels <= 0) {
    opt_.output_channels = channels_from_format(opt_.output_img_type, 3);
  }

  auto holder = std::make_shared<PreprocConfigHolder>();
  if (opt_.config_json.has_value()) {
    holder->config = *opt_.config_json;
    holder->has_config = true;
    if (!opt_.next_cpu.empty()) {
      holder->config["next_cpu"] = opt_.next_cpu;
      set_next_cpu_manual(holder->config);
    }
    if (!opt_.config_path.empty()) {
      config_path_ = opt_.config_path;
      write_json_file(holder->config, config_path_, "Preproc");
      holder->keep = true;
    } else {
      config_path_ = write_json_temp(holder->config, opt_.config_dir);
      holder->keep = opt_.keep_config;
    }
    holder->path = config_path_;
    config_holder_ = std::move(holder);
    return;
  }

  if (!opt_.config_path.empty()) {
    config_path_ = opt_.config_path;
    holder->config = load_json_file(config_path_, "Preproc");
    holder->has_config = true;
    holder->path = config_path_;
    holder->keep = true;
    config_holder_ = std::move(holder);
    return;
  }

  holder->config = build_preproc_json(opt_, in_w, in_h, out_w, out_h, scaled_w, scaled_h);
  holder->has_config = true;
  config_path_ = write_json_temp(holder->config, opt_.config_dir);
  holder->path = config_path_;
  holder->keep = opt_.keep_config;
  config_holder_ = std::move(holder);
}

std::string Preproc::gst_fragment(int node_index) const {
  std::ostringstream ss;
  ss << "simaaiprocesscvu name=n" << node_index << "_preproc";
  if (!config_path_.empty()) {
    ss << " config=\"" << config_path_ << "\"";
  }
  if (opt_.num_buffers > 0) {
    ss << " num-buffers=" << opt_.num_buffers;
  }
  return ss.str();
}

std::vector<std::string> Preproc::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_preproc"};
}

OutputSpec Preproc::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "video/x-raw";
  out.format = upper_copy(opt_.output_img_type);
  out.width = (opt_.output_width > 0) ? opt_.output_width : opt_.input_width;
  out.height = (opt_.output_height > 0) ? opt_.output_height : opt_.input_height;
  out.depth = (opt_.output_channels > 0)
      ? opt_.output_channels
      : channels_from_format(out.format, 3);
  out.layout = (out.format == "GRAY" || out.format == "GRAY8") ? "HW" : "HWC";
  out.dtype = "UInt8";
  if (upper_copy(opt_.next_cpu) == "APU") {
    out.memory = "SystemMemory";
  } else if (!input.memory.empty()) {
    out.memory = input.memory;
  } else {
    out.memory = "SimaAI";
  }
  out.certainty = SpecCertainty::Derived;
  out.note = "simaaiprocesscvu";
  out.byte_size = expected_byte_size(out);
  return out;
}

bool Preproc::set_next_cpu_if_auto(const std::string& next_cpu) {
  if (!config_holder_ || !config_holder_->has_config) return false;
  if (next_cpu.empty()) return false;
  json& j = config_holder_->config;
  if (has_next_cpu_manual(j)) return false;
  j["next_cpu"] = next_cpu;
  opt_.next_cpu = next_cpu;
  if (!config_path_.empty()) {
    write_json_file(j, config_path_, "Preproc");
  }
  return true;
}

const nlohmann::json* Preproc::config_json() const {
  if (!config_holder_ || !config_holder_->has_config) return nullptr;
  return &config_holder_->config;
}

bool Preproc::override_config_json(const std::function<void(json&)>& edit,
                                   const std::string& tag) {
  (void)tag;
  if (!config_holder_ || !config_holder_->has_config) return false;
  json cfg = config_holder_->config;
  edit(cfg);
  config_path_ = write_json_temp(cfg, opt_.config_dir);
  auto holder = std::make_shared<PreprocConfigHolder>();
  holder->config = std::move(cfg);
  holder->has_config = true;
  holder->path = config_path_;
  holder->keep = false;
  config_holder_ = std::move(holder);
  return true;
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> Preproc(PreprocOptions opt) {
  return std::make_shared<sima::Preproc>(std::move(opt));
}

} // namespace sima::nodes
