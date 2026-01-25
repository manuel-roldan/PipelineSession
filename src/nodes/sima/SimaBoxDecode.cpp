#include "nodes/sima/SimaBoxDecode.h"

#include "mpk/ModelMPK.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace sima {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct BoxDecodeConfigHolder {
  std::string path;
  bool keep = false;
  json config;
  bool has_config = false;
  ~BoxDecodeConfigHolder() {
    if (!keep && !path.empty()) {
      std::remove(path.c_str());
    }
  }
};

struct BoxDecodeOptionsInternal {
  std::string config_path;
  int sima_allocator_type = 2;
  bool silent = true;
  bool emit_signals = false;
  bool transmit = false;
  std::string config_dir;
  bool keep_config = false;
  std::optional<json> config_json;
  std::string decode_type;
  int top_k = 0;
  int num_classes = 0;
  double detection_threshold = 0.0;
  double nms_iou_threshold = 0.0;
  int original_width = 0;
  int original_height = 0;
  int memory_cpu = -1;
  int memory_next_cpu = -1;
};

namespace {

std::string make_temp_json_path(const std::string& dir) {
  std::string root = dir.empty() ? "/tmp" : dir;
  std::error_code ec;
  fs::create_directories(root, ec);
  if (ec) {
    throw std::runtime_error("SimaBoxDecode: failed to create config dir: " + root);
  }

  std::string templ = (fs::path(root) / "sima_boxdecode_XXXXXX.json").string();
  std::vector<char> buf(templ.begin(), templ.end());
  buf.push_back('\0');

  int fd = mkstemps(buf.data(), 5);
  if (fd < 0) {
    throw std::runtime_error("SimaBoxDecode: mkstemps failed");
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
  out << j.dump(2);
}

std::string write_json_temp(const json& j, const std::string& dir) {
  const std::string path = make_temp_json_path(dir);
  write_json_file(j, path, "SimaBoxDecode");
  return path;
}

std::string lower_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string find_boxdecode_config_path(const sima::mpk::ModelMPK& model) {
  std::string path = model.find_config_path_by_plugin("boxdecode");
  if (path.empty()) path = model.find_config_path_by_plugin("box_decode");
  return path;
}

std::string read_node_name(const json& j) {
  if (j.contains("node_name") && j["node_name"].is_string()) {
    return j["node_name"].get<std::string>();
  }
  return "";
}

bool copy_array_if_present(json& dst,
                           const json& src,
                           const char* dst_key,
                           const char* src_key) {
  if (!src.contains(src_key) || !src[src_key].is_array()) return false;
  dst[dst_key] = src[src_key];
  return true;
}

bool copy_number_if_present(json& dst,
                            const json& src,
                            const char* dst_key,
                            const char* src_key) {
  if (!src.contains(src_key) || !src[src_key].is_number()) return false;
  dst[dst_key] = src[src_key];
  return true;
}

void apply_overrides(json& j, const BoxDecodeOptionsInternal& opt) {
  if (!opt.decode_type.empty()) j["decode_type"] = opt.decode_type;
  if (opt.top_k > 0) {
    j["topk"] = opt.top_k;
    if (!j.contains("buffers") || !j["buffers"].is_object()) {
      j["buffers"] = json::object();
    }
    if (!j["buffers"].contains("output") || !j["buffers"]["output"].is_object()) {
      j["buffers"]["output"] = json::object();
    }
    const std::size_t out_size =
        sizeof(std::uint32_t) + static_cast<std::size_t>(opt.top_k) * 24;
    j["buffers"]["output"]["size"] = out_size;
  }
  if (opt.num_classes > 0) j["num_classes"] = opt.num_classes;
  if (opt.detection_threshold > 0.0) j["detection_threshold"] = opt.detection_threshold;
  if (opt.nms_iou_threshold > 0.0) j["nms_iou_threshold"] = opt.nms_iou_threshold;
  if (opt.original_width > 0) j["original_width"] = opt.original_width;
  if (opt.original_height > 0) j["original_height"] = opt.original_height;
  if (opt.memory_cpu >= 0 || opt.memory_next_cpu >= 0) {
    if (!j.contains("memory") || !j["memory"].is_object()) {
      j["memory"] = json::object();
    }
    if (opt.memory_cpu >= 0) j["memory"]["cpu"] = opt.memory_cpu;
    if (opt.memory_next_cpu >= 0) j["memory"]["next_cpu"] = opt.memory_next_cpu;
  }
}

void sanitize_boxdecode_config(json& j) {
  if (j.contains("debug") && j["debug"].is_string()) {
    j["debug"] = false;
  }
  if (!j.contains("system") || !j["system"].is_object()) {
    j["system"] = json::object();
  }
  if (j["system"].contains("debug") && j["system"]["debug"].is_string()) {
    j["system"]["debug"] = 0;
  } else if (!j["system"].contains("debug")) {
    j["system"]["debug"] = 0;
  }
  if (j["system"].contains("dump_data") && j["system"]["dump_data"].is_string()) {
    j["system"]["dump_data"] = 0;
  }
}

} // namespace

static BoxDecodeOptionsInternal options_from_model(const sima::mpk::ModelMPK& model) {
  BoxDecodeOptionsInternal opt;
  const std::string path = find_boxdecode_config_path(model);
  if (path.empty()) {
    throw std::runtime_error("SimaBoxDecode: failed to locate boxdecode config");
  }
  opt.config_json = load_json_file(path, "SimaBoxDecode");
  if (!opt.config_json->contains("decode_type") ||
      !(*opt.config_json)["decode_type"].is_string()) {
    const std::string guess = lower_copy(model.etc_dir());
    if (guess.find("yolov8") != std::string::npos ||
        guess.find("yolo_v8") != std::string::npos) {
      opt.decode_type = "yolov8";
    } else if (guess.find("yolov9") != std::string::npos ||
               guess.find("yolo_v9") != std::string::npos) {
      opt.decode_type = "yolov9";
    }
  }
  return opt;
}

static std::shared_ptr<BoxDecodeConfigHolder> init_config_holder(
    const BoxDecodeOptionsInternal& opt,
    std::string& config_path_out) {
  auto holder = std::make_shared<BoxDecodeConfigHolder>();
  if (opt.config_json.has_value()) {
    holder->config = *opt.config_json;
    holder->has_config = true;
  } else if (!opt.config_path.empty()) {
    config_path_out = opt.config_path;
    holder->config = load_json_file(config_path_out, "SimaBoxDecode");
    holder->has_config = true;
    holder->keep = true;
  }

  if (!holder->has_config) {
    return holder;
  }

  apply_overrides(holder->config, opt);
  sanitize_boxdecode_config(holder->config);
  if (!opt.config_path.empty()) {
    config_path_out = opt.config_path;
    write_json_file(holder->config, config_path_out, "SimaBoxDecode");
    holder->keep = true;
  } else {
    config_path_out = write_json_temp(holder->config, opt.config_dir);
    holder->keep = opt.keep_config;
  }
  holder->path = config_path_out;
  return holder;
}

SimaBoxDecode::SimaBoxDecode(const sima::mpk::ModelMPK& model,
                             const std::string& decode_type,
                             int original_width,
                             int original_height,
                             double detection_threshold,
                             double nms_iou_threshold,
                             int top_k) {
  auto opt = std::make_unique<BoxDecodeOptionsInternal>(options_from_model(model));
  if (!decode_type.empty()) opt->decode_type = decode_type;
  if (original_width > 0) opt->original_width = original_width;
  if (original_height > 0) opt->original_height = original_height;
  if (detection_threshold > 0.0) opt->detection_threshold = detection_threshold;
  if (nms_iou_threshold > 0.0) opt->nms_iou_threshold = nms_iou_threshold;
  if (top_k > 0) opt->top_k = top_k;

  config_holder_ = init_config_holder(*opt, config_path_);
  opt_ = std::move(opt);
}

std::string SimaBoxDecode::gst_fragment(int node_index) const {
  std::ostringstream ss;
  ss << "simaaiboxdecode name=n" << node_index << "_boxdecode";
  const std::string& cfg = config_path_.empty() ? opt_->config_path : config_path_;
  if (!cfg.empty()) {
    ss << " config=\"" << cfg << "\"";
  }
  ss << " silent=" << (opt_->silent ? "true" : "false");
  ss << " emit-signals=" << (opt_->emit_signals ? "true" : "false");
  if (opt_->sima_allocator_type > 0) {
    ss << " sima-allocator-type=" << opt_->sima_allocator_type;
  }
  ss << " transmit=" << (opt_->transmit ? "true" : "false");
  return ss.str();
}

std::vector<std::string> SimaBoxDecode::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_boxdecode"};
}

OutputSpec SimaBoxDecode::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "BBOX";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "simaaiboxdecode";
  return out;
}

const nlohmann::json* SimaBoxDecode::config_json() const {
  if (!config_holder_ || !config_holder_->has_config) return nullptr;
  return &config_holder_->config;
}

bool SimaBoxDecode::override_config_json(const std::function<void(json&)>& edit,
                                         const std::string& tag) {
  (void)tag;
  if (!config_holder_ || !config_holder_->has_config) return false;
  json cfg = config_holder_->config;
  edit(cfg);
  config_path_ = write_json_temp(cfg, opt_->config_dir);
  auto holder = std::make_shared<BoxDecodeConfigHolder>();
  holder->config = std::move(cfg);
  holder->has_config = true;
  holder->path = config_path_;
  holder->keep = false;
  config_holder_ = std::move(holder);
  return true;
}

void SimaBoxDecode::apply_upstream_config(const nlohmann::json& upstream,
                                          const std::string&) {
  if (!config_holder_ || !config_holder_->has_config) return;
  json& j = config_holder_->config;
  copy_array_if_present(j, upstream, "input_width", "output_width");
  copy_array_if_present(j, upstream, "input_height", "output_height");
  copy_array_if_present(j, upstream, "input_depth", "output_depth");
  copy_array_if_present(j, upstream, "slice_width", "slice_width");
  copy_array_if_present(j, upstream, "slice_height", "slice_height");
  copy_array_if_present(j, upstream, "slice_depth", "slice_depth");
  copy_array_if_present(j, upstream, "data_type", "data_type");
  copy_number_if_present(j, upstream, "model_width", "model_width");
  copy_number_if_present(j, upstream, "model_height", "model_height");

  std::string node_name = read_node_name(upstream);
  if (node_name.empty() && upstream.contains("output_width")) {
    node_name = "mla";
  }
  if (!node_name.empty()) {
    if (!j.contains("buffers") || !j["buffers"].is_object()) {
      j["buffers"] = json::object();
    }
    if (!j["buffers"].contains("input") || !j["buffers"]["input"].is_array()) {
      j["buffers"]["input"] = json::array();
    }
    if (j["buffers"]["input"].empty()) {
      j["buffers"]["input"].push_back(json::object());
    }
    j["buffers"]["input"][0]["name"] = "simaai_process_" + node_name;
  }

  if (j.contains("input_width") && j["input_width"].is_array()) {
    j["num_in_tensor"] = static_cast<int>(j["input_width"].size());
  }

  apply_overrides(j, *opt_);
  sanitize_boxdecode_config(j);

  if (config_path_.empty()) {
    config_path_ = write_json_temp(j, opt_->config_dir);
    config_holder_->path = config_path_;
    config_holder_->keep = opt_->keep_config;
  } else {
    write_json_file(j, config_path_, "SimaBoxDecode");
  }
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> SimaBoxDecode(const sima::mpk::ModelMPK& model,
                                          const std::string& decode_type,
                                          int original_width,
                                          int original_height,
                                          double detection_threshold,
                                          double nms_iou_threshold,
                                          int top_k) {
  return std::make_shared<sima::SimaBoxDecode>(model,
                                               decode_type,
                                               original_width,
                                               original_height,
                                               detection_threshold,
                                               nms_iou_threshold,
                                               top_k);
}

} // namespace sima::nodes
