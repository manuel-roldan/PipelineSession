#include "mpk/ModelMPK.h"

#include <nlohmann/json.hpp>

#include <array>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sima::mpk {
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr const char* kDefaultBaseOutputDir = "/data/simaai/coprocessing/models/";
constexpr const char* kDirConf = "etc";
constexpr const char* kDirLib = "lib";
constexpr const char* kDirShare = "share";

constexpr const char* kDefaultPreviousNodeName = "decoder";

constexpr int kQueueMaxSizeBuffers = 2;
constexpr int kQueueMaxSizeBytes = 0;
constexpr int kQueueMaxSizeTime = 10000000;
constexpr const char* kQueueLeakyMode = "downstream";
constexpr bool kQueueSilent = true;

static std::string to_upper(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

static std::array<float, 3> materialize3(const std::vector<float>& v, float defv) {
  if (v.empty()) return {defv, defv, defv};
  if (v.size() == 1) return {v[0], v[0], v[0]};
  if (v.size() == 3) return {v[0], v[1], v[2]};
  throw std::invalid_argument("mean/stddev must have 0, 1, or 3 values.");
}

static std::string append_model_paths_if_exists(json& json_data, const std::string& append_path) {
  if (json_data.contains("simaai__params") && json_data["simaai__params"].contains("model_path")) {
    std::string model_path = json_data["simaai__params"]["model_path"];
    json_data["simaai__params"]["model_path"] = append_path + "/share/" + model_path;
  } else if (json_data.contains("model_info") && json_data["model_info"].contains("path")) {
    std::string model_info_path = json_data["model_info"]["path"];
    json_data["model_info"]["path"] = append_path + "/lib/" + model_info_path;
  }
  return "";
}

static bool is_valid_tarball(const std::string& tar_path) noexcept {
  if (!fs::exists(tar_path) || tar_path.size() < 7 ||
      tar_path.substr(tar_path.size() - 7) != ".tar.gz") {
    return false;
  }

  std::string cmd = "tar -tzf \"" + tar_path + "\"";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return false;

  char buffer[512];
  bool found_elf_or_so = false;
  while (fgets(buffer, sizeof(buffer), pipe)) {
    std::string line(buffer);
    if (line.find(".elf") != std::string::npos || line.find(".so") != std::string::npos) {
      found_elf_or_so = true;
      break;
    }
  }
  pclose(pipe);
  return found_elf_or_so;
}

static std::string extract_and_organize(const std::string& tar_path) {
  if (!is_valid_tarball(tar_path)) {
    throw std::runtime_error("ModelMPK: invalid tar.gz: " + tar_path);
  }

  std::string base_name = fs::path(tar_path).filename().string();
  base_name = base_name.substr(0, base_name.find(".tar.gz"));

  fs::path target_dir = fs::path(kDefaultBaseOutputDir) / base_name;
  fs::create_directories(target_dir / kDirConf);
  fs::create_directories(target_dir / kDirLib);
  fs::create_directories(target_dir / kDirShare);

  fs::path temp_extract_dir = target_dir / "temp_extract";
  fs::create_directories(temp_extract_dir);
  std::string extract_cmd =
      "tar -xzf \"" + tar_path + "\" -C \"" + temp_extract_dir.string() + "\"";
  if (std::system(extract_cmd.c_str()) != 0) {
    throw std::runtime_error("ModelMPK: tar extraction failed: " + tar_path);
  }

  for (const auto& entry : fs::recursive_directory_iterator(temp_extract_dir)) {
    if (!entry.is_regular_file()) continue;
    std::string ext = entry.path().extension().string();
    if (ext == ".json") {
      std::ifstream json_file(entry.path());
      json json_data;
      json_file >> json_data;
      append_model_paths_if_exists(json_data, target_dir.string());
      std::ofstream updated_json_file(target_dir / kDirConf / entry.path().filename());
      updated_json_file << json_data.dump(4);
    } else if (ext == ".so") {
      fs::copy(entry.path(), target_dir / kDirLib / entry.path().filename(),
               fs::copy_options::overwrite_existing);
    } else if (ext == ".elf") {
      fs::copy(entry.path(), target_dir / kDirShare / entry.path().filename(),
               fs::copy_options::overwrite_existing);
    }
  }

  fs::remove_all(temp_extract_dir);
  return target_dir.string();
}

static std::string update_input_buffers_name(const std::string& file_path,
                                             const std::string& previous_node_name) {
  std::ifstream json_file(file_path);
  if (!json_file.is_open()) {
    return "Failed to open the JSON file.";
  }

  json json_data;
  json_file >> json_data;

  if (json_data.contains("input_buffers") && json_data["input_buffers"][0].contains("name")) {
    json_data["input_buffers"][0]["name"] = previous_node_name;
    std::ofstream updated_json_file(file_path);
    if (!updated_json_file.is_open()) {
      return "Failed to open the file for writing.";
    }
    updated_json_file << json_data.dump(4);
    return "";
  }

  return "input_buffers->name not found.";
}

static std::tuple<int, int, std::string> read_and_update_preproc_json(
    const std::string& etc_dir,
    int inWidth,
    int inHeight,
    const std::string& inFormat,
    bool normalize,
    const std::vector<float>& channel_mean,
    const std::vector<float>& channel_stddev) {
  std::string json_path = (fs::path(etc_dir) / "0_preproc.json").string();
  std::ifstream json_file(json_path);
  if (!json_file.is_open()) {
    throw std::runtime_error("ModelMPK: failed to open " + json_path);
  }

  json json_data;
  json_file >> json_data;

  int out_w = 0;
  int out_h = 0;
  std::string out_img_type;
  if (inWidth > 0) {
    json_data["input_width"] = inWidth;
  } else if (json_data.contains("output_width")) {
    out_w = json_data["output_width"].get<int>();
    json_data["input_width"] = out_w;
  }
  if (inHeight > 0) {
    json_data["input_height"] = inHeight;
  } else if (json_data.contains("output_height")) {
    out_h = json_data["output_height"].get<int>();
    json_data["input_height"] = out_h;
  }
  if (!inFormat.empty()) {
    json_data["input_img_type"] = inFormat;
  } else if (json_data.contains("output_img_type")) {
    out_img_type = json_data["output_img_type"].get<std::string>();
    json_data["input_img_type"] = out_img_type;
  }

  if (normalize) {
    json_data["normalize"] = normalize;
    json_data["channel_mean"] = channel_mean;
    json_data["channel_stddev"] = channel_stddev;
  }

  std::ofstream updated_json_file(json_path);
  if (!updated_json_file.is_open()) {
    throw std::runtime_error("ModelMPK: failed to write " + json_path);
  }
  updated_json_file << json_data.dump(4);

  return std::make_tuple(out_w, out_h, out_img_type);
}

static void patch_quanttess_caps_to_tensor_sink(const std::string& etc_dir) {
  const std::string qp = (fs::path(etc_dir) / "0_quanttess.json").string();
  std::ifstream in(qp);
  if (!in.is_open()) throw std::runtime_error("patch: cannot open " + qp);
  json j;
  in >> j;
  in.close();

  j["caps"]["sink_pads"] = json::array({
      {
          {"media_type", "application/vnd.simaai.tensor"},
          {"params", json::array({
              json{{"name", "format"}, {"type", "string"},
                   {"values", json::array({"FP32"})}},
              json{{"json_field", "input_width"}, {"name", "width"},
                   {"type", "int"}, {"values", "1 - 4096"}},
              json{{"json_field", "input_height"}, {"name", "height"},
                   {"type", "int"}, {"values", "1 - 4096"}},
              json{{"json_field", "input_depth"}, {"name", "depth"},
                   {"type", "int"}, {"values", "1 - 4096"}},
          })},
      },
  });

  j["tile_depth"] = std::max(1, j.value("tile_depth", 3));

  std::ofstream out(qp);
  out << j.dump(4);
}

static std::tuple<bool, int, int, int> rewrite_preproc_to_quanttess(
    const std::string& etc_dir) {
  const fs::path seq_path = fs::path(etc_dir) / "pipeline_sequence.json";
  std::ifstream in(seq_path);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open " + seq_path.string());
  }

  json j;
  in >> j;
  in.close();

  if (!j.contains("pipelines") || !j["pipelines"].is_array() || j["pipelines"].empty()) {
    throw std::runtime_error("pipeline_sequence.json: missing/empty 'pipelines'");
  }

  json& pipe = j["pipelines"][0];
  if (!pipe.contains("sequence") || !pipe["sequence"].is_array()) {
    throw std::runtime_error("pipeline_sequence.json: missing 'sequence'");
  }

  auto& seq = pipe["sequence"];

  int preproc_idx = -1;
  std::string old_name;
  int old_seq_id = 0;
  for (int i = 0; i < (int)seq.size(); ++i) {
    auto& s = seq[i];
    if (s.value("kernel", "") == "preproc") {
      preproc_idx = i;
      old_name = s.value("name", "simaaiprocesspreproc_1");
      old_seq_id = s.value("sequence_id", i + 1);
      break;
    }
  }
  if (preproc_idx < 0) {
    throw std::runtime_error("Could not find preproc on first segment");
  }

  const fs::path quanttess_path = fs::path(etc_dir) / "0_quanttess.json";
  if (!fs::exists(quanttess_path)) {
    throw std::runtime_error("INT8 path requested but 0_quanttess.json not found at " +
                             quanttess_path.string());
  }

  json qj;
  {
    std::ifstream qin(quanttess_path);
    if (!qin.is_open()) throw std::runtime_error("Cannot open " + quanttess_path.string());
    qj = json::parse(qin, nullptr, /*allow_exceptions=*/true);
  }

  if (qj.contains("input_buffers") && qj["input_buffers"].is_array() &&
      !qj["input_buffers"].empty()) {
    qj["input_buffers"][0]["name"] = kDefaultPreviousNodeName;
  }

  {
    std::ofstream qout(quanttess_path);
    qout << qj.dump(4);
  }

  const int out_w = qj.value("input_width", 0);
  const int out_h = qj.value("input_height", 0);
  const int out_d = qj.value("input_depth", 0);
  if (out_w <= 0 || out_h <= 0 || out_d <= 0) {
    throw std::runtime_error("Invalid input dims in 0_quanttess.json");
  }

  json new_stage = {
      {"configPath", "0_quanttess.json"},
      {"executable", nullptr},
      {"input", kDefaultPreviousNodeName},
      {"kernel", "quanttess"},
      {"name", "simaaiprocessquanttess_1"},
      {"params", json::object()},
      {"pluginId", "processcvu"},
      {"processor", "CVU"},
      {"sequence_id", old_seq_id},
  };

  seq[preproc_idx] = new_stage;

  const std::string new_name = "simaaiprocessquanttess_1";
  for (auto& s : seq) {
    if (!s.contains("input")) continue;
    auto& inp = s["input"];
    if (inp.is_string()) {
      if (inp.get<std::string>() == old_name) inp = new_name;
    } else if (inp.is_array()) {
      for (auto& e : inp) {
        if (e.is_string() && e.get<std::string>() == old_name) e = new_name;
      }
    }
  }

  {
    std::ofstream out(seq_path);
    out << j.dump(4);
  }
  return {true, out_w, out_h, out_d};
}

static PipelineType get_pipeline_type(const std::string& folder_path,
                                      const std::string& input_format) {
  if (input_format.empty()) {
    const fs::path seq_path = fs::path(folder_path) / "0_process_mla.json";
    std::ifstream in(seq_path);
    if (!in.is_open()) throw std::runtime_error("Cannot open " + seq_path.string());

    json j;
    in >> j;
    in.close();

    if (!j.contains("data_type") || !j["data_type"].is_array() || j["data_type"].empty() ||
        !j["data_type"][0].is_string()) {
      throw std::runtime_error(
          "0_process_mla.json: 'data_type' must be an array with a string element");
    }

    std::string data_type = j["data_type"][0].get<std::string>();
    if (data_type == "EVXX_BFLOAT16") return PipelineType::CastTess;
    if (data_type == "INT8") return PipelineType::QuantTess;
    throw std::runtime_error("Unknown MLA data type '" + data_type + "'");
  }

  return PipelineType::Preproc;
}

struct SeqEntry {
  int sequence_id = 0;
  std::string name;
  std::string plugin_id;
  std::string config_path;
  std::string processor;
  std::string kernel;
};

static std::vector<SeqEntry> load_sequence(const std::string& etc_dir) {
  const std::string pipelineJsonPath = (fs::path(etc_dir) / "pipeline_sequence.json").string();
  std::ifstream inFile(pipelineJsonPath);
  if (!inFile.is_open()) {
    throw std::runtime_error("ModelMPK: Error opening JSON file: " + pipelineJsonPath);
  }

  json pipelineJson;
  inFile >> pipelineJson;

  if (!pipelineJson.contains("pipelines") || !pipelineJson["pipelines"].is_array() ||
      pipelineJson["pipelines"].empty()) {
    throw std::runtime_error("ModelMPK: Invalid pipeline format in JSON");
  }

  const auto& pipeline = pipelineJson["pipelines"][0];
  const auto& sequence = pipeline["sequence"];
  std::vector<SeqEntry> ordered;
  ordered.reserve(sequence.size());
  for (const auto& elem : sequence) {
    SeqEntry e;
    e.sequence_id = elem.value("sequence_id", 0);
    e.name = elem.value("name", "");
    e.plugin_id = elem.value("pluginId", "");
    e.config_path = elem.value("configPath", "");
    e.processor = elem.value("processor", "");
    e.kernel = elem.value("kernel", "");
    ordered.push_back(std::move(e));
  }

  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const SeqEntry& a, const SeqEntry& b) {
                     return a.sequence_id < b.sequence_id;
                   });
  return ordered;
}

static std::pair<int, int> mla_range(const std::vector<SeqEntry>& seq) {
  int first = -1;
  int last = -1;
  for (size_t i = 0; i < seq.size(); ++i) {
    if (seq[i].processor == "MLA") {
      if (first < 0) first = static_cast<int>(i);
      last = static_cast<int>(i);
    }
  }
  return {first, last};
}

static std::vector<SeqEntry> select_stage(const std::vector<SeqEntry>& seq, ModelStage stage) {
  if (stage == ModelStage::Full) return seq;
  auto [first_mla, last_mla] = mla_range(seq);

  std::vector<SeqEntry> out;
  if (stage == ModelStage::MlaOnly) {
    for (const auto& e : seq) {
      if (e.processor == "MLA") out.push_back(e);
    }
    return out;
  }

  if (first_mla < 0 || last_mla < 0) {
    return out;
  }

  if (stage == ModelStage::Preprocess) {
    for (int i = 0; i < first_mla; ++i) out.push_back(seq[static_cast<size_t>(i)]);
    return out;
  }

  if (stage == ModelStage::Postprocess) {
    for (int i = last_mla + 1; i < static_cast<int>(seq.size()); ++i) {
      out.push_back(seq[static_cast<size_t>(i)]);
    }
  }
  return out;
}

static const char* stage_label(ModelStage stage) {
  switch (stage) {
    case ModelStage::Preprocess: return "preprocess";
    case ModelStage::MlaOnly: return "mla_only";
    case ModelStage::Postprocess: return "postprocess";
    case ModelStage::Full: return "full";
  }
  return "full";
}

static std::string upstream_name_for_stage(const std::vector<SeqEntry>& seq, ModelStage stage) {
  if (seq.empty()) return kDefaultPreviousNodeName;
  auto [first_mla, last_mla] = mla_range(seq);
  if (stage == ModelStage::MlaOnly) {
    if (first_mla > 0 && first_mla <= static_cast<int>(seq.size())) {
      const std::string& name = seq[static_cast<size_t>(first_mla - 1)].name;
      if (!name.empty()) return name;
    }
    return kDefaultPreviousNodeName;
  }
  if (stage == ModelStage::Postprocess) {
    if (last_mla >= 0 && last_mla < static_cast<int>(seq.size())) {
      const std::string& name = seq[static_cast<size_t>(last_mla)].name;
      if (!name.empty()) return name;
    }
    return kDefaultPreviousNodeName;
  }
  return kDefaultPreviousNodeName;
}

static ModelFragment build_fragment_linear(const std::string& etc_dir,
                                           const std::vector<SeqEntry>& seq,
                                           const std::string& initial_input_name,
                                           const std::string& queue_prefix) {
  ModelFragment frag;
  if (seq.empty()) return frag;

  std::ostringstream pipelineStr;
  std::string previous_node_name =
      initial_input_name.empty() ? kDefaultPreviousNodeName : initial_input_name;

  const std::string qprefix = queue_prefix.empty() ? "q" : queue_prefix;
  int qnum = 0;
  auto add_queue = [&](std::ostringstream& ss) {
    ss << "! queue name=" << qprefix << (++qnum)
       << " max-size-buffers=" << kQueueMaxSizeBuffers
       << " max-size-bytes=" << kQueueMaxSizeBytes
       << " max-size-time=" << kQueueMaxSizeTime
       << " leaky=" << kQueueLeakyMode
       << " silent=" << (kQueueSilent ? "true" : "false") << " ";
    frag.elements.push_back(qprefix + std::to_string(qnum));
  };

  for (size_t i = 0; i < seq.size(); ++i) {
    const auto& elem = seq[i];
    const std::string plugin = "simaai" + elem.plugin_id;
    const std::string name = elem.name;
    const std::string config = (fs::path(etc_dir) / elem.config_path).string();

    update_input_buffers_name(config, previous_node_name);

    if (i) pipelineStr << "! ";
    pipelineStr << plugin
                << " name=" << name
                << " config=" << config << " ";
    if (plugin == "simaaiprocessmla") {
      pipelineStr << "multi-pipeline=true ";
    }

    frag.elements.push_back(name);
    previous_node_name = name;
  }

  add_queue(pipelineStr);
  frag.gst = pipelineStr.str();
  return frag;
}

} // namespace

ModelMPK ModelMPK::load(const std::string& tar_gz, const ModelMPKOptions& opt) {
  ModelMPK out;
  out.options_ = opt;

  std::string extracted = extract_and_organize(tar_gz);
  out.etc_dir_ = (fs::path(extracted) / kDirConf).string();

  std::string fmt = to_upper(out.options_.input_format);
  if (fmt == "I420") fmt = "IYUV";
  out.options_.input_format = fmt;

  out.pipeline_type_ = get_pipeline_type(out.etc_dir_, out.options_.input_format);
  if (out.pipeline_type_ == PipelineType::CastTess) {
    throw std::runtime_error("ModelMPK: BF16/CastTess is not supported yet");
  }

  int out_w = 0;
  int out_h = 0;
  int out_c = 0;
  std::string out_fmt;

  if (out.pipeline_type_ == PipelineType::QuantTess) {
    std::tie(std::ignore, out_w, out_h, out_c) = rewrite_preproc_to_quanttess(out.etc_dir_);
    patch_quanttess_caps_to_tensor_sink(out.etc_dir_);
  } else if (out.pipeline_type_ == PipelineType::Preproc) {
    std::array<float, 3> mean3 = materialize3(out.options_.mean, 0.0f);
    std::array<float, 3> std3 = materialize3(out.options_.stddev, 1.0f);
    auto [pw, ph, pformat] =
        read_and_update_preproc_json(out.etc_dir_,
                                     out.options_.input_width,
                                     out.options_.input_height,
                                     out.options_.input_format,
                                     out.options_.normalize,
                                     out.options_.normalize
                                         ? std::vector<float>{mean3[0], mean3[1], mean3[2]}
                                         : std::vector<float>{},
                                     out.options_.normalize
                                         ? std::vector<float>{std3[0], std3[1], std3[2]}
                                         : std::vector<float>{});
    out_w = pw;
    out_h = ph;
    out_fmt = to_upper(pformat);
    if (!out_fmt.empty()) {
      out_c = (out_fmt == "GRAY") ? 1 : 3;
    }
  }

  if (out_w > 0) out.options_.input_width = out_w;
  if (out_h > 0) out.options_.input_height = out_h;
  if (!out_fmt.empty()) out.options_.input_format = out_fmt;
  if (out_c > 0) out.options_.input_depth = out_c;
  if (out_c > 0 && out.options_.input_format.empty()) {
    out.options_.input_format = (out_c == 1) ? "GRAY" : "RGB";
  }
  if (out.options_.input_depth == 0 && !out.options_.input_format.empty()) {
    out.options_.input_depth = (out.options_.input_format == "GRAY") ? 1 : 3;
  }

  return out;
}

ModelFragment ModelMPK::fragment(ModelStage stage) const {
  std::vector<SeqEntry> seq = load_sequence(etc_dir_);
  std::vector<SeqEntry> sel = select_stage(seq, stage);
  if (sel.empty()) return {};

  std::string upstream = options_.upstream_name.empty()
      ? upstream_name_for_stage(seq, stage)
      : options_.upstream_name;
  const std::string queue_prefix = std::string("q_") + stage_label(stage) + "_";
  return build_fragment_linear(etc_dir_, sel, upstream, queue_prefix);
}

std::string ModelMPK::gst_fragment(ModelStage stage) const {
  return fragment(stage).gst;
}

NodeGroup ModelMPK::to_node_group(ModelStage stage) const {
  ModelFragment frag = fragment(stage);
  if (frag.gst.empty()) return NodeGroup{};

  class ModelFragmentNode final : public Node {
  public:
    ModelFragmentNode(std::string kind,
                      std::string label,
                      std::string fragment,
                      std::vector<std::string> elements)
        : kind_(std::move(kind)),
          label_(std::move(label)),
          fragment_(std::move(fragment)),
          elements_(std::move(elements)) {}

    std::string kind() const override { return kind_; }
    std::string user_label() const override { return label_; }
    std::string gst_fragment(int) const override { return fragment_; }
    std::vector<std::string> element_names(int) const override { return elements_; }

  private:
    std::string kind_;
    std::string label_;
    std::string fragment_;
    std::vector<std::string> elements_;
  };

  std::string label = stage_label(stage);

  std::vector<std::shared_ptr<Node>> nodes;
  nodes.push_back(std::make_shared<ModelFragmentNode>(
      "ModelFragment", label, frag.gst, frag.elements));
  return NodeGroup(std::move(nodes));
}

InputAppSrcOptions ModelMPK::input_appsrc_options(bool tensor_mode) const {
  InputAppSrcOptions opt;

  if (tensor_mode) {
    if (options_.input_width <= 0 || options_.input_height <= 0 || options_.input_depth <= 0) {
      throw std::runtime_error("ModelMPK: missing input dims for tensor mode");
    }
    opt.media_type = "application/vnd.simaai.tensor";
    opt.format = "FP32";
    opt.width = options_.input_width;
    opt.height = options_.input_height;
    opt.depth = options_.input_depth;
    return opt;
  }

  if (options_.input_width <= 0 || options_.input_height <= 0) {
    throw std::runtime_error("ModelMPK: missing input dims for image mode");
  }

  std::string fmt = options_.input_format;
  if (fmt.empty()) fmt = "RGB";
  if (fmt == "GRAY") fmt = "GRAY8";

  opt.media_type = "video/x-raw";
  opt.format = fmt;
  opt.width = options_.input_width;
  opt.height = options_.input_height;
  return opt;
}

} // namespace sima::mpk
