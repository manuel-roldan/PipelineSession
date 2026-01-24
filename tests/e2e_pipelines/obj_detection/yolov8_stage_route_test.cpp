#include "pipeline/PipelineSession.h"
#include "pipeline/StageRun.h"
#include "pipeline/TessellatedTensor.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/InputAppSrc.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "mpk/ModelMPK.h"
#if defined(SIMA_WITH_OPENCV)
#include "mpk/ModelMPKOpenCV.h"
#endif

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <opencv2/imgcodecs.hpp>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

enum class StageRoute {
  StageOnly,
  PreprocPipeline,
};

struct StageTestConfig {
  int iters = 1;
  float min_score = 0.52f;
  float min_iou = 0.30f;
};

struct PreprocWireInfo {
  std::string format;
  int width = 0;
  int height = 0;
  int depth = 0;
};

std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string find_preproc_config_path(const sima::mpk::ModelMPK& model) {
  std::string path = model.find_config_path_by_plugin("process_cvu");
  if (path.empty()) path = model.find_config_path_by_plugin("preproc");
  if (path.empty()) path = model.find_config_path_by_processor("CVU");
  return path;
}

nlohmann::json load_json_file(const std::string& path, const char* label) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error(std::string(label) + ": failed to open config: " + path);
  }
  nlohmann::json j;
  in >> j;
  return j;
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

PreprocWireInfo read_preproc_wire_info(const sima::mpk::ModelMPK& model,
                                       const sima::NeatTensor& tensor,
                                       const cv::Mat& img) {
  PreprocWireInfo info;
  const std::string path = find_preproc_config_path(model);
  if (!path.empty()) {
    const nlohmann::json j = load_json_file(path, "yolov8_stage_route_test");
    if (j.contains("output_img_type") && j["output_img_type"].is_string()) {
      info.format = j["output_img_type"].get<std::string>();
    }
    info.width = read_int_field(j, "output_width");
    info.height = read_int_field(j, "output_height");
    info.depth = read_int_field(j, "output_channels");
    if (info.depth <= 0) info.depth = read_int_field(j, "tile_channels");
  }

  if (info.format.empty()) info.format = "RGB";
  info.format = upper_copy(info.format);

  if (info.width <= 0 && tensor.shape.size() > 1) {
    info.width = static_cast<int>(tensor.shape[1]);
  }
  if (info.height <= 0 && tensor.shape.size() > 0) {
    info.height = static_cast<int>(tensor.shape[0]);
  }
  if (info.width <= 0) info.width = img.cols;
  if (info.height <= 0) info.height = img.rows;
  if (info.depth <= 0 && tensor.shape.size() >= 3) {
    info.depth = static_cast<int>(tensor.shape[2]);
  }
  if (info.depth <= 0) info.depth = img.channels();

  require(info.width > 0 && info.height > 0,
          "Preproc wire info missing width/height");
  return info;
}

StageRoute parse_route(int argc, char** argv, fs::path& root) {
  StageRoute route = StageRoute::StageOnly;
  bool root_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--route=", 0) == 0) {
      const std::string val = arg.substr(std::string("--route=").size());
      if (val == "stage") {
        route = StageRoute::StageOnly;
      } else if (val == "preproc-pipeline") {
        route = StageRoute::PreprocPipeline;
      } else {
        throw std::invalid_argument("Unknown route: " + val);
      }
      continue;
    }
    if (!root_set && !arg.empty() && arg[0] != '-') {
      root = fs::path(arg);
      root_set = true;
    }
  }
  if (!root_set) root = fs::current_path();
  return route;
}

std::string route_name(StageRoute route) {
  switch (route) {
    case StageRoute::StageOnly:
      return "stage";
    case StageRoute::PreprocPipeline:
      return "preproc-pipeline";
  }
  return "unknown";
}

std::vector<objdet::Box> to_objdet_boxes(const std::vector<sima::Box>& boxes) {
  std::vector<objdet::Box> out;
  out.reserve(boxes.size());
  for (const auto& b : boxes) {
    out.push_back({b.x1, b.y1, b.x2, b.y2, b.score, b.class_id});
  }
  return out;
}

std::string neat_format(const sima::NeatTensor& tensor) {
  if (tensor.semantic.tess.has_value()) {
    return upper_copy(tensor.semantic.tess->format);
  }
  if (tensor.semantic.image.has_value()) {
    switch (tensor.semantic.image->format) {
      case sima::NeatImageSpec::PixelFormat::RGB: return "RGB";
      case sima::NeatImageSpec::PixelFormat::BGR: return "BGR";
      case sima::NeatImageSpec::PixelFormat::GRAY8: return "GRAY8";
      case sima::NeatImageSpec::PixelFormat::NV12: return "NV12";
      case sima::NeatImageSpec::PixelFormat::I420: return "I420";
      case sima::NeatImageSpec::PixelFormat::UNKNOWN: break;
    }
  }
  return "";
}

int neat_width(const sima::NeatTensor& tensor) {
  return (tensor.shape.size() > 1) ? static_cast<int>(tensor.shape[1]) : -1;
}

int neat_height(const sima::NeatTensor& tensor) {
  return (tensor.shape.size() > 0) ? static_cast<int>(tensor.shape[0]) : -1;
}

size_t neat_dtype_bytes(sima::TensorDType dtype) {
  switch (dtype) {
    case sima::TensorDType::UInt8: return 1;
    case sima::TensorDType::Int8: return 1;
    case sima::TensorDType::UInt16: return 2;
    case sima::TensorDType::Int16: return 2;
    case sima::TensorDType::Int32: return 4;
    case sima::TensorDType::BFloat16: return 2;
    case sima::TensorDType::Float32: return 4;
    case sima::TensorDType::Float64: return 8;
  }
  return 1;
}

bool is_int8_tensor(const sima::NeatTensor& tensor) {
  const std::string fmt = neat_format(tensor);
  return sima::is_tessellated_int8_format(fmt) ||
         tensor.dtype == sima::TensorDType::Int8;
}

void step_log(const char* label) {
  std::cout << "[STEP] " << label << std::endl;
}

bool extract_bbox_payload(const sima::RunInputResult& result,
                          std::vector<uint8_t>& payload,
                          std::string& err) {
  if (result.kind != sima::RunOutputKind::Tensor) {
    err = "capture_expected_tensor";
    return false;
  }
  if (result.neat.has_value()) {
    const auto& tensor = result.neat.value();
    const std::string fmt = !result.format.empty()
        ? upper_copy(result.format)
        : neat_format(tensor);
    if (!fmt.empty() && fmt != "BBOX") {
      err = "capture_expected_bbox format=" + fmt;
      return false;
    }
    if (!tensor.storage) {
      err = "capture_missing_storage";
      return false;
    }
    sima::NeatMapping mapping = tensor.map(sima::NeatMapMode::Read);
    if (!mapping.data) {
      err = "capture_missing_mapping";
      return false;
    }
    size_t bytes = 0;
    if (!tensor.shape.empty()) {
      bytes = neat_dtype_bytes(tensor.dtype);
      for (auto dim : tensor.shape) {
        if (dim <= 0) {
          bytes = 0;
          break;
        }
        bytes *= static_cast<size_t>(dim);
      }
    }
    if (bytes == 0) bytes = mapping.size_bytes;
    if (bytes > mapping.size_bytes) {
      err = "capture_size_mismatch";
      return false;
    }
    payload.assign(static_cast<const uint8_t*>(mapping.data),
                   static_cast<const uint8_t*>(mapping.data) + bytes);
  } else {
    err = "capture_missing_tensor";
    return false;
  }
  if (payload.empty()) {
    err = "capture_empty_payload";
    return false;
  }
  return true;
}

void run_stage_route_once(const cv::Mat& img_bgr,
                          const sima::mpk::ModelMPK& model,
                          const StageTestConfig& cfg,
                          const sima::stages::BoxDecodeOptions& box_opt,
                          const std::vector<objdet::ExpectedBox>& expected,
                          int iter) {
  step_log("stage_route: begin");
  auto pre = sima::stages::Preproc(img_bgr, model);
  require(is_int8_tensor(pre),
          "Preproc output is not INT8 tessellated");
  require(neat_width(pre) > 0 && neat_height(pre) > 0,
          "Preproc output missing width/height");
  require(pre.shape.size() >= 3,
          "Preproc output missing shape dims");
  std::cout << "Preproc passed\n" << std::endl;

  auto mla = sima::stages::MLA(pre, model);
  require(is_int8_tensor(mla),
          "MLA output is not INT8 tessellated");
  require(neat_width(mla) > 0 && neat_height(mla) > 0,
          "MLA output missing width/height");
  require(mla.shape.size() >= 3,
          "MLA output missing shape dims");
  std::cout << "MLA passed\n" << std::endl;

  step_log("stage_route: before BoxDecode");
  const auto out = sima::stages::BoxDecode(mla, model, box_opt);
  std::cout << "BoxDecode passed\n" << std::endl;
  step_log("stage_route: after BoxDecode");
  const auto boxes = to_objdet_boxes(out.boxes);
  const objdet::MatchResult match = objdet::match_expected_boxes(
      boxes, expected, cfg.min_score, cfg.min_iou);
  require(match.ok, "verify_mismatch iter=" + std::to_string(iter) + " " + match.note);
}

void run_preproc_pipeline_once(const cv::Mat& img_bgr,
                               const sima::mpk::ModelMPK& model,
                               const StageTestConfig& cfg,
                               const sima::stages::BoxDecodeOptions& box_opt,
                               const std::vector<objdet::ExpectedBox>& expected,
                               int iter) {
  step_log("preproc_pipeline: begin");
  auto pre = sima::stages::Preproc(img_bgr, model);
  require(is_int8_tensor(pre),
          "Preproc output is not INT8 tessellated");
  require(neat_width(pre) > 0 && neat_height(pre) > 0,
          "Preproc output missing width/height");
  require(pre.shape.size() >= 3,
          "Preproc output missing shape dims");
  std::cout << "Preproc passed\n" << std::endl;

  const PreprocWireInfo wire_info = read_preproc_wire_info(model, pre, img_bgr);

  sima::NeatTensor wire = pre;
  wire.dtype = sima::TensorDType::UInt8;
  wire.semantic.tess.reset();
  {
    sima::NeatImageSpec image;
    image.format = sima::NeatImageSpec::PixelFormat::RGB;
    if (upper_copy(wire_info.format) == "BGR") {
      image.format = sima::NeatImageSpec::PixelFormat::BGR;
    }
    wire.semantic.image = image;
  }
  wire.shape = {wire_info.height, wire_info.width, wire_info.depth};
  wire.strides_bytes.clear();

  sima::InputAppSrcOptions src_opt = model.input_appsrc_options(false);
  src_opt.media_type = "video/x-raw";
  src_opt.format = wire_info.format;
  src_opt.width = wire_info.width;
  src_opt.height = wire_info.height;
  src_opt.depth = -1;

  const int topk = (box_opt.top_k > 0) ? box_opt.top_k : 100;
  sima::PipelineSession p;
  p.add(sima::nodes::InputAppSrc(src_opt));
  p.add(sima::nodes::groups::MLA(model));
  p.add(sima::nodes::SimaBoxDecode(
      model,
      "yolov8",
      img_bgr.cols,
      img_bgr.rows,
      cfg.min_score,
      0.5f,
      topk));
  p.add(sima::nodes::OutputAppSink());

  step_log("preproc_pipeline: before p.run");
  const sima::RunInputResult out = p.run(wire);
  step_log("preproc_pipeline: after p.run");
  std::vector<uint8_t> payload;
  std::string err;
  require(extract_bbox_payload(out, payload, err), err);

  const auto boxes = objdet::parse_boxes_strict(
      payload, img_bgr.cols, img_bgr.rows, topk, false);
  const objdet::MatchResult match = objdet::match_expected_boxes(
      boxes, expected, cfg.min_score, cfg.min_iou);
  require(match.ok, "verify_mismatch iter=" + std::to_string(iter) + " " + match.note);
}

} // namespace

int main(int argc, char** argv) {
  try {
    fs::path root;
    const StageRoute route = parse_route(argc, argv, root);
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_e2e::resolve_yolov8s_tar(root);
    require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

    const fs::path img_path = root / "tmp" / "yolov8s_people.jpg";
    cv::Mat img_bgr = cv::imread(img_path.string(), cv::IMREAD_COLOR);
    require(!img_bgr.empty(), "Failed to read image: " + img_path.string());

#if defined(SIMA_WITH_OPENCV)
    auto model = sima::mpk::ModelMPK(tar_gz, img_bgr);
#else
    auto model = sima::mpk::ModelMPK(
        tar_gz,
        "video/x-raw",
        "BGR",
        img_bgr.cols,
        img_bgr.rows,
        3);
#endif

    StageTestConfig cfg;

    sima::stages::BoxDecodeOptions box_opt;
    box_opt.decode_type = "yolov8";
    box_opt.original_width = img_bgr.cols;
    box_opt.original_height = img_bgr.rows;
    box_opt.detection_threshold = cfg.min_score;
    box_opt.nms_iou_threshold = 0.5f;
    box_opt.top_k = 100;

    const std::vector<objdet::ExpectedBox> expected = objdet::expected_people_boxes();

    for (int i = 0; i < cfg.iters; ++i) {
      if (route == StageRoute::StageOnly) {
        run_stage_route_once(img_bgr, model, cfg, box_opt, expected, i);
      } else {
        run_preproc_pipeline_once(img_bgr, model, cfg, box_opt, expected, i);
      }
    }

    std::cout << "[OK] yolov8_stage_route_test passed route="
              << route_name(route) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
