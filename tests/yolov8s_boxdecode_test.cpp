#include "pipeline/PipelineSession.h"
#include "nodes/common/AppSink.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/InputAppSrc.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "mpk/ModelMPK.h"

#include "example_utils.h"
#include "test_utils.h"

// Requires network access, sima-cli, and simaai plugins to run end-to-end.

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool env_bool(const char* key, bool def = false) {
  const char* v = std::getenv(key);
  if (!v) return def;
  return std::string(v) != "0";
}

int read_boxdecode_topk(const std::string& path) {
  std::ifstream in(path);
  require(in.is_open(), "Failed to open boxdecode config: " + path);
  nlohmann::json j;
  in >> j;
  return j.value("topk", 0);
}

struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

struct GoldenBox {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

struct RawBox {
  int32_t x = 0;
  int32_t y = 0;
  int32_t w = 0;
  int32_t h = 0;
  float score = 0.0f;
  int32_t cls = 0;
};

float box_iou(const Box& a, const GoldenBox& b) {
  const float ix1 = std::max(a.x1, b.x1);
  const float iy1 = std::max(a.y1, b.y1);
  const float ix2 = std::min(a.x2, b.x2);
  const float iy2 = std::min(a.y2, b.y2);
  const float iw = std::max(0.0f, ix2 - ix1);
  const float ih = std::max(0.0f, iy2 - iy1);
  const float inter = iw * ih;
  const float a_area = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float b_area = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float uni = a_area + b_area - inter;
  if (uni <= 0.0f) return 0.0f;
  return inter / uni;
}

std::vector<Box> parse_boxes(const std::vector<uint8_t>& bytes,
                             int img_w,
                             int img_h,
                             int expected_topk) {
  require(bytes.size() >= 4, "bbox buffer too small");
  const size_t payload = bytes.size() - 4;
  require(payload % 24 == 0, "bbox buffer size mismatch");

  uint32_t header = 0;
  std::memcpy(&header, bytes.data(), sizeof(header));

  if (env_bool("SIMA_BOXDECODE_DEBUG")) {
    std::cerr << "[DBG] bbox header=" << header
              << " expected_topk=" << expected_topk
              << " payload=" << payload << "\n";
  }

  const size_t max_boxes = payload / 24;
  require(header <= max_boxes, "bbox header exceeds payload count");
  if (expected_topk > 0) {
    const size_t expected_size = 4 + static_cast<size_t>(expected_topk) * 24;
    require(bytes.size() == expected_size, "bbox buffer size != expected topk");
  }

  const size_t count = header;
  std::vector<Box> out;
  out.reserve(count);

  const uint8_t* base = bytes.data() + 4;
  for (size_t i = 0; i < count; ++i) {
    RawBox r{};
    std::memcpy(&r, base + i * 24, sizeof(r));

    float x1 = static_cast<float>(r.x);
    float y1 = static_cast<float>(r.y);
    float x2 = static_cast<float>(r.x + r.w);
    float y2 = static_cast<float>(r.y + r.h);

    x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_w)));
    y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_h)));
    x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_w)));
    y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_h)));

    out.push_back(Box{x1, y1, x2, y2, r.score, r.cls});
    if (env_bool("SIMA_BOXDECODE_DEBUG") && i < 4) {
      std::cerr << "[DBG] box[" << i << "]="
                << r.x << "," << r.y << "," << r.w << "," << r.h
                << " score=" << r.score << " class=" << r.cls << "\n";
    }
  }
  return out;
}

void draw_boxes(cv::Mat& img, const std::vector<Box>& boxes, float min_score) {
  for (const auto& b : boxes) {
    if (b.score < min_score) continue;
    const int x1 = std::max(0, static_cast<int>(std::round(b.x1)));
    const int y1 = std::max(0, static_cast<int>(std::round(b.y1)));
    const int x2 = std::min(img.cols - 1, static_cast<int>(std::round(b.x2)));
    const int y2 = std::min(img.rows - 1, static_cast<int>(std::round(b.y2)));
    if (x2 <= x1 || y2 <= y1) continue;

    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
    const std::string label =
        "id=" + std::to_string(b.class_id) + " score=" + std::to_string(b.score);
    cv::putText(img, label, cv::Point(x1, std::max(0, y1 - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const fs::path image_path = sima_examples::ensure_coco_sample(root);
    require(!image_path.empty(), "Failed to download people image");

    const std::string tar_gz = sima_examples::resolve_yolov8s_tar(root);
    require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

    cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    require(!bgr.empty(), "Failed to read image");

    sima::OutputSpec input_spec;
    input_spec.media_type = "video/x-raw";
    input_spec.format = "BGR";
    input_spec.width = bgr.cols;
    input_spec.height = bgr.rows;
    input_spec.depth = 3;

    sima::mpk::ModelMPKOptions mpk_opt = sima::mpk::options_from_output_spec(input_spec);
    auto model = sima::mpk::ModelMPK::load(tar_gz, mpk_opt);

    const std::string config_path = sima_examples::find_boxdecode_config(model.etc_dir());
    require(!config_path.empty(), "Failed to locate simaaiboxdecode config JSON");
    const std::string runtime_config =
        sima_examples::prepare_yolo_boxdecode_config(
            config_path, root, bgr.cols, bgr.rows, 0.5f, 0.5f);
    const int box_topk = read_boxdecode_topk(runtime_config);

    sima::PipelineSession p;
    sima::InputAppSrcOptions src_opt;
    src_opt.format = "BGR";
    src_opt.width = bgr.cols;
    src_opt.height = bgr.rows;
    p.add(sima::nodes::InputAppSrc(src_opt));
    p.add(sima::nodes::groups::Preprocess(model));
    p.add(sima::nodes::groups::MLA(model));

    sima::SimaBoxDecodeOptions box_opt;
    box_opt.config_path = runtime_config;
    p.add(sima::nodes::SimaBoxDecode(box_opt));
    p.add(sima::nodes::OutputAppSink());

    sima::RunInputResult out = p.run(bgr);
    require(out.kind == sima::RunOutputKind::Tensor, "Expected tensor output from boxdecode");
    require(out.tensor.has_value(), "Missing boxdecode tensor output");
    require(out.tensor->format == "BBOX", "Expected BBOX tensor output");
    require(!out.tensor->planes.empty(), "BBOX tensor missing data");

    const auto boxes = parse_boxes(out.tensor->planes[0],
                                   bgr.cols,
                                   bgr.rows,
                                   box_topk);
    const float kGoldenMinScore = 0.6f;
    const float kGoldenScoreTol = 0.1f;
    const float kGoldenIou = 0.5f;
    const std::vector<GoldenBox> golden = {
        {747.0f, 42.0f, 1131.0f, 711.0f, 0.70576f, 0},
        {148.0f, 202.0f, 1096.0f, 713.0f, 0.70576f, 0},
    };

    std::vector<bool> matched(golden.size(), false);
    for (const auto& pred : boxes) {
      if (pred.score < kGoldenMinScore) continue;
      for (size_t i = 0; i < golden.size(); ++i) {
        if (matched[i]) continue;
        const GoldenBox& g = golden[i];
        if (pred.class_id != g.class_id) continue;
        if (std::abs(pred.score - g.score) > kGoldenScoreTol) continue;
        if (box_iou(pred, g) >= kGoldenIou) {
          matched[i] = true;
          break;
        }
      }
    }

    for (size_t i = 0; i < golden.size(); ++i) {
      require(matched[i], "Golden box " + std::to_string(i) + " not matched");
    }

    cv::Mat overlay = bgr.clone();
    draw_boxes(overlay, boxes, kGoldenMinScore);
    const fs::path out_path = root / "tmp" / "yolov8s_boxes.jpg";
    require(cv::imwrite(out_path.string(), overlay), "Failed to write overlay image");

    std::cout << "[OK] yolov8s_boxdecode_test wrote " << out_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
