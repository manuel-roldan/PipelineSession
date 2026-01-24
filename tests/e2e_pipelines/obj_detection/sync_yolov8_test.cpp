#include "pipeline/PipelineSession.h"
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

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct SyncTestConfig {
  int iters = 20;
  float min_score = 0.52f;
  float min_iou = 0.30f;
};

struct RunSummary {
  bool ok = false;
  int outputs = 0;
  double avg_fps = 0.0;
  std::string note;
  std::string diagnostics;
};

void step_log(const char* label) {
  std::cout << "[STEP] " << label << std::endl;
}

void append_note(std::string& note, const std::string& part) {
  if (part.empty()) return;
  if (!note.empty()) note += ";";
  note += part;
}

std::string sanitize_note(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back((c == '\n' || c == '\r') ? '|' : c);
  }
  return out;
}

int env_int(const char* name, int def) {
  const char* val = std::getenv(name);
  if (!val || !*val) return def;
  return std::atoi(val);
}

bool extract_bbox_payload(const sima::RunInputResult& result,
                          int iter,
                          std::vector<uint8_t>& payload,
                          std::string& err) {
  if (result.kind != sima::RunOutputKind::Tensor) {
    err = "capture_expected_tensor iter=" + std::to_string(iter);
    return false;
  }
  if (result.neat.has_value()) {
    const auto& tensor = result.neat.value();
    std::string fmt = result.format;
    if (fmt.empty() && tensor.semantic.tess.has_value()) {
      fmt = tensor.semantic.tess->format;
    }
    if (!fmt.empty() && fmt != "BBOX") {
      err = "capture_expected_bbox iter=" + std::to_string(iter) +
            " format=" + fmt;
      return false;
    }
    sima::NeatMapping mapping = tensor.map(sima::NeatMapMode::Read);
    if (!mapping.data) {
      err = "capture_missing_mapping iter=" + std::to_string(iter);
      return false;
    }
    size_t bytes = 0;
    if (!tensor.shape.empty()) {
      size_t elem = 1;
      switch (tensor.dtype) {
        case sima::TensorDType::UInt8: elem = 1; break;
        case sima::TensorDType::Int8: elem = 1; break;
        case sima::TensorDType::UInt16: elem = 2; break;
        case sima::TensorDType::Int16: elem = 2; break;
        case sima::TensorDType::Int32: elem = 4; break;
        case sima::TensorDType::BFloat16: elem = 2; break;
        case sima::TensorDType::Float32: elem = 4; break;
        case sima::TensorDType::Float64: elem = 8; break;
      }
      bytes = elem;
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
      err = "capture_size_mismatch iter=" + std::to_string(iter);
      return false;
    }
    payload.assign(static_cast<const uint8_t*>(mapping.data),
                   static_cast<const uint8_t*>(mapping.data) + bytes);
  } else {
    err = "capture_missing_tensor iter=" + std::to_string(iter);
    return false;
  }
  if (payload.empty()) {
    err = "capture_empty_payload iter=" + std::to_string(iter);
    return false;
  }
  return true;
}

RunSummary run_yolov8_sync(const fs::path& root,
                           const cv::Mat& img,
                           const SyncTestConfig& cfg) {
  RunSummary res;

  const std::string tar_gz = sima_e2e::resolve_yolov8s_tar(root);
  require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

  const int num_both = env_int("SIMA_SYNC_NUM_BUFFERS", -1);
  int num_cvu = env_int("SIMA_SYNC_NUM_BUFFERS_CVU", num_both);
  int num_mla = env_int("SIMA_SYNC_NUM_BUFFERS_MLA", num_both);
  const bool override_num = (num_cvu >= 0 || num_mla >= 0);
  if (override_num) {
    if (num_cvu < 0 || num_mla < 0) {
      append_note(res.note, "num_buffers_requires_both");
      return res;
    }
    if (!((num_cvu == 0 || num_cvu == 1) && (num_mla == 0 || num_mla == 1))) {
      append_note(res.note, "num_buffers_invalid");
      return res;
    }
  }

#if defined(SIMA_WITH_OPENCV)
  auto model = override_num
      ? sima::mpk::ModelMPK(tar_gz,
                            img,
                            false,
                            {},
                            {},
                            {},
                            "decoder",
                            num_cvu,
                            num_mla)
      : sima::mpk::ModelMPK(tar_gz, img);
#else
  auto model = override_num
      ? sima::mpk::ModelMPK(tar_gz,
                            "video/x-raw",
                            "BGR",
                            img.cols,
                            img.rows,
                            3,
                            false,
                            {},
                            {},
                            {},
                            "decoder",
                            num_cvu,
                            num_mla)
      : sima::mpk::ModelMPK(tar_gz, "video/x-raw", "BGR", img.cols, img.rows, 3);
#endif
  const int topk = 100;

  sima::PipelineSession p;
  p.add(sima::nodes::InputAppSrc());
  p.add(sima::nodes::groups::Preprocess(model));
  p.add(sima::nodes::groups::MLA(model));
  p.add(sima::nodes::SimaBoxDecode(
      model,
      "yolov8",
      img.cols,
      img.rows,
      cfg.min_score,
      0.5f,
      topk));
  p.add(sima::nodes::OutputAppSink());

  const std::vector<objdet::ExpectedBox> expected = objdet::expected_people_boxes();

  step_log("sync: before build");
  (void)p.build(img, sima::PipelineRunMode::Sync);
  step_log("sync: after build");

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < cfg.iters; ++i) {
    std::cout << "SYNC_YOLOV8 iter " << (i + 1) << "/" << cfg.iters << "\n";
    std::cout.flush();
    sima::RunInputResult out;
    try {
      step_log("sync: before run");
      out = p.run(img);
      step_log("sync: after run");
    } catch (const std::exception& e) {
      append_note(res.note, "run_error=" + sanitize_note(e.what()));
      break;
    }

    std::vector<uint8_t> payload;
    std::string err;
    if (!extract_bbox_payload(out, i, payload, err)) {
      append_note(res.note, err);
      break;
    }

    const auto boxes = objdet::parse_boxes_strict(
        payload, img.cols, img.rows, topk, false);
    const objdet::MatchResult match = objdet::match_expected_boxes(
        boxes, expected, cfg.min_score, cfg.min_iou);
    if (!match.ok) {
      append_note(res.note,
                  "verify_mismatch iter=" + std::to_string(i) + " " + match.note);
      break;
    }

    res.outputs += 1;
  }
  const auto end = std::chrono::steady_clock::now();

  res.diagnostics = p.last_pipeline();

  const double elapsed_s =
      std::chrono::duration<double>(end - start).count();
  res.avg_fps = (elapsed_s > 0.0)
      ? (static_cast<double>(res.outputs) / elapsed_s)
      : 0.0;
  res.ok = (res.outputs == cfg.iters);
  if (elapsed_s <= 0.0) {
    append_note(res.note, "sync_timing_incomplete");
    res.ok = false;
  }

  return res;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const fs::path img_path =
        "/home/sima/stable_pipeline_session/PipelineSession/tmp/yolov8s_people.jpg";
    cv::Mat img_bgr = cv::imread(img_path.string(), cv::IMREAD_COLOR);
    require(!img_bgr.empty(), "Failed to read image: " + img_path.string());

    SyncTestConfig cfg;
    RunSummary res = run_yolov8_sync(root, img_bgr, cfg);

    std::cout << "SYNC_YOLOV8 outputs=" << res.outputs
              << " avg_fps=" << res.avg_fps
              << " ok=" << (res.ok ? "1" : "0")
              << " note=" << res.note << "\n";
    if (!res.diagnostics.empty()) {
      std::cout << "SYNC_YOLOV8 diagnostics\n" << res.diagnostics << "\n";
    }

    return res.ok ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
