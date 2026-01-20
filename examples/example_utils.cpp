#include "example_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <system_error>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace sima_examples {
namespace {

std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

bool move_to_tmp(const fs::path& src, const fs::path& dst) {
  std::error_code ec;
  fs::create_directories(dst.parent_path(), ec);
  ec.clear();
  fs::rename(src, dst, ec);
  if (!ec) return true;

  ec.clear();
  fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
  if (ec) return false;
  fs::remove(src, ec);
  return true;
}

std::string lower_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

bool env_int(const char* key, int& out) {
  const char* v = std::getenv(key);
  if (!v || !*v) return false;
  char* end = nullptr;
  long val = std::strtol(v, &end, 10);
  if (!end || *end != '\0') return false;
  out = static_cast<int>(val);
  return true;
}

bool env_double(const char* key, double& out) {
  const char* v = std::getenv(key);
  if (!v || !*v) return false;
  char* end = nullptr;
  double val = std::strtod(v, &end);
  if (!end || *end != '\0') return false;
  out = val;
  return true;
}

bool env_string(const char* key, std::string& out) {
  const char* v = std::getenv(key);
  if (!v || !*v) return false;
  out = v;
  return true;
}

} // namespace

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) { out = argv[i + 1]; return true; }
  }
  return false;
}

bool download_file(const std::string& url, const fs::path& out_path) {
  if (fs::exists(out_path)) {
    std::error_code ec;
    if (fs::file_size(out_path, ec) > 0 && !ec) return true;
  }

  std::error_code ec;
  fs::create_directories(out_path.parent_path(), ec);

  const std::string qurl = shell_quote(url);
  const std::string qout = shell_quote(out_path.string());

  std::string cmd = "curl -L --fail --silent --show-error -o " + qout + " " + qurl;
  if (std::system(cmd.c_str()) == 0) return true;

  cmd = "wget -O " + qout + " " + qurl;
  if (std::system(cmd.c_str()) == 0) return true;

  std::error_code rm_ec;
  fs::remove(out_path, rm_ec);
  return false;
}

fs::path default_goldfish_path() {
  try {
    return fs::temp_directory_path() / "sima_imagenet_goldfish.jpg";
  } catch (...) {
    return fs::path("tmp") / "imagenet_goldfish.jpg";
  }
}

std::string resolve_resnet50_tar() {
  const char* env = std::getenv("SIMA_RESNET50_TAR");
  if (env && *env && fs::exists(env)) {
    return std::string(env);
  }

  const fs::path local = fs::path("tmp") / "resnet_50_mpk.tar.gz";
  if (fs::exists(local)) return local.string();

  auto move_to_tmp = [&](const fs::path& src) -> bool {
    std::error_code ec;
    fs::create_directories(local.parent_path(), ec);
    ec.clear();
    fs::rename(src, local, ec);
    if (!ec) return true;

    ec.clear();
    fs::copy_file(src, local, fs::copy_options::overwrite_existing, ec);
    if (ec) return false;
    fs::remove(src, ec);
    return true;
  };

  const int rc = std::system("sima-cli modelzoo get resnet_50");
  if (rc != 0) return "";

  if (fs::exists(local)) return local.string();

  const std::vector<fs::path> candidates = {
      "resnet_50_mpk.tar.gz",
      "resnet-50_mpk.tar.gz",
  };
  for (const auto& candidate : candidates) {
    if (fs::exists(candidate) && move_to_tmp(candidate)) {
      return local.string();
    }
  }

  return "";
}

std::string resolve_yolov8s_tar(const fs::path& root_in) {
  const fs::path root = root_in.empty() ? fs::current_path() : root_in;
  const fs::path tmp_tar = root / "tmp" / "yolo_v8s_mpk.tar.gz";

  const char* env = std::getenv("SIMA_YOLO_TAR");
  if (env && *env && fs::exists(env)) {
    return std::string(env);
  }

  const int rc = std::system("sima-cli modelzoo get yolo_v8s");
  if (rc == 0 && fs::exists(tmp_tar)) return tmp_tar.string();

  const char* home = std::getenv("HOME");
  const fs::path home_path = home ? fs::path(home) : fs::path();
  const std::vector<fs::path> search_dirs = {
      root,
      fs::current_path(),
      root / "tmp",
      home_path / ".simaai",
      home_path / ".simaai" / "modelzoo",
      home_path / ".sima" / "modelzoo",
      "/data/simaai/modelzoo",
  };

  const std::vector<std::string> names = {
      "yolo_v8s_mpk.tar.gz",
      "yolo-v8s_mpk.tar.gz",
      "yolov8s_mpk.tar.gz",
      "yolov8_s_mpk.tar.gz",
  };

  for (const auto& dir : search_dirs) {
    if (dir.empty()) continue;
    for (const auto& name : names) {
      fs::path candidate = dir / name;
      if (fs::exists(candidate) && move_to_tmp(candidate, tmp_tar)) {
        return tmp_tar.string();
      }
    }
  }

  return "";
}

fs::path ensure_coco_sample(const fs::path& root_in) {
  const fs::path root = root_in.empty() ? fs::current_path() : root_in;
  const char* url_env = std::getenv("SIMA_COCO_URL");
  const std::string url = (url_env && *url_env)
      ? std::string(url_env)
      : "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg";
  const fs::path out_path = root / "tmp" / "coco_sample.jpg";
  if (!download_file(url, out_path)) return {};
  return out_path;
}

std::string find_boxdecode_config(const fs::path& etc_dir) {
  if (!fs::exists(etc_dir)) return "";

  for (const auto& entry : fs::directory_iterator(etc_dir)) {
    if (!entry.is_regular_file()) continue;
    const fs::path p = entry.path();
    if (p.extension() == ".json" &&
        lower_copy(p.filename().string()).find("boxdecode") != std::string::npos) {
      return p.string();
    }
  }

  for (const auto& entry : fs::directory_iterator(etc_dir)) {
    if (!entry.is_regular_file()) continue;
    const fs::path p = entry.path();
    if (p.extension() != ".json") continue;
    std::ifstream in(p);
    if (!in.is_open()) continue;
    std::string content((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
    if (lower_copy(content).find("boxdecode") != std::string::npos) {
      return p.string();
    }
  }

  return "";
}

std::string prepare_yolo_boxdecode_config(const std::string& src_path,
                                          const fs::path& root_in,
                                          int img_w,
                                          int img_h,
                                          float conf,
                                          float nms) {
  const fs::path root = root_in.empty() ? fs::current_path() : root_in;
  std::ifstream in(src_path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open boxdecode config: " + src_path);
  }

  json j;
  in >> j;

  j["original_width"] = img_w;
  j["original_height"] = img_h;
  j["detection_threshold"] = conf;
  j["nms_iou_threshold"] = nms;
  if (!j.contains("memory") || !j["memory"].is_object()) {
    j["memory"] = json::object();
  }
  std::string override_decode;
  if (env_string("SIMA_BOXDECODE_DECODE_TYPE", override_decode)) {
    j["decode_type"] = override_decode;
  }
  int override_topk = 0;
  if (env_int("SIMA_BOXDECODE_TOPK", override_topk)) {
    j["topk"] = override_topk;
  }
  int override_classes = 0;
  if (env_int("SIMA_BOXDECODE_NUM_CLASSES", override_classes)) {
    j["num_classes"] = override_classes;
  }
  int override_num_in = 0;
  if (env_int("SIMA_BOXDECODE_NUM_IN_TENSOR", override_num_in)) {
    j["num_in_tensor"] = override_num_in;
  }
  double override_det = 0.0;
  if (env_double("SIMA_BOXDECODE_DETECTION_THRESHOLD", override_det)) {
    j["detection_threshold"] = override_det;
  }
  double override_nms = 0.0;
  if (env_double("SIMA_BOXDECODE_NMS_IOU", override_nms)) {
    j["nms_iou_threshold"] = override_nms;
  }
  int override_mem_cpu = 0;
  if (env_int("SIMA_BOXDECODE_MEMORY_CPU", override_mem_cpu)) {
    j["memory"]["cpu"] = override_mem_cpu;
  }
  int override_mem_next = 0;
  if (env_int("SIMA_BOXDECODE_MEMORY_NEXT_CPU", override_mem_next)) {
    j["memory"]["next_cpu"] = override_mem_next;
    if (j.contains("next_cpu")) {
      j["next_cpu"] = override_mem_next;
    }
  }
  int override_debug = 0;
  const bool override_debug_set = env_int("SIMA_BOXDECODE_DEBUG", override_debug);
  if (override_debug_set) {
    j["debug"] = (override_debug != 0);
    if (!j.contains("system") || !j["system"].is_object()) {
      j["system"] = json::object();
    }
    j["system"]["debug"] = override_debug;
  }
  int override_outq = 0;
  if (env_int("SIMA_BOXDECODE_OUT_QUEUE", override_outq)) {
    if (!j.contains("system") || !j["system"].is_object()) {
      j["system"] = json::object();
    }
    j["system"]["out_buf_queue"] = override_outq;
  }
  int override_coi = 0;
  if (env_int("SIMA_BOXDECODE_CHANNEL", override_coi)) {
    j["channel_of_interest"] = override_coi;
  }
  if (j.contains("debug") && j["debug"].is_string()) {
    j["debug"] = false;
  }
  if (!override_debug_set && j.contains("system") && j["system"].is_object()) {
    j["system"]["debug"] = 0;
  }

  const fs::path out_path = root / "tmp" / "boxdecode_runtime.json";
  std::ofstream out(out_path);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to write boxdecode runtime config");
  }
  out << j.dump(2);
  return out_path.string();
}

cv::Mat load_rgb_resized(const std::string& image_path, int w, int h) {
  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path);
  }

  if (w > 0 && h > 0 && (bgr.cols != w || bgr.rows != h)) {
    cv::resize(bgr, bgr, cv::Size(w, h), 0, 0, cv::INTER_AREA);
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

std::vector<float> tensor_to_floats(const sima::FrameTensor& t) {
  if (t.dtype != sima::TensorDType::Float32) {
    throw std::runtime_error("Expected Float32 tensor output");
  }
  if (t.planes.empty() || t.planes[0].empty()) {
    throw std::runtime_error("Tensor output is empty");
  }

  const size_t bytes = t.planes[0].size();
  if (bytes % sizeof(float) != 0) {
    throw std::runtime_error("Tensor plane size is not a multiple of float");
  }

  const size_t elems = bytes / sizeof(float);
  std::vector<float> out(elems);
  std::memcpy(out.data(), t.planes[0].data(), elems * sizeof(float));
  return out;
}

std::vector<float> scores_from_tensor(const sima::FrameTensor& t,
                                      const std::string& label) {
  auto scores_full = tensor_to_floats(t);
  if (scores_full.empty()) {
    throw std::runtime_error(label + ": empty tensor output");
  }
  if (scores_full.size() < 1000) {
    throw std::runtime_error(label + ": expected at least 1000 scores, got " +
                             std::to_string(scores_full.size()));
  }
  if (scores_full.size() > 1000) {
    scores_full.resize(1000);
  }
  return scores_full;
}

std::vector<ScoredIndex> topk_with_softmax(const std::vector<float>& v, int k) {
  if (v.empty() || k <= 0) return {};
  const int n = static_cast<int>(v.size());
  k = std::min(k, n);

  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&v](int a, int b) { return v[a] > v[b]; });

  const float maxv = *std::max_element(v.begin(), v.end());
  double sum = 0.0;
  for (float x : v) {
    sum += std::exp(static_cast<double>(x - maxv));
  }

  std::vector<ScoredIndex> out;
  out.reserve(k);
  for (int i = 0; i < k; ++i) {
    const int id = idx[i];
    const double prob = std::exp(static_cast<double>(v[id] - maxv)) / sum;
    out.push_back(ScoredIndex{id, v[id], static_cast<float>(prob)});
  }
  return out;
}

void check_top1(const std::vector<float>& scores,
                int expected_id,
                float min_prob,
                const std::string& label) {
  const auto top = topk_with_softmax(scores, 5);
  std::cout << "[" << label << "] top1 index=" << top[0].index
            << " score=" << top[0].value
            << " prob=" << top[0].prob << "\n";
  std::cout << "[" << label << "] top5:";
  for (const auto& t : top) {
    std::cout << " " << t.index << ":" << t.prob;
  }
  std::cout << "\n";

  if (expected_id < 0) return;

  if (top[0].index != expected_id) {
    throw std::runtime_error(label + ": top-1 mismatch: expected " +
                             std::to_string(expected_id) + " got " +
                             std::to_string(top[0].index));
  }
  if (min_prob > 0.0f && top[0].prob < min_prob) {
    throw std::runtime_error(label + ": top-1 probability too low: " +
                             std::to_string(top[0].prob) + " < " +
                             std::to_string(min_prob));
  }
  std::cout << "[" << label << "] top-1 matches expected class "
            << expected_id << "\n";
}

sima::FrameTensor pull_tensor_with_retry(sima::TensorStream& ts,
                                         const std::string& label,
                                         int per_try_ms,
                                         int tries) {
  for (int i = 0; i < tries; ++i) {
    auto t = ts.next_copy(per_try_ms);
    if (t.has_value()) return *t;
  }
  throw std::runtime_error(label + ": no tensor received (timeout/EOS)");
}

} // namespace sima_examples
