#include "ModelSession.hpp"

#include "pipeline/internal/SimaaiGuard.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) { out = argv[i + 1]; return true; }
  }
  return false;
}

static bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) return true;
  }
  return false;
}

static std::string shell_quote(const std::string& s) {
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

static bool download_file(const std::string& url, const fs::path& out_path) {
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

static fs::path default_goldfish_path() {
  try {
    return fs::temp_directory_path() / "sima_imagenet_goldfish.jpg";
  } catch (...) {
    return fs::path("tmp") / "imagenet_goldfish.jpg";
  }
}

static std::string resolve_resnet50_tar() {
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

static cv::Mat load_rgb_resized(const std::string& image_path, int w, int h) {
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

static std::vector<float> tensor_to_floats(const sima::FrameTensor& t) {
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

struct ScoredIndex {
  int index = -1;
  float value = 0.0f;
  float prob = 0.0f;
};

static std::vector<ScoredIndex> topk_with_softmax(const std::vector<float>& v, int k) {
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

static std::vector<float> scores_from_tensor(const sima::FrameTensor& t,
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

static void check_top1(const std::vector<float>& scores,
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

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  constexpr const char* kGoldfishUrl =
      "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
      "n01443537_goldfish.JPEG";
  constexpr int kGoldfishId = 1; // ILSVRC2012 0-based index for "goldfish"
  constexpr int kInferWidth = 224;
  constexpr int kInferHeight = 224;
  const std::vector<float> kMean = {0.485f, 0.456f, 0.406f};
  const std::vector<float> kStd = {0.229f, 0.224f, 0.225f};

  std::string image_path;
  std::string tar_gz;
  std::string goldfish_url = kGoldfishUrl;
  int expected_id = kGoldfishId;
  float min_prob = 0.2f;

  std::string tmp;
  if (get_arg(argc, argv, "--image", tmp)) image_path = tmp;
  if (get_arg(argc, argv, "--model", tmp)) tar_gz = tmp;
  if (get_arg(argc, argv, "--goldfish-url", tmp)) goldfish_url = tmp;
  if (get_arg(argc, argv, "--expect-id", tmp)) expected_id = std::stoi(tmp);
  if (get_arg(argc, argv, "--min-prob", tmp)) min_prob = std::stof(tmp);

  const bool skip_second = has_flag(argc, argv, "--skip-second");
  const bool skip_third = has_flag(argc, argv, "--skip-third");

  unsetenv("SIMA_GUARD_TEST_HOLD_MS");
  unsetenv("SIMA_GUARD_TEST_SIGNAL_PATH");
  sima::pipeline_internal::reset_simaai_guard_for_test();

  if (image_path.empty()) {
    const fs::path out_path = default_goldfish_path();
    if (!download_file(goldfish_url, out_path)) {
      std::cerr << "Failed to download goldfish image.\n";
      std::cerr << "URL was: " << goldfish_url << "\n";
      std::cerr << "Tip: supply --image <path> and --expect-id <id> instead.\n";
      return 3;
    }
    image_path = out_path.string();
    std::cout << "Using goldfish image: " << image_path << "\n";
  }

  if (tar_gz.empty()) {
    tar_gz = resolve_resnet50_tar();
    if (tar_gz.empty()) {
      std::cerr << "Failed to resolve resnet50 tar.gz. "
                << "Set SIMA_RESNET50_TAR or run 'sima-cli modelzoo get resnet_50'.\n";
      return 3;
    }
  }

  try {
    std::cerr << "[stage] load image\n";
    cv::Mat rgb = load_rgb_resized(image_path, kInferWidth, kInferHeight);

    std::cerr << "[stage] session1 init\n";
    simaai::ModelSession session1;
    require(session1.init(tar_gz, kInferWidth, kInferHeight, "RGB", true, kMean, kStd),
            "ModelSession init failed: " + session1.last_error());
    std::cerr << "[stage] session1 run\n";
    auto first_out = session1.run_tensor(rgb);
    auto first_scores = scores_from_tensor(first_out, "first");
    check_top1(first_scores, expected_id, min_prob, "first");

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    if (!skip_second) {
      std::cerr << "[stage] session2 init (expect failure)\n";
      simaai::ModelSession session2;
      const bool ok2 = session2.init(tar_gz, kInferWidth, kInferHeight, "RGB", true, kMean, kStd);
      require(!ok2, "expected ModelSession #2 init to fail");
      if (!session2.last_error().empty()) {
        require_contains(session2.last_error(),
                         "single-owner per process",
                         "guard error missing");
      }
      session2.close();
    }
    std::cerr << "[stage] session1 close\n";
    session1.close();

    if (!skip_third) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      std::cerr << "[stage] session3 init (expect failure)\n";
      simaai::ModelSession session3;
      const bool ok3 = session3.init(tar_gz, kInferWidth, kInferHeight, "RGB", true, kMean, kStd);
      require(!ok3, "expected ModelSession #3 init to fail after single-use guard");
      if (!session3.last_error().empty()) {
        require_contains(session3.last_error(),
                         "single-use per process",
                         "guard error missing");
      }
      session3.close();
    }

    std::cout << "[OK] modelsession_resnet50_guard_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
