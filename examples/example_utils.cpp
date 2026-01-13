#include "example_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <system_error>

namespace fs = std::filesystem;

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
