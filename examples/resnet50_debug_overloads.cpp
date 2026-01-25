#include "sima/debug.h"

#include "pipeline/PipelineSession.h"
#include "mpk/ModelMPK.h"
#include "nodes/groups/ImageInputGroup.h"
#include "nodes/groups/RtspInputGroup.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/AppSrcImage.h"
#include "nodes/io/InputAppSrc.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/RtpH264Pay.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
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

static cv::Mat load_rgb(const std::string& image_path) {
  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path);
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

static std::vector<float> tensor_to_floats(const sima::NeatTensor& t) {
  if (t.dtype != sima::TensorDType::Float32) {
    throw std::runtime_error("Expected Float32 tensor output");
  }
  std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  if (raw.empty()) {
    throw std::runtime_error("Tensor output is empty");
  }

  const size_t bytes = raw.size();
  if (bytes % sizeof(float) != 0) {
    throw std::runtime_error("Tensor plane size is not a multiple of float");
  }

  const size_t elems = bytes / sizeof(float);
  std::vector<float> out(elems);
  std::memcpy(out.data(), raw.data(), elems * sizeof(float));
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

static sima::NeatTensor require_tensor(const sima::debug::DebugOutput& out,
                                       const std::string& label) {
  if (!out.tensor.has_value()) {
    throw std::runtime_error(label + ": expected tensor output");
  }
  return *out.tensor;
}

static std::vector<float> scores_from_tensor(const sima::NeatTensor& t,
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
  auto top = topk_with_softmax(scores, 5);
  if (top.empty()) {
    throw std::runtime_error(label + ": topk failed");
  }
  std::cout << "[" << label << "] top1 index=" << top[0].index
            << " score=" << top[0].value << " prob=" << top[0].prob << "\n";

  if (expected_id >= 0 && top[0].index != expected_id) {
    throw std::runtime_error(label + ": top-1 mismatch");
  }
  if (min_prob > 0.0f && top[0].prob < min_prob) {
    throw std::runtime_error(label + ": top-1 prob too low");
  }
}

constexpr int kInferFps = 30;
constexpr int kRtspPort = 8557;

static sima::debug::DebugOutput run_direct_debug(const std::string& tar_gz,
                                                 const cv::Mat& rgb) {
  sima::InputAppSrcOptions src_opt;
  src_opt.format = "RGB";
  auto input = sima::nodes::InputAppSrc(rgb, src_opt);
  sima::mpk::ModelMPKOptions mpk_opt = sima::mpk::options_from_output_spec(input.observed);
  auto model = sima::mpk::ModelMPK::load(tar_gz, mpk_opt);
  return sima::nodes::groups::Infer(input, model);
}

static sima::debug::DebugOutput run_image_group_debug(const std::string& tar_gz,
                                                      const std::string& image_path,
                                                      int fps) {
  sima::nodes::groups::ImageInputGroupOptions opt;
  opt.path = image_path;
  opt.imagefreeze_num_buffers = 8;
  opt.fps = fps;
  auto frame = sima::nodes::groups::ImageInputGroup(opt, sima::debug::output);
  // frame.observed captures the negotiated caps (format/size) used to configure preproc.
  sima::mpk::ModelMPKOptions mpk_opt = sima::mpk::options_from_output_spec(frame.observed);
  auto model = sima::mpk::ModelMPK::load(tar_gz, mpk_opt);
  return sima::nodes::groups::Infer(frame, model);
}

struct RtspServerContext {
  sima::PipelineSession session;
  sima::RtspServerHandle handle;
};

static RtspServerContext start_rtsp_server(const std::string& image_path,
                                           int content_w,
                                           int content_h,
                                           int enc_w,
                                           int enc_h,
                                           int fps,
                                           int port) {
  RtspServerContext ctx;
  ctx.session.add(sima::nodes::AppSrcImage(image_path,
                                           content_w, content_h,
                                           enc_w, enc_h,
                                           fps));
  ctx.session.add(sima::nodes::H264EncodeSima(
      enc_w, enc_h, fps, /*bitrate_kbps=*/400, "baseline", "4.0"));
  ctx.session.add(sima::nodes::H264Parse(/*config_interval=*/1));
  ctx.session.add(sima::nodes::RtpH264Pay(/*pt=*/96, /*config_interval=*/1));

  ctx.handle = ctx.session.run_rtsp({ .mount = "image", .port = port });
  std::cout << "[rtsp] url=" << ctx.handle.url() << "\n";
  return ctx;
}

static sima::debug::DebugOutput run_rtsp_debug(const std::string& tar_gz,
                                               const std::string& url) {
  sima::nodes::groups::RtspInputGroupOptions opt;
  opt.url = url;
  opt.latency_ms = 200;
  opt.tcp = true;
  opt.payload_type = 96;
  opt.decoder_name = "decoder";

  auto frames = sima::nodes::groups::RtspInputGroup(opt, sima::debug::stream);
  auto first = frames.next(1000);
  frames.close();
  if (!first.has_value()) {
    throw std::runtime_error("rtsp: no output received");
  }
  sima::mpk::ModelMPKOptions mpk_opt = sima::mpk::options_from_output_spec(first->observed);
  auto model = sima::mpk::ModelMPK::load(tar_gz, mpk_opt);
  return sima::nodes::groups::Infer(*first, model);
}

int main(int argc, char** argv) {
  constexpr const char* kGoldfishUrl =
      "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
      "n01443537_goldfish.JPEG";
  constexpr int kGoldfishId = 1;

  const bool use_goldfish = has_flag(argc, argv, "--goldfish");
  const bool only_direct = has_flag(argc, argv, "--only-direct");
  const bool only_rtsp = has_flag(argc, argv, "--only-rtsp");
  const bool only_jpeg = has_flag(argc, argv, "--only-jpeg");

  std::string image_path;
  std::string tar_gz;
  std::string tmp;

  if (get_arg(argc, argv, "--image", tmp)) image_path = tmp;
  if (get_arg(argc, argv, "--model", tmp)) tar_gz = tmp;

  if (use_goldfish) {
    const fs::path out_path = default_goldfish_path();
    if (!download_file(kGoldfishUrl, out_path)) {
      std::cerr << "Failed to download goldfish image.\n";
      return 3;
    }
    image_path = out_path.string();
    std::cout << "Using goldfish image: " << image_path << "\n";
  }

  if (image_path.empty()) {
    std::cerr << "Usage: " << argv[0] << " --image <path> --model <tar.gz>\n";
    std::cerr << "       " << argv[0] << " --goldfish --model <tar.gz>\n";
    return 2;
  }
  if (tar_gz.empty()) {
    const char* env = std::getenv("SIMA_RESNET50_TAR");
    if (env && *env) tar_gz = env;
  }
  if (tar_gz.empty()) {
    std::cerr << "Missing model tar.gz (set --model or SIMA_RESNET50_TAR)\n";
    return 2;
  }

  const bool run_direct = only_direct ? true : (!only_rtsp && !only_jpeg);
  const bool run_jpeg = only_jpeg ? true : (!only_rtsp && !only_direct);
  const bool run_rtsp = only_rtsp ? true : (!only_jpeg && !only_direct);

  const int expected_id = use_goldfish ? kGoldfishId : -1;
  const float min_prob = use_goldfish ? 0.2f : 0.0f;

  cv::Mat rgb = load_rgb(image_path);
  const int img_w = rgb.cols;
  const int img_h = rgb.rows;

  if (run_direct) {
    auto out = run_direct_debug(tar_gz, rgb);
    auto scores = scores_from_tensor(require_tensor(out, "direct"), "direct");
    check_top1(scores, expected_id, min_prob, "direct");
  }

  if (run_jpeg) {
    auto out = run_image_group_debug(tar_gz, image_path, kInferFps);
    auto scores = scores_from_tensor(require_tensor(out, "jpeg"), "jpeg");
    check_top1(scores, expected_id, min_prob, "jpeg");
  }

  if (run_rtsp) {
    auto ctx = start_rtsp_server(image_path,
                                 img_w, img_h,
                                 img_w, img_h,
                                 kInferFps, kRtspPort);
    auto out = run_rtsp_debug(tar_gz, ctx.handle.url());
    auto scores = scores_from_tensor(require_tensor(out, "rtsp"), "rtsp");
    check_top1(scores, expected_id, min_prob, "rtsp");
  }

  return 0;
}
