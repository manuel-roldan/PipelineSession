#include "pipeline/PipelineSession.h"
#include "mpk/ModelMPK.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/groups/ImageInputGroup.h"
#include "nodes/groups/RtspInputGroup.h"
#include "nodes/io/AppSrcImage.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/RtpH264Pay.h"

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

static bool env_flag(const char* key) {
  const char* v = std::getenv(key);
  if (!v || !*v) return false;
  return std::string(v) != "0";
}

static int env_int(const char* key, int def_val) {
  const char* v = std::getenv(key);
  if (!v || !*v) return def_val;
  return std::atoi(v);
}

static void usage(const char* prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " <image> [model.tar.gz]\n"
      << "  " << prog << " --image <path> [--model <model.tar.gz>]\n"
      << "  " << prog << " --goldfish [--model <model.tar.gz>]\n"
      << "\n"
      << "Options:\n"
      << "  --goldfish             Download ImageNet goldfish sample and run accuracy check.\n"
      << "  --goldfish-url <url>   Override goldfish image URL.\n"
      << "  --expect-id <int>      Expected top-1 class id (0-based).\n"
      << "  --min-prob <float>     Minimum softmax probability for top-1 when checking accuracy.\n"
      << "  --image <path>         Input image path.\n"
      << "  --model <path>         Model pack tar.gz path.\n"
      << "  --only-direct          Skip JPEG/RTSP paths; run direct model only.\n"
      << "  --only-rtsp            Skip direct and JPEG paths; run RTSP only.\n"
      << "  --only-jpeg            Skip direct and RTSP paths; run JPEG only.\n"
      << "  --skip-direct          Skip the direct InputAppSrc path.\n"
      << "  --skip-jpeg            Skip the JPEG ImageInputGroup path.\n";
}

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

constexpr int kInferWidth = 224;
constexpr int kInferHeight = 224;
constexpr int kInferFps = 30;
constexpr int kJpegInputWidth = 256;
constexpr int kJpegInputHeight = 256;
constexpr int kRtspInputWidth = 256;
constexpr int kRtspInputHeight = 256;
constexpr int kRtspPort = 8557;

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

static sima::FrameTensor require_tensor(const sima::RunInputResult& out,
                                        const std::string& label) {
  if (out.kind != sima::RunOutputKind::Tensor || !out.tensor.has_value()) {
    throw std::runtime_error(label + ": expected tensor output");
  }
  return *out.tensor;
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

  if (expected_id < 0) {
    return;
  }

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

static sima::FrameTensor pull_tensor_with_retry(sima::TensorStream& ts,
                                                const std::string& label,
                                                int per_try_ms,
                                                int tries) {
  for (int i = 0; i < tries; ++i) {
    auto t = ts.next_copy(per_try_ms);
    if (t.has_value()) return *t;
  }
  throw std::runtime_error(label + ": no tensor received (timeout/EOS)");
}

static sima::FrameTensor run_direct_infer(const sima::mpk::ModelMPK& model,
                                          const cv::Mat& rgb,
                                          bool use_debug,
                                          int debug_timeout_ms) {
  sima::PipelineSession p;
  p.add(sima::nodes::InputAppSrc(model.input_appsrc_options(/*tensor_mode=*/false)));
  p.add(sima::nodes::groups::Infer(model));
  p.add(sima::nodes::OutputAppSink());

  if (use_debug) {
    sima::RunDebugOptions opt;
    opt.timeout_ms = debug_timeout_ms;
    auto dbg = p.run_debug(opt, rgb);
    for (const auto& tap : dbg.taps) {
      if (!tap.error.empty()) {
        throw std::runtime_error("run_debug tap '" + tap.name + "' error: " + tap.error);
      }
    }
    std::cout << "[direct] run_debug taps=" << dbg.taps.size() << "\n";
  }

  auto out = p.run(rgb);
  return require_tensor(out, "direct");
}

static sima::FrameTensor run_image_group_infer(const sima::mpk::ModelMPK& model,
                                               const std::string& image_path,
                                               int w,
                                               int h,
                                               int fps,
                                               bool print_pipeline) {
  sima::nodes::groups::ImageInputGroupOptions opt;
  opt.path = image_path;
  opt.imagefreeze_num_buffers = 8;
  opt.fps = fps;
  opt.use_videorate = true;
  opt.use_videoscale = true;
  opt.output_caps.enable = true;
  opt.output_caps.format = "NV12";
  opt.output_caps.width = w;
  opt.output_caps.height = h;
  opt.output_caps.fps = fps;
  opt.output_caps.memory = sima::CapsMemory::Any;
  opt.sima_decoder.enable = true;
  opt.sima_decoder.decoder_name = "decoder";
  opt.sima_decoder.raw_output = true;

  sima::PipelineSession p;
  p.add(sima::nodes::groups::ImageInputGroup(opt));
  p.add(sima::nodes::groups::Infer(model));
  p.add(sima::nodes::OutputAppSink());

  if (print_pipeline) {
    std::cout << "[jpeg] pipeline:\n" << p.to_gst() << "\n";
  }

  auto ts = p.run_tensor();
  auto t = pull_tensor_with_retry(ts, "jpeg_group", 500, 20);
  ts.close();
  return t;
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

static sima::FrameTensor run_rtsp_infer(const sima::mpk::ModelMPK& model,
                                        const std::string& url,
                                        int w,
                                        int h,
                                        int fps,
                                        bool print_pipeline) {
  sima::nodes::groups::RtspInputGroupOptions opt;
  opt.url = url;
  opt.latency_ms = 200;
  opt.tcp = true;
  opt.payload_type = 96;
  opt.use_videoconvert = false;
  opt.use_videoscale = false;
  opt.output_caps.enable = false;
  opt.output_caps.memory = sima::CapsMemory::Any;
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;

  sima::PipelineSession p;
  p.add(sima::nodes::groups::RtspInputGroup(opt));
  p.add(sima::nodes::groups::Infer(model));
  p.add(sima::nodes::OutputAppSink());

  if (print_pipeline) {
    std::cout << "[rtsp] pipeline:\n" << p.to_gst() << "\n";
  }

  auto ts = p.run_tensor();
  auto t = pull_tensor_with_retry(ts, "rtsp", 1000, 20);
  ts.close();
  return t;
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

int main(int argc, char** argv) {
  constexpr const char* kGoldfishUrl =
      "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
      "n01443537_goldfish.JPEG";
  constexpr int kGoldfishId = 1; // ILSVRC2012 0-based index for "goldfish"

  const bool use_goldfish = has_flag(argc, argv, "--goldfish");
  const bool only_direct = has_flag(argc, argv, "--only-direct");
  const bool only_rtsp = has_flag(argc, argv, "--only-rtsp");
  const bool only_jpeg = has_flag(argc, argv, "--only-jpeg");
  const bool skip_direct = has_flag(argc, argv, "--skip-direct");
  const bool skip_jpeg = has_flag(argc, argv, "--skip-jpeg");

  std::string image_path;
  std::string tar_gz;
  std::string goldfish_url = kGoldfishUrl;
  bool have_expected = false;
  int expected_id = -1;
  float min_prob = 0.0f;
  bool have_min_prob = false;

  std::string tmp;
  if (get_arg(argc, argv, "--image", tmp)) image_path = tmp;
  if (get_arg(argc, argv, "--model", tmp)) tar_gz = tmp;
  if (get_arg(argc, argv, "--goldfish-url", tmp)) goldfish_url = tmp;
  if (get_arg(argc, argv, "--expect-id", tmp)) {
    expected_id = std::stoi(tmp);
    have_expected = true;
  }
  if (get_arg(argc, argv, "--min-prob", tmp)) {
    min_prob = std::stof(tmp);
    have_min_prob = true;
  }

  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--image" || arg == "--model" || arg == "--goldfish-url" ||
        arg == "--expect-id" || arg == "--min-prob") {
      ++i;
      continue;
    }
    if (arg == "--goldfish" || arg == "--only-direct" ||
        arg == "--only-rtsp" || arg == "--only-jpeg" ||
        arg == "--skip-direct" || arg == "--skip-jpeg") {
      continue;
    }
    if (!arg.empty() && arg[0] == '-') {
      usage(argv[0]);
      return 2;
    }
    positional.push_back(arg);
  }

  if (image_path.empty() && !use_goldfish && !positional.empty()) {
    image_path = positional[0];
  }
  if (tar_gz.empty() && positional.size() >= 2) {
    tar_gz = positional[1];
  }

  if (use_goldfish) {
    if (!have_expected) {
      expected_id = kGoldfishId;
      have_expected = true;
    }
    if (!have_min_prob) {
      min_prob = 0.2f;
    }

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

  if (image_path.empty()) {
    usage(argv[0]);
    return 2;
  }

  if (tar_gz.empty()) {
    tar_gz = resolve_resnet50_tar();
    if (tar_gz.empty()) {
      std::cerr << "Failed to resolve resnet50 tar.gz. "
                << "Set SIMA_RESNET50_TAR or run 'sima-cli modelzoo get resnet_50'.\n";
      return 3;
    }
  }

  const bool use_debug = env_flag("SIMA_RUN_DEBUG");
  const int debug_timeout_ms =
      env_int("SIMA_RUN_DEBUG_TIMEOUT_MS", sima::RunDebugOptions{}.timeout_ms);

  auto model_rgb = sima::mpk::ModelMPK::load(
      tar_gz,
      sima::mpk::ModelMPKOptions{false, {}, {}, kInferWidth, kInferHeight, "RGB", 0});

  try {
    cv::Mat rgb = load_rgb_resized(image_path, kInferWidth, kInferHeight);

    const bool run_direct = only_direct
        ? true
        : (!only_rtsp && !only_jpeg && !skip_direct);
    const bool run_jpeg = only_jpeg
        ? true
        : (!only_rtsp && !only_direct && !skip_jpeg);
    const bool run_rtsp = only_rtsp
        ? true
        : (!only_jpeg && !only_direct);

    if (use_debug && run_direct) {
      (void)run_direct_infer(model_rgb, rgb, true, debug_timeout_ms);
      return 0;
    }
    if (use_debug && !run_direct) {
      std::cout << "[direct] SIMA_RUN_DEBUG set but direct path disabled.\n";
    }

    const int expect = have_expected ? expected_id : -1;
    const float prob = have_expected ? min_prob : 0.0f;

    if (run_direct) {
      auto direct_t = run_direct_infer(model_rgb, rgb, false, debug_timeout_ms);
      auto direct_scores = scores_from_tensor(direct_t, "direct");
      check_top1(direct_scores, expect, prob, "direct");
    }

    if (run_jpeg) {
      auto model_nv12_jpeg = sima::mpk::ModelMPK::load(
          tar_gz,
          sima::mpk::ModelMPKOptions{
              false, {}, {}, kJpegInputWidth, kJpegInputHeight, "NV12", 0});
      auto jpeg_t = run_image_group_infer(model_nv12_jpeg, image_path,
                                          kJpegInputWidth, kJpegInputHeight, kInferFps,
                                          /*print_pipeline=*/only_jpeg);
      auto jpeg_scores = scores_from_tensor(jpeg_t, "jpeg_group");
      check_top1(jpeg_scores, expect, prob, "jpeg_group");
    }

    if (run_rtsp) {
      auto model_nv12_rtsp = sima::mpk::ModelMPK::load(
          tar_gz,
          sima::mpk::ModelMPKOptions{false, {}, {}, kRtspInputWidth, kRtspInputHeight, "NV12", 0});

      auto rtsp_ctx = start_rtsp_server(image_path,
                                        kRtspInputWidth, kRtspInputHeight,
                                        kRtspInputWidth, kRtspInputHeight,
                                        kInferFps, kRtspPort);
      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      auto rtsp_t = run_rtsp_infer(model_nv12_rtsp, rtsp_ctx.handle.url(),
                                   kRtspInputWidth, kRtspInputHeight, kInferFps,
                                   /*print_pipeline=*/true);
      auto rtsp_scores = scores_from_tensor(rtsp_t, "rtsp");
      check_top1(rtsp_scores, expect, prob, "rtsp");

      rtsp_ctx.handle.stop();
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 5;
  }
}
