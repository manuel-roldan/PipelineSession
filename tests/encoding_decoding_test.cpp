#include "pipeline/PipelineSession.h"

#include <gst/gst.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>


using namespace sima::nodes;

static void dump_nv12_raw(const sima::FrameNV12& f, const std::string& path_nv12) {
  std::ofstream ofs(path_nv12, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(f.nv12.data()),
            static_cast<std::streamsize>(f.nv12.size()));
  std::cerr << "[DUMP] " << path_nv12 << " (" << f.nv12.size() << " bytes)\n";
}

static cv::Mat nv12_to_bgr(const sima::FrameNV12& f) {
  const size_t expected = (size_t)f.width * (size_t)f.height * 3 / 2;
  if (f.nv12.size() < expected) {
    throw std::runtime_error("FrameNV12 nv12 buffer too small: got " +
                             std::to_string(f.nv12.size()) + " expected >= " +
                             std::to_string(expected));
  }

  cv::Mat yuv(f.height + f.height / 2, f.width, CV_8UC1, (void*)f.nv12.data());
  cv::Mat bgr;
  cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
  return bgr;
}

static sima::FrameNV12 copy_nv12_ref(const sima::FrameNV12Ref& ref) {
  sima::FrameNV12 out;
  out.width = ref.width;
  out.height = ref.height;
  out.pts_ns = ref.pts_ns;
  out.dts_ns = ref.dts_ns;
  out.duration_ns = ref.duration_ns;
  out.keyframe = ref.keyframe;

  const size_t y_bytes = static_cast<size_t>(ref.width) * ref.height;
  const size_t uv_bytes = static_cast<size_t>(ref.width) * ref.height / 2;
  out.nv12.resize(y_bytes + uv_bytes);

  uint8_t* dst = out.nv12.data();
  for (int y = 0; y < ref.height; ++y) {
    std::memcpy(dst + static_cast<size_t>(y) * ref.width,
                ref.y + static_cast<size_t>(y) * ref.y_stride,
                ref.width);
  }
  uint8_t* dst_uv = dst + y_bytes;
  const int uv_h = ref.height / 2;
  for (int y = 0; y < uv_h; ++y) {
    std::memcpy(dst_uv + static_cast<size_t>(y) * ref.width,
                ref.uv + static_cast<size_t>(y) * ref.uv_stride,
                ref.width);
  }

  return out;
}

static void save_bgr(const cv::Mat& bgr, const std::string& path) {
  if (!cv::imwrite(path, bgr)) throw std::runtime_error("OpenCV imwrite failed: " + path);
  std::cerr << "[SAVE] " << path << "\n";
}

struct Metrics {
  double mae = 0.0;
  double mse = 0.0;
  double psnr = 0.0;
  double max_abs = 0.0;
};

static Metrics compare_bgr(const cv::Mat& a, const cv::Mat& b) {
  if (a.empty() || b.empty()) throw std::runtime_error("compare_bgr: empty image");
  if (a.size() != b.size() || a.type() != b.type()) throw std::runtime_error("compare_bgr: mismatch");

  cv::Mat diff;
  cv::absdiff(a, b, diff);

  cv::Scalar mean_abs = cv::mean(diff);
  const double mae = (mean_abs[0] + mean_abs[1] + mean_abs[2]) / 3.0;

  cv::Mat diff_f;
  diff.convertTo(diff_f, CV_32F);
  diff_f = diff_f.mul(diff_f);
  cv::Scalar mean_sq = cv::mean(diff_f);
  const double mse = (mean_sq[0] + mean_sq[1] + mean_sq[2]) / 3.0;

  const double psnr = (mse <= 1e-12) ? 1e9 : (10.0 * std::log10((255.0 * 255.0) / mse));

  cv::Mat diff_gray;
  cv::cvtColor(diff, diff_gray, cv::COLOR_BGR2GRAY);
  double minv = 0.0, maxv = 0.0;
  cv::minMaxLoc(diff_gray, &minv, &maxv);

  return {mae, mse, psnr, maxv};
}

static bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) { out = argv[i + 1]; return true; }
  }
  return false;
}
static int get_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string s;
  if (!get_arg(argc, argv, key, s)) return def;
  return std::stoi(s);
}
static double get_double_arg(int argc, char** argv, const std::string& key, double def) {
  std::string s;
  if (!get_arg(argc, argv, key, s)) return def;
  return std::stod(s);
}

static std::atomic<bool> g_keep_running{true};
static void on_sigint(int) { g_keep_running.store(false); }

static void wait_forever_until_ctrl_c() {
  std::signal(SIGINT, on_sigint);
  std::cerr << "[SERVER] Press Ctrl+C to stop.\n";
  while (g_keep_running.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

struct DecodedResult {
  bool ok = false;
  sima::FrameNV12 frame{};
};

static DecodedResult run_decode_sima_rtsp(const std::string& url) {
  sima::PipelineSessionOptions opt;
  opt.callback_timeout_ms = 12000;
  sima::PipelineSession p(opt);

  p.add(RTSPInput(url, /*latency_ms=*/200, /*tcp=*/true));
  p.add(H264DepayParse());
  p.add(H264Decode(/*sima_allocator_type=*/2, /*out_format=*/"NV12"));
  p.add(OutputAppSink());

  std::this_thread::sleep_for(std::chrono::milliseconds(600));

  // Wait for keyframe (best-effort)
  sima::FrameNV12Ref f{};
  bool got_any = false;
  bool saw_keyframe = false;
  int after_key = 0;

  p.set_frame_callback([&](const sima::FrameNV12Ref& frame) {
    f = frame;
    got_any = true;
    if (frame.keyframe && !saw_keyframe) {
      saw_keyframe = true;
      after_key = 0;
    }
    if (!saw_keyframe) return true;
    if (after_key >= 8) return false;
    ++after_key;
    return true;
  });
  p.run();

  if (!got_any) {
    return {false, {}};
  }
  return {true, copy_nv12_ref(f)};
}

int main(int argc, char** argv) {
  try {
    

    if (argc < 2) {
      std::cerr << "Usage: " << argv[0]
                << " /path/to/static.jpg [--server-only] [--enc-w N] [--enc-h N] [--mae X] [--psnr Y]\n";
      return 2;
    }

    std::string image_path = argv[1];
    bool server_only = false;
    int enc_w_override = -1;
    int enc_h_override = -1;

    for (int i = 2; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--server-only") server_only = true;
      else if (a == "--enc-w" && i + 1 < argc) enc_w_override = std::stoi(argv[++i]);
      else if (a == "--enc-h" && i + 1 < argc) enc_h_override = std::stoi(argv[++i]);
    }

    const double mae_thr = get_double_arg(argc, argv, "--mae", 25.0);
    const double psnr_thr = get_double_arg(argc, argv, "--psnr", 22.0);

    // Content dims for this test
    const int content_w = 224;
    const int content_h = 224;

    // Encoder padding defaults
    int enc_w = (enc_w_override > 0) ? enc_w_override : 256;
    int enc_h = (enc_h_override > 0) ? enc_h_override : 256;

    // -------------------------
    // Server session (RTSP)
    // -------------------------
    sima::PipelineSession s;

    const int fps = 30;

    s.add(AppSrcImage(image_path, content_w, content_h, enc_w, enc_h, fps));
    s.add(H264EncodeSima(enc_w, enc_h, fps, /*bitrate_kbps=*/400, "baseline", "4.0"));
    s.add(H264Parse(/*config_interval=*/1));
    s.add(RtpH264Pay(/*pt=*/96, /*config_interval=*/1));


    auto server = s.run_rtsp({ .mount = "image", .port = 8554 });

    std::cerr << "[INFO] RTSP URL: " << server.url() << "\n";
    std::cerr << "[INFO] Server pipeline: " << s.last_pipeline() << "\n";

    if (server_only) {
      wait_forever_until_ctrl_c();
      server.stop();
      return 0;
    }

    // -------------------------
    // Client session (decode)
    // -------------------------
    DecodedResult dec = run_decode_sima_rtsp(server.url());
    if (!dec.ok) {
      server.stop();
      std::cerr << "[FAIL] Could not decode any frame from RTSP.\n";
      return 3;
    }

    dump_nv12_raw(dec.frame, "decoded_sima.nv12");

    cv::Mat bgr_full = nv12_to_bgr(dec.frame);
    save_bgr(bgr_full, "decoded_sima_full.jpg");

    // Crop back to content if encoded padded
    cv::Mat bgr_crop;
    if (dec.frame.width == enc_w && dec.frame.height == enc_h &&
        (enc_w != content_w || enc_h != content_h)) {

      int off_x = (enc_w - content_w) / 2;
      int off_y = (enc_h - content_h) / 2;
      off_x &= ~1;
      off_y &= ~1;

      bgr_crop = bgr_full(cv::Rect(off_x, off_y, content_w, content_h)).clone();
    } else {
      bgr_crop = bgr_full.clone();
      if (bgr_crop.cols != content_w || bgr_crop.rows != content_h) {
        cv::resize(bgr_crop, bgr_crop, cv::Size(content_w, content_h), 0, 0, cv::INTER_LINEAR);
      }
    }
    save_bgr(bgr_crop, "decoded_sima_crop.jpg");

    // Reference: OpenCV decode + resize to content
    cv::Mat ref = cv::imread(image_path, cv::IMREAD_COLOR);
    if (ref.empty()) {
      server.stop();
      throw std::runtime_error("OpenCV imread failed: " + image_path);
    }
    cv::Mat ref_rs;
    int interp = (content_w < ref.cols || content_h < ref.rows) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(ref, ref_rs, cv::Size(content_w, content_h), 0, 0, interp);
    save_bgr(ref_rs, "reference_content.jpg");

    Metrics m = compare_bgr(bgr_crop, ref_rs);

    std::cerr << "[METRICS] MAE=" << m.mae
              << "  PSNR=" << m.psnr
              << " dB  MaxAbs=" << m.max_abs << "\n";

    if (m.mae > mae_thr || m.psnr < psnr_thr) {
      std::cerr << "[FAIL] Encoding/decoding mismatch: requires MAE <= " << mae_thr
                << " and PSNR >= " << psnr_thr << " dB\n";
      server.stop();
      return 4;
    }

    std::cout << "[OK] encoding_decoding_test passed.\n";
    server.stop();
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "[FATAL] " << e.what() << "\n";
    return 1;
  }
}
