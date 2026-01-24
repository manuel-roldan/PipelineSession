#include "pipeline/PipelineSession.h"

#include <gst/gst.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace sima::nodes;

static std::vector<uint8_t> copy_nv12_from_neat(const sima::NeatTensor& t,
                                                int& out_w,
                                                int& out_h) {
  if (!t.is_nv12()) {
    throw std::runtime_error("Expected NV12 NeatTensor output");
  }
  out_h = t.height();
  out_w = t.width();
  if (out_w <= 0 || out_h <= 0) {
    throw std::runtime_error("NV12 tensor invalid dimensions");
  }
  return t.copy_nv12_contiguous();
}

static cv::Mat nv12_to_bgr(const std::vector<uint8_t>& nv12, int w, int h) {
  const size_t expected = static_cast<size_t>(w) * static_cast<size_t>(h) * 3 / 2;
  if (nv12.size() < expected) {
    throw std::runtime_error("NV12 buffer too small: got " +
                             std::to_string(nv12.size()) + " expected >= " +
                             std::to_string(expected));
  }

  cv::Mat yuv(h + h / 2, w, CV_8UC1, const_cast<uint8_t*>(nv12.data()));
  cv::Mat bgr;
  cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
  return bgr;
}

static void dump_nv12_raw(const std::vector<uint8_t>& nv12,
                          const std::string& path_nv12) {
  std::ofstream ofs(path_nv12, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(nv12.data()),
            static_cast<std::streamsize>(nv12.size()));
  std::cerr << "[DUMP] " << path_nv12 << " (" << nv12.size() << " bytes)\n";
}

static void save_bgr(const cv::Mat& bgr, const std::string& path) {
  if (!cv::imwrite(path, bgr)) {
    throw std::runtime_error("OpenCV imwrite failed: " + path);
  }
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
  if (a.size() != b.size() || a.type() != b.type())
    throw std::runtime_error("compare_bgr: size/type mismatch");

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

static double get_double_arg(int argc, char** argv, const std::string& key, double def) {
  std::string s;
  if (!get_arg(argc, argv, key, s)) return def;
  return std::stod(s);
}

static int run_image_once(const std::string& image_path,
                          int out_w_in, int out_h_in,
                          double mae_thr, double psnr_thr) {
  // Deterministic defaults for CI
  int out_w = (out_w_in > 0) ? out_w_in : 224;
  int out_h = (out_h_in > 0) ? out_h_in : 224;

  // NV12 requires even dims
  out_w &= ~1;
  out_h &= ~1;

  const int fps = 30;
  const int bitrate_kbps = 400;

  sima::PipelineSession p;

  // JPEG -> raw video frames
  p.add(FileSrc(image_path));
  p.add(JpegDec());

  // CRITICAL: turn single image into a timestamped stream
  // Keep it short for ctest, but long enough for encode/decode to start.
  p.gst("imagefreeze num-buffers=45"); // ~1.5s @ 30fps
  p.gst("videorate");

  // Normalize format/size/rate for SIMA encoder contract
  p.add(VideoConvert());
  p.add(VideoScale());
  p.add(CapsNV12SysMem(out_w, out_h, fps));
  p.add(Queue());

  // Encode -> Parse -> Decode (SIMA)
  p.add(H264EncodeSima(out_w, out_h, fps, bitrate_kbps, "baseline", "4.0"));
  p.add(H264Parse(/*config_interval=*/1));
  p.add(H264Decode(/*sima_allocator_type=*/2, /*out_format=*/"NV12"));

  // Make sure appsink pull isn't blocked by upstream scheduling
  p.add(Queue());
  p.add(OutputAppSink());

  bool got = false;
  sima::NeatTensor captured{};
  p.set_tensor_callback([&](const sima::NeatTensor& t) {
    captured = t;
    got = true;
    return false; // first frame is fine for this test
  });
  p.run();

  if (!got) {
    std::cerr << "[ERR] No frame received from image pipeline.\n";
    std::cerr << "[DBG] Pipeline: " << p.last_pipeline() << "\n";
    return 1;
  }

  int out_w = 0;
  int out_h = 0;
  std::vector<uint8_t> nv12 = copy_nv12_from_neat(captured, out_w, out_h);

  // Decode output -> BGR
  cv::Mat dec_bgr = nv12_to_bgr(nv12, out_w, out_h);
  dump_nv12_raw(nv12, "decoded_image.nv12");
  save_bgr(dec_bgr, "decoded_image_full.jpg");

  // Reference via OpenCV decode + resize to decoded size
  cv::Mat ref = cv::imread(image_path, cv::IMREAD_COLOR);
  if (ref.empty()) throw std::runtime_error("OpenCV imread failed: " + image_path);

  const int tgt_w = out_w;
  const int tgt_h = out_h;
  int interp = (tgt_w < ref.cols || tgt_h < ref.rows) ? cv::INTER_AREA : cv::INTER_LINEAR;

  cv::Mat ref_rs;
  cv::resize(ref, ref_rs, cv::Size(tgt_w, tgt_h), 0, 0, interp);

  Metrics m = compare_bgr(dec_bgr, ref_rs);

  std::cerr << "[METRICS] MAE=" << m.mae
            << "  PSNR=" << m.psnr
            << " dB  MaxAbs=" << m.max_abs << "\n";

  if (m.mae > mae_thr || m.psnr < psnr_thr) {
    save_bgr(ref_rs, "reference_resized.jpg");
    std::cerr << "[FAIL] Image mismatch: requires MAE <= " << mae_thr
              << " and PSNR >= " << psnr_thr << " dB\n";
    return 3;
  }

  std::cout << "[OK] file_read_test passed.\n";
  return 0;
}

static int run_video_frames(const std::string& video_path, int nframes) {
  sima::PipelineSession p;

  p.add(FileSrc(video_path));
  p.add(QtDemuxVideoPad(0));
  p.add(Queue());
  p.add(H264ParseAu());
  p.add(H264Decode(2, "NV12"));
  p.add(Queue());
  p.add(OutputAppSink());

  int got = 0;
  std::vector<sima::NeatTensor> frames;
  p.set_tensor_callback([&](const sima::NeatTensor& t) {
    frames.push_back(t);
    ++got;
    return got < nframes;
  });
  p.run();
  for (int i = 0; i < static_cast<int>(frames.size()); ++i) {
    int out_w = 0;
    int out_h = 0;
    std::vector<uint8_t> nv12 = copy_nv12_from_neat(frames[i], out_w, out_h);
    cv::Mat bgr = nv12_to_bgr(nv12, out_w, out_h);
    char path[256];
    std::snprintf(path, sizeof(path), "frame_%03d.jpg", i);
    save_bgr(bgr, path);
  }
  std::cerr << "[DONE] Saved " << got << " frames.\n";
  return (got > 0) ? 0 : 2;
}

int main(int argc, char** argv) {
  try {
    gst_init(nullptr, nullptr);

    if (argc < 3) {
      std::cerr
        << "Usage:\n"
        << "  " << argv[0] << " --image /path/to.jpg [--w N --h N] [--mae X] [--psnr Y]\n"
        << "  " << argv[0] << " --video /path/to.mp4 [--nframes N]\n";
      return 2;
    }

    std::string mode = argv[1];
    std::string path = argv[2];

    int w = -1, h = -1;
    int nframes = 30;

    double mae_thr = get_double_arg(argc, argv, "--mae", 18.0);
    double psnr_thr = get_double_arg(argc, argv, "--psnr", 25.0);

    for (int i = 3; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--w" && i + 1 < argc) w = std::stoi(argv[++i]);
      else if (a == "--h" && i + 1 < argc) h = std::stoi(argv[++i]);
      else if (a == "--nframes" && i + 1 < argc) nframes = std::stoi(argv[++i]);
    }

    if (mode == "--image") {
      return run_image_once(path, w, h, mae_thr, psnr_thr);
    } else if (mode == "--video") {
      return run_video_frames(path, nframes);
    } else {
      std::cerr << "[ERR] Unknown mode: " << mode << "\n";
      return 2;
    }

  } catch (const std::exception& e) {
    std::cerr << "[FATAL] " << e.what() << "\n";
    return 1;
  }
}
