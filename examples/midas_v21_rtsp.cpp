#include "example_utils.h"

#include "ModelSession.hpp"
#include "nodes/groups/RtspInputGroup.h"
#include "pipeline/PipelineSession.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) return true;
  }
  return false;
}

int get_int_arg(int argc, char** argv, const std::string& key, int def_val) {
  std::string tmp;
  if (!sima_examples::get_arg(argc, argv, key, tmp)) return def_val;
  try {
    return std::stoi(tmp);
  } catch (...) {
    return def_val;
  }
}

float get_float_arg(int argc, char** argv, const std::string& key, float def_val) {
  std::string tmp;
  if (!sima_examples::get_arg(argc, argv, key, tmp)) return def_val;
  try {
    return std::stof(tmp);
  } catch (...) {
    return def_val;
  }
}

std::vector<float> parse_floats_csv(const std::string& s) {
  std::vector<float> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') {
      if (!cur.empty()) {
        out.push_back(std::stof(cur));
        cur.clear();
      }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(std::stof(cur));
  return out;
}

std::string to_upper(std::string s) {
  for (char& c : s) {
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  }
  return s;
}

std::string resolve_midas_tar() {
  const char* env = std::getenv("SIMA_MIDAS_TAR");
  if (env && *env && fs::exists(env)) return std::string(env);

  const fs::path local = fs::path("tmp") / "midas_v21_small_256_mpk.tar.gz";
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

  const int rc = std::system("sima-cli modelzoo get midas_v21_small_256");
  if (rc != 0) return "";

  if (fs::exists(local)) return local.string();

  const std::vector<fs::path> candidates = {
      "midas_v21_small_256_mpk.tar.gz",
      "midas-v21_small_256_mpk.tar.gz",
      "midas_v21_small_256.tar.gz",
  };
  for (const auto& candidate : candidates) {
    if (fs::exists(candidate) && move_to_tmp(candidate)) {
      return local.string();
    }
  }

  return "";
}

std::string resolve_sample_video() {
  const char* env = std::getenv("SIMA_MIDAS_SAMPLE_VIDEO");
  if (env && *env && fs::exists(env)) return std::string(env);

  const fs::path local = fs::path("tmp") / "midas_sample.mp4";
  if (fs::exists(local)) return local.string();

  const std::string url = "https://sample-videos.com/video123/mp4/240/"
                          "big_buck_bunny_240p_1mb.mp4";
  if (!sima_examples::download_file(url, local)) return "";
  return local.string();
}

size_t dtype_bytes(sima::TensorDType dtype) {
  switch (dtype) {
    case sima::TensorDType::UInt8:
    case sima::TensorDType::Int8:
      return 1;
    case sima::TensorDType::UInt16:
    case sima::TensorDType::Int16:
      return 2;
    case sima::TensorDType::Int32:
    case sima::TensorDType::Float32:
      return 4;
    case sima::TensorDType::Float64:
      return 8;
  }
  return 1;
}

float read_elem(const uint8_t* data, size_t idx, sima::TensorDType dtype) {
  switch (dtype) {
    case sima::TensorDType::UInt8:
      return static_cast<float>(reinterpret_cast<const uint8_t*>(data)[idx]);
    case sima::TensorDType::Int8:
      return static_cast<float>(reinterpret_cast<const int8_t*>(data)[idx]);
    case sima::TensorDType::UInt16:
      return static_cast<float>(reinterpret_cast<const uint16_t*>(data)[idx]);
    case sima::TensorDType::Int16:
      return static_cast<float>(reinterpret_cast<const int16_t*>(data)[idx]);
    case sima::TensorDType::Int32:
      return static_cast<float>(reinterpret_cast<const int32_t*>(data)[idx]);
    case sima::TensorDType::Float32:
      return reinterpret_cast<const float*>(data)[idx];
    case sima::TensorDType::Float64:
      return static_cast<float>(reinterpret_cast<const double*>(data)[idx]);
  }
  return 0.0f;
}

bool depth_tensor_to_u8(const sima::FrameTensor& t,
                        int fallback_w,
                        int fallback_h,
                        cv::Mat& depth_u8,
                        float* out_min,
                        float* out_max) {
  if (t.planes.empty()) return false;

  int w = t.width > 0 ? t.width : fallback_w;
  int h = t.height > 0 ? t.height : fallback_h;
  if ((w <= 0 || h <= 0) && t.shape.size() >= 2) {
    h = static_cast<int>(t.shape[0]);
    w = static_cast<int>(t.shape[1]);
  }
  if (w <= 0 || h <= 0) return false;

  int depth = 1;
  if (t.shape.size() >= 3) {
    depth = static_cast<int>(t.shape[2]);
  }

  const size_t elem_size = dtype_bytes(t.dtype);
  const size_t needed = static_cast<size_t>(w) * static_cast<size_t>(h) *
                        static_cast<size_t>(depth) * elem_size;
  if (t.planes[0].size() < needed) return false;

  const uint8_t* data = t.planes[0].data();
  cv::Mat depth_f(h, w, CV_32FC1);

  float minv = std::numeric_limits<float>::infinity();
  float maxv = -std::numeric_limits<float>::infinity();

  for (int y = 0; y < h; ++y) {
    float* row = depth_f.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(w) +
                          static_cast<size_t>(x)) * static_cast<size_t>(depth);
      float v = read_elem(data, idx, t.dtype);
      row[x] = v;
      minv = std::min(minv, v);
      maxv = std::max(maxv, v);
    }
  }

  if (out_min) *out_min = minv;
  if (out_max) *out_max = maxv;

  if (std::isfinite(minv) && std::isfinite(maxv) && maxv > minv) {
    cv::normalize(depth_f, depth_u8, 0, 255, cv::NORM_MINMAX);
    depth_u8.convertTo(depth_u8, CV_8U);
  } else {
    depth_u8 = cv::Mat(h, w, CV_8U, cv::Scalar(0));
  }
  return true;
}

bool tensor_to_depth_bgr(const sima::FrameTensor& t,
                         int fallback_w,
                         int fallback_h,
                         bool use_colormap,
                         cv::Mat& bgr_out,
                         float* out_min,
                         float* out_max) {
  cv::Mat depth_u8;
  if (!depth_tensor_to_u8(t, fallback_w, fallback_h, depth_u8, out_min, out_max)) {
    return false;
  }

  if (use_colormap) {
    cv::applyColorMap(depth_u8, bgr_out, cv::COLORMAP_TURBO);
  } else {
    cv::cvtColor(depth_u8, bgr_out, cv::COLOR_GRAY2BGR);
  }
  return true;
}

bool tensor_ref_to_bgr(const sima::FrameTensorRef& t, cv::Mat& bgr_out) {
  if (t.planes.empty()) return false;
  if (t.format != "BGR") return false;
  const auto& plane = t.planes[0];
  const int stride = plane.stride > 0 ? plane.stride : t.width * 3;
  bgr_out = cv::Mat(t.height, t.width, CV_8UC3,
                    const_cast<uint8_t*>(plane.data),
                    stride);
  return true;
}

bool process_stream(simaai::ModelSession& session,
                    const std::function<bool(cv::Mat&)>& next_frame,
                    cv::VideoWriter& writer,
                    int max_frames,
                    float alpha,
                    bool use_colormap) {
  int frames = 0;
  for (; frames < max_frames; ++frames) {
    cv::Mat frame;
    if (!next_frame(frame)) break;

    if (frame.empty()) continue;
    if (frame.type() != CV_8UC3) {
      cv::Mat converted;
      frame.convertTo(converted, CV_8UC3);
      frame = converted;
    }

    sima::FrameTensor t = session.run_tensor(frame);

    cv::Mat depth_bgr;
    float minv = 0.0f;
    float maxv = 0.0f;
    if (!tensor_to_depth_bgr(t, frame.cols, frame.rows, use_colormap, depth_bgr, &minv, &maxv)) {
      std::cerr << "Failed to convert depth tensor\n";
      return false;
    }

    if (depth_bgr.size() != frame.size()) {
      cv::resize(depth_bgr, depth_bgr, frame.size(), 0, 0, cv::INTER_NEAREST);
    }

    const double a = std::max(0.0f, std::min(1.0f, alpha));
    cv::Mat blended;
    cv::addWeighted(frame, 1.0 - a, depth_bgr, a, 0.0, blended);
    writer.write(blended);

    if (frames == 0 || (frames + 1) % 10 == 0) {
      std::cout << "frame " << (frames + 1) << " depth range: ["
                << minv << ", " << maxv << "]\n";
    }
  }

  std::cout << "Processed " << frames << " frames\n";
  return true;
}

void usage(const char* prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " --url <rtsp://...> [options]\n"
      << "  " << prog << " --self-test [options]\n"
      << "\n"
      << "Options:\n"
      << "  --model <tar.gz>     Path to midas_v21_small_256 MPK tar.gz\n"
      << "  --out <file.mp4>     Output mp4 path (default: midas_depth.mp4)\n"
      << "  --frames <n>         Number of frames to record (default: 120)\n"
      << "  --width <n>          Input width (default: 256)\n"
      << "  --height <n>         Input height (default: 256)\n"
      << "  --fps <n>            Output fps (default: 30)\n"
      << "  --alpha <f>          Overlay alpha (default: 0.5)\n"
      << "  --format <BGR>       Input format (default: BGR)\n"
      << "  --colormap           Apply color map to depth output\n"
      << "  --video <file.mp4>   Local video for --self-test\n"
      << "  --normalize          Enable normalization in ModelSession\n"
      << "  --mean <a,b,c>       Channel mean when --normalize is set\n"
      << "  --std <a,b,c>        Channel stddev when --normalize is set\n"
      << "  --latency <ms>       RTSP latency (default: 200)\n"
      << "  --tcp                Force TCP (default)\n"
      << "  --udp                Use UDP\n";
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  const bool self_test = has_flag(argc, argv, "--self-test");

  std::string url;
  sima_examples::get_arg(argc, argv, "--url", url);

  if (!self_test && url.empty()) {
    usage(argv[0]);
    return 2;
  }

  std::string tar_gz;
  sima_examples::get_arg(argc, argv, "--model", tar_gz);
  if (tar_gz.empty()) {
    tar_gz = resolve_midas_tar();
    if (tar_gz.empty()) {
      std::cerr << "Missing midas_v21_small_256 MPK tarball.\n";
      std::cerr << "Set SIMA_MIDAS_TAR or run 'sima-cli modelzoo get midas_v21_small_256'.\n";
      return 3;
    }
  }

  std::string out_path = "midas_depth.mp4";
  sima_examples::get_arg(argc, argv, "--out", out_path);

  const int frames = get_int_arg(argc, argv, "--frames", 120);
  const int width = get_int_arg(argc, argv, "--width", 256);
  const int height = get_int_arg(argc, argv, "--height", 256);
  const int fps = get_int_arg(argc, argv, "--fps", 30);
  const float alpha = get_float_arg(argc, argv, "--alpha", 0.5f);
  const int latency_ms = get_int_arg(argc, argv, "--latency", 200);
  const bool use_colormap = has_flag(argc, argv, "--colormap");
  const bool tcp = !has_flag(argc, argv, "--udp");

  std::string format = "BGR";
  sima_examples::get_arg(argc, argv, "--format", format);
  format = to_upper(format);
  if (format != "BGR") {
    std::cerr << "Only BGR input is supported in this example. Got: " << format << "\n";
    return 9;
  }

  const bool normalize = has_flag(argc, argv, "--normalize");
  std::vector<float> mean;
  std::vector<float> stddev;
  std::string mean_arg;
  std::string std_arg;
  if (sima_examples::get_arg(argc, argv, "--mean", mean_arg)) {
    mean = parse_floats_csv(mean_arg);
  }
  if (sima_examples::get_arg(argc, argv, "--std", std_arg)) {
    stddev = parse_floats_csv(std_arg);
  }

  std::cout << "Using model: " << tar_gz << "\n";
  if (!self_test) std::cout << "RTSP url: " << url << "\n";
  std::cout << "Output: " << out_path << "\n";

  simaai::ModelSession session;
  if (!session.init(tar_gz, width, height, "BGR", normalize, mean, stddev)) {
    std::cerr << "ModelSession init failed: " << session.last_error() << "\n";
    return 4;
  }

  cv::VideoWriter writer;
  writer.open(out_path,
              cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
              fps,
              cv::Size(width, height),
              true);
  if (!writer.isOpened()) {
    std::cerr << "Failed to open VideoWriter: " << out_path << "\n";
    return 5;
  }

  bool ok = false;
  if (self_test) {
    std::string video_path;
    sima_examples::get_arg(argc, argv, "--video", video_path);
    if (video_path.empty()) {
      video_path = resolve_sample_video();
    }
    if (video_path.empty()) {
      std::cerr << "Failed to resolve sample video.\n";
      return 6;
    }
    std::cout << "Self-test video: " << video_path << "\n";

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
      std::cerr << "Failed to open video: " << video_path << "\n";
      return 7;
    }

    auto next_frame = [&](cv::Mat& frame) -> bool {
      cv::Mat bgr;
      if (!cap.read(bgr)) return false;
      if (bgr.empty()) return false;
      if (bgr.cols != width || bgr.rows != height) {
        cv::resize(bgr, bgr, cv::Size(width, height), 0, 0, cv::INTER_AREA);
      }
      frame = bgr;
      return true;
    };

    ok = process_stream(session, next_frame, writer, frames, alpha, use_colormap);
  } else {
    sima::nodes::groups::RtspInputGroupOptions ro;
    ro.url = url;
    ro.latency_ms = latency_ms;
    ro.tcp = tcp;
    ro.payload_type = 96;
    ro.insert_queue = true;
    ro.out_format = "BGR";
    ro.decoder_raw_output = false;
    ro.decoder_name = "decoder";
    ro.use_videoconvert = false;
    ro.use_videoscale = true;
    ro.output_caps.enable = true;
    ro.output_caps.format = "BGR";
    ro.output_caps.width = width;
    ro.output_caps.height = height;
    ro.output_caps.fps = fps;
    ro.output_caps.memory = sima::CapsMemory::SystemMemory;

    sima::PipelineSession p;
    p.add(sima::nodes::groups::RtspInputGroup(ro));
    p.add(sima::nodes::OutputAppSink(sima::OutputAppSinkOptions::EveryFrame(5)));

    auto ts = p.run_tensor();
    auto next_frame = [&](cv::Mat& frame) -> bool {
      auto ref_opt = ts.next(/*timeout_ms=*/2000);
      if (!ref_opt.has_value()) return false;
      cv::Mat view;
      if (!tensor_ref_to_bgr(*ref_opt, view)) return false;
      frame = view.clone();
      return true;
    };

    ok = process_stream(session, next_frame, writer, frames, alpha, use_colormap);
    ts.close();
  }

  writer.release();
  session.close();

  if (!ok) {
    std::cerr << "Processing failed.\n";
    return 8;
  }

  std::cout << "Wrote depth overlay video to: " << out_path << "\n";
  return 0;
}
