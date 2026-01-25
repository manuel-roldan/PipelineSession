#include "nodes/common/AppSink.h"
#include "nodes/groups/RtspInputGroup.h"
#include "nodes/io/AppSrcImage.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/RtpH264Pay.h"
#include "pipeline/PipelineSession.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

static bool has_arg(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) {
      return true;
    }
  }
  return false;
}

static int arg_int(int argc, char** argv, const std::string& key, int defval) {
  std::string raw;
  if (!get_arg(argc, argv, key, raw)) return defval;
  try {
    return std::stoi(raw);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid " + key + ": " + raw);
  }
}

static std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

static std::string neat_format(const sima::NeatTensor& t) {
  if (!t.semantic.image.has_value()) return {};
  switch (t.semantic.image->format) {
    case sima::NeatImageSpec::PixelFormat::RGB: return "RGB";
    case sima::NeatImageSpec::PixelFormat::BGR: return "BGR";
    case sima::NeatImageSpec::PixelFormat::GRAY8: return "GRAY8";
    case sima::NeatImageSpec::PixelFormat::NV12: return "NV12";
    case sima::NeatImageSpec::PixelFormat::I420: return "I420";
    case sima::NeatImageSpec::PixelFormat::UNKNOWN: return {};
  }
  return {};
}

static size_t plane_bytes(const sima::NeatPlane& p, sima::TensorDType dtype) {
  size_t elem = 1;
  switch (dtype) {
    case sima::TensorDType::UInt8: elem = 1; break;
    case sima::TensorDType::Int8: elem = 1; break;
    case sima::TensorDType::UInt16: elem = 2; break;
    case sima::TensorDType::Int16: elem = 2; break;
    case sima::TensorDType::Int32: elem = 4; break;
    case sima::TensorDType::BFloat16: elem = 2; break;
    case sima::TensorDType::Float32: elem = 4; break;
    case sima::TensorDType::Float64: elem = 8; break;
  }
  if (p.shape.size() >= 2 && !p.strides_bytes.empty()) {
    return static_cast<size_t>(p.strides_bytes[0]) *
           static_cast<size_t>(p.shape[0]);
  }
  if (p.shape.size() >= 2) {
    return static_cast<size_t>(p.shape[0]) *
           static_cast<size_t>(p.shape[1]) * elem;
  }
  if (p.shape.size() == 1) {
    return static_cast<size_t>(p.shape[0]) * elem;
  }
  return 0;
}

static bool format_matches(const std::string& expect, const std::string& actual) {
  if (expect == "NV12") return actual == "NV12";
  if (expect == "I420") return actual == "I420" || actual == "YUV420P" || actual == "YUV420";
  return false;
}

static std::atomic<bool> g_keep_running{true};
static void on_sigint(int) { g_keep_running.store(false); }

static void wait_forever_until_ctrl_c() {
  std::signal(SIGINT, on_sigint);
  std::cerr << "[SERVER] Press Ctrl+C to stop.\n";
  while (g_keep_running.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

int main(int argc, char** argv) {
  sima::RtspServerHandle server;
  bool server_started = false;
  try {
    std::string url;
    get_arg(argc, argv, "--rtsp", url);
    std::string image_path;
    get_arg(argc, argv, "--image", image_path);
    const bool local = has_arg(argc, argv, "--local");
    const bool server_only = has_arg(argc, argv, "--server-only");
    const int timeout_ms = arg_int(argc, argv, "--timeout-ms", 1000);
    const int tries = arg_int(argc, argv, "--tries", 20);
    int h264_fps = arg_int(argc, argv, "--h264-fps", -1);
    int h264_width = arg_int(argc, argv, "--h264-width", -1);
    int h264_height = arg_int(argc, argv, "--h264-height", -1);
    const bool print_pipeline = has_arg(argc, argv, "--print-pipeline");
    const bool diagnose = has_arg(argc, argv, "--diagnose");
    std::string expect_format;
    get_arg(argc, argv, "--expect-format", expect_format);
    expect_format = upper_copy(expect_format);

    if (local && !url.empty()) {
      throw std::runtime_error("Use either --local or --rtsp, not both.");
    }
    if (server_only && !url.empty()) {
      throw std::runtime_error("--server-only cannot be used with --rtsp.");
    }
    if (tries <= 0) {
      throw std::runtime_error("Invalid --tries: must be > 0");
    }
    if (!expect_format.empty() && expect_format != "NV12" && expect_format != "I420") {
      throw std::runtime_error("Invalid --expect-format: " + expect_format + " (expected NV12|I420)");
    }

    if (url.empty() || local) {
      if (image_path.empty()) {
        image_path = "test.jpg";
      }
      if (!std::filesystem::exists(image_path)) {
        throw std::runtime_error(
            "Missing image for local RTSP server. Use --image <path>.");
      }

      const int content_w = 256;
      const int content_h = 256;
      const int enc_w = 256;
      const int enc_h = 256;
      const int fps = 30;

      sima::PipelineSession s;
      s.add(sima::nodes::AppSrcImage(image_path, content_w, content_h, enc_w, enc_h, fps));
      s.add(sima::nodes::H264EncodeSima(enc_w, enc_h, fps, /*bitrate_kbps=*/400, "baseline", "4.0"));
      s.add(sima::nodes::H264Parse(/*config_interval=*/1));
      s.add(sima::nodes::RtpH264Pay(/*pt=*/96, /*config_interval=*/1));

      server = s.run_rtsp({ .mount = "image", .port = 8554 });
      server_started = true;
      url = server.url();
      std::cerr << "[INFO] RTSP URL: " << url << "\n";
      if (print_pipeline) {
        std::cerr << "[INFO] Server pipeline: " << s.last_pipeline() << "\n";
      }

      if (server_only) {
        wait_forever_until_ctrl_c();
        server.stop();
        return 0;
      }
    } else if (server_only) {
      throw std::runtime_error("--server-only requires --local or no --rtsp.");
    }

    sima::PipelineSession p;
    sima::nodes::groups::RtspInputGroupOptions ropt;
    ropt.url = url;
    ropt.latency_ms = 200;
    ropt.tcp = true;
    ropt.h264_parse_config_interval = 1;
    if (h264_fps > 0) ropt.h264_fps = h264_fps;
    if (h264_width > 0) ropt.h264_width = h264_width;
    if (h264_height > 0) ropt.h264_height = h264_height;
    p.add(sima::nodes::groups::RtspInputGroup(ropt));
    p.add(sima::nodes::OutputAppSink());

    if (print_pipeline) {
      std::cout << "[rtsp] pipeline:\n" << p.to_gst() << "\n";
    }

    sima::PipelineRunOptions run_opt;
    run_opt.copy_output = true;
    sima::PipelineRun runner = p.build(run_opt);

    if (!runner.can_pull()) {
      throw std::runtime_error(
          "rtsp: pipeline cannot pull (missing OutputAppSink).");
    }
    if (runner.can_push()) {
      throw std::runtime_error(
          "rtsp: pipeline expects input (AppSrc present). This test only supports RTSP source pipelines.");
    }

    sima::RunOutput out;
    sima::PullError err;
    sima::PullStatus status = sima::PullStatus::Timeout;
    for (int i = 0; i < tries; ++i) {
      status = runner.pull(timeout_ms, out, &err);
      if (status == sima::PullStatus::Timeout) {
        continue;
      }
      break;
    }

    if (status == sima::PullStatus::Timeout) {
      throw std::runtime_error(
          "rtsp: no output received (timeout). Check RTSP server and try "
          "higher --timeout-ms/--tries. Use --print-pipeline/--diagnose for details.");
    }
    if (status == sima::PullStatus::Closed) {
      throw std::runtime_error(
          "rtsp: pipeline closed before any output (EOS/teardown). Check RTSP server or credentials.");
    }
    if (status == sima::PullStatus::Error) {
      std::string msg = err.message.empty() ? "rtsp: pull failed" : err.message;
      if (!err.code.empty()) {
        msg += " (code=" + err.code + ")";
      }
      if (diagnose && err.report.has_value()) {
        msg += "\nreport_json=" + err.report->to_json();
      } else if (diagnose) {
        msg += "\nreport=" + runner.report();
      }
      throw std::runtime_error(msg);
    }

    const sima::NeatTensor* neat = out.neat.has_value() ? &out.neat.value() : nullptr;
    if (!neat) {
      throw std::runtime_error("rtsp: expected NeatTensor output but none was produced");
    }

    std::string actual_format = neat_format(*neat);
    if (actual_format.empty()) {
      actual_format = upper_copy(out.payload_tag);
    }

    if (!expect_format.empty() && !format_matches(expect_format, actual_format)) {
      std::string extra;
      if (!out.caps_string.empty()) {
        extra += " caps=\"" + out.caps_string + "\"";
      }
      if (!out.media_type.empty()) {
        extra += " media=\"" + out.media_type + "\"";
      }
      throw std::runtime_error("rtsp: unexpected format: got " + actual_format +
                               " expected " + expect_format + extra);
    }

    const int64_t h = neat->shape.size() > 0 ? neat->shape[0] : 0;
    const int64_t w = neat->shape.size() > 1 ? neat->shape[1] : 0;
    size_t total = 0;
    for (const auto& p : neat->planes) total += plane_bytes(p, neat->dtype);
    const size_t bytes = neat->storage ? neat->storage->size_bytes : total;
    std::cout << "[rtsp] neat format=" << actual_format
              << " w=" << w
              << " h=" << h
              << " planes=" << neat->planes.size()
              << " bytes=" << bytes;
    if (!out.payload_tag.empty()) {
      std::cout << " tag=" << out.payload_tag;
    }
    if (!out.caps_string.empty()) {
      std::cout << " caps=\"" << out.caps_string << "\"";
    }
    std::cout << "\n";

    runner.close();
    if (server_started) server.stop();
    return 0;
  } catch (const std::exception& e) {
    if (server_started) server.stop();
    std::cerr << "Error: " << e.what() << "\n";
    return 5;
  }
}
