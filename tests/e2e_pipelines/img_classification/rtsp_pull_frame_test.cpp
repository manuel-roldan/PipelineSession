#include "nodes/common/AppSink.h"
#include "nodes/groups/RtspInputGroup.h"
#include "pipeline/PipelineSession.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
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

int main(int argc, char** argv) {
  try {
    std::string url;
    get_arg(argc, argv, "--rtsp", url);
    const int timeout_ms = arg_int(argc, argv, "--timeout-ms", 1000);
    const int tries = arg_int(argc, argv, "--tries", 20);
    std::string expect_format;
    get_arg(argc, argv, "--expect-format", expect_format);
    expect_format = upper_copy(expect_format);

    if (url.empty()) {
      throw std::runtime_error("Missing RTSP url. Pass --rtsp <url>.");
    }
    if (!expect_format.empty() && expect_format != "NV12" && expect_format != "I420") {
      throw std::runtime_error("Invalid --expect-format: " + expect_format + " (expected NV12|I420)");
    }

    sima::PipelineSession p;
    sima::nodes::groups::RtspInputGroupOptions ropt;
    ropt.url = url;
    ropt.latency_ms = 200;
    ropt.tcp = true;
    ropt.h264_parse_config_interval = 1;
    p.add(sima::nodes::groups::RtspInputGroup(ropt));
    p.add(sima::nodes::OutputAppSink());

    sima::PipelineRunOptions run_opt;
    run_opt.copy_output = true;
    sima::PipelineRun runner = p.build(run_opt);

    std::optional<sima::RunInputResult> got;
    for (int i = 0; i < tries; ++i) {
      auto out = runner.pull(timeout_ms);
      if (out.has_value()) {
        got = std::move(out);
        break;
      }
    }

    if (!got.has_value()) {
      throw std::runtime_error("rtsp: no output received (timeout/EOS)");
    }

    const sima::RunInputResult& out = *got;
    const sima::NeatTensor* neat = out.neat.has_value() ? &out.neat.value() : nullptr;
    if (!neat) {
      throw std::runtime_error("rtsp: expected NeatTensor output but none was produced");
    }

    std::string actual_format = neat_format(*neat);
    if (actual_format.empty()) {
      actual_format = upper_copy(out.format);
    }

    if (!expect_format.empty() && !format_matches(expect_format, actual_format)) {
      throw std::runtime_error("rtsp: unexpected format: got " + actual_format +
                               " expected " + expect_format);
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
              << " bytes=" << bytes << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 5;
  }
}
