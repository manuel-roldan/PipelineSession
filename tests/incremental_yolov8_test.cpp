#include "pipeline/PipelineSession.h"
#include "pipeline/AsyncStream.h"
#include "nodes/common/AppSink.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/InputAppSrc.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "mpk/ModelMPK.h"

#include "example_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct RunResult {
  bool ok = false;
  double latency_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double mean_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  int latency_samples = 0;
  sima::InputStreamStats stats{};
  std::string pipeline;
  std::string diag;
  std::string note;
  std::string out_format;
  std::string out_caps;
  std::string out_dtype;
  int out_width = 0;
  int out_height = 0;
  bool byte_match = false;
  bool solid_match = false;
};

struct ThroughputResult {
  bool ok = false;
  int inputs = 0;
  int outputs = 0;
  double avg_latency_ms = 0.0;
  double min_latency_ms = 0.0;
  double max_latency_ms = 0.0;
  double avg_fps = 0.0;
  std::string pipeline;
  std::string note;
};

const char* dtype_name(sima::TensorDType dtype) {
  switch (dtype) {
    case sima::TensorDType::UInt8:
      return "UInt8";
    case sima::TensorDType::Int8:
      return "Int8";
    case sima::TensorDType::UInt16:
      return "UInt16";
    case sima::TensorDType::Int16:
      return "Int16";
    case sima::TensorDType::Int32:
      return "Int32";
    case sima::TensorDType::Float32:
      return "Float32";
    case sima::TensorDType::Float64:
      return "Float64";
  }
  return "Unknown";
}

struct PreprocConfig {
  bool aspect_ratio = false;
  bool normalize = false;
  int input_width = 0;
  int input_height = 0;
  int output_width = 0;
  int output_height = 0;
  int scaled_width = 0;
  int scaled_height = 0;
  std::string scaling_type;
  std::string padding_type;
  std::string output_img_type;
  std::string output_dtype;
  double q_scale = 1.0;
  int q_zp = 0;
};

PreprocConfig preproc_config_from_options(const sima::PreprocOptions& opt) {
  PreprocConfig cfg;
  cfg.aspect_ratio = opt.aspect_ratio;
  cfg.normalize = opt.normalize;
  cfg.input_width = opt.input_width;
  cfg.input_height = opt.input_height;
  cfg.output_width = opt.output_width;
  cfg.output_height = opt.output_height;
  cfg.scaled_width = opt.scaled_width;
  cfg.scaled_height = opt.scaled_height;
  cfg.scaling_type = opt.scaling_type;
  cfg.padding_type = opt.padding_type;
  cfg.output_img_type = opt.output_img_type;
  cfg.output_dtype = opt.output_dtype;
  cfg.q_scale = opt.q_scale;
  cfg.q_zp = opt.q_zp;
  return cfg;
}

struct LatencySummary {
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  int samples = 0;
};

LatencySummary summarize_latencies(const std::vector<double>& latencies) {
  LatencySummary s;
  if (latencies.empty()) return s;
  s.samples = static_cast<int>(latencies.size());
  s.min_ms = *std::min_element(latencies.begin(), latencies.end());
  s.max_ms = *std::max_element(latencies.begin(), latencies.end());
  double sum = 0.0;
  for (double v : latencies) sum += v;
  s.mean_ms = sum / static_cast<double>(latencies.size());
  std::vector<double> sorted = latencies;
  std::sort(sorted.begin(), sorted.end());
  auto at_pct = [&](double pct) -> double {
    if (sorted.empty()) return 0.0;
    const double pos = (pct / 100.0) * (sorted.size() - 1);
    const size_t idx = static_cast<size_t>(std::lround(pos));
    return sorted[std::min(idx, sorted.size() - 1)];
  };
  s.p50_ms = at_pct(50.0);
  s.p95_ms = at_pct(95.0);
  return s;
}

bool env_bool(const char* key, bool def = false) {
  const char* v = std::getenv(key);
  if (!v) return def;
  return std::string(v) != "0";
}

int env_int(const char* key, int def) {
  const char* v = std::getenv(key);
  if (!v || !*v) return def;
  return std::atoi(v);
}

std::string env_str(const char* key, const std::string& def) {
  const char* v = std::getenv(key);
  if (!v) return def;
  return std::string(v);
}

sima::DropPolicy parse_drop_policy(const std::string& value) {
  std::string v;
  v.reserve(value.size());
  for (char c : value) {
    v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  if (v == "drop_newest" || v == "newest") return sima::DropPolicy::DropNewest;
  if (v == "drop_oldest" || v == "oldest") return sima::DropPolicy::DropOldest;
  return sima::DropPolicy::Block;
}

enum class FullPostMode {
  DetessDequant,
  Boxdecode,
  None,
};

enum class TputMode {
  Default,
  Preproc,
  FullBoxdecode,
  FullDetessDequant,
  FullNone,
};

std::string lower_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

FullPostMode parse_full_post_mode(const std::string& value) {
  const std::string v = lower_copy(value);
  if (v == "boxdecode" || v == "bbox") return FullPostMode::Boxdecode;
  if (v == "none" || v == "mla" || v == "mla_only") return FullPostMode::None;
  return FullPostMode::DetessDequant;
}

TputMode parse_tput_mode(const std::string& value) {
  if (value.empty()) return TputMode::Default;
  const std::string v = lower_copy(value);
  if (v == "preproc") return TputMode::Preproc;
  if (v == "full_boxdecode" || v == "boxdecode") return TputMode::FullBoxdecode;
  if (v == "full_detessdequant" || v == "detessdequant" || v == "detess") {
    return TputMode::FullDetessDequant;
  }
  if (v == "full_none" || v == "none" || v == "mla" || v == "mla_only") {
    return TputMode::FullNone;
  }
  return TputMode::Default;
}

bool tput_mode_is_full(TputMode mode) {
  return mode == TputMode::FullBoxdecode ||
         mode == TputMode::FullDetessDequant ||
         mode == TputMode::FullNone;
}

const char* full_post_mode_name(FullPostMode mode) {
  switch (mode) {
    case FullPostMode::DetessDequant:
      return "DetessDequant";
    case FullPostMode::Boxdecode:
      return "Boxdecode";
    case FullPostMode::None:
      return "MLA";
  }
  return "Unknown";
}

bool fragment_mentions_detessdequant(const sima::mpk::ModelMPK& model) {
  const auto frag = model.fragment(sima::mpk::ModelStage::Postprocess);
  std::string combined = lower_copy(frag.gst);
  for (const auto& elem : frag.elements) {
    combined += " ";
    combined += lower_copy(elem);
  }
  return combined.find("detess") != std::string::npos ||
         combined.find("dequant") != std::string::npos;
}

PreprocConfig read_preproc_config(const fs::path& json_path) {
  std::ifstream in(json_path);
  require(in.is_open(), "Failed to open preproc config: " + json_path.string());
  nlohmann::json j;
  in >> j;

  PreprocConfig cfg;
  cfg.aspect_ratio = j.value("aspect_ratio", false);
  cfg.normalize = j.value("normalize", false);
  cfg.input_width = j.value("input_width", 0);
  cfg.input_height = j.value("input_height", 0);
  cfg.output_width = j.value("output_width", 0);
  cfg.output_height = j.value("output_height", 0);
  cfg.scaled_width = j.value("scaled_width", 0);
  cfg.scaled_height = j.value("scaled_height", 0);
  cfg.scaling_type = j.value("scaling_type", std::string("BILINEAR"));
  cfg.padding_type = j.value("padding_type", std::string("CENTER"));
  cfg.output_img_type = j.value("output_img_type", std::string("RGB"));
  cfg.output_dtype = j.value("output_dtype", std::string());
  cfg.q_scale = j.value("q_scale", 1.0);
  cfg.q_zp = j.value("q_zp", 0);
  return cfg;
}

bool patch_preproc_rgb_only(const fs::path& json_path, bool rgb_only) {
  std::ifstream in(json_path);
  if (!in.is_open()) return false;
  nlohmann::json j;
  in >> j;
  j["normalize"] = false;
  j["channel_mean"] = nlohmann::json::array({0.0, 0.0, 0.0});
  j["channel_stddev"] = nlohmann::json::array({1.0, 1.0, 1.0});
  if (rgb_only) {
    j["output_memory_order"] =
        nlohmann::json::array({"output_rgb_image", "output_tessellated_image"});
  }
  std::ofstream out(json_path);
  if (!out.is_open()) return false;
  out << j.dump(4);
  return true;
}

int cv_interp_for_scaling(const std::string& scaling) {
  if (scaling == "NEAREST") return cv::INTER_NEAREST;
  if (scaling == "BICUBIC") return cv::INTER_CUBIC;
  if (scaling == "AREA") return cv::INTER_AREA;
  return cv::INTER_LINEAR;
}

const char* interp_name(int interp) {
  switch (interp) {
    case cv::INTER_NEAREST:
      return "nearest";
    case cv::INTER_LINEAR:
      return "linear";
    case cv::INTER_AREA:
      return "area";
    case cv::INTER_CUBIC:
      return "cubic";
    case cv::INTER_LINEAR_EXACT:
      return "linear_exact";
    default:
      return "interp";
  }
}

cv::Mat build_preproc_reference(const cv::Mat& input_rgb,
                                const PreprocConfig& cfg,
                                int pad_value,
                                int interp_override = -1) {
  const int out_w = cfg.output_width > 0 ? cfg.output_width : 640;
  const int out_h = cfg.output_height > 0 ? cfg.output_height : 640;

  cv::Mat resized;
  const int interp = (interp_override >= 0)
      ? interp_override
      : cv_interp_for_scaling(cfg.scaling_type);
  if (cfg.aspect_ratio) {
    const double scale_w = static_cast<double>(out_w) / input_rgb.cols;
    const double scale_h = static_cast<double>(out_h) / input_rgb.rows;
    const double scale = std::min(scale_w, scale_h);
    int scaled_w = static_cast<int>(std::lround(input_rgb.cols * scale));
    int scaled_h = static_cast<int>(std::lround(input_rgb.rows * scale));

    const bool json_scaled_ok =
        cfg.scaled_width > 0 && cfg.scaled_height > 0 &&
        cfg.scaled_width <= out_w && cfg.scaled_height <= out_h &&
        (cfg.scaled_width * input_rgb.rows == cfg.scaled_height * input_rgb.cols);
    if (json_scaled_ok) {
      scaled_w = cfg.scaled_width;
      scaled_h = cfg.scaled_height;
    }

    cv::resize(input_rgb, resized, cv::Size(scaled_w, scaled_h),
               0.0, 0.0, interp);
    cv::Mat out(out_h, out_w, CV_8UC3, cv::Scalar(pad_value, pad_value, pad_value));
    int pad_x = (out_w - scaled_w) / 2;
    int pad_y = (out_h - scaled_h) / 2;
    resized.copyTo(out(cv::Rect(pad_x, pad_y, scaled_w, scaled_h)));
    return out;
  }

  cv::resize(input_rgb, resized, cv::Size(out_w, out_h),
             0.0, 0.0, interp);
  return resized;
}

void fill_output_meta(const sima::RunInputResult& out, RunResult& res) {
  res.out_caps = out.caps_string;
  if (out.tensor.has_value()) {
    const auto& t = out.tensor.value();
    res.out_format = t.format;
    res.out_width = t.width;
    res.out_height = t.height;
    res.out_dtype = dtype_name(t.dtype);
  } else if (out.tensor_ref.has_value()) {
    const auto& t = out.tensor_ref.value();
    res.out_format = t.format;
    res.out_width = t.width;
    res.out_height = t.height;
    res.out_dtype = dtype_name(t.dtype);
  } else if (out.frame_nv12.has_value()) {
    const auto& f = out.frame_nv12.value();
    res.out_format = "NV12";
    res.out_width = f.width;
    res.out_height = f.height;
  }
}

bool compare_rgb_tensor(const cv::Mat& img,
                        const sima::FrameTensor& t,
                        const std::string& expected_format,
                        std::string& err) {
  if (t.dtype != sima::TensorDType::UInt8) {
    err = std::string("Output tensor dtype mismatch: expected UInt8 got ") +
          dtype_name(t.dtype);
    return false;
  }
  if (t.format != expected_format) {
    err = "Output tensor format mismatch: expected " + expected_format +
          " got " + t.format;
    return false;
  }
  if (t.width != img.cols || t.height != img.rows) {
    err = "Output tensor size mismatch";
    return false;
  }
  if (t.planes.empty()) {
    err = "Output tensor has no planes";
    return false;
  }
  const auto& plane = t.planes[0];
  const int row_bytes = img.cols * img.channels();
  if (static_cast<int>(plane.size()) < row_bytes * img.rows) {
    err = "Output tensor plane too small";
    return false;
  }
  const uint8_t* in = img.data;
  const uint8_t* out = plane.data();
  for (int r = 0; r < img.rows; ++r) {
    if (std::memcmp(in + static_cast<size_t>(r) * img.step,
                    out + static_cast<size_t>(r) * row_bytes,
                    static_cast<size_t>(row_bytes)) != 0) {
      err = "Output tensor does not match input bytes";
      return false;
    }
  }
  return true;
}

bool compare_preproc_tensor(const sima::FrameTensor& t,
                            const cv::Mat& expected,
                            std::string& err) {
  if (t.format != "RGB") {
    err = "Preproc output format mismatch: expected RGB got " + t.format;
    return false;
  }
  if (t.dtype != sima::TensorDType::UInt8) {
    err = std::string("Preproc output dtype mismatch: expected UInt8 got ") +
          dtype_name(t.dtype);
    return false;
  }
  if (t.width != expected.cols || t.height != expected.rows) {
    err = "Preproc output size mismatch";
    return false;
  }
  if (t.planes.empty()) {
    err = "Preproc output tensor has no planes";
    return false;
  }
  const auto& plane = t.planes[0];
  const int row_bytes = expected.cols * expected.channels();
  if (static_cast<int>(plane.size()) < row_bytes * expected.rows) {
    err = "Preproc output tensor plane too small";
    return false;
  }
  for (int r = 0; r < expected.rows; ++r) {
    const uint8_t* exp_row = expected.ptr<uint8_t>(r);
    const uint8_t* out_row = plane.data() + static_cast<size_t>(r) * row_bytes;
    if (std::memcmp(exp_row, out_row, static_cast<size_t>(row_bytes)) != 0) {
      err = "Preproc output bytes do not match reference";
      return false;
    }
  }
  return true;
}

struct DiffStats {
  size_t mismatches = 0;
  int max_abs = 0;
  double mean_abs = 0.0;
};

struct MatchResult {
  bool matched = false;
  bool best_set = false;
  std::string match_label;
  std::string best_label;
  std::string error;
  DiffStats best_stats;
};

DiffStats diff_preproc_tensor(const sima::FrameTensor& t,
                              const cv::Mat& expected,
                              std::string& err) {
  DiffStats stats;
  if (t.format != "RGB" || t.dtype != sima::TensorDType::UInt8) {
    err = "diff_preproc_tensor: unsupported tensor format/dtype";
    return stats;
  }
  if (t.width != expected.cols || t.height != expected.rows) {
    err = "diff_preproc_tensor: size mismatch";
    return stats;
  }
  if (t.planes.empty()) {
    err = "diff_preproc_tensor: missing planes";
    return stats;
  }
  const auto& plane = t.planes[0];
  const int row_bytes = expected.cols * expected.channels();
  if (static_cast<int>(plane.size()) < row_bytes * expected.rows) {
    err = "diff_preproc_tensor: plane too small";
    return stats;
  }
  uint64_t sum_abs = 0;
  const size_t total = static_cast<size_t>(row_bytes) * expected.rows;
  for (int r = 0; r < expected.rows; ++r) {
    const uint8_t* exp_row = expected.ptr<uint8_t>(r);
    const uint8_t* out_row = plane.data() + static_cast<size_t>(r) * row_bytes;
    for (int c = 0; c < row_bytes; ++c) {
      const int diff = static_cast<int>(exp_row[c]) - static_cast<int>(out_row[c]);
      const int abs_diff = diff < 0 ? -diff : diff;
      if (abs_diff != 0) {
        stats.mismatches++;
        if (abs_diff > stats.max_abs) stats.max_abs = abs_diff;
        sum_abs += static_cast<uint64_t>(abs_diff);
      }
    }
  }
  stats.mean_abs = total ? static_cast<double>(sum_abs) / static_cast<double>(total) : 0.0;
  return stats;
}

MatchResult match_preproc_output(const sima::FrameTensor& tensor,
                                 const cv::Mat& input_rgb,
                                 const PreprocConfig& cfg,
                                 int pad_value) {
  MatchResult out;
  cv::Mat input_bgr;
  const cv::Mat* input_ptr = &input_rgb;

  std::vector<int> interps = {
      cv::INTER_NEAREST,
      cv::INTER_LINEAR,
      cv::INTER_AREA,
      cv::INTER_CUBIC,
      cv::INTER_LINEAR_EXACT,
  };

  std::vector<bool> aspect_opts = {cfg.aspect_ratio, !cfg.aspect_ratio};

  for (bool aspect_ratio : aspect_opts) {
    for (bool use_bgr : {false, true}) {
      if (use_bgr) {
        if (input_bgr.empty()) {
          cv::cvtColor(input_rgb, input_bgr, cv::COLOR_RGB2BGR);
        }
        input_ptr = &input_bgr;
      } else {
        input_ptr = &input_rgb;
      }
      for (int interp : interps) {
        PreprocConfig cfg_local = cfg;
        cfg_local.aspect_ratio = aspect_ratio;
        cv::Mat ref = build_preproc_reference(*input_ptr, cfg_local, pad_value, interp);
        std::string err;
        if (compare_preproc_tensor(tensor, ref, err)) {
          out.matched = true;
          out.match_label = aspect_ratio ? "aspect_ratio" : "stretch";
          out.match_label += use_bgr ? " BGR " : " RGB ";
          out.match_label += interp_name(interp);
          return out;
        }

        std::string diff_err;
        DiffStats diff = diff_preproc_tensor(tensor, ref, diff_err);
        if (!diff_err.empty()) {
          out.error = diff_err;
        } else if (!out.best_set || diff.mean_abs < out.best_stats.mean_abs) {
          out.best_set = true;
          out.best_stats = diff;
          out.best_label = aspect_ratio ? "aspect_ratio" : "stretch";
          out.best_label += use_bgr ? " BGR " : " RGB ";
          out.best_label += interp_name(interp);
        }
      }
    }
  }

  return out;
}

RunResult run_baseline_noop(const cv::Mat& img, bool use_pool) {
  RunResult res;

  sima::PipelineSession p;
  sima::InputAppSrcOptions src_opt;
  src_opt.format = "RGB";
  src_opt.width = 640;
  src_opt.height = 640;
  src_opt.is_live = true;
  src_opt.do_timestamp = true;
  src_opt.block = true;
  src_opt.use_simaai_pool = use_pool;
  src_opt.pool_min_buffers = 1;
  src_opt.pool_max_buffers = 2;
  const size_t frame_bytes = static_cast<size_t>(src_opt.width) *
                             static_cast<size_t>(src_opt.height) * 3;
  src_opt.max_bytes = static_cast<std::uint64_t>(frame_bytes);

  p.add(sima::nodes::InputAppSrc(src_opt));
  // Identity to mirror contracts used in other run paths.
  p.gst("identity silent=true");
  sima::OutputAppSinkOptions sink_opt;
  sink_opt.sync = false;
  sink_opt.drop = false;
  sink_opt.max_buffers = 1;
  p.add(sima::nodes::OutputAppSink(sink_opt));

  sima::InputStreamOptions opt;
  opt.timeout_ms = env_int("SIMA_INPUT_TIMEOUT_MS", 20000);
  opt.appsink_sync = false;
  opt.appsink_drop = false;
  opt.appsink_max_buffers = 1;
  opt.copy_output = true;
  opt.reuse_input_buffer = false;
  opt.enable_timings = true;
  opt.allow_mismatched_input = false;

  auto stream = p.run_input_stream(img, opt);
  const auto t0 = std::chrono::steady_clock::now();
  auto out = stream.push_and_pull(img, opt.timeout_ms);
  const auto t1 = std::chrono::steady_clock::now();
  std::vector<double> latencies;
  latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  const LatencySummary summary = summarize_latencies(latencies);
  res.latency_ms = summary.mean_ms;
  res.mean_ms = summary.mean_ms;
  res.p50_ms = summary.p50_ms;
  res.p95_ms = summary.p95_ms;
  res.min_ms = summary.min_ms;
  res.max_ms = summary.max_ms;
  res.latency_samples = summary.samples;
  res.stats = stream.stats();
  res.pipeline = p.to_gst(false);
  res.diag = stream.diagnostics_summary();
  fill_output_meta(out, res);
  if (out.kind == sima::RunOutputKind::Tensor && out.tensor.has_value()) {
    std::string err;
    res.ok = compare_rgb_tensor(img, out.tensor.value(), "RGB", err);
    if (!res.ok) res.note = err;
  } else {
    res.ok = false;
    res.note = "Unexpected output kind in baseline";
  }
  stream.close();
  return res;
}

RunResult run_baseline_noop_run_api(const cv::Mat& img, bool use_pool) {
  RunResult res;

  sima::PipelineSession p;
  sima::InputAppSrcOptions src_opt;
  src_opt.format = "RGB";
  src_opt.width = 640;
  src_opt.height = 640;
  src_opt.is_live = true;
  src_opt.do_timestamp = true;
  src_opt.block = true;
  src_opt.use_simaai_pool = use_pool;
  src_opt.pool_min_buffers = 1;
  src_opt.pool_max_buffers = 2;
  const size_t frame_bytes = static_cast<size_t>(src_opt.width) *
                             static_cast<size_t>(src_opt.height) * 3;
  src_opt.max_bytes = static_cast<std::uint64_t>(frame_bytes);

  p.add(sima::nodes::InputAppSrc(src_opt));
  p.gst("identity silent=true");
  sima::OutputAppSinkOptions sink_opt;
  sink_opt.sync = false;
  sink_opt.drop = false;
  sink_opt.max_buffers = 1;
  p.add(sima::nodes::OutputAppSink(sink_opt));

  sima::RunInputOptions run_opt;
  run_opt.copy_output = true;
  run_opt.reuse_input_buffer = false;
  run_opt.strict = true;

  const auto t_build0 = std::chrono::steady_clock::now();
  p.build(img, run_opt);
  const auto t_build1 = std::chrono::steady_clock::now();
  const auto t0 = std::chrono::steady_clock::now();
  auto out = p.run(img, run_opt);
  const auto t1 = std::chrono::steady_clock::now();
  std::vector<double> latencies;
  latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  const LatencySummary summary = summarize_latencies(latencies);
  res.latency_ms = summary.mean_ms;
  res.mean_ms = summary.mean_ms;
  res.p50_ms = summary.p50_ms;
  res.p95_ms = summary.p95_ms;
  res.min_ms = summary.min_ms;
  res.max_ms = summary.max_ms;
  res.latency_samples = summary.samples;
  const double build_ms =
      std::chrono::duration<double, std::milli>(t_build1 - t_build0).count();
  res.note = "build_ms=" + std::to_string(build_ms);
  res.stats = p.last_input_stats();
  res.pipeline = p.last_pipeline();
  res.diag = p.last_input_diagnostics();
  fill_output_meta(out, res);

  const bool is_video = out.kind == sima::RunOutputKind::FrameNV12 ||
                        out.kind == sima::RunOutputKind::Tensor;
  if (!is_video) {
    res.ok = false;
    res.note = "Unexpected output kind in run(input)";
  } else if (out.tensor.has_value()) {
    std::string err;
    res.ok = compare_rgb_tensor(img, out.tensor.value(), "RGB", err);
    if (!res.ok) res.note = err;
  } else {
    res.ok = false;
    res.note = "run(input) output missing tensor copy";
  }
  return res;
}

RunResult run_full_detessdequant(const fs::path& root,
                                 const cv::Mat& img,
                                 bool use_pool) {
  RunResult res;

  const std::string tar_gz = sima_examples::resolve_yolov8s_tar(root);
  require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

  sima::OutputSpec input_spec;
  input_spec.media_type = "video/x-raw";
  input_spec.format = "BGR";
  input_spec.width = img.cols;
  input_spec.height = img.rows;
  input_spec.depth = 3;

  sima::mpk::ModelMPKOptions mpk_opt = sima::mpk::options_from_output_spec(input_spec);
  mpk_opt.fast_mode = true;
  mpk_opt.num_buffers_cvu = 1;
  mpk_opt.num_buffers_mla = 1;
  mpk_opt.upstream_name = "decoder";

  auto model = sima::mpk::ModelMPK::load(tar_gz, mpk_opt);

  sima::PipelineSession p;
  sima::InputAppSrcOptions src_opt;
  src_opt.format = "BGR";
  src_opt.width = img.cols;
  src_opt.height = img.rows;
  src_opt.depth = 3;
  src_opt.is_live = true;
  src_opt.do_timestamp = true;
  src_opt.block = true;
  src_opt.use_simaai_pool = use_pool;
  src_opt.pool_min_buffers = 1;
  src_opt.pool_max_buffers = 2;
  src_opt.buffer_name = "decoder";
  src_opt.max_bytes = static_cast<std::uint64_t>(img.total() * img.elemSize());

  p.add(sima::nodes::InputAppSrc(src_opt));
  p.add(sima::nodes::groups::Preprocess(model));
  p.add(sima::nodes::groups::MLA(model));
  // Postprocess: detessdequant placeholder via ModelStage::Postprocess (matches legacy flow)
  p.add(sima::nodes::groups::Postprocess(model));
  sima::OutputAppSinkOptions sink_opt;
  sink_opt.sync = false;
  sink_opt.drop = true;
  sink_opt.max_buffers = 1;
  p.add(sima::nodes::OutputAppSink(sink_opt));

  sima::InputStreamOptions opt;
  opt.timeout_ms = env_int("SIMA_INPUT_TIMEOUT_MS", 20000);
  opt.appsink_sync = false;
  opt.appsink_drop = true;
  opt.appsink_max_buffers = 1;
  opt.copy_output = true;
  opt.reuse_input_buffer = false;
  opt.enable_timings = true;

  auto stream = p.run_input_stream(img, opt);
  const auto t0 = std::chrono::steady_clock::now();
  auto out = stream.push_and_pull(img, opt.timeout_ms);
  const auto t1 = std::chrono::steady_clock::now();
  res.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  res.stats = stream.stats();
  res.pipeline = p.to_gst(false);
  res.diag = stream.diagnostics_summary();
  res.ok = (out.kind == sima::RunOutputKind::Tensor ||
            out.kind == sima::RunOutputKind::FrameNV12);
  if (!res.ok) {
    res.note = "Unexpected output kind in full pipeline";
  }
  stream.close();
  return res;
}

RunResult run_preproc_to_appsink_apu(const fs::path& root,
                                     const cv::Mat& img,
                                     bool use_pool,
                                     bool use_standalone) {
  RunResult res;
  int cvu_buffers = env_int("SIMA_PREPROC_NUM_BUFFERS", 5);
  if (cvu_buffers <= 0) cvu_buffers = 5;
  PreprocConfig pre_cfg;

  sima::PipelineSession p;
  sima::InputAppSrcOptions src_opt;
  src_opt.format = "RGB";
  src_opt.width = img.cols;
  src_opt.height = img.rows;
  src_opt.depth = 3;
  src_opt.is_live = true;
  src_opt.do_timestamp = true;
  src_opt.block = false;
  src_opt.use_simaai_pool = use_pool;
  src_opt.pool_min_buffers = cvu_buffers;
  src_opt.pool_max_buffers = cvu_buffers;
  src_opt.buffer_name = "decoder";
  src_opt.max_bytes = 0;

  p.add(sima::nodes::InputAppSrc(src_opt));
  if (use_standalone) {
    sima::PreprocOptions pre_opt;
    pre_opt.input_width = img.cols;
    pre_opt.input_height = img.rows;
    pre_opt.output_width = 640;
    pre_opt.output_height = 640;
    pre_opt.scaled_width = 640;
    pre_opt.scaled_height = 640;
    pre_opt.input_channels = 3;
    pre_opt.output_channels = 3;
    pre_opt.input_img_type = "RGB";
    pre_opt.output_img_type = "RGB";
    pre_opt.normalize = env_bool("SIMA_PREPROC_NORMALIZE", false);
    pre_opt.aspect_ratio = env_bool("SIMA_PREPROC_ASPECT", true);
    pre_opt.scaling_type = "BILINEAR";
    pre_opt.padding_type = "CENTER";
    pre_opt.output_dtype = "EVXX_INT8";
    pre_opt.q_scale = 255.06967737092486;
    pre_opt.q_zp = -128;
    pre_opt.next_cpu = env_str("SIMA_PREPROC_NEXT_CPU", "APU");
    pre_opt.upstream_name = "decoder";
    pre_opt.num_buffers = cvu_buffers;
    pre_opt.config_dir = env_str("SIMA_PREPROC_CONFIG_DIR", (root / "tmp").string());
    pre_opt.keep_config = env_bool("SIMA_PREPROC_KEEP_CONFIG", false);
    if (env_bool("SIMA_PREPROC_RGB_ONLY", true)) {
      pre_opt.output_memory_order = {"output_rgb_image", "output_tessellated_image"};
    }
    pre_cfg = preproc_config_from_options(pre_opt);
    p.add(sima::nodes::Preproc(pre_opt));
  } else {
    const std::string tar_gz = sima_examples::resolve_yolov8s_tar(root);
    require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

    sima::OutputSpec input_spec;
    input_spec.media_type = "video/x-raw";
    input_spec.format = "RGB";
    input_spec.width = img.cols;
    input_spec.height = img.rows;
    input_spec.depth = 3;

    sima::mpk::ModelMPKOptions mpk_opt = sima::mpk::options_from_output_spec(input_spec);
    mpk_opt.preproc_next_cpu = env_str("SIMA_PREPROC_NEXT_CPU", "APU");
    mpk_opt.normalize = env_bool("SIMA_PREPROC_NORMALIZE", false);
    mpk_opt.fast_mode = false;
    mpk_opt.num_buffers_cvu = cvu_buffers;
    mpk_opt.disable_internal_queues = false;
    mpk_opt.upstream_name = "decoder";

    auto model = sima::mpk::ModelMPK::load(tar_gz, mpk_opt);
    const fs::path preproc_json = fs::path(model.etc_dir()) / "0_preproc.json";
    const bool rgb_only = env_bool("SIMA_PREPROC_RGB_ONLY", true);
    if (!patch_preproc_rgb_only(preproc_json, rgb_only)) {
      throw std::runtime_error("Failed to patch preproc JSON for rgb-only output");
    }
    pre_cfg = read_preproc_config(preproc_json);
    if (mpk_opt.num_buffers_cvu > 0) {
      cvu_buffers = mpk_opt.num_buffers_cvu;
    }
    p.add(sima::nodes::groups::Preprocess(model));
  }
  sima::OutputAppSinkOptions sink_opt;
  sink_opt.sync = false;
  sink_opt.drop = true;
  sink_opt.max_buffers = 1;
  p.add(sima::nodes::OutputAppSink(sink_opt));

  sima::InputStreamOptions stream_opt;
  stream_opt.timeout_ms = env_int("SIMA_INPUT_TIMEOUT_MS", 20000);
  stream_opt.appsink_sync = false;
  stream_opt.appsink_drop = true;
  stream_opt.appsink_max_buffers = 1;
  stream_opt.copy_output = true;
  stream_opt.reuse_input_buffer = false;
  stream_opt.enable_timings = true;
  stream_opt.allow_mismatched_input = false;

  const auto t_build0 = std::chrono::steady_clock::now();
  auto stream = p.run_input_stream(img, stream_opt);
  const auto t_build1 = std::chrono::steady_clock::now();

  const int warm = env_int("SIMA_PREPROC_WARM", 2);
  int iters = env_int("SIMA_PREPROC_ITERS", 10);
  if (iters < 1) iters = 1;
  for (int i = 0; i < warm; ++i) {
    (void)stream.push_and_pull(img, stream_opt.timeout_ms);
  }

  sima::RunInputResult out;
  std::vector<double> latencies;
  latencies.reserve(static_cast<size_t>(iters));
  try {
    for (int i = 0; i < iters; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      auto out_i = stream.push_and_pull(img, stream_opt.timeout_ms);
      const auto t1 = std::chrono::steady_clock::now();
      latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      if (i == 0) out = std::move(out_i);
    }
  } catch (const std::exception& e) {
    res.ok = false;
    res.note = e.what();
    res.stats = stream.stats();
    res.pipeline = p.to_gst(false);
    res.diag = stream.diagnostics_summary();
    stream.close();
    return res;
  }
  const LatencySummary summary = summarize_latencies(latencies);
  res.latency_ms = summary.mean_ms;
  res.mean_ms = summary.mean_ms;
  res.p50_ms = summary.p50_ms;
  res.p95_ms = summary.p95_ms;
  res.min_ms = summary.min_ms;
  res.max_ms = summary.max_ms;
  res.latency_samples = summary.samples;
  const double build_ms =
      std::chrono::duration<double, std::milli>(t_build1 - t_build0).count();
  res.note = "build_ms=" + std::to_string(build_ms) +
             " warm=" + std::to_string(warm) +
             " iters=" + std::to_string(iters);
  res.stats = stream.stats();
  res.pipeline = p.to_gst(false);
  res.diag = stream.diagnostics_summary();
  fill_output_meta(out, res);

  auto append_note = [&](const std::string& msg) {
    if (!res.note.empty()) res.note += "; ";
    res.note += msg;
  };

  res.ok = (res.out_width == 640 && res.out_height == 640);
  if (!res.ok) {
    append_note("Unexpected preproc output size");
  } else if (!res.out_format.empty() && res.out_format != "RGB") {
    res.ok = false;
    append_note("Unexpected preproc output format: " + res.out_format);
  } else if (pre_cfg.normalize) {
    res.ok = false;
    append_note("Preproc normalize=true (expected false)");
  } else if (!pre_cfg.normalize) {
    // normalize=false should keep output in uint8 range.
    if (res.out_dtype != "UInt8") {
      res.ok = false;
      append_note("Unexpected preproc dtype with normalize=false: " + res.out_dtype);
    }
  }
  const bool basic_ok = res.ok;

  sima::FrameTensor owned_tensor;
  const sima::FrameTensor* tensor = nullptr;
  if (out.tensor.has_value()) {
    tensor = &out.tensor.value();
  } else if (out.tensor_ref.has_value()) {
    owned_tensor = out.tensor_ref.value().to_copy();
    tensor = &owned_tensor;
  }

  if (tensor) {
    int pad_value = 0;
    if (pre_cfg.aspect_ratio) {
      if (!tensor->planes.empty()) {
        pad_value = tensor->planes[0].empty() ? 0 : tensor->planes[0][0];
      }
    }
    MatchResult match = match_preproc_output(*tensor, img, pre_cfg, pad_value);
    if (match.matched) {
      res.byte_match = true;
      append_note("Matched reference: " + match.match_label);
    } else {
      res.ok = false;
      if (match.best_set) {
        std::ostringstream oss;
        oss << "Closest reference (" << match.best_label << "): mean_abs_diff="
            << match.best_stats.mean_abs << " max_abs_diff=" << match.best_stats.max_abs
            << " mismatches=" << match.best_stats.mismatches;
        append_note(oss.str());
      } else if (!match.error.empty()) {
        append_note(match.error);
      }
    }

    if (env_bool("SIMA_PREPROC_SOLID_TEST", true)) {
      try {
        cv::Mat solid(img.rows, img.cols, CV_8UC3, cv::Scalar(64, 128, 192));
        if (!solid.isContinuous()) solid = solid.clone();
        auto solid_out = stream.push_and_pull(solid, stream_opt.timeout_ms);
        sima::FrameTensor solid_tensor;
        const sima::FrameTensor* solid_ptr = nullptr;
        if (solid_out.tensor.has_value()) {
          solid_ptr = &solid_out.tensor.value();
        } else if (solid_out.tensor_ref.has_value()) {
          solid_tensor = solid_out.tensor_ref.value().to_copy();
          solid_ptr = &solid_tensor;
        }
        if (solid_ptr) {
          int solid_pad = 0;
          if (pre_cfg.aspect_ratio && !solid_ptr->planes.empty()) {
            solid_pad = solid_ptr->planes[0].empty() ? 0 : solid_ptr->planes[0][0];
          }
          MatchResult solid_match = match_preproc_output(*solid_ptr, solid, pre_cfg, solid_pad);
          res.solid_match = solid_match.matched;
          append_note(std::string("solid_match=") + (res.solid_match ? "true" : "false"));
        }
      } catch (const std::exception& e) {
        append_note(std::string("solid_test_error=") + e.what());
      }
    }

    if (!res.byte_match && res.solid_match && basic_ok) {
      res.ok = true;
      append_note("Using solid_match for byte-exact validation");
    }
  } else {
    res.ok = false;
    append_note("Preproc output missing tensor data");
  }
  return res;
}

ThroughputResult run_preproc_to_appsink_async(const fs::path& root,
                                              const cv::Mat& img,
                                              bool use_pool,
                                              bool use_standalone) {
  ThroughputResult res;

  int cvu_buffers = env_int("SIMA_PREPROC_NUM_BUFFERS", 5);
  if (cvu_buffers <= 0) cvu_buffers = 5;

  PreprocConfig pre_cfg;

  sima::PipelineSession p;
  sima::InputAppSrcOptions src_opt;
  src_opt.format = "RGB";
  src_opt.width = img.cols;
  src_opt.height = img.rows;
  src_opt.depth = 3;
  src_opt.is_live = true;
  src_opt.do_timestamp = true;
  src_opt.block = true;
  src_opt.use_simaai_pool = use_pool;
  src_opt.pool_min_buffers = cvu_buffers;
  src_opt.pool_max_buffers = cvu_buffers;
  src_opt.buffer_name = "decoder";
  src_opt.max_bytes = 0;

  p.add(sima::nodes::InputAppSrc(src_opt));

  if (use_standalone) {
    sima::PreprocOptions pre_opt;
    pre_opt.input_width = img.cols;
    pre_opt.input_height = img.rows;
    pre_opt.output_width = 640;
    pre_opt.output_height = 640;
    pre_opt.scaled_width = 640;
    pre_opt.scaled_height = 640;
    pre_opt.input_channels = 3;
    pre_opt.output_channels = 3;
    pre_opt.input_img_type = "RGB";
    pre_opt.output_img_type = "RGB";
    pre_opt.normalize = env_bool("SIMA_PREPROC_NORMALIZE", false);
    pre_opt.aspect_ratio = env_bool("SIMA_PREPROC_ASPECT", true);
    pre_opt.scaling_type = "BILINEAR";
    pre_opt.padding_type = "CENTER";
    pre_opt.output_dtype = "EVXX_INT8";
    pre_opt.q_scale = 255.06967737092486;
    pre_opt.q_zp = -128;
    pre_opt.next_cpu = env_str("SIMA_PREPROC_NEXT_CPU", "APU");
    pre_opt.upstream_name = "decoder";
    pre_opt.num_buffers = cvu_buffers;
    pre_opt.config_dir = env_str("SIMA_PREPROC_CONFIG_DIR", (root / "tmp").string());
    pre_opt.keep_config = env_bool("SIMA_PREPROC_KEEP_CONFIG", false);
    if (env_bool("SIMA_PREPROC_RGB_ONLY", true)) {
      pre_opt.output_memory_order = {"output_rgb_image", "output_tessellated_image"};
    }
    pre_cfg = preproc_config_from_options(pre_opt);
    p.add(sima::nodes::Preproc(pre_opt));
  } else {
    const std::string tar_gz = sima_examples::resolve_yolov8s_tar(root);
    require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

    sima::OutputSpec input_spec;
    input_spec.media_type = "video/x-raw";
    input_spec.format = "RGB";
    input_spec.width = img.cols;
    input_spec.height = img.rows;
    input_spec.depth = 3;

    sima::mpk::ModelMPKOptions mpk_opt = sima::mpk::options_from_output_spec(input_spec);
    mpk_opt.preproc_next_cpu = env_str("SIMA_PREPROC_NEXT_CPU", "APU");
    mpk_opt.normalize = env_bool("SIMA_PREPROC_NORMALIZE", false);
    mpk_opt.fast_mode = false;
    mpk_opt.num_buffers_cvu = cvu_buffers;
    mpk_opt.disable_internal_queues = false;
    mpk_opt.upstream_name = "decoder";

    auto model = sima::mpk::ModelMPK::load(tar_gz, mpk_opt);
    const fs::path preproc_json = fs::path(model.etc_dir()) / "0_preproc.json";
    const bool rgb_only = env_bool("SIMA_PREPROC_RGB_ONLY", true);
    if (!patch_preproc_rgb_only(preproc_json, rgb_only)) {
      throw std::runtime_error("Failed to patch preproc JSON for rgb-only output");
    }
    pre_cfg = read_preproc_config(preproc_json);
    if (mpk_opt.num_buffers_cvu > 0) {
      cvu_buffers = mpk_opt.num_buffers_cvu;
    }
    p.add(sima::nodes::groups::Preprocess(model));
  }

  sima::OutputAppSinkOptions sink_opt;
  sink_opt.sync = false;
  sink_opt.drop = false;
  sink_opt.max_buffers = env_int("SIMA_ASYNC_OUTQ", 8);
  p.add(sima::nodes::OutputAppSink(sink_opt));

  sima::AsyncOptions async_opt;
  async_opt.input_queue = env_int("SIMA_ASYNC_INQ", 8);
  async_opt.output_queue = env_int("SIMA_ASYNC_OUTQ", 8);
  async_opt.drop = parse_drop_policy(env_str("SIMA_ASYNC_DROP", "block"));
  async_opt.copy_output = env_bool("SIMA_ASYNC_COPY_OUT", false);
  async_opt.copy_input = env_bool("SIMA_ASYNC_COPY_IN", false);
  async_opt.allow_mismatched_input = false;
  async_opt.timeout_ms = -1;

  const int iters = env_int("SIMA_ASYNC_ITERS", 200);
  if (iters <= 0) {
    res.note = "SIMA_ASYNC_ITERS <= 0";
    return res;
  }

  auto async = p.run_async(img, async_opt);

  int outputs = 0;
  RunResult meta_out;
  bool got_meta = false;
  std::mutex meta_mu;
  std::string consumer_error;
  const int pull_timeout = env_int("SIMA_ASYNC_TIMEOUT_MS", 20000);

  std::thread consumer([&]() {
    try {
      while (true) {
        auto out = async.pull(pull_timeout);
        if (!out.has_value()) break;
        outputs += 1;
        if (!got_meta) {
          std::lock_guard<std::mutex> lock(meta_mu);
          if (!got_meta) {
            fill_output_meta(out.value(), meta_out);
            got_meta = true;
          }
        }
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(meta_mu);
      consumer_error = e.what();
    }
  });

  const auto t0 = std::chrono::steady_clock::now();
  int pushed = 0;
  for (int i = 0; i < iters; ++i) {
    if (async.push(img)) pushed += 1;
  }
  async.close_input();
  consumer.join();
  const auto t1 = std::chrono::steady_clock::now();
  const double elapsed_s =
      std::chrono::duration<double>(t1 - t0).count();

  const sima::AsyncStats st = async.stats();
  res.inputs = pushed;
  res.outputs = outputs;
  res.avg_latency_ms = st.avg_latency_ms;
  res.min_latency_ms = st.min_latency_ms;
  res.max_latency_ms = st.max_latency_ms;
  res.avg_fps = (elapsed_s > 0.0) ? (static_cast<double>(outputs) / elapsed_s) : 0.0;
  res.pipeline = p.to_gst(false);
  res.ok = (outputs == static_cast<int>(st.inputs_pushed));
  if (!consumer_error.empty()) {
    if (!res.note.empty()) res.note += "; ";
    res.note += "async_pull_error=" + consumer_error;
    res.ok = false;
  }

  if (!pre_cfg.normalize && got_meta && !meta_out.out_dtype.empty()) {
    if (meta_out.out_dtype != "UInt8") {
      res.note = "Unexpected async dtype with normalize=false: " + meta_out.out_dtype;
    }
  }

  return res;
}

ThroughputResult run_preproc_to_appsink_sync_tput(const fs::path& root,
                                                  const cv::Mat& img,
                                                  bool use_pool,
                                                  bool use_standalone,
                                                  int iters,
                                                  int warm) {
  ThroughputResult res;

  int cvu_buffers = env_int("SIMA_PREPROC_NUM_BUFFERS", 5);
  if (cvu_buffers <= 0) cvu_buffers = 5;

  sima::PipelineSession p;
  sima::InputAppSrcOptions src_opt;
  src_opt.format = "RGB";
  src_opt.width = img.cols;
  src_opt.height = img.rows;
  src_opt.depth = 3;
  src_opt.is_live = true;
  src_opt.do_timestamp = true;
  src_opt.block = true;
  src_opt.use_simaai_pool = use_pool;
  src_opt.pool_min_buffers = cvu_buffers;
  src_opt.pool_max_buffers = cvu_buffers;
  src_opt.buffer_name = "decoder";
  src_opt.max_bytes = 0;

  p.add(sima::nodes::InputAppSrc(src_opt));

  if (use_standalone) {
    sima::PreprocOptions pre_opt;
    pre_opt.input_width = img.cols;
    pre_opt.input_height = img.rows;
    pre_opt.output_width = 640;
    pre_opt.output_height = 640;
    pre_opt.scaled_width = 640;
    pre_opt.scaled_height = 640;
    pre_opt.input_channels = 3;
    pre_opt.output_channels = 3;
    pre_opt.input_img_type = "RGB";
    pre_opt.output_img_type = "RGB";
    pre_opt.normalize = env_bool("SIMA_PREPROC_NORMALIZE", false);
    pre_opt.aspect_ratio = env_bool("SIMA_PREPROC_ASPECT", true);
    pre_opt.scaling_type = "BILINEAR";
    pre_opt.padding_type = "CENTER";
    pre_opt.output_dtype = "EVXX_INT8";
    pre_opt.q_scale = 255.06967737092486;
    pre_opt.q_zp = -128;
    pre_opt.next_cpu = env_str("SIMA_PREPROC_NEXT_CPU", "APU");
    pre_opt.upstream_name = "decoder";
    pre_opt.num_buffers = cvu_buffers;
    pre_opt.config_dir = env_str("SIMA_PREPROC_CONFIG_DIR", (root / "tmp").string());
    pre_opt.keep_config = env_bool("SIMA_PREPROC_KEEP_CONFIG", false);
    if (env_bool("SIMA_PREPROC_RGB_ONLY", true)) {
      pre_opt.output_memory_order = {"output_rgb_image", "output_tessellated_image"};
    }
    p.add(sima::nodes::Preproc(pre_opt));
  } else {
    const std::string tar_gz = sima_examples::resolve_yolov8s_tar(root);
    require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

    sima::OutputSpec input_spec;
    input_spec.media_type = "video/x-raw";
    input_spec.format = "RGB";
    input_spec.width = img.cols;
    input_spec.height = img.rows;
    input_spec.depth = 3;

    sima::mpk::ModelMPKOptions mpk_opt = sima::mpk::options_from_output_spec(input_spec);
    mpk_opt.preproc_next_cpu = env_str("SIMA_PREPROC_NEXT_CPU", "APU");
    mpk_opt.normalize = env_bool("SIMA_PREPROC_NORMALIZE", false);
    mpk_opt.fast_mode = false;
    mpk_opt.num_buffers_cvu = cvu_buffers;
    mpk_opt.disable_internal_queues = false;
    mpk_opt.upstream_name = "decoder";

    auto model = sima::mpk::ModelMPK::load(tar_gz, mpk_opt);
    const fs::path preproc_json = fs::path(model.etc_dir()) / "0_preproc.json";
    const bool rgb_only = env_bool("SIMA_PREPROC_RGB_ONLY", true);
    if (!patch_preproc_rgb_only(preproc_json, rgb_only)) {
      throw std::runtime_error("Failed to patch preproc JSON for rgb-only output");
    }
    if (mpk_opt.num_buffers_cvu > 0) {
      cvu_buffers = mpk_opt.num_buffers_cvu;
    }
    p.add(sima::nodes::groups::Preprocess(model));
  }

  sima::OutputAppSinkOptions sink_opt;
  sink_opt.sync = false;
  sink_opt.drop = false;
  sink_opt.max_buffers = 1;
  p.add(sima::nodes::OutputAppSink(sink_opt));

  sima::InputStreamOptions stream_opt;
  stream_opt.timeout_ms = env_int("SIMA_TPUT_TIMEOUT_MS", 20000);
  stream_opt.appsink_sync = false;
  stream_opt.appsink_drop = false;
  stream_opt.appsink_max_buffers = 1;
  stream_opt.copy_output = false;
  stream_opt.reuse_input_buffer = false;
  stream_opt.enable_timings = false;

  auto stream = p.run_input_stream(img, stream_opt);
  for (int i = 0; i < warm; ++i) {
    (void)stream.push_and_pull(img, stream_opt.timeout_ms);
  }

  double sum_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  int outputs = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    const auto t1 = std::chrono::steady_clock::now();
    (void)stream.push_and_pull(img, stream_opt.timeout_ms);
    const auto t2 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    sum_ms += ms;
    if (i == 0) {
      min_ms = ms;
      max_ms = ms;
    } else {
      min_ms = std::min(min_ms, ms);
      max_ms = std::max(max_ms, ms);
    }
    outputs += 1;
  }
  const auto t3 = std::chrono::steady_clock::now();
  const double elapsed_s =
      std::chrono::duration<double>(t3 - t0).count();

  stream.close();

  res.inputs = iters;
  res.outputs = outputs;
  res.avg_latency_ms = (outputs > 0) ? (sum_ms / outputs) : 0.0;
  res.min_latency_ms = min_ms;
  res.max_latency_ms = max_ms;
  res.avg_fps = (elapsed_s > 0.0) ? (static_cast<double>(outputs) / elapsed_s) : 0.0;
  res.pipeline = p.to_gst(false);
  res.ok = (outputs == iters);
  return res;
}

ThroughputResult run_preproc_to_appsink_async_tput(const fs::path& root,
                                                   const cv::Mat& img,
                                                   bool use_pool,
                                                   bool use_standalone,
                                                   int iters,
                                                   int warm) {
  ThroughputResult res;

  int cvu_buffers = env_int("SIMA_PREPROC_NUM_BUFFERS", 5);
  if (cvu_buffers <= 0) cvu_buffers = 5;

  PreprocConfig pre_cfg;

  sima::PipelineSession p;
  sima::InputAppSrcOptions src_opt;
  src_opt.format = "RGB";
  src_opt.width = img.cols;
  src_opt.height = img.rows;
  src_opt.depth = 3;
  src_opt.is_live = true;
  src_opt.do_timestamp = true;
  src_opt.block = true;
  src_opt.use_simaai_pool = use_pool;
  src_opt.pool_min_buffers = cvu_buffers;
  src_opt.pool_max_buffers = cvu_buffers;
  src_opt.buffer_name = "decoder";
  src_opt.max_bytes = 0;

  p.add(sima::nodes::InputAppSrc(src_opt));

  if (use_standalone) {
    sima::PreprocOptions pre_opt;
    pre_opt.input_width = img.cols;
    pre_opt.input_height = img.rows;
    pre_opt.output_width = 640;
    pre_opt.output_height = 640;
    pre_opt.scaled_width = 640;
    pre_opt.scaled_height = 640;
    pre_opt.input_channels = 3;
    pre_opt.output_channels = 3;
    pre_opt.input_img_type = "RGB";
    pre_opt.output_img_type = "RGB";
    pre_opt.normalize = env_bool("SIMA_PREPROC_NORMALIZE", false);
    pre_opt.aspect_ratio = env_bool("SIMA_PREPROC_ASPECT", true);
    pre_opt.scaling_type = "BILINEAR";
    pre_opt.padding_type = "CENTER";
    pre_opt.output_dtype = "EVXX_INT8";
    pre_opt.q_scale = 255.06967737092486;
    pre_opt.q_zp = -128;
    pre_opt.next_cpu = env_str("SIMA_PREPROC_NEXT_CPU", "APU");
    pre_opt.upstream_name = "decoder";
    pre_opt.num_buffers = cvu_buffers;
    pre_opt.config_dir = env_str("SIMA_PREPROC_CONFIG_DIR", (root / "tmp").string());
    pre_opt.keep_config = env_bool("SIMA_PREPROC_KEEP_CONFIG", false);
    if (env_bool("SIMA_PREPROC_RGB_ONLY", true)) {
      pre_opt.output_memory_order = {"output_rgb_image", "output_tessellated_image"};
    }
    pre_cfg = preproc_config_from_options(pre_opt);
    p.add(sima::nodes::Preproc(pre_opt));
  } else {
    const std::string tar_gz = sima_examples::resolve_yolov8s_tar(root);
    require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

    sima::OutputSpec input_spec;
    input_spec.media_type = "video/x-raw";
    input_spec.format = "RGB";
    input_spec.width = img.cols;
    input_spec.height = img.rows;
    input_spec.depth = 3;

    sima::mpk::ModelMPKOptions mpk_opt = sima::mpk::options_from_output_spec(input_spec);
    mpk_opt.preproc_next_cpu = env_str("SIMA_PREPROC_NEXT_CPU", "APU");
    mpk_opt.normalize = env_bool("SIMA_PREPROC_NORMALIZE", false);
    mpk_opt.fast_mode = false;
    mpk_opt.num_buffers_cvu = cvu_buffers;
    mpk_opt.disable_internal_queues = false;
    mpk_opt.upstream_name = "decoder";

    auto model = sima::mpk::ModelMPK::load(tar_gz, mpk_opt);
    const fs::path preproc_json = fs::path(model.etc_dir()) / "0_preproc.json";
    const bool rgb_only = env_bool("SIMA_PREPROC_RGB_ONLY", true);
    if (!patch_preproc_rgb_only(preproc_json, rgb_only)) {
      throw std::runtime_error("Failed to patch preproc JSON for rgb-only output");
    }
    pre_cfg = read_preproc_config(preproc_json);
    if (mpk_opt.num_buffers_cvu > 0) {
      cvu_buffers = mpk_opt.num_buffers_cvu;
    }
    p.add(sima::nodes::groups::Preprocess(model));
  }

  const int min_q = iters + warm;
  const int inq_env = env_int("SIMA_TPUT_INQ", min_q);
  const int outq_env = env_int("SIMA_TPUT_OUTQ", min_q);
  const int inq = (inq_env > min_q) ? inq_env : min_q;
  const int outq = (outq_env > min_q) ? outq_env : min_q;

  sima::OutputAppSinkOptions sink_opt;
  sink_opt.sync = false;
  sink_opt.drop = false;
  sink_opt.max_buffers = outq;
  p.add(sima::nodes::OutputAppSink(sink_opt));

  sima::AsyncOptions async_opt;
  async_opt.input_queue = inq;
  async_opt.output_queue = outq;
  async_opt.drop = sima::DropPolicy::Block;
  async_opt.copy_output = env_bool("SIMA_ASYNC_COPY_OUT", false);
  async_opt.copy_input = env_bool("SIMA_ASYNC_COPY_IN", false);
  async_opt.allow_mismatched_input = false;
  async_opt.timeout_ms = -1;

  auto async = p.run_async(img, async_opt);

  std::mutex warm_mu;
  std::condition_variable warm_cv;
  bool warm_ready = (warm <= 0);

  std::mutex time_mu;
  bool measured_done = false;
  std::chrono::steady_clock::time_point t_end{};

  std::atomic<int> total_out{0};
  std::atomic<int> measured_out{0};
  std::string consumer_error;
  const int pull_timeout = env_int("SIMA_TPUT_TIMEOUT_MS", 20000);

  std::thread consumer([&]() {
    try {
      while (true) {
        auto out = async.pull(pull_timeout);
        if (!out.has_value()) break;
        const int cur = total_out.fetch_add(1) + 1;
        if (cur == warm && warm > 0) {
          std::lock_guard<std::mutex> lock(warm_mu);
          warm_ready = true;
          warm_cv.notify_one();
        }
        if (cur > warm) {
          const int m = measured_out.fetch_add(1) + 1;
          if (m == iters) {
            std::lock_guard<std::mutex> lock(time_mu);
            if (!measured_done) {
              t_end = std::chrono::steady_clock::now();
              measured_done = true;
            }
          }
        }
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(time_mu);
      consumer_error = e.what();
    }
  });

  for (int i = 0; i < warm; ++i) {
    (void)async.push(img);
  }

  if (warm > 0) {
    std::unique_lock<std::mutex> lock(warm_mu);
    warm_cv.wait_for(lock, std::chrono::seconds(20), [&]() { return warm_ready; });
  }

  const auto t_start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    (void)async.push(img);
  }
  async.close_input();

  consumer.join();

  const bool done = measured_done;
  const auto t_stop = done ? t_end : std::chrono::steady_clock::now();
  const double elapsed_s =
      std::chrono::duration<double>(t_stop - t_start).count();

  const sima::AsyncStats st = async.stats();
  res.inputs = iters;
  res.outputs = measured_out.load();
  res.avg_latency_ms = st.avg_latency_ms;
  res.min_latency_ms = st.min_latency_ms;
  res.max_latency_ms = st.max_latency_ms;
  res.avg_fps = (elapsed_s > 0.0) ? (static_cast<double>(res.outputs) / elapsed_s) : 0.0;
  res.pipeline = p.to_gst(false);
  res.ok = (res.outputs == iters);

  if (!consumer_error.empty()) {
    res.note = "async_pull_error=" + consumer_error;
    res.ok = false;
  }
  if (!done) {
    if (!res.note.empty()) res.note += "; ";
    res.note += "async_timing_incomplete";
    res.ok = false;
  }

  return res;
}

ThroughputResult run_yolov8_full_sync_tput(const fs::path& root,
                                           const cv::Mat& img,
                                           bool use_pool,
                                           int iters,
                                           int warm,
                                           FullPostMode post_mode) {
  ThroughputResult res;
  const bool enable_timings = env_bool("SIMA_TPUT_FULL_TIMINGS", false);

  const std::string tar_gz = sima_examples::resolve_yolov8s_tar(root);
  require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

  sima::OutputSpec input_spec;
  input_spec.media_type = "video/x-raw";
  input_spec.format = "BGR";
  input_spec.width = img.cols;
  input_spec.height = img.rows;
  input_spec.depth = 3;

  sima::mpk::ModelMPKOptions mpk_opt = sima::mpk::options_from_output_spec(input_spec);
  mpk_opt.preproc_next_cpu = "MLA";
  mpk_opt.fast_mode = env_bool("SIMA_TPUT_FULL_FAST_MODE", true);
  mpk_opt.disable_internal_queues = env_bool("SIMA_TPUT_FULL_DISABLE_QUEUES", false);
  mpk_opt.num_buffers_cvu = env_int("SIMA_TPUT_FULL_NUM_BUFFERS_CVU", 0);
  mpk_opt.num_buffers_mla = env_int("SIMA_TPUT_FULL_NUM_BUFFERS_MLA", 0);
  mpk_opt.upstream_name = "decoder";

  auto model = sima::mpk::ModelMPK::load(tar_gz, mpk_opt);
  std::string runtime_config;
  if (post_mode == FullPostMode::Boxdecode) {
    const std::string config_path = sima_examples::find_boxdecode_config(model.etc_dir());
    require(!config_path.empty(), "Failed to locate simaaiboxdecode config JSON");
    runtime_config = sima_examples::prepare_yolo_boxdecode_config(
        config_path, root, img.cols, img.rows, 0.5f, 0.5f);
  } else if (post_mode == FullPostMode::DetessDequant) {
    require(fragment_mentions_detessdequant(model),
            "Model postprocess fragment does not mention detess/dequant");
  }

  sima::PipelineSession p;
  sima::InputAppSrcOptions src_opt;
  src_opt.format = "BGR";
  src_opt.width = img.cols;
  src_opt.height = img.rows;
  src_opt.depth = 3;
  src_opt.is_live = true;
  src_opt.do_timestamp = true;
  src_opt.block = true;
  src_opt.use_simaai_pool = use_pool;
  src_opt.pool_min_buffers = env_int("SIMA_TPUT_FULL_POOL_MIN", 1);
  src_opt.pool_max_buffers = env_int("SIMA_TPUT_FULL_POOL_MAX", 2);
  src_opt.buffer_name = "decoder";
  src_opt.max_bytes = static_cast<std::uint64_t>(img.total() * img.elemSize());

  p.add(sima::nodes::InputAppSrc(src_opt));
  p.add(sima::nodes::groups::Preprocess(model));
  p.add(sima::nodes::groups::MLA(model));

  if (post_mode == FullPostMode::Boxdecode) {
    sima::SimaBoxDecodeOptions box_opt;
    box_opt.config_path = runtime_config;
    p.add(sima::nodes::SimaBoxDecode(box_opt));
  } else if (post_mode == FullPostMode::DetessDequant) {
    p.add(sima::nodes::groups::Postprocess(model));
  }

  sima::OutputAppSinkOptions sink_opt;
  sink_opt.sync = false;
  sink_opt.drop = false;
  sink_opt.max_buffers = 1;
  p.add(sima::nodes::OutputAppSink(sink_opt));

  sima::InputStreamOptions stream_opt;
  stream_opt.timeout_ms = env_int("SIMA_TPUT_FULL_TIMEOUT_MS", 20000);
  stream_opt.appsink_sync = false;
  stream_opt.appsink_drop = false;
  stream_opt.appsink_max_buffers = 1;
  stream_opt.copy_output = false;
  stream_opt.reuse_input_buffer = false;
  stream_opt.enable_timings = enable_timings;
  stream_opt.allow_mismatched_input = false;

  auto stream = p.run_input_stream(img, stream_opt);
  for (int i = 0; i < warm; ++i) {
    (void)stream.push_and_pull(img, stream_opt.timeout_ms);
  }

  double sum_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  int outputs = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    const auto t1 = std::chrono::steady_clock::now();
    (void)stream.push_and_pull(img, stream_opt.timeout_ms);
    const auto t2 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    sum_ms += ms;
    if (i == 0) {
      min_ms = ms;
      max_ms = ms;
    } else {
      min_ms = std::min(min_ms, ms);
      max_ms = std::max(max_ms, ms);
    }
    outputs += 1;
  }
  const auto t3 = std::chrono::steady_clock::now();
  const double elapsed_s =
      std::chrono::duration<double>(t3 - t0).count();

  const sima::InputStreamStats stats = stream.stats();
  const std::string diag = enable_timings ? stream.diagnostics_summary() : "";
  stream.close();

  res.inputs = iters;
  res.outputs = outputs;
  res.avg_latency_ms = (outputs > 0) ? (sum_ms / outputs) : 0.0;
  res.min_latency_ms = min_ms;
  res.max_latency_ms = max_ms;
  res.avg_fps = (elapsed_s > 0.0) ? (static_cast<double>(outputs) / elapsed_s) : 0.0;
  res.pipeline = p.to_gst(false);
  res.ok = (outputs == iters);
  if (enable_timings) {
    std::ostringstream oss;
    oss << "push=" << stats.push_count
        << " pull=" << stats.pull_count
        << " avg_alloc_us=" << stats.avg_alloc_us
        << " avg_map_us=" << stats.avg_map_us
        << " avg_copy_us=" << stats.avg_copy_us
        << " avg_push_us=" << stats.avg_push_us
        << " avg_pull_wait_us=" << stats.avg_pull_wait_us
        << " avg_decode_us=" << stats.avg_decode_us;
    if (!diag.empty()) {
      oss << "\n" << diag;
    }
    res.note = oss.str();
  }
  if (!res.note.empty()) res.note += "\n";
  res.note += std::string("post_mode=") + full_post_mode_name(post_mode);
  return res;
}

ThroughputResult run_yolov8_full_async_tput(const fs::path& root,
                                            const cv::Mat& img,
                                            bool use_pool,
                                            int iters,
                                            int warm,
                                            FullPostMode post_mode) {
  ThroughputResult res;

  const std::string tar_gz = sima_examples::resolve_yolov8s_tar(root);
  require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

  sima::OutputSpec input_spec;
  input_spec.media_type = "video/x-raw";
  input_spec.format = "BGR";
  input_spec.width = img.cols;
  input_spec.height = img.rows;
  input_spec.depth = 3;

  sima::mpk::ModelMPKOptions mpk_opt = sima::mpk::options_from_output_spec(input_spec);
  mpk_opt.preproc_next_cpu = "MLA";
  mpk_opt.fast_mode = env_bool("SIMA_TPUT_FULL_FAST_MODE", true);
  mpk_opt.disable_internal_queues = env_bool("SIMA_TPUT_FULL_DISABLE_QUEUES", false);
  mpk_opt.num_buffers_cvu = env_int("SIMA_TPUT_FULL_NUM_BUFFERS_CVU", 0);
  mpk_opt.num_buffers_mla = env_int("SIMA_TPUT_FULL_NUM_BUFFERS_MLA", 0);
  mpk_opt.upstream_name = "decoder";

  auto model = sima::mpk::ModelMPK::load(tar_gz, mpk_opt);
  std::string runtime_config;
  if (post_mode == FullPostMode::Boxdecode) {
    const std::string config_path = sima_examples::find_boxdecode_config(model.etc_dir());
    require(!config_path.empty(), "Failed to locate simaaiboxdecode config JSON");
    runtime_config = sima_examples::prepare_yolo_boxdecode_config(
        config_path, root, img.cols, img.rows, 0.5f, 0.5f);
  } else if (post_mode == FullPostMode::DetessDequant) {
    require(fragment_mentions_detessdequant(model),
            "Model postprocess fragment does not mention detess/dequant");
  }

  sima::PipelineSession p;
  sima::InputAppSrcOptions src_opt;
  src_opt.format = "BGR";
  src_opt.width = img.cols;
  src_opt.height = img.rows;
  src_opt.depth = 3;
  src_opt.is_live = true;
  src_opt.do_timestamp = true;
  src_opt.block = true;
  src_opt.use_simaai_pool = use_pool;
  src_opt.pool_min_buffers = env_int("SIMA_TPUT_FULL_POOL_MIN", 1);
  src_opt.pool_max_buffers = env_int("SIMA_TPUT_FULL_POOL_MAX", 2);
  src_opt.buffer_name = "decoder";
  src_opt.max_bytes = static_cast<std::uint64_t>(img.total() * img.elemSize());

  p.add(sima::nodes::InputAppSrc(src_opt));
  p.add(sima::nodes::groups::Preprocess(model));
  p.add(sima::nodes::groups::MLA(model));

  if (post_mode == FullPostMode::Boxdecode) {
    sima::SimaBoxDecodeOptions box_opt;
    box_opt.config_path = runtime_config;
    p.add(sima::nodes::SimaBoxDecode(box_opt));
  } else if (post_mode == FullPostMode::DetessDequant) {
    p.add(sima::nodes::groups::Postprocess(model));
  }

  const int min_q = iters + warm;
  const int inq_env = env_int("SIMA_TPUT_FULL_INQ", min_q);
  const int outq_env = env_int("SIMA_TPUT_FULL_OUTQ", min_q);
  const int inq = (inq_env > min_q) ? inq_env : min_q;
  const int outq = (outq_env > min_q) ? outq_env : min_q;

  sima::OutputAppSinkOptions sink_opt;
  sink_opt.sync = false;
  sink_opt.drop = false;
  sink_opt.max_buffers = outq;
  p.add(sima::nodes::OutputAppSink(sink_opt));

  sima::AsyncOptions async_opt;
  async_opt.input_queue = inq;
  async_opt.output_queue = outq;
  async_opt.drop = sima::DropPolicy::Block;
  async_opt.copy_output = false;
  async_opt.copy_input = false;
  async_opt.allow_mismatched_input = false;
  async_opt.timeout_ms = -1;

  auto async = p.run_async(img, async_opt);

  std::mutex warm_mu;
  std::condition_variable warm_cv;
  bool warm_ready = (warm <= 0);

  std::mutex time_mu;
  bool measured_done = false;
  std::chrono::steady_clock::time_point t_end{};

  std::atomic<int> total_out{0};
  std::atomic<int> measured_out{0};
  std::string consumer_error;
  const int pull_timeout = env_int("SIMA_TPUT_FULL_TIMEOUT_MS", 20000);

  std::thread consumer([&]() {
    try {
      while (true) {
        auto out = async.pull(pull_timeout);
        if (!out.has_value()) break;
        const int cur = total_out.fetch_add(1) + 1;
        if (cur == warm && warm > 0) {
          std::lock_guard<std::mutex> lock(warm_mu);
          warm_ready = true;
          warm_cv.notify_one();
        }
        if (cur > warm) {
          const int m = measured_out.fetch_add(1) + 1;
          if (m == iters) {
            std::lock_guard<std::mutex> lock(time_mu);
            if (!measured_done) {
              t_end = std::chrono::steady_clock::now();
              measured_done = true;
            }
          }
        }
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(time_mu);
      consumer_error = e.what();
    }
  });

  for (int i = 0; i < warm; ++i) {
    (void)async.push(img);
  }

  if (warm > 0) {
    std::unique_lock<std::mutex> lock(warm_mu);
    warm_cv.wait_for(lock, std::chrono::seconds(20), [&]() { return warm_ready; });
  }

  const auto t_start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    (void)async.push(img);
  }
  async.close_input();

  consumer.join();

  const bool done = measured_done;
  const auto t_stop = done ? t_end : std::chrono::steady_clock::now();
  const double elapsed_s =
      std::chrono::duration<double>(t_stop - t_start).count();

  const sima::AsyncStats st = async.stats();
  res.inputs = iters;
  res.outputs = measured_out.load();
  res.avg_latency_ms = st.avg_latency_ms;
  res.min_latency_ms = st.min_latency_ms;
  res.max_latency_ms = st.max_latency_ms;
  res.avg_fps = (elapsed_s > 0.0) ? (static_cast<double>(res.outputs) / elapsed_s) : 0.0;
  res.pipeline = p.to_gst(false);
  res.ok = (res.outputs == iters);

  if (!consumer_error.empty()) {
    res.note = "async_pull_error=" + consumer_error;
    res.ok = false;
  }
  if (!done) {
    if (!res.note.empty()) res.note += "; ";
    res.note += "async_timing_incomplete";
    res.ok = false;
  }
  if (!res.note.empty()) res.note += "; ";
  res.note += std::string("post_mode=") + full_post_mode_name(post_mode);

  return res;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    // Enable diagnostics to capture stage timings.
    setenv("SIMA_GST_STAGE_TIMINGS", "1", 1);
    setenv("SIMA_GST_BOUNDARY_PROBES", "1", 1);

    const std::string img_path = env_str("SIMA_YOLO_IMAGE", (root / "test.jpg").string());
    cv::Mat img_bgr = cv::imread(img_path, cv::IMREAD_COLOR);
    require(!img_bgr.empty(), "Failed to read image: " + img_path);
    cv::Mat img_rgb_640;
    cv::Mat img_rgb_1280;
    cv::Mat img_bgr_1280;
    cv::resize(img_bgr, img_rgb_640, cv::Size(640, 640), 0.0, 0.0, cv::INTER_LINEAR);
    cv::resize(img_bgr, img_bgr_1280, cv::Size(1280, 720), 0.0, 0.0, cv::INTER_LINEAR);
    if (!img_bgr_1280.isContinuous()) img_bgr_1280 = img_bgr_1280.clone();
    cv::cvtColor(img_rgb_640, img_rgb_640, cv::COLOR_BGR2RGB);
    cv::cvtColor(img_bgr_1280, img_rgb_1280, cv::COLOR_BGR2RGB);
    if (!img_rgb_640.isContinuous()) img_rgb_640 = img_rgb_640.clone();
    if (!img_rgb_1280.isContinuous()) img_rgb_1280 = img_rgb_1280.clone();

    const bool use_pool = env_bool("SIMA_USE_POOL", false);
    const bool use_pool_preproc = env_bool("SIMA_PREPROC_USE_POOL", true);
    const bool use_pool_full = env_bool("SIMA_FULL_USE_POOL", use_pool_preproc);
    const std::string preproc_next_cpu = env_str("SIMA_PREPROC_NEXT_CPU", "APU");
    const bool use_standalone_preproc = env_bool("SIMA_PREPROC_STANDALONE", true);
    const std::string tput_mode_env = env_str("SIMA_TPUT_MODE", "");
    const TputMode tput_mode = parse_tput_mode(tput_mode_env);
    const bool tput_mode_forced = (tput_mode != TputMode::Default);
    const bool tput_only_env = env_bool("SIMA_TPUT_RUN_ONLY", false);
    const bool tput_only = tput_only_env || tput_mode_forced;
    const bool tput_enable = env_bool("SIMA_TPUT_ENABLE", false) || tput_only;
    const bool tput_full_enable =
        env_bool("SIMA_TPUT_FULL_ENABLE", false) || tput_mode_is_full(tput_mode);
    const bool tput_full_skip_sync = env_bool("SIMA_TPUT_FULL_SKIP_SYNC", false);
    const bool legacy_mla_only = env_bool("SIMA_TPUT_FULL_MLA_ONLY", false);
    const std::string full_post_env = env_str("SIMA_TPUT_FULL_POST", "boxdecode");
    FullPostMode full_post_mode = parse_full_post_mode(full_post_env);
    if (tput_mode == TputMode::FullBoxdecode) {
      full_post_mode = FullPostMode::Boxdecode;
    } else if (tput_mode == TputMode::FullDetessDequant) {
      full_post_mode = FullPostMode::DetessDequant;
    } else if (tput_mode == TputMode::FullNone) {
      full_post_mode = FullPostMode::None;
    }
    if (legacy_mla_only) {
      full_post_mode = FullPostMode::None;
    }
    setenv("SIMA_RUN_INPUT_TIMINGS", "1", 1);

    RunResult base;
    RunResult base_run;
    RunResult preproc;

    if (!tput_only) {
      std::cout << "[RUN] Baseline InputAppSrc -> OutputAppSink\n";
      base = run_baseline_noop(img_rgb_640, use_pool);
      std::cout << "pipeline:\n" << base.pipeline << "\n";
      std::cout << "latency_ms=" << base.latency_ms
                << " mean_ms=" << base.mean_ms
                << " p50_ms=" << base.p50_ms
                << " p95_ms=" << base.p95_ms
                << " samples=" << base.latency_samples
                << " push=" << base.stats.push_count
                << " pull=" << base.stats.pull_count
                << " avg_alloc_us=" << base.stats.avg_alloc_us
                << " avg_map_us=" << base.stats.avg_map_us
                << " avg_copy_us=" << base.stats.avg_copy_us
                << " avg_push_us=" << base.stats.avg_push_us
                << " avg_pull_wait_us=" << base.stats.avg_pull_wait_us
                << " avg_decode_us=" << base.stats.avg_decode_us << "\n";
      if (!base.diag.empty()) {
        std::cout << "[DIAG]\n" << base.diag << "\n";
      }
      if (!base.ok) {
        std::cout << "[WARN] Baseline output unexpected: " << base.note << "\n";
      }

      std::cout << "[RUN] Baseline run(input) InputAppSrc -> OutputAppSink\n";
      base_run = run_baseline_noop_run_api(img_rgb_640, use_pool);
      std::cout << "pipeline:\n" << base_run.pipeline << "\n";
      std::cout << "latency_ms=" << base_run.latency_ms
                << " mean_ms=" << base_run.mean_ms
                << " p50_ms=" << base_run.p50_ms
                << " p95_ms=" << base_run.p95_ms
                << " samples=" << base_run.latency_samples
                << " push=" << base_run.stats.push_count
                << " pull=" << base_run.stats.pull_count
                << " avg_alloc_us=" << base_run.stats.avg_alloc_us
                << " avg_map_us=" << base_run.stats.avg_map_us
                << " avg_copy_us=" << base_run.stats.avg_copy_us
                << " avg_push_us=" << base_run.stats.avg_push_us
                << " avg_pull_wait_us=" << base_run.stats.avg_pull_wait_us
                << " avg_decode_us=" << base_run.stats.avg_decode_us << "\n";
      if (!base_run.diag.empty()) {
        std::cout << "[DIAG]\n" << base_run.diag << "\n";
      }
      if (!base_run.note.empty()) {
        std::cout << "[INFO] Baseline run(input) " << base_run.note << "\n";
      }
      if (!base_run.ok) {
        std::cout << "[WARN] Baseline run(input) output unexpected: " << base_run.note << "\n";
      }

      std::cout << "[RUN] Preproc -> AppSink (next_cpu=" << preproc_next_cpu
                << " standalone=" << (use_standalone_preproc ? "true" : "false") << ")\n";
      preproc =
          run_preproc_to_appsink_apu(root, img_rgb_1280, use_pool_preproc, use_standalone_preproc);
      std::cout << "pipeline:\n" << preproc.pipeline << "\n";
      std::cout << "latency_ms=" << preproc.latency_ms
                << " mean_ms=" << preproc.mean_ms
                << " p50_ms=" << preproc.p50_ms
                << " p95_ms=" << preproc.p95_ms
                << " samples=" << preproc.latency_samples
                << " push=" << preproc.stats.push_count
                << " pull=" << preproc.stats.pull_count
                << " avg_alloc_us=" << preproc.stats.avg_alloc_us
                << " avg_map_us=" << preproc.stats.avg_map_us
                << " avg_copy_us=" << preproc.stats.avg_copy_us
                << " avg_push_us=" << preproc.stats.avg_push_us
                << " avg_pull_wait_us=" << preproc.stats.avg_pull_wait_us
                << " avg_decode_us=" << preproc.stats.avg_decode_us << "\n";
      if (!preproc.diag.empty()) {
        std::cout << "[DIAG]\n" << preproc.diag << "\n";
      }
      if (!preproc.ok) {
        std::cout << "[WARN] Preproc pipeline output unexpected: " << preproc.note << "\n";
      }
      if (!preproc.note.empty()) {
        std::cout << "[INFO] Preproc note: " << preproc.note << "\n";
      }
      if (!preproc.out_dtype.empty()) {
        std::cout << "[INFO] Preproc output dtype: " << preproc.out_dtype << "\n";
      }
      std::cout << "[INFO] Preproc byte_match=" << (preproc.byte_match ? "true" : "false") << "\n";
      std::cout << "[INFO] Preproc solid_match=" << (preproc.solid_match ? "true" : "false") << "\n";
      if (preproc.out_width == 640 && preproc.out_height == 640 &&
          !preproc.out_format.empty() && base_run.out_format == preproc.out_format) {
        std::cout << "[OK] Baseline output matches preproc output format/size: "
                  << preproc.out_format << " " << preproc.out_width << "x"
                  << preproc.out_height << "\n";
      } else {
        std::cout << "[WARN] Baseline output does not match preproc output: "
                  << "baseline=" << base_run.out_format << " "
                  << base_run.out_width << "x" << base_run.out_height
                  << " preproc=" << preproc.out_format << " "
                  << preproc.out_width << "x" << preproc.out_height << "\n";
      }

      const bool run_async = env_bool("SIMA_ASYNC_ENABLE", true);
      ThroughputResult async_res;
      if (run_async) {
        std::cout << "[RUN] Async Preproc -> AppSink (iters="
                  << env_int("SIMA_ASYNC_ITERS", 200)
                  << " inq=" << env_int("SIMA_ASYNC_INQ", 8)
                  << " outq=" << env_int("SIMA_ASYNC_OUTQ", 8)
                  << " drop=" << env_str("SIMA_ASYNC_DROP", "block") << ")\n";
        async_res = run_preproc_to_appsink_async(
            root, img_rgb_1280, use_pool_preproc, use_standalone_preproc);
        std::cout << "pipeline:\n" << async_res.pipeline << "\n";
        std::cout << "avg_fps=" << async_res.avg_fps
                  << " avg_latency_ms=" << async_res.avg_latency_ms
                  << " min_latency_ms=" << async_res.min_latency_ms
                  << " max_latency_ms=" << async_res.max_latency_ms
                  << " inputs=" << async_res.inputs
                  << " outputs=" << async_res.outputs << "\n";
        if (!async_res.note.empty()) {
          std::cout << "[INFO] Async note: " << async_res.note << "\n";
        }
        if (!async_res.ok) {
          std::cout << "[WARN] Async output count mismatch\n";
        }
        std::cout << "[COMPARE] sync_mean_ms=" << preproc.mean_ms
                  << " async_avg_ms=" << async_res.avg_latency_ms
                  << " sync_samples=" << preproc.latency_samples
                  << " async_outputs=" << async_res.outputs << "\n";
      }
    }

    ThroughputResult sync_tput;
    ThroughputResult async_tput;
    ThroughputResult full_sync_tput;
    ThroughputResult full_async_tput;
    bool ran_full = false;
    const bool run_preproc_tput = (tput_mode == TputMode::Default ||
                                   tput_mode == TputMode::Preproc);
    const bool run_full_tput = tput_full_enable &&
                               (tput_mode == TputMode::Default ||
                                tput_mode_is_full(tput_mode));
    if (tput_enable) {
      const int tput_iters = env_int("SIMA_TPUT_ITERS", 200);
      const int tput_warm = env_int("SIMA_TPUT_WARM", 10);
      if (run_preproc_tput && tput_iters > 0) {
        std::cout << "[RUN] Throughput Sync Preproc (iters=" << tput_iters
                  << " warm=" << tput_warm << ")\n";
        sync_tput = run_preproc_to_appsink_sync_tput(
            root, img_rgb_1280, use_pool_preproc, use_standalone_preproc,
            tput_iters, tput_warm);
        std::cout << "avg_fps=" << sync_tput.avg_fps
                  << " avg_latency_ms=" << sync_tput.avg_latency_ms
                  << " min_latency_ms=" << sync_tput.min_latency_ms
                  << " max_latency_ms=" << sync_tput.max_latency_ms
                  << " outputs=" << sync_tput.outputs << "\n";
        if (!sync_tput.ok) {
          std::cout << "[WARN] Sync throughput output count mismatch\n";
        }

        std::cout << "[RUN] Throughput Async Preproc (iters=" << tput_iters
                  << " warm=" << tput_warm << ")\n";
        async_tput = run_preproc_to_appsink_async_tput(
            root, img_rgb_1280, use_pool_preproc, use_standalone_preproc,
            tput_iters, tput_warm);
        std::cout << "avg_fps=" << async_tput.avg_fps
                  << " avg_latency_ms=" << async_tput.avg_latency_ms
                  << " min_latency_ms=" << async_tput.min_latency_ms
                  << " max_latency_ms=" << async_tput.max_latency_ms
                  << " outputs=" << async_tput.outputs << "\n";
        if (!async_tput.note.empty()) {
          std::cout << "[INFO] Async tput note: " << async_tput.note << "\n";
        }
        if (!async_tput.ok) {
          std::cout << "[WARN] Async throughput output count mismatch\n";
        }
      }
      if (run_full_tput) {
        const int full_iters = env_int("SIMA_TPUT_FULL_ITERS", tput_iters);
        const int full_warm = env_int("SIMA_TPUT_FULL_WARM", tput_warm);
        if (full_iters > 0) {
          ran_full = true;
          if (!tput_full_skip_sync) {
            std::cout << "[RUN] Throughput Sync Full "
                      << full_post_mode_name(full_post_mode)
                      << " (iters=" << full_iters
                      << " warm=" << full_warm << ")\n";
            full_sync_tput = run_yolov8_full_sync_tput(
                root, img_bgr_1280, use_pool_full, full_iters, full_warm,
                full_post_mode);
            std::cout << "avg_fps=" << full_sync_tput.avg_fps
                      << " avg_latency_ms=" << full_sync_tput.avg_latency_ms
                      << " min_latency_ms=" << full_sync_tput.min_latency_ms
                      << " max_latency_ms=" << full_sync_tput.max_latency_ms
                      << " outputs=" << full_sync_tput.outputs << "\n";
            if (!full_sync_tput.note.empty()) {
              std::cout << "[INFO] Sync full note:\n" << full_sync_tput.note << "\n";
            }
            if (!full_sync_tput.ok) {
              std::cout << "[WARN] Sync full-pipeline throughput output count mismatch\n";
            }
          } else {
            std::cout << "[INFO] SIMA_TPUT_FULL_SKIP_SYNC=1: running async-only full pipeline\n";
            std::cout << "[RUN] Throughput Async Full "
                      << full_post_mode_name(full_post_mode)
                      << " (iters=" << full_iters
                      << " warm=" << full_warm << ")\n";
            full_async_tput = run_yolov8_full_async_tput(
                root, img_bgr_1280, use_pool_full, full_iters, full_warm,
                full_post_mode);
            std::cout << "avg_fps=" << full_async_tput.avg_fps
                      << " avg_latency_ms=" << full_async_tput.avg_latency_ms
                      << " min_latency_ms=" << full_async_tput.min_latency_ms
                      << " max_latency_ms=" << full_async_tput.max_latency_ms
                      << " outputs=" << full_async_tput.outputs << "\n";
            if (!full_async_tput.note.empty()) {
              std::cout << "[INFO] Async full tput note: " << full_async_tput.note << "\n";
            }
            if (!full_async_tput.ok) {
              std::cout << "[WARN] Async full-pipeline throughput output count mismatch\n";
            }
          }
        }
      }
    }

    bool ok = true;
    if (!tput_only) {
      ok = base.ok && base_run.ok && preproc.ok;
    }
    if (tput_enable && run_preproc_tput) {
      ok = ok && sync_tput.ok && async_tput.ok;
    }
    if (ran_full) {
      if (!tput_full_skip_sync) {
        ok = ok && full_sync_tput.ok;
      } else {
        ok = ok && full_async_tput.ok;
      }
    }
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
