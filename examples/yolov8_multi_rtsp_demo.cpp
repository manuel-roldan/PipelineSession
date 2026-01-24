#include "example_utils.h"

#include "nodes/common/AppSink.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoScale.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/groups/RtspInputGroup.h"
#include "nodes/io/InputAppSrc.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/PipelineSession.h"
#include "mpk/ModelMPK.h"
#if defined(SIMA_WITH_OPENCV)
#include "mpk/ModelMPKOpenCV.h"
#endif

#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "test_utils.h"

#include <gst/gst.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool is_debug_enabled() {
  const char* env = std::getenv("SIMA_RTSP_DEMO_DEBUG");
  return env && *env != '\0';
}

bool is_trace_enabled() {
  const char* env = std::getenv("SIMA_RTSP_DEMO_TRACE");
  return env && *env != '\0';
}

std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

const char* get_env(const char* name) {
  const char* env = std::getenv(name);
  return (env && *env != '\0') ? env : nullptr;
}

bool has_arg(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) return true;
  }
  return false;
}

void configure_gst_logging() {
  if (const char* gst_dbg = get_env("SIMA_RTSP_DEMO_GST_DEBUG")) {
    ::setenv("GST_DEBUG", gst_dbg, 1);
    ::setenv("GST_DEBUG_NO_COLOR", "1", 1);
    return;
  }
  ::setenv("GST_DEBUG", "2", 1);
  ::setenv("GST_DEBUG_NO_COLOR", "1", 1);
}

std::string dtype_to_string(sima::TensorDType dtype) {
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
    case sima::TensorDType::BFloat16:
      return "BFloat16";
    case sima::TensorDType::Float32:
      return "Float32";
    case sima::TensorDType::Float64:
      return "Float64";
    default:
      return "Unknown";
  }
}

void log_tensor_info(const std::string& label, const sima::FrameTensor& t) {
  std::cout << "[dbg] " << label << " format=" << t.format
            << " dtype=" << dtype_to_string(t.dtype)
            << " w=" << t.width
            << " h=" << t.height
            << " shape=" << t.shape.size()
            << " planes=" << t.planes.size();
  size_t total = 0;
  for (size_t i = 0; i < t.planes.size(); ++i) {
    total += t.planes[i].size();
    std::cout << " p" << i << "=" << t.planes[i].size();
  }
  if (is_trace_enabled() && !t.caps_string.empty()) {
    std::cout << " caps=\"" << t.caps_string << "\"";
  }
  std::cout << " total=" << total << "\n";
}

double elapsed_seconds() {
  static const auto start = std::chrono::steady_clock::now();
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(now - start).count();
}

void log_debug(const std::string& tag, int stream_id, const std::string& msg) {
  if (!is_debug_enabled()) return;
  std::cout << "[dbg +" << std::fixed << std::setprecision(3)
            << elapsed_seconds() << "s]";
  if (stream_id >= 0) std::cout << " stream=" << stream_id;
  std::cout << " " << tag << ": " << msg << "\n";
}

void log_trace(const std::string& tag, int stream_id, const std::string& msg) {
  if (!is_trace_enabled()) return;
  std::cout << "[trace +" << std::fixed << std::setprecision(3)
            << elapsed_seconds() << "s]";
  if (stream_id >= 0) std::cout << " stream=" << stream_id;
  std::cout << " " << tag << ": " << msg << "\n";
}

void log_warn(const std::string& tag, int stream_id, const std::string& msg) {
  std::cerr << "[warn]";
  if (stream_id >= 0) std::cerr << " stream=" << stream_id;
  std::cerr << " " << tag << ": " << msg << "\n";
}

bool infer_dims(const sima::FrameTensor& t, int& w, int& h) {
  w = t.width;
  h = t.height;
  if ((w <= 0 || h <= 0) && t.shape.size() >= 2) {
    h = static_cast<int>(t.shape[0]);
    w = static_cast<int>(t.shape[1]);
  }
  return (w > 0 && h > 0);
}

size_t total_plane_bytes(const sima::FrameTensor& t) {
  size_t total = 0;
  for (const auto& p : t.planes) total += p.size();
  return total;
}

std::string hex_prefix(const std::vector<uint8_t>& payload, size_t max_bytes) {
  std::ostringstream ss;
  ss << std::hex << std::setfill('0');
  const size_t count = std::min(payload.size(), max_bytes);
  for (size_t i = 0; i < count; ++i) {
    ss << std::setw(2) << static_cast<int>(payload[i]);
    if (i + 1 < count) ss << " ";
  }
  return ss.str();
}

void warn_rtsp_options(const sima::nodes::groups::RtspInputGroupOptions& opt, int stream_id) {
  if (!opt.output_caps.enable) return;
  const bool needs_scale = opt.output_caps.width > 0 || opt.output_caps.height > 0;
  if (needs_scale && !opt.use_videoscale) {
    log_warn("rtsp-caps",
             stream_id,
             "output_caps width/height set but use_videoscale=false; negotiation may fail");
  }
  if (!opt.output_caps.format.empty() && !opt.out_format.empty() &&
      upper_copy(opt.output_caps.format) != upper_copy(opt.out_format) &&
      !opt.use_videoconvert) {
    log_warn("rtsp-caps",
             stream_id,
             "output_caps format != decoder out_format without videoconvert; negotiation may fail");
  }
  if (opt.output_caps.width <= 0 || opt.output_caps.height <= 0) {
    log_warn("rtsp-caps",
             stream_id,
             "output_caps enabled but width/height <= 0; caps filter may be too loose");
  }
}

bool output_caps_enabled(const sima::nodes::groups::RtspInputGroupOptions::OutputCaps& c) {
  return c.enable || c.width > 0 || c.height > 0 || c.fps > 0;
}

const char* rtsp_stage_label(int stage) {
  switch (stage) {
    case 0:
      return "resnet-raw";
    case 1:
      return "decode-sysmem";
    case 2:
      return "caps-wh";
    case 3:
      return "scale-wh";
    case 4:
      return "caps-fps";
    default:
      return "default";
  }
}

void apply_rtsp_stage(int stage,
                      sima::nodes::groups::RtspInputGroupOptions& opt,
                      int target_w,
                      int target_h,
                      int caps_fps) {
  switch (stage) {
    case 0:  // Resnet-like: raw decoder output, no post-decode caps/scale.
      opt.decoder_raw_output = true;
      opt.use_videoconvert = false;
      opt.use_videoscale = false;
      opt.output_caps.enable = false;
      opt.output_caps.format = opt.out_format;
      opt.output_caps.width = -1;
      opt.output_caps.height = -1;
      opt.output_caps.fps = -1;
      opt.output_caps.memory = sima::CapsMemory::Any;
      break;
    case 1:  // Decode to SystemMemory, but no caps/scale.
      opt.decoder_raw_output = false;
      opt.use_videoconvert = false;
      opt.use_videoscale = false;
      opt.output_caps.enable = false;
      opt.output_caps.format = opt.out_format;
      opt.output_caps.width = -1;
      opt.output_caps.height = -1;
      opt.output_caps.fps = -1;
      break;
    case 2:  // Add width/height caps only.
      opt.decoder_raw_output = false;
      opt.use_videoconvert = false;
      opt.use_videoscale = false;
      opt.output_caps.enable = true;
      opt.output_caps.format = opt.out_format;
      opt.output_caps.width = target_w;
      opt.output_caps.height = target_h;
      opt.output_caps.fps = -1;
      break;
    case 3:  // Add videoscale for width/height.
      opt.decoder_raw_output = false;
      opt.use_videoconvert = false;
      opt.use_videoscale = true;
      opt.output_caps.enable = true;
      opt.output_caps.format = opt.out_format;
      opt.output_caps.width = target_w;
      opt.output_caps.height = target_h;
      opt.output_caps.fps = -1;
      break;
    case 4:  // Add fps caps (most strict).
      opt.decoder_raw_output = false;
      opt.use_videoconvert = false;
      opt.use_videoscale = true;
      opt.output_caps.enable = true;
      opt.output_caps.format = opt.out_format;
      opt.output_caps.width = target_w;
      opt.output_caps.height = target_h;
      opt.output_caps.fps = caps_fps;
      break;
    default:
      break;
  }
}

struct DecoderOverrides {
  std::string next_element;
  std::string dec_mode;
  std::string apu_mem_pool;
  int dec_fps = -1;
  int dec_width = -1;
  int dec_height = -1;
  bool enabled = false;
};

std::string build_decoder_fragment(int stream_id,
                                   const sima::nodes::groups::RtspInputGroupOptions& opt,
                                   const DecoderOverrides& dec) {
  std::ostringstream ss;
  ss << "simaaidecoder name=" << (opt.decoder_name.empty()
                                     ? ("decoder_s" + std::to_string(stream_id))
                                     : opt.decoder_name)
     << " sima-allocator-type=" << opt.sima_allocator_type;
  if (!opt.decoder_name.empty()) ss << " op-buff-name=" << opt.decoder_name;
  if (!opt.out_format.empty()) {
    std::string dec_fmt = opt.out_format;
    if (dec_fmt == "I420") dec_fmt = "YUV420P";
    if (dec_fmt == "NV12" || dec_fmt == "YUV420P") {
      ss << " dec-fmt=" << dec_fmt;
    }
  }
  if (!dec.next_element.empty()) ss << " next-element=" << dec.next_element;
  if (!dec.dec_mode.empty()) ss << " dec-mode=" << dec.dec_mode;
  if (dec.dec_fps >= 0) ss << " dec-fps=" << dec.dec_fps;
  if (dec.dec_width > 0) ss << " dec-width=" << dec.dec_width;
  if (dec.dec_height > 0) ss << " dec-height=" << dec.dec_height;
  if (!dec.apu_mem_pool.empty()) ss << " apu-mem-pool=" << dec.apu_mem_pool;

  if (opt.decoder_raw_output) return ss.str();

  ss << " ! videoconvert name=dec_vc_s" << stream_id
     << " ! capsfilter name=dec_raw_caps_s" << stream_id
     << " caps=\"video/x-raw(memory:SystemMemory),format=" << opt.out_format << "\"";
  return ss.str();
}

void warn_nv12_layout(const sima::FrameTensor& t, int stream_id) {
  if (upper_copy(t.format) != "NV12") return;
  if (t.planes.size() < 2) {
    log_warn("rtsp-nv12",
             stream_id,
             "NV12 expects 2 planes; got " + std::to_string(t.planes.size()));
  }
  int w = 0;
  int h = 0;
  if (!infer_dims(t, w, h)) {
    log_warn("rtsp-nv12", stream_id, "NV12 missing dimensions");
    return;
  }
  if ((w % 2) != 0 || (h % 2) != 0) {
    log_warn("rtsp-nv12",
             stream_id,
             "NV12 requires even dimensions; got " + std::to_string(w) + "x" +
                 std::to_string(h));
  }
  const size_t expected = static_cast<size_t>(w) * static_cast<size_t>(h) * 3 / 2;
  const size_t actual = total_plane_bytes(t);
  if (actual != expected) {
    log_warn("rtsp-nv12",
             stream_id,
             "NV12 bytes mismatch expected=" + std::to_string(expected) +
                 " actual=" + std::to_string(actual));
  }
}

void warn_frame_mismatch(const sima::FrameTensor& frame,
                         const std::string& expected_format,
                         int expected_w,
                         int expected_h,
                         int stream_id,
                         const char* tag) {
  int w = 0;
  int h = 0;
  if (!infer_dims(frame, w, h)) {
    log_warn(tag, stream_id, "frame missing width/height; check caps negotiation");
    return;
  }
  if (expected_w > 0 && expected_w != w) {
    log_warn(tag,
             stream_id,
             "width mismatch expected=" + std::to_string(expected_w) +
                 " actual=" + std::to_string(w));
  }
  if (expected_h > 0 && expected_h != h) {
    log_warn(tag,
             stream_id,
             "height mismatch expected=" + std::to_string(expected_h) +
                 " actual=" + std::to_string(h));
  }
  if (!expected_format.empty()) {
    if (frame.format.empty()) {
      log_warn(tag, stream_id, "frame format missing; expected " + expected_format);
    } else if (upper_copy(expected_format) != upper_copy(frame.format)) {
      log_warn(tag,
               stream_id,
               "format mismatch expected=" + expected_format +
                   " actual=" + frame.format);
    }
  }
}

void warn_yolo_appsrc_options(const sima::InputAppSrcOptions& opt,
                              int expected_w,
                              int expected_h) {
  if (opt.media_type.empty() || upper_copy(opt.media_type) != "VIDEO/X-RAW") {
    log_warn("yolo-input", -1, "unexpected media_type: " + opt.media_type);
  }
  if (opt.format.empty()) {
    log_warn("yolo-input", -1, "InputAppSrc format is empty");
  } else if (upper_copy(opt.format) != "NV12") {
    log_warn("yolo-input", -1, "InputAppSrc format is " + opt.format);
  }
  if (opt.width <= 0 || opt.height <= 0) {
    log_warn("yolo-input",
             -1,
             "InputAppSrc missing width/height (" + std::to_string(opt.width) +
                 "x" + std::to_string(opt.height) + ")");
  }
  if (expected_w > 0 && opt.width > 0 && opt.width != expected_w) {
    log_warn("yolo-input",
             -1,
             "InputAppSrc width mismatch expected=" + std::to_string(expected_w) +
                 " actual=" + std::to_string(opt.width));
  }
  if (expected_h > 0 && opt.height > 0 && opt.height != expected_h) {
    log_warn("yolo-input",
             -1,
             "InputAppSrc height mismatch expected=" + std::to_string(expected_h) +
                 " actual=" + std::to_string(opt.height));
  }
}

void warn_boxdecode_config(const sima::mpk::ModelMPK& model) {
  std::string path = model.find_config_path_by_plugin("boxdecode");
  if (path.empty()) path = model.find_config_path_by_plugin("box_decode");
  if (path.empty()) {
    log_warn("yolo-config",
             -1,
             "boxdecode config not found in model; SimaBoxDecode will fail");
  } else if (is_debug_enabled()) {
    log_debug("yolo-config", -1, "boxdecode config=" + path);
  }
}

std::string build_h264_caps(int width, int height, int fps) {
  std::ostringstream caps;
  caps << "video/x-h264,parsed=true,stream-format=(string)byte-stream,alignment=(string)au";
  if (width > 0) {
    caps << ",width=(int)" << width;
  } else {
    caps << ",width=(int)[1,4096]";
  }
  if (height > 0) {
    caps << ",height=(int)" << height;
  } else {
    caps << ",height=(int)[1,4096]";
  }
  if (fps > 0) {
    caps << ",framerate=(fraction)" << fps << "/1";
  }
  return caps.str();
}

std::string build_h264_dims_caps(int width, int height) {
  std::ostringstream caps;
  caps << "video/x-h264";
  if (width > 0) {
    caps << ",width=(int)" << width;
  } else {
    caps << ",width=(int)[1,4096]";
  }
  if (height > 0) {
    caps << ",height=(int)" << height;
  } else {
    caps << ",height=(int)[1,4096]";
  }
  return caps.str();
}

std::string build_rtp_h264_fragment(int stream_id,
                                    int payload_type,
                                    bool add_rtp_caps,
                                    bool add_h264_caps,
                                    bool add_h264_dims_caps,
                                    bool wait_for_keyframe,
                                    int h264_parse_config_interval,
                                    int h264_width,
                                    int h264_height,
                                    int h264_fps) {
  std::ostringstream ss;
  if (add_rtp_caps) {
    ss << "capsfilter name=rtp_caps_s" << stream_id
       << " caps=\"application/x-rtp,media=video,encoding-name=H264,payload="
       << payload_type << "\" ! ";
  }
  ss << "rtph264depay name=rtp_depay_s" << stream_id;
  if (wait_for_keyframe) ss << " wait-for-keyframe=true";
  ss << " ! h264parse name=h264_parse_s" << stream_id << " disable-passthrough=true";
  if (h264_parse_config_interval >= 0) {
    ss << " config-interval=" << h264_parse_config_interval;
  }
  if (add_h264_caps) {
    ss << " ! capsfilter name=h264_caps_s" << stream_id
       << " caps=\"" << build_h264_caps(h264_width, h264_height, h264_fps) << "\"";
  } else if (add_h264_dims_caps) {
    ss << " ! capsfilter name=h264_dims_s" << stream_id
       << " caps=\"" << build_h264_dims_caps(h264_width, h264_height) << "\"";
  }
  return ss.str();
}

struct InFlight {
  int stream_id = -1;
  int frame_index = 0;
  cv::Mat bgr;
  std::chrono::steady_clock::time_point t_pull;
  std::chrono::steady_clock::time_point t_push;
};

struct StreamMetrics {
  std::mutex mu;
  int pulled = 0;
  int saved = 0;
  double pull_ms_sum = 0.0;
  double detect_ms_sum = 0.0;
  double e2e_ms_sum = 0.0;
  std::chrono::steady_clock::time_point first_pull{};
  std::chrono::steady_clock::time_point last_output{};
};

struct StreamContext {
  int id = 0;
  std::string url;
  std::string expected_format;
  int expected_width = -1;
  int expected_height = -1;
  sima::PipelineSession session;
  sima::TensorStream stream;
  std::thread th;
  int pushed = 0;
  int saved = 0;
  std::atomic<int> pull_timeouts{0};
  std::atomic<int> pull_success{0};
  cv::VideoWriter writer;
  bool writer_open = false;
  StreamMetrics metrics;
};

bool tensor_to_bgr_mat(const sima::FrameTensor& t, cv::Mat& out, std::string& err) {
  if (t.dtype != sima::TensorDType::UInt8) {
    err = "expected uint8 tensor";
    return false;
  }
  int w = t.width;
  int h = t.height;
  if ((w <= 0 || h <= 0) && t.shape.size() >= 2) {
    h = static_cast<int>(t.shape[0]);
    w = static_cast<int>(t.shape[1]);
  }
  if (w <= 0 || h <= 0) {
    err = "invalid tensor dimensions";
    return false;
  }

  if (t.format == "NV12") {
    if (t.planes.size() < 2) {
      err = "NV12 tensor missing planes";
      return false;
    }
    cv::Mat y(h, w, CV_8UC1, const_cast<uint8_t*>(t.planes[0].data()));
    cv::Mat uv(h / 2, w, CV_8UC1, const_cast<uint8_t*>(t.planes[1].data()));
    cv::cvtColorTwoPlane(y, uv, out, cv::COLOR_YUV2BGR_NV12);
    return true;
  }

  if (t.planes.empty()) {
    err = "missing tensor plane";
    return false;
  }
  const size_t needed = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
  if (t.planes[0].size() < needed) {
    err = "tensor plane too small";
    return false;
  }
  cv::Mat tmp(h, w, CV_8UC3);
  std::memcpy(tmp.data, t.planes[0].data(), needed);

  if (t.format == "RGB") {
    cv::cvtColor(tmp, out, cv::COLOR_RGB2BGR);
    return true;
  }
  if (t.format == "BGR") {
    out = std::move(tmp);
    return true;
  }
  err = "unexpected format: " + t.format;
  return false;
}

bool extract_bbox_payload(const sima::RunInputResult& result,
                          std::vector<uint8_t>& payload,
                          std::string& err) {
  if (result.kind != sima::RunOutputKind::Tensor) {
    err = "expected tensor output";
    return false;
  }
  if (result.tensor.has_value()) {
    if (result.tensor->planes.empty()) {
      err = "missing tensor planes";
      return false;
    }
    if (result.tensor->format != "BBOX") {
      err = "unexpected tensor format: " + result.tensor->format;
      return false;
    }
    payload = result.tensor->planes[0];
    return !payload.empty();
  }
  if (result.tensor_ref.has_value()) {
    sima::FrameTensor owned = result.tensor_ref.value().to_copy();
    if (owned.planes.empty()) {
      err = "missing tensor planes";
      return false;
    }
    if (owned.format != "BBOX") {
      err = "unexpected tensor format: " + owned.format;
      return false;
    }
    payload = owned.planes[0];
    return !payload.empty();
  }
  err = "missing tensor output";
  return false;
}

bool parse_int_arg(int argc, char** argv, const std::string& key, int& out) {
  std::string raw;
  if (!sima_examples::get_arg(argc, argv, key, raw)) return false;
  try {
    out = std::stoi(raw);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid " + key + ": " + raw);
  }
  return true;
}

bool parse_double_arg(int argc, char** argv, const std::string& key, double& out) {
  std::string raw;
  if (!sima_examples::get_arg(argc, argv, key, raw)) return false;
  try {
    out = std::stod(raw);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid " + key + ": " + raw);
  }
  return true;
}

std::vector<std::string> resolve_rtsp_urls(int argc, char** argv, int stream_count) {
  if (stream_count < 1 || stream_count > 4) {
    throw std::runtime_error("Invalid --streams: " + std::to_string(stream_count) +
                             " (expected 1..4)");
  }
  std::vector<std::string> urls(static_cast<size_t>(stream_count));
  std::string shared;
  sima_examples::get_arg(argc, argv, "--rtsp", shared);
  for (int i = 0; i < stream_count; ++i) {
    const std::string key = "--rtsp" + std::to_string(i);
    sima_examples::get_arg(argc, argv, key, urls[i]);
  }

  if (!shared.empty()) {
    for (auto& url : urls) {
      if (url.empty()) url = shared;
    }
  }

  bool any = false;
  for (const auto& url : urls) {
    if (!url.empty()) {
      any = true;
      break;
    }
  }
  if (!any) {
    const int last_idx = stream_count - 1;
    const std::string range =
        (last_idx == 0) ? "--rtsp0" : "--rtsp0..--rtsp" + std::to_string(last_idx);
    throw std::runtime_error("Missing RTSP url: pass --rtsp or " + range);
  }

  for (int i = 0; i < stream_count; ++i) {
    if (urls[i].empty()) {
      throw std::runtime_error("Missing --rtsp" + std::to_string(i) +
                               " (or use --rtsp to apply one URL to all)");
    }
  }
  return urls;
}

} // namespace

int main(int argc, char** argv) {
  try {
    configure_gst_logging();
    gst_init(nullptr, nullptr);
    if (is_debug_enabled()) {
      if (get_env("SIMA_RTSP_DEMO_GST_DEBUG")) {
        log_debug("gst", -1, "GST_DEBUG set from SIMA_RTSP_DEMO_GST_DEBUG");
      } else {
        log_debug("gst", -1,
                  "GST_DEBUG defaulted to warning level; "
                  "set SIMA_RTSP_DEMO_GST_DEBUG for verbose logs");
      }
    }

    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_examples::resolve_yolov8s_tar(root);
    require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

    int stream_count = 4;
    parse_int_arg(argc, argv, "--streams", stream_count);
    if (stream_count < 1 || stream_count > 4) {
      throw std::runtime_error("Invalid --streams: " + std::to_string(stream_count) +
                               " (expected 1..4)");
    }
    const std::vector<std::string> urls = resolve_rtsp_urls(argc, argv, stream_count);

    int rtsp_stage = -1;
    parse_int_arg(argc, argv, "--rtsp-stage", rtsp_stage);
    const bool rtsp_resnet = has_arg(argc, argv, "--rtsp-resnet");
    const bool rtsp_working = has_arg(argc, argv, "--rtsp-working");
    const bool print_rtsp_pipeline = has_arg(argc, argv, "--rtsp-print");
    bool rtsp_only = has_arg(argc, argv, "--rtsp-only");

    bool bypass_rtp_caps = has_arg(argc, argv, "--rtsp-no-rtp-caps");
    bool bypass_h264_caps = has_arg(argc, argv, "--rtsp-no-h264-caps");
    bool force_manual_rtsp = has_arg(argc, argv, "--rtsp-manual");
    const bool wait_for_keyframe = !has_arg(argc, argv, "--rtsp-no-wait-keyframe");
    int h264_parse_ci = -1;
    parse_int_arg(argc, argv, "--rtsp-h264parse-ci", h264_parse_ci);
    DecoderOverrides dec_override;
    dec_override.enabled = false;
    if (sima_examples::get_arg(argc, argv, "--rtsp-dec-next", dec_override.next_element)) {
      dec_override.enabled = true;
    }
    if (sima_examples::get_arg(argc, argv, "--rtsp-dec-mode", dec_override.dec_mode)) {
      dec_override.enabled = true;
    }
    if (sima_examples::get_arg(argc, argv, "--rtsp-dec-apu-mem", dec_override.apu_mem_pool)) {
      dec_override.enabled = true;
    }
    if (parse_int_arg(argc, argv, "--rtsp-dec-fps", dec_override.dec_fps)) {
      dec_override.enabled = true;
    }
    if (parse_int_arg(argc, argv, "--rtsp-dec-width", dec_override.dec_width)) {
      dec_override.enabled = true;
    }
    if (parse_int_arg(argc, argv, "--rtsp-dec-height", dec_override.dec_height)) {
      dec_override.enabled = true;
    }

    if (rtsp_resnet && rtsp_stage < 0) rtsp_stage = 0;
    if (rtsp_stage >= 0 && rtsp_stage > 4) {
      throw std::runtime_error("Invalid --rtsp-stage: " + std::to_string(rtsp_stage) +
                               " (expected 0..4)");
    }
    if (rtsp_working) {
      bypass_rtp_caps = true;
      bypass_h264_caps = false;
      force_manual_rtsp = true;
    }
    if (rtsp_resnet) {
      bypass_rtp_caps = false;
      bypass_h264_caps = false;
      force_manual_rtsp = false;
    }
    const bool add_h264_dims_caps = bypass_h264_caps;
    int default_payload = 96;
    parse_int_arg(argc, argv, "--payload", default_payload);
    std::vector<int> payloads(static_cast<size_t>(stream_count), default_payload);
    for (int i = 0; i < stream_count; ++i) {
      parse_int_arg(argc, argv, "--payload" + std::to_string(i), payloads[i]);
    }

    int target_w = 1280;
    int target_h = 720;
    double target_fps = 30.0;
    parse_int_arg(argc, argv, "--width", target_w);
    parse_int_arg(argc, argv, "--height", target_h);
    parse_double_arg(argc, argv, "--fps", target_fps);
    if (target_w <= 0 || target_h <= 0) {
      log_warn("args",
               -1,
               "target width/height must be > 0; got " + std::to_string(target_w) +
                   "x" + std::to_string(target_h));
    }
    if (is_debug_enabled()) {
      std::cout << "[dbg] target_w=" << target_w
                << " target_h=" << target_h
                << " target_fps=" << target_fps << "\n";
    }
    const int caps_fps = (target_fps > 0.0)
        ? static_cast<int>(target_fps + 0.5)
        : -1;
    int h264_caps_w = target_w;
    int h264_caps_h = target_h;
    int h264_caps_fps = caps_fps;
    if (rtsp_working) {
      h264_caps_w = -1;
      h264_caps_h = -1;
      h264_caps_fps = -1;
    }
    if (rtsp_stage == 0 && !rtsp_only) {
      log_warn("args", -1, "--rtsp-stage 0 uses raw decoder output; enabling --rtsp-only");
      rtsp_only = true;
    }

    std::vector<StreamContext> streams(static_cast<size_t>(stream_count));
    for (int i = 0; i < stream_count; ++i) {
      streams[i].id = i;
      streams[i].url = urls[i];
      std::cout << "[rtsp] stream " << i << " url=" << urls[i] << "\n";

      sima::nodes::groups::RtspInputGroupOptions opt;
      opt.url = urls[i];
      opt.latency_ms = 200;
      opt.tcp = true;
      opt.payload_type = payloads[i];
      opt.decoder_name = "decoder";
      opt.out_format = "NV12";
      opt.decoder_raw_output = false;
      opt.use_videoconvert = false;
      opt.use_videoscale = true;
      opt.output_caps.enable = true;
      opt.output_caps.format = "NV12";
      opt.output_caps.width = target_w;
      opt.output_caps.height = target_h;
      opt.output_caps.fps = caps_fps;
      if (rtsp_stage >= 0) {
        apply_rtsp_stage(rtsp_stage, opt, target_w, target_h, caps_fps);
      }
      if (rtsp_working) {
        apply_rtsp_stage(0, opt, target_w, target_h, caps_fps);
      }
      streams[i].expected_format =
          opt.output_caps.format.empty() ? opt.out_format : opt.output_caps.format;
      if (output_caps_enabled(opt.output_caps)) {
        streams[i].expected_width = opt.output_caps.width;
        streams[i].expected_height = opt.output_caps.height;
      } else {
        streams[i].expected_width = -1;
        streams[i].expected_height = -1;
      }
      warn_rtsp_options(opt, i);
      if (is_debug_enabled()) {
        std::cout << "[dbg] rtsp opt stream=" << i
                  << " url=" << opt.url
                  << " out_format=" << opt.out_format
                  << " latency_ms=" << opt.latency_ms
                  << " caps=" << opt.output_caps.width << "x"
                  << opt.output_caps.height << "@"
                  << opt.output_caps.fps << "\n";
        log_debug("rtsp-caps",
                  i,
                  "payload=" + std::to_string(opt.payload_type) +
                      " rtp_caps=" + std::string(bypass_rtp_caps ? "off" : "on") +
                      " h264_caps=" + std::string(bypass_h264_caps ? "off" : "on") +
                      " h264_dims=" + std::string(add_h264_dims_caps ? "on" : "off") +
                      " manual=" + std::string(force_manual_rtsp ? "on" : "off") +
                      " wait_kf=" + std::string(wait_for_keyframe ? "on" : "off"));
        if (rtsp_stage >= 0) {
          log_debug("rtsp-stage", i,
                    std::to_string(rtsp_stage) + " (" + rtsp_stage_label(rtsp_stage) + ")");
        }
        if (rtsp_resnet) log_debug("rtsp-mode", i, "resnet");
        if (rtsp_working) log_debug("rtsp-mode", i, "working");
        if (rtsp_only) log_debug("rtsp-mode", i, "rtsp-only");
        if (h264_parse_ci >= 0) {
          log_debug("rtsp-h264parse", i,
                    "config-interval=" + std::to_string(h264_parse_ci));
        }
        if (dec_override.enabled) {
          std::ostringstream ss;
          ss << "next=" << (dec_override.next_element.empty() ? "-" : dec_override.next_element)
             << " mode=" << (dec_override.dec_mode.empty() ? "-" : dec_override.dec_mode)
             << " fps=" << dec_override.dec_fps
             << " w=" << dec_override.dec_width
             << " h=" << dec_override.dec_height
             << " apu=" << (dec_override.apu_mem_pool.empty() ? "-" : dec_override.apu_mem_pool);
          log_debug("rtsp-decoder", i, ss.str());
        }
      }

      if (!force_manual_rtsp && !bypass_rtp_caps && !bypass_h264_caps) {
        streams[i].session.add(sima::nodes::groups::RtspInputGroup(opt));
      } else {
        streams[i].session.add(sima::nodes::RTSPInput(opt.url, opt.latency_ms, opt.tcp));
        if (opt.insert_queue) streams[i].session.add(sima::nodes::Queue());
        const std::string depay = build_rtp_h264_fragment(
            i,
            opt.payload_type,
            !bypass_rtp_caps,
            !bypass_h264_caps,
            add_h264_dims_caps,
            wait_for_keyframe,
            h264_parse_ci,
            h264_caps_w,
            h264_caps_h,
            h264_caps_fps);
        streams[i].session.add(sima::nodes::Gst(depay));
        if (opt.insert_queue) streams[i].session.add(sima::nodes::Queue());
        if (dec_override.enabled) {
          streams[i].session.add(sima::nodes::Gst(
              build_decoder_fragment(i, opt, dec_override)));
        } else {
          streams[i].session.add(sima::nodes::H264Decode(opt.sima_allocator_type,
                                                         opt.out_format,
                                                         opt.decoder_name,
                                                         opt.decoder_raw_output));
        }
        if (opt.use_videoconvert) streams[i].session.add(sima::nodes::VideoConvert());
        if (opt.use_videoscale) streams[i].session.add(sima::nodes::VideoScale());
        if (output_caps_enabled(opt.output_caps)) {
          const auto& c = opt.output_caps;
          streams[i].session.add(
              sima::nodes::CapsRaw(c.format, c.width, c.height, c.fps, c.memory));
        }
      }
      streams[i].session.add(sima::nodes::OutputAppSink());
      if (is_trace_enabled()) {
        log_trace("rtsp-gst", i, streams[i].session.to_gst(false));
      }
      if (print_rtsp_pipeline) {
        std::cout << "[rtsp] pipeline s" << i << ":\n"
                  << streams[i].session.to_gst(false) << "\n";
      }
      streams[i].stream = streams[i].session.run_tensor();
    }

    const int target_frames = 100;
    const int total_frames = target_frames * stream_count;
    const int topk = 100;
    const float min_score = 0.52f;

    log_debug("rtsp", 0, "pulling first frame (timeout=5000ms)");
    auto first_pull_start = std::chrono::steady_clock::now();
    auto first_ref = streams[0].stream.next(5000);
    auto first_pull_end = std::chrono::steady_clock::now();
    if (!first_ref.has_value()) {
      log_debug("rtsp", 0, "first frame timeout");
    }
    require(first_ref.has_value(), "RTSP stream 0: failed to pull first frame");

    sima::FrameTensor first = first_ref->to_copy();
    if (is_debug_enabled()) {
      log_tensor_info("rtsp first tensor", first);
    }
    warn_frame_mismatch(first,
                        streams[0].expected_format,
                        streams[0].expected_width,
                        streams[0].expected_height,
                        0,
                        "rtsp-frame");
    warn_nv12_layout(first, 0);
    if (rtsp_only) {
      std::cout << "[rtsp] --rtsp-only set; stopping after first frame\n";
      for (auto& s : streams) {
        s.stream.close();
      }
      return 0;
    }
    cv::Mat first_bgr;
    std::string err;
    require(tensor_to_bgr_mat(first, first_bgr, err), err);

    auto model = sima::mpk::ModelMPK(
        tar_gz, "video/x-raw", "NV12", target_w, target_h, 1);
    warn_boxdecode_config(model);

    sima::PipelineSession yolo;
    sima::InputAppSrcOptions src_opt = model.input_appsrc_options(false);
    if (!src_opt.format.empty() && upper_copy(src_opt.format) != "NV12") {
      log_warn("yolo-input",
               -1,
               "model input format is " + src_opt.format + "; overriding to NV12");
    }
    src_opt.format = "NV12";
    warn_yolo_appsrc_options(src_opt, target_w, target_h);
    warn_frame_mismatch(first,
                        src_opt.format,
                        src_opt.width,
                        src_opt.height,
                        0,
                        "yolo-input");
    if (first.planes.size() != 1) {
      log_warn("yolo-input",
               -1,
               "FrameTensor has " + std::to_string(first.planes.size()) +
                   " planes; build(input) expects a single-plane FrameTensor");
    }
    yolo.add(sima::nodes::InputAppSrc(src_opt));
    yolo.add(sima::nodes::groups::Preprocess(model));
    yolo.add(sima::nodes::groups::MLA(model));
    yolo.add(sima::nodes::SimaBoxDecode(
        model,
        "yolov8",
        target_w,
        target_h,
        min_score,
        0.5f,
        topk));
    yolo.add(sima::nodes::OutputAppSink());
    if (is_trace_enabled()) {
      log_trace("yolo-gst", -1, yolo.to_gst(false));
    }

    sima::PipelineRunOptions yolo_opt;
    yolo_opt.input_queue = 16;
    yolo_opt.output_queue = 16;
    yolo_opt.drop = sima::DropPolicy::Block;
    yolo_opt.copy_output = true;
    yolo_opt.appsink_max_buffers = 16;
    yolo_opt.appsink_drop = false;
    yolo_opt.appsink_sync = false;

    auto yolo_run = yolo.build(first, sima::PipelineRunMode::Async, yolo_opt);

    std::deque<InFlight> inflight;
    std::mutex inflight_mu;
    std::condition_variable inflight_cv;
    std::mutex push_mu;

    std::atomic<bool> done_pushing{false};
    std::atomic<int> outputs_done{0};
    std::atomic<int> pull_timeouts{0};
    std::atomic<int> yolo_push_ok{0};
    std::atomic<int> yolo_push_fail{0};
    std::atomic<bool> logged_first_output{false};

    auto record_pull = [&](StreamContext& s, double pull_ms,
                           const std::chrono::steady_clock::time_point& pull_end) {
      std::lock_guard<std::mutex> lock(s.metrics.mu);
      s.metrics.pull_ms_sum += pull_ms;
      s.metrics.pulled += 1;
      if (s.metrics.pulled == 1) s.metrics.first_pull = pull_end;
    };

    auto record_output = [&](StreamContext& s,
                             double detect_ms,
                             double e2e_ms,
                             const std::chrono::steady_clock::time_point& t_out) {
      std::lock_guard<std::mutex> lock(s.metrics.mu);
      s.metrics.detect_ms_sum += detect_ms;
      s.metrics.e2e_ms_sum += e2e_ms;
      s.metrics.saved += 1;
      s.metrics.last_output = t_out;
    };

    std::thread output_thread([&]() {
      while (true) {
        auto out_opt = yolo_run.pull(1000);
        if (!out_opt.has_value()) {
          if (done_pushing.load()) {
            if (outputs_done.load() >= total_frames) break;
            std::lock_guard<std::mutex> lock(inflight_mu);
            if (inflight.empty()) break;
          }
          if (is_debug_enabled()) {
            int t = pull_timeouts.fetch_add(1) + 1;
            if (t % 20 == 0) {
              std::cout << "[dbg] yolo pull timeout count=" << t << "\n";
            }
          }
          continue;
        }
        if (is_debug_enabled() && !logged_first_output.exchange(true)) {
          if (out_opt->kind == sima::RunOutputKind::Tensor) {
            if (out_opt->tensor.has_value()) {
              log_tensor_info("yolo output tensor", out_opt->tensor.value());
            } else if (out_opt->tensor_ref.has_value()) {
              sima::FrameTensor copy = out_opt->tensor_ref.value().to_copy();
              log_tensor_info("yolo output tensor_ref", copy);
            } else {
              log_debug("yolo-output", -1, "tensor output missing payload");
            }
          } else {
            log_debug("yolo-output", -1, "non-tensor output kind");
          }
        }

        InFlight item;
        {
          std::unique_lock<std::mutex> lock(inflight_mu);
          inflight_cv.wait(lock, [&]() { return !inflight.empty() || done_pushing.load(); });
          if (inflight.empty()) {
            if (done_pushing.load()) break;
            continue;
          }
          item = std::move(inflight.front());
          inflight.pop_front();
        }

        auto t_out = std::chrono::steady_clock::now();
        const double detect_ms = std::chrono::duration<double, std::milli>(
            t_out - item.t_push).count();
        const double e2e_ms = std::chrono::duration<double, std::milli>(
            t_out - item.t_pull).count();

        std::vector<uint8_t> payload;
        std::string err_msg;
        if (!extract_bbox_payload(*out_opt, payload, err_msg)) {
          std::cerr << "[warn] bbox extract failed: " << err_msg << "\n";
          continue;
        }
        if (is_debug_enabled() && outputs_done.load() == 0) {
          std::cout << "[dbg] bbox payload bytes=" << payload.size() << "\n";
        }
        if (is_trace_enabled()) {
          log_trace("bbox-bytes", item.stream_id,
                    "prefix=" + hex_prefix(payload, 16));
        }

        std::vector<objdet::Box> boxes =
            objdet::parse_boxes_strict(payload, item.bgr.cols, item.bgr.rows, topk, false);
        objdet::draw_boxes(item.bgr, boxes, min_score, cv::Scalar(0, 255, 0), "det");

        StreamContext& s = streams[item.stream_id];
        if (s.saved < target_frames) {
          if (!s.writer_open) {
            const fs::path out_dir = root / "tmp" / "yolov8_multi_rtsp";
            std::error_code out_ec;
            fs::create_directories(out_dir, out_ec);
            const fs::path out_path = out_dir /
                ("stream_" + std::to_string(item.stream_id) + ".mp4");
            const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            const double out_fps = (target_fps > 0.0) ? target_fps : 30.0;
            s.writer.open(out_path.string(),
                          fourcc,
                          out_fps,
                          item.bgr.size(),
                          true);
            if (!s.writer.isOpened()) {
              std::cerr << "[warn] video writer open failed: " << out_path << "\n";
            } else {
              s.writer_open = true;
            }
          }
          if (s.writer_open) {
            s.writer.write(item.bgr);
            s.saved += 1;
            record_output(s, detect_ms, e2e_ms, t_out);
          }
        }

        outputs_done.fetch_add(1);
        if (is_debug_enabled() && outputs_done.load() % 20 == 0) {
          std::cout << "[dbg] yolo outputs_done=" << outputs_done.load() << "\n";
        }
        if (outputs_done.load() >= total_frames) {
          break;
        }
      }
    });

    std::atomic<bool> status_stop{false};
    std::thread status_thread;
    if (is_debug_enabled()) {
      status_thread = std::thread([&]() {
        while (!status_stop.load()) {
          std::this_thread::sleep_for(std::chrono::seconds(5));
          if (status_stop.load()) break;
          size_t inflight_size = 0;
          {
            std::lock_guard<std::mutex> lock(inflight_mu);
            inflight_size = inflight.size();
          }
          std::ostringstream ss;
          ss << "inflight=" << inflight_size
             << " yolo_push_ok=" << yolo_push_ok.load()
             << " yolo_push_fail=" << yolo_push_fail.load()
             << " yolo_out=" << outputs_done.load();
          for (size_t idx = 0; idx < streams.size(); ++idx) {
            StreamContext& s = streams[idx];
            int pulled = 0;
            int saved = 0;
            {
              std::lock_guard<std::mutex> lock(s.metrics.mu);
              pulled = s.metrics.pulled;
              saved = s.metrics.saved;
            }
            ss << " s" << s.id
               << "(pulled=" << pulled
               << " saved=" << saved
               << " timeouts=" << s.pull_timeouts.load() << ")";
          }
          log_debug("status", -1, ss.str());
        }
      });
    }

    auto push_frame = [&](StreamContext& s,
                          const sima::FrameTensor& nv12_frame,
                          cv::Mat frame,
                          const std::chrono::steady_clock::time_point& t_pull) {
      InFlight item;
      item.stream_id = s.id;
      item.frame_index = s.pushed;
      item.bgr = std::move(frame);
      item.t_pull = t_pull;

      {
        std::lock_guard<std::mutex> lock(push_mu);
        if (!yolo_run.push(nv12_frame)) {
          const int fail = yolo_push_fail.fetch_add(1) + 1;
          log_debug("yolo-push", s.id, "failed count=" + std::to_string(fail));
          return;
        }
        item.t_push = std::chrono::steady_clock::now();
      }
      const int ok = yolo_push_ok.fetch_add(1) + 1;
      if (is_trace_enabled()) {
        log_trace("yolo-push", s.id, "ok count=" + std::to_string(ok));
      } else if (is_debug_enabled() && ok % 50 == 0) {
        log_debug("yolo-push", s.id, "ok count=" + std::to_string(ok));
      }

      {
        std::lock_guard<std::mutex> lock(inflight_mu);
        inflight.push_back(std::move(item));
      }
      inflight_cv.notify_one();
      s.pushed += 1;
    };

    {
      StreamContext& s = streams[0];
      const double pull_ms = std::chrono::duration<double, std::milli>(
          first_pull_end - first_pull_start).count();
      record_pull(s, pull_ms, first_pull_end);
      push_frame(s, first, first_bgr, first_pull_end);
    }

    for (int i = 0; i < stream_count; ++i) {
      streams[i].th = std::thread([&, i]() {
        StreamContext& s = streams[i];
        log_debug("rtsp-thread", s.id, "start");
        while (s.pushed < target_frames) {
          auto t0 = std::chrono::steady_clock::now();
          auto frame_ref = s.stream.next(1000);
          auto t1 = std::chrono::steady_clock::now();
          if (!frame_ref.has_value()) {
            const int count = s.pull_timeouts.fetch_add(1) + 1;
            if (is_debug_enabled() && count % 20 == 0) {
              log_debug("rtsp-timeout", s.id, "count=" + std::to_string(count));
            }
            continue;
          }

          sima::FrameTensor frame = frame_ref->to_copy();
          s.pull_success.fetch_add(1);
          if (is_trace_enabled()) {
            log_trace("rtsp-pull", s.id, "ok");
          }
          if (is_debug_enabled() && s.pushed == 0) {
            log_tensor_info("rtsp stream " + std::to_string(s.id), frame);
          }
          if (s.pushed == 0) {
            warn_frame_mismatch(frame,
                                s.expected_format,
                                s.expected_width,
                                s.expected_height,
                                s.id,
                                "rtsp-frame");
            warn_nv12_layout(frame, s.id);
          }
          cv::Mat bgr;
          std::string err;
          if (!tensor_to_bgr_mat(frame, bgr, err)) {
            std::cerr << "[warn] stream " << s.id << " tensor->bgr failed: " << err << "\n";
            continue;
          }
          const double pull_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
          record_pull(s, pull_ms, t1);
          push_frame(s, frame, std::move(bgr), t1);
        }
        log_debug("rtsp-thread",
                  s.id,
                  "done pushed=" + std::to_string(s.pushed));
      });
    }

    for (auto& s : streams) {
      if (s.th.joinable()) s.th.join();
    }
    done_pushing.store(true);
    yolo_run.close_input();
    inflight_cv.notify_all();

    if (output_thread.joinable()) output_thread.join();
    status_stop.store(true);
    if (status_thread.joinable()) status_thread.join();

    for (auto& s : streams) {
      if (s.writer_open) s.writer.release();
      s.stream.close();
    }

    std::cout << "\n=== Summary ===\n";
    for (auto& s : streams) {
      std::lock_guard<std::mutex> lock(s.metrics.mu);
      const int saved = s.metrics.saved;
      const double pull_ms = (saved > 0) ? (s.metrics.pull_ms_sum / saved) : 0.0;
      const double detect_ms = (saved > 0) ? (s.metrics.detect_ms_sum / saved) : 0.0;
      const double e2e_ms = (saved > 0) ? (s.metrics.e2e_ms_sum / saved) : 0.0;
      double tput = 0.0;
      if (saved > 0) {
        const double dur_s =
            std::chrono::duration<double>(s.metrics.last_output -
                                          s.metrics.first_pull).count();
        if (dur_s > 0.0) tput = static_cast<double>(saved) / dur_s;
      }
      std::cout << "stream " << s.id
                << " saved=" << saved
                << " pull_ms=" << pull_ms
                << " detect_ms=" << detect_ms
                << " e2e_ms=" << e2e_ms
                << " fps=" << tput << "\n";
    }

    auto run_stats = yolo_run.stats();
    auto in_stats = yolo_run.input_stats();
    std::cout << "\nYOLO stats: inputs=" << run_stats.inputs_pushed
              << " outputs=" << run_stats.outputs_pulled
              << " avg_latency_ms=" << run_stats.avg_latency_ms
              << " avg_push_us=" << in_stats.avg_push_us
              << " avg_pull_wait_us=" << in_stats.avg_pull_wait_us << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
