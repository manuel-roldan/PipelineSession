#include "example_utils.h"
#include "test_utils.h"

#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "mpk/ModelMPK.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/groups/RtspInputGroup.h"
#include "nodes/io/InputAppSrc.h"
#include "nodes/io/UdpSink.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/RtpH264Pay.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/PipelineSession.h"
#include "pipeline/NeatTensorCore.h"
#include "pipeline/internal/TensorUtil.h"

#include <gst/gst.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Config {
  std::string url;
  std::string mpk;
  int frames = 300;
  bool frames_set = false;
  bool debug = false;
  bool udp = false;
  bool forever = false;
};

bool has_arg(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) return true;
  }
  return false;
}

bool parse_int_arg(int argc, char** argv, const std::string& key, int& out) {
  std::string raw;
  if (!sima_examples::get_arg(argc, argv, key, raw)) return false;
  out = std::stoi(raw);
  return true;
}

Config parse_config(int argc, char** argv) {
  Config cfg;
  sima_examples::get_arg(argc, argv, "--rtsp", cfg.url);
  sima_examples::get_arg(argc, argv, "--mpk", cfg.mpk);
  cfg.frames_set = parse_int_arg(argc, argv, "--frames", cfg.frames);
  cfg.debug = has_arg(argc, argv, "--debug");
  cfg.udp = has_arg(argc, argv, "--udp");
  cfg.forever = has_arg(argc, argv, "--forever");
  return cfg;
}

bool infer_dims(const sima::NeatTensor& t, int& w, int& h) {
  w = t.width();
  h = t.height();
  if ((w <= 0 || h <= 0) && t.shape.size() >= 2) {
    h = static_cast<int>(t.shape[0]);
    w = static_cast<int>(t.shape[1]);
  }
  return (w > 0 && h > 0);
}

bool nv12_to_bgr(const sima::NeatTensor& t, cv::Mat& out, std::string& err) {
  if (!t.is_nv12()) {
    err = "expected NV12 tensor";
    return false;
  }
  int w = 0;
  int h = 0;
  if (!infer_dims(t, w, h)) {
    err = "invalid tensor dimensions";
    return false;
  }
  std::vector<uint8_t> nv12 = t.copy_nv12_contiguous();
  if (nv12.empty()) {
    err = "NV12 copy failed";
    return false;
  }
  cv::Mat yuv(h + h / 2, w, CV_8UC1, nv12.data());
  cv::cvtColor(yuv, out, cv::COLOR_YUV2BGR_NV12);
  return true;
}

bool init_nv12_tensor_meta(sima::NeatTensor& out, int w, int h, std::string& err) {
  if (w <= 0 || h <= 0) {
    err = "invalid NV12 dimensions";
    return false;
  }
  if ((w % 2) != 0 || (h % 2) != 0) {
    err = "NV12 requires even width/height";
    return false;
  }
  out.dtype = sima::TensorDType::UInt8;
  out.layout = sima::TensorLayout::HW;
  out.shape = {h, w};
  out.strides_bytes = {w, 1};
  out.byte_offset = 0;
  out.device = {sima::NeatDeviceType::CPU, 0};
  out.read_only = true;
  sima::NeatImageSpec image;
  image.format = sima::NeatImageSpec::PixelFormat::NV12;
  out.semantic.image = image;

  sima::NeatPlane y;
  y.role = sima::NeatPlaneRole::Y;
  y.shape = {h, w};
  y.strides_bytes = {w, 1};
  y.byte_offset = 0;

  sima::NeatPlane uv;
  uv.role = sima::NeatPlaneRole::UV;
  uv.shape = {h / 2, w};
  uv.strides_bytes = {w, 1};
  uv.byte_offset = static_cast<int64_t>(w) * static_cast<int64_t>(h);

  out.planes.clear();
  out.planes.push_back(std::move(y));
  out.planes.push_back(std::move(uv));
  return true;
}

bool bgr_to_nv12_tensor(const cv::Mat& bgr, sima::NeatTensor& out, std::string& err) {
  if (bgr.empty() || bgr.data == nullptr) {
    err = "empty BGR frame";
    return false;
  }
  if (bgr.type() != CV_8UC3) {
    err = "expected CV_8UC3 BGR frame";
    return false;
  }

  const int w = bgr.cols;
  const int h = bgr.rows;
  if ((w % 2) != 0 || (h % 2) != 0) {
    err = "NV12 requires even width/height";
    return false;
  }

  cv::Mat i420;
  cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
  if (!i420.isContinuous()) i420 = i420.clone();
  const size_t expected = static_cast<size_t>(w) * static_cast<size_t>(h) * 3 / 2;
  const size_t bytes = i420.total() * i420.elemSize();
  if (bytes < expected) {
    err = "I420 buffer too small";
    return false;
  }

  auto storage = sima::make_cpu_owned_storage(expected);
  uint8_t* dst = static_cast<uint8_t*>(storage->data);
  const uint8_t* src = i420.data;
  const size_t y_bytes = static_cast<size_t>(w) * static_cast<size_t>(h);
  std::memcpy(dst, src, y_bytes);

  const uint8_t* src_u = src + y_bytes;
  const uint8_t* src_v = src_u + (static_cast<size_t>(w) / 2) * (static_cast<size_t>(h) / 2);
  uint8_t* dst_uv = dst + y_bytes;
  for (int row = 0; row < h / 2; ++row) {
    const uint8_t* u_row = src_u + row * (w / 2);
    const uint8_t* v_row = src_v + row * (w / 2);
    uint8_t* uv_row = dst_uv + row * w;
    for (int col = 0; col < w / 2; ++col) {
      uv_row[col * 2] = u_row[col];
      uv_row[col * 2 + 1] = v_row[col];
    }
  }

  out = sima::NeatTensor{};
  out.storage = std::move(storage);
  if (!init_nv12_tensor_meta(out, w, h, err)) return false;
  return true;
}

bool make_blank_nv12_tensor(int w, int h, sima::NeatTensor& out, std::string& err) {
  if ((w % 2) != 0 || (h % 2) != 0) {
    err = "NV12 requires even width/height";
    return false;
  }
  const size_t bytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 3 / 2;
  auto storage = sima::make_cpu_owned_storage(bytes);
  uint8_t* data = static_cast<uint8_t*>(storage->data);
  if (!data) {
    err = "NV12 storage allocation failed";
    return false;
  }
  const size_t y_bytes = static_cast<size_t>(w) * static_cast<size_t>(h);
  std::memset(data, 0, y_bytes);
  std::memset(data + y_bytes, 128, bytes - y_bytes);

  out = sima::NeatTensor{};
  out.storage = std::move(storage);
  if (!init_nv12_tensor_meta(out, w, h, err)) return false;
  return true;
}

bool extract_bbox_payload(const sima::RunInputResult& result,
                          std::vector<uint8_t>& payload,
                          std::string& err) {
  if (result.kind == sima::RunOutputKind::Bundle) {
    for (const auto& field : result.fields) {
      if (extract_bbox_payload(field, payload, err)) return true;
    }
    err = "bundle missing BBOX field";
    return false;
  }
  if (result.kind != sima::RunOutputKind::Tensor) {
    err = "expected tensor output";
    return false;
  }
  if (!result.neat.has_value()) {
    err = "missing tensor output";
    return false;
  }
  const std::string tag =
      !result.payload_tag.empty() ? result.payload_tag : result.format;
  if (tag != "BBOX" && tag != "bbox") {
    err = "unexpected tensor format: " + tag;
    return false;
  }
  try {
    payload = result.neat->copy_payload_bytes();
  } catch (const std::exception& ex) {
    err = std::string("bbox payload copy failed: ") + ex.what();
    return false;
  }
  if (payload.empty()) {
    err = "bbox payload empty";
    return false;
  }
  return true;
}

double time_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void print_time(const char* label, double ms, bool enabled) {
  if (!enabled) return;
  std::cout << label << " " << ms << "\n";
}

struct FrameItem {
  int index = 0;
  sima::NeatTensor frame;
  double pull_ts_ms = 0.0;
};

struct FrameQueue {
  explicit FrameQueue(size_t max_size_in) : max_size(max_size_in) {}

  bool push(FrameItem item) {
    std::unique_lock<std::mutex> lock(mu);
    cond.wait(lock, [&]() { return closed || items.size() < max_size; });
    if (closed) return false;
    items.push_back(std::move(item));
    lock.unlock();
    cond.notify_all();
    return true;
  }

  bool pop(FrameItem& out) {
    std::unique_lock<std::mutex> lock(mu);
    cond.wait(lock, [&]() { return closed || !items.empty(); });
    if (items.empty()) return false;
    out = std::move(items.front());
    items.pop_front();
    lock.unlock();
    cond.notify_all();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu);
    closed = true;
    cond.notify_all();
  }

private:
  size_t max_size = 0;
  std::mutex mu;
  std::condition_variable cond;
  std::deque<FrameItem> items;
  bool closed = false;
};

struct ProducerTiming {
  int count = 0;
  double rtsp_pull_sum = 0.0;
  double rtsp_pull_max = 0.0;
  double queue_push_sum = 0.0;
  double queue_push_max = 0.0;

  void add_rtsp_pull(double ms) {
    rtsp_pull_sum += ms;
    if (ms > rtsp_pull_max) rtsp_pull_max = ms;
  }

  void add_queue_push(double ms) {
    queue_push_sum += ms;
    if (ms > queue_push_max) queue_push_max = ms;
  }

  void print() const {
    if (count <= 0) return;
    std::cout << "producer_avg_rtsp_pull_ms " << (rtsp_pull_sum / count)
              << " producer_max_rtsp_pull_ms " << rtsp_pull_max
              << " producer_avg_queue_push_ms " << (queue_push_sum / count)
              << " producer_max_queue_push_ms " << queue_push_max << "\n";
  }
};

struct ConsumerTiming {
  int count = 0;
  double queue_pop_sum = 0.0;
  double queue_pop_max = 0.0;
  double convert_sum = 0.0;
  double convert_max = 0.0;
  double udp_convert_sum = 0.0;
  double udp_convert_max = 0.0;
  double yolo_push_sum = 0.0;
  double yolo_push_max = 0.0;
  double yolo_pull_sum = 0.0;
  double yolo_pull_max = 0.0;
  double udp_push_sum = 0.0;
  double udp_push_max = 0.0;
  double bbox_extract_sum = 0.0;
  double bbox_extract_max = 0.0;
  double bbox_parse_sum = 0.0;
  double bbox_parse_max = 0.0;
  double overlay_sum = 0.0;
  double overlay_max = 0.0;
  double write_sum = 0.0;
  double write_max = 0.0;
  double e2e_sum = 0.0;
  double e2e_max = 0.0;

  void add_queue_pop(double ms) {
    queue_pop_sum += ms;
    if (ms > queue_pop_max) queue_pop_max = ms;
  }

  void add_convert(double ms) {
    convert_sum += ms;
    if (ms > convert_max) convert_max = ms;
  }

  void add_udp_convert(double ms) {
    udp_convert_sum += ms;
    if (ms > udp_convert_max) udp_convert_max = ms;
  }

  void add_yolo_push(double ms) {
    yolo_push_sum += ms;
    if (ms > yolo_push_max) yolo_push_max = ms;
  }

  void add_yolo_pull(double ms) {
    yolo_pull_sum += ms;
    if (ms > yolo_pull_max) yolo_pull_max = ms;
  }

  void add_udp_push(double ms) {
    udp_push_sum += ms;
    if (ms > udp_push_max) udp_push_max = ms;
  }

  void add_bbox_extract(double ms) {
    bbox_extract_sum += ms;
    if (ms > bbox_extract_max) bbox_extract_max = ms;
  }

  void add_bbox_parse(double ms) {
    bbox_parse_sum += ms;
    if (ms > bbox_parse_max) bbox_parse_max = ms;
  }

  void add_overlay(double ms) {
    overlay_sum += ms;
    if (ms > overlay_max) overlay_max = ms;
  }

  void add_write(double ms) {
    write_sum += ms;
    if (ms > write_max) write_max = ms;
  }

  void add_e2e(double ms) {
    e2e_sum += ms;
    if (ms > e2e_max) e2e_max = ms;
  }

  void print() const {
    if (count <= 0) return;
    std::cout << "consumer_avg_queue_pop_ms " << (queue_pop_sum / count)
              << " consumer_max_queue_pop_ms " << queue_pop_max
              << " consumer_avg_convert_ms " << (convert_sum / count)
              << " consumer_max_convert_ms " << convert_max
              << " consumer_avg_udp_convert_ms " << (udp_convert_sum / count)
              << " consumer_max_udp_convert_ms " << udp_convert_max
              << " consumer_avg_yolo_push_ms " << (yolo_push_sum / count)
              << " consumer_max_yolo_push_ms " << yolo_push_max
              << " consumer_avg_yolo_pull_ms " << (yolo_pull_sum / count)
              << " consumer_max_yolo_pull_ms " << yolo_pull_max
              << " consumer_avg_udp_push_ms " << (udp_push_sum / count)
              << " consumer_max_udp_push_ms " << udp_push_max
              << " consumer_avg_bbox_extract_ms " << (bbox_extract_sum / count)
              << " consumer_max_bbox_extract_ms " << bbox_extract_max
              << " consumer_avg_bbox_parse_ms " << (bbox_parse_sum / count)
              << " consumer_max_bbox_parse_ms " << bbox_parse_max
              << " consumer_avg_overlay_ms " << (overlay_sum / count)
              << " consumer_max_overlay_ms " << overlay_max
              << " consumer_avg_write_ms " << (write_sum / count)
              << " consumer_max_write_ms " << write_max
              << " consumer_avg_e2e_ms " << (e2e_sum / count)
              << " consumer_max_e2e_ms " << e2e_max << "\n";
  }
};

} // namespace

int main(int argc, char** argv) {
  try {
    gst_init(nullptr, nullptr);

    Config cfg = parse_config(argc, argv);
    require(!cfg.url.empty(), "Missing --rtsp <url>");
    if (cfg.mpk.empty()) cfg.mpk = sima_examples::resolve_yolov8s_tar(fs::current_path());
    require(!cfg.mpk.empty(), "Failed to locate yolo_v8s MPK tarball");

    sima::PipelineSession camera;
    sima::nodes::groups::RtspInputGroupOptions cam_opt;
    cam_opt.url = cfg.url;
    cam_opt.decoder_name = "decoder";
    camera.add(sima::nodes::groups::RtspInputGroup(cam_opt));
    camera.add(sima::nodes::OutputAppSink());
    auto cam = camera.build();

    const double first_pull_start = time_ms();
    sima::NeatTensor first = cam.pull_neat_or_throw(5000);
    const double first_pull_end = time_ms();
    const double first_pull_ms = first_pull_end - first_pull_start;
    const double first_pull_ts = first_pull_end;
    int frame_w = 0;
    int frame_h = 0;
    require(infer_dims(first, frame_w, frame_h), "first frame missing dimensions");

    const bool use_udp = cfg.udp;
    const std::string udp_host = "192.168.193.205";
    const int udp_port = 5000;
    fs::path out_path;
    cv::VideoWriter writer;
    sima::PipelineRun udp_run;

    if (use_udp) {
      sima::PipelineSession udp;
      sima::InputAppSrcOptions udp_src;
      udp_src.format = "NV12";
      udp_src.width = frame_w;
      udp_src.height = frame_h;
      udp_src.caps_override = "video/x-raw,format=NV12,width=" +
          std::to_string(frame_w) + ",height=" + std::to_string(frame_h) +
          ",framerate=30/1";
      udp_src.use_simaai_pool = false;
      udp.add(sima::nodes::InputAppSrc(udp_src));
      udp.add(sima::nodes::H264EncodeSima(frame_w, frame_h, 30, 4000));
      udp.add(sima::nodes::H264Parse());
      udp.add(sima::nodes::RtpH264Pay(96, 1));
      sima::UdpSinkOptions udp_opt;
      udp_opt.host = udp_host;
      udp_opt.port = udp_port;
      udp.add(sima::nodes::UdpSink(udp_opt));

      sima::NeatTensor udp_dummy;
      std::string udp_err;
      require(make_blank_nv12_tensor(frame_w, frame_h, udp_dummy, udp_err), udp_err);
      udp_run = udp.build(udp_dummy, sima::PipelineRunMode::Async);
      std::cout << "udp=" << udp_host << ":" << udp_port << "\n";
    } else {
      out_path = fs::path("out") / "stream_0.mp4";
      std::error_code out_ec;
      fs::create_directories(out_path.parent_path(), out_ec);
      fs::remove(out_path, out_ec);
      std::string writer_err;
      require(sima_examples::open_h264_writer(writer, out_path, frame_w, frame_h, 30.0, 4000, &writer_err),
              writer_err);
    }

    sima::mpk::ModelMPK model(cfg.mpk, "video/x-raw", "NV12", frame_w, frame_h, 1);
    sima::PipelineSession yolo;
    sima::InputAppSrcOptions src_opt = model.input_appsrc_options(false);
    src_opt.format = "NV12";
    src_opt.width = frame_w;
    src_opt.height = frame_h;
    yolo.add(sima::nodes::InputAppSrc(src_opt));
    yolo.add(sima::nodes::groups::Preprocess(model));
    yolo.add(sima::nodes::groups::MLA(model));
    const int topk = 100;
    const float min_score = 0.52f;
    yolo.add(sima::nodes::SimaBoxDecode(model, "yolov8", frame_w, frame_h, min_score, 0.5f, topk));
    yolo.add(sima::nodes::OutputAppSink());
    auto det = yolo.build(first, sima::PipelineRunMode::Async);

    if (!use_udp && cfg.forever) {
      require(false, "Video output cannot run forever; pass --frames N");
    }
    if (cfg.frames_set && cfg.frames <= 0) {
      require(false, "--frames must be > 0");
    }

    std::optional<int> frame_limit;
    if (cfg.frames_set) {
      frame_limit = cfg.frames;
    } else if (cfg.forever || use_udp) {
      frame_limit = std::nullopt;
    } else {
      frame_limit = cfg.frames;
    }
    std::cout << "mode=" << (use_udp ? "udp" : "video")
              << " frame_limit=" << (frame_limit ? std::to_string(*frame_limit) : "inf")
              << " frames_set=" << (cfg.frames_set ? "1" : "0")
              << " forever=" << (cfg.forever ? "1" : "0") << "\n";

    FrameQueue queue(300);
    ProducerTiming producer_stats;
    ConsumerTiming consumer_stats;
    std::atomic<bool> stop{false};
    std::atomic<int> saved{0};

    std::thread producer([&]() {
      int produced = 0;
      bool use_first = true;
      while (!stop.load() && (!frame_limit || produced < *frame_limit)) {
        sima::NeatTensor frame;
        double pull_ms = 0.0;
        double pull_ts = 0.0;
        if (use_first) {
          frame = std::move(first);
          use_first = false;
          pull_ms = first_pull_ms;
          pull_ts = first_pull_ts;
        } else {
          const double t0 = time_ms();
          auto frame_opt = cam.pull_neat();
          if (!frame_opt.has_value()) continue;
          const double t1 = time_ms();
          frame = std::move(*frame_opt);
          pull_ms = t1 - t0;
          pull_ts = t1;
        }
        print_time("rtsp_pull_ms", pull_ms, cfg.debug);
        producer_stats.add_rtsp_pull(pull_ms);

        FrameItem item;
        item.index = produced;
        item.frame = std::move(frame);
        item.pull_ts_ms = pull_ts;

        const double q0 = time_ms();
        if (!queue.push(std::move(item))) break;
        const double q1 = time_ms();
        const double queue_ms = q1 - q0;
        print_time("queue_push_ms", queue_ms, cfg.debug);
        producer_stats.add_queue_push(queue_ms);

        produced += 1;
        producer_stats.count = produced;
      }
      queue.close();
    });

    std::thread consumer([&]() {
      int out_pulls = 0;
      while (!stop.load() && (!frame_limit || saved.load() < *frame_limit)) {
        FrameItem item;
        const double q0 = time_ms();
        if (!queue.pop(item)) break;
        const double q1 = time_ms();
        const double queue_ms = q1 - q0;
        print_time("queue_pop_ms", queue_ms, cfg.debug);
        consumer_stats.add_queue_pop(queue_ms);

        const double t_convert0 = time_ms();
        cv::Mat bgr;
        std::string bgr_err;
        if (!nv12_to_bgr(item.frame, bgr, bgr_err)) {
          std::cerr << "[warn] nv12->bgr failed: " << bgr_err << "\n";
          continue;
        }
        const double t_convert1 = time_ms();
        const double convert_ms = t_convert1 - t_convert0;
        print_time("nv12_to_bgr_ms", convert_ms, cfg.debug);
        consumer_stats.add_convert(convert_ms);

        const double t_push0 = time_ms();
        bool pushed = false;
        auto holder = sima::pipeline_internal::holder_from_neat_tensor(item.frame);
        if (holder) pushed = det.push_holder(holder);
        if (!holder || !pushed) pushed = det.push(item.frame);
        const double t_push1 = time_ms();
        const double push_ms = t_push1 - t_push0;
        print_time("yolo_push_ms", push_ms, cfg.debug);
        consumer_stats.add_yolo_push(push_ms);
        if (!pushed) {
          std::cerr << "[warn] push failed\n";
          continue;
        }

        const double t_pull0 = time_ms();
        auto out_opt = det.pull();
        const double t_pull1 = time_ms();
        const double pull_ms = t_pull1 - t_pull0;
        print_time("yolo_pull_ms", pull_ms, cfg.debug);
        consumer_stats.add_yolo_pull(pull_ms);
        if (!out_opt.has_value()) continue;
        out_pulls += 1;
        if (cfg.debug) {
          std::cout << "[dbg] det pull=" << out_pulls
                    << " kind=" << static_cast<int>(out_opt->kind)
                    << " tag=" << out_opt->payload_tag
                    << " format=" << out_opt->format
                    << " frame_id=" << out_opt->frame_id
                    << " input_seq=" << out_opt->input_seq << "\n";
        }

        const double t_extract0 = time_ms();
        std::vector<uint8_t> payload;
        std::string err;
        if (!extract_bbox_payload(*out_opt, payload, err)) {
          std::cerr << "[warn] bbox extract failed: " << err << "\n";
          continue;
        }
        const double t_extract1 = time_ms();
        const double extract_ms = t_extract1 - t_extract0;
        print_time("bbox_extract_ms", extract_ms, cfg.debug);
        consumer_stats.add_bbox_extract(extract_ms);

        const double t_parse0 = time_ms();
        std::vector<objdet::Box> boxes;
        try {
          boxes = objdet::parse_boxes_strict(payload, frame_w, frame_h, topk, false);
        } catch (const std::exception& ex) {
          std::cerr << "[warn] bbox parse failed: " << ex.what() << "\n";
          continue;
        }
        const double t_parse1 = time_ms();
        const double parse_ms = t_parse1 - t_parse0;
        print_time("bbox_parse_ms", parse_ms, cfg.debug);
        consumer_stats.add_bbox_parse(parse_ms);

        const double t_overlay0 = time_ms();
        objdet::draw_boxes(bgr, boxes, min_score, cv::Scalar(0, 255, 0), "det");
        const double t_overlay1 = time_ms();
        const double overlay_ms = t_overlay1 - t_overlay0;
        print_time("overlay_ms", overlay_ms, cfg.debug);
        consumer_stats.add_overlay(overlay_ms);

        std::cout << "boxes=" << boxes.size() << "\n";

        double output_ts = 0.0;
        if (use_udp) {
          const double t_udp_conv0 = time_ms();
          sima::NeatTensor nv12_frame;
          std::string nv12_err;
          if (!bgr_to_nv12_tensor(bgr, nv12_frame, nv12_err)) {
            std::cerr << "[warn] bgr->nv12 failed: " << nv12_err << "\n";
            continue;
          }
          const double t_udp_conv1 = time_ms();
          const double udp_conv_ms = t_udp_conv1 - t_udp_conv0;
          print_time("bgr_to_nv12_ms", udp_conv_ms, cfg.debug);
          consumer_stats.add_udp_convert(udp_conv_ms);

          const double t_udp_push0 = time_ms();
          if (!udp_run.push(nv12_frame)) {
            std::cerr << "[warn] udp push failed\n";
            continue;
          }
          const double t_udp_push1 = time_ms();
          const double udp_push_ms = t_udp_push1 - t_udp_push0;
          print_time("udp_push_ms", udp_push_ms, cfg.debug);
          consumer_stats.add_udp_push(udp_push_ms);
          output_ts = t_udp_push1;
        } else {
          const double t_write0 = time_ms();
          writer.write(bgr);
          const double t_write1 = time_ms();
          const double write_ms = t_write1 - t_write0;
          print_time("write_ms", write_ms, cfg.debug);
          consumer_stats.add_write(write_ms);
          output_ts = t_write1;
        }

        const double e2e_ms = output_ts - item.pull_ts_ms;
        print_time("e2e_ms", e2e_ms, cfg.debug);
        consumer_stats.add_e2e(e2e_ms);

        const int saved_now = saved.fetch_add(1) + 1;
        consumer_stats.count = saved_now;
      }
      stop.store(true);
      queue.close();
    });

    if (producer.joinable()) producer.join();
    if (consumer.joinable()) consumer.join();

    if (!use_udp) {
      writer.release();
      std::cout << "saved=" << saved.load() << " video=" << out_path << "\n";
    } else {
      std::cout << "saved=" << saved.load()
                << " udp=" << udp_host << ":" << udp_port << "\n";
    }
    producer_stats.print();
    consumer_stats.print();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
