#include "ModelSession.hpp"

#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "nodes/io/InputAppSrc.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/DispatcherRecovery.h"
#include "pipeline/internal/SimaaiGuard.h"
#include "pipeline/NeatTensorAdapters.h"
#include "pipeline/Errors.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

#if __has_include(<simaai/gstsimaaibufferpool.h>)
#include <simaai/gstsimaaibufferpool.h>
#define SIMA_HAS_SIMAAI_POOL 1
#else
#define SIMA_HAS_SIMAAI_POOL 0
#endif

namespace sima {
GstCaps* build_caps_with_override(const char* where,
                                  const std::string& media_type,
                                  const std::string& format,
                                  int width,
                                  int height,
                                  int depth,
                                  const std::string& caps_override);
} // namespace sima

namespace simaai {
namespace {

std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

struct InputCapsConfig {
  std::string media_type;
  std::string format;
  int width = -1;
  int height = -1;
  int depth = -1;
  size_t bytes = 0;
};

InputCapsConfig infer_input_caps(const sima::InputAppSrcOptions& opt,
                                 const cv::Mat& input) {
  if (input.empty()) {
    throw std::invalid_argument("ModelSession::run_tensor: input frame is empty");
  }

  InputCapsConfig out;
  out.media_type = opt.media_type.empty() ? "video/x-raw" : opt.media_type;

  const int in_w = input.cols;
  const int in_h = input.rows;
  const int in_c = input.channels();

  const bool is_video = (out.media_type == "video/x-raw");
  const bool is_tensor = (out.media_type == "application/vnd.simaai.tensor");

  if (!is_video && !is_tensor) {
    throw std::invalid_argument(
        "ModelSession::run_tensor: unsupported media_type: " + out.media_type);
  }

  std::string fmt = upper_copy(opt.format);
  if (is_video) {
    if (fmt.empty()) {
      fmt = (in_c == 1) ? "GRAY8" : "BGR";
    }
    if (fmt == "GRAY") fmt = "GRAY8";

    if (fmt != "RGB" && fmt != "BGR" && fmt != "GRAY8") {
      throw std::invalid_argument(
          "ModelSession::run_tensor: unsupported video format: " + fmt);
    }

    if ((fmt == "RGB" || fmt == "BGR") && input.type() != CV_8UC3) {
      throw std::invalid_argument(
          "ModelSession::run_tensor: expected CV_8UC3 for video/x-raw " + fmt);
    }
    if (fmt == "GRAY8" && input.type() != CV_8UC1) {
      throw std::invalid_argument(
          "ModelSession::run_tensor: expected CV_8UC1 for video/x-raw GRAY8");
    }

    out.width = (opt.width > 0) ? opt.width : in_w;
    out.height = (opt.height > 0) ? opt.height : in_h;
    if (opt.width > 0 && opt.width != in_w) {
      throw std::invalid_argument(
          "ModelSession::run_tensor: input width does not match InputAppSrcOptions");
    }
    if (opt.height > 0 && opt.height != in_h) {
      throw std::invalid_argument(
          "ModelSession::run_tensor: input height does not match InputAppSrcOptions");
    }
  } else {
    if (fmt.empty()) fmt = "FP32";
    if (fmt != "FP32") {
      throw std::invalid_argument(
          "ModelSession::run_tensor: only FP32 tensor input is supported");
    }
    if (input.type() != CV_32FC1 && input.type() != CV_32FC3) {
      throw std::invalid_argument(
          "ModelSession::run_tensor: tensor input must be CV_32FC1 or CV_32FC3");
    }
    out.width = (opt.width > 0) ? opt.width : in_w;
    out.height = (opt.height > 0) ? opt.height : in_h;
    out.depth = (opt.depth > 0) ? opt.depth : in_c;
    if (opt.width > 0 && opt.width != in_w) {
      throw std::invalid_argument(
          "ModelSession::run_tensor: tensor input width does not match InputAppSrcOptions");
    }
    if (opt.height > 0 && opt.height != in_h) {
      throw std::invalid_argument(
          "ModelSession::run_tensor: tensor input height does not match InputAppSrcOptions");
    }
    if (opt.depth > 0 && opt.depth != in_c) {
      throw std::invalid_argument(
          "ModelSession::run_tensor: tensor input depth does not match InputAppSrcOptions");
    }
  }

  out.format = fmt;
  out.bytes = input.total() * input.elemSize();
  return out;
}

bool caps_equal(const InputCapsConfig& a, const InputCapsConfig& b) {
  return a.media_type == b.media_type &&
         a.format == b.format &&
         a.width == b.width &&
         a.height == b.height &&
         a.depth == b.depth &&
         a.bytes == b.bytes;
}

GstCaps* build_input_caps(const InputCapsConfig& cfg,
                          const sima::InputAppSrcOptions& opt) {
  return sima::build_caps_with_override("ModelSession::run_tensor",
                                        cfg.media_type,
                                        cfg.format,
                                        cfg.width,
                                        cfg.height,
                                        cfg.depth,
                                        opt.caps_override);
}

void configure_appsrc(GstElement* appsrc, const sima::InputAppSrcOptions& opt) {
  if (!appsrc) return;
  g_object_set(G_OBJECT(appsrc),
               "is-live", opt.is_live ? TRUE : FALSE,
               "format", GST_FORMAT_TIME,
               "do-timestamp", opt.do_timestamp ? TRUE : FALSE,
               "block", opt.block ? TRUE : FALSE,
               "stream-type", opt.stream_type,
               "max-bytes", static_cast<guint64>(opt.max_bytes),
               nullptr);
}

bool appsrc_has_explicit_caps(const sima::InputAppSrcOptions& opt) {
  return !opt.format.empty() || opt.width > 0 || opt.height > 0 || opt.depth > 0;
}

std::uint64_t resolve_appsrc_max_bytes(const sima::InputAppSrcOptions& opt,
                                       const InputCapsConfig& cfg) {
  if (opt.max_bytes > 0) return opt.max_bytes;
  if (appsrc_has_explicit_caps(opt)) return opt.max_bytes;
  return static_cast<std::uint64_t>(cfg.bytes);
}

void configure_appsink_for_input(GstElement* appsink) {
  if (!appsink) return;
  g_object_set(G_OBJECT(appsink),
               "emit-signals", FALSE,
               "max-buffers", 1,
               "drop", FALSE,
               "sync", TRUE,
               "enable-last-sample", FALSE,
               "qos", FALSE,
               nullptr);
}

struct InputBufferPoolGuard {
#if SIMA_HAS_SIMAAI_POOL
  std::unique_ptr<GstBufferPool, decltype(&gst_simaai_free_buffer_pool)> pool{
      nullptr, gst_simaai_free_buffer_pool};
#else
  std::unique_ptr<GstBufferPool, void(*)(GstBufferPool*)> pool{nullptr, +[](GstBufferPool*) {}};
#endif
};

GstBuffer* allocate_input_buffer(size_t bytes,
                                 const sima::InputAppSrcOptions& opt,
                                 InputBufferPoolGuard& guard) {
#if SIMA_HAS_SIMAAI_POOL
  if (opt.use_simaai_pool) {
    GstBufferPool* pool = guard.pool.get();
    if (!pool) {
      gst_simaai_segment_memory_init_once();
      GstMemoryFlags flags = static_cast<GstMemoryFlags>(
          GST_SIMAAI_MEMORY_TARGET_EV74 | GST_SIMAAI_MEMORY_FLAG_CACHED);
      GstBufferPool* new_pool = gst_simaai_allocate_buffer_pool(
          /*allocator_user_data=*/nullptr,
          gst_simaai_memory_get_segment_allocator(),
          bytes,
          opt.pool_min_buffers,
          opt.pool_max_buffers,
          flags);
      if (new_pool) {
        guard.pool.reset(new_pool);
        pool = new_pool;
      }
    }

    if (pool) {
      GstBuffer* buf = nullptr;
      if (gst_buffer_pool_acquire_buffer(pool, &buf, nullptr) == GST_FLOW_OK && buf) {
        return buf;
      }
      return nullptr;
    }
    return nullptr;
  }
#else
  (void)opt;
  (void)guard;
#endif

  return gst_buffer_new_allocate(nullptr, bytes, nullptr);
}

int64_t next_input_frame_id() {
  static std::atomic<int64_t> next_id{0};
  return next_id.fetch_add(1);
}

void maybe_add_simaai_meta(GstBuffer* buffer,
                           int64_t frame_id,
                           const sima::InputAppSrcOptions& opt) {
#if SIMA_HAS_SIMAAI_POOL
  if (!buffer || !opt.use_simaai_pool) return;
  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  if (!meta) return;
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) return;
  const std::string name = opt.buffer_name.empty() ? "decoder" : opt.buffer_name;
  gint64 phys_addr =
      gst_simaai_segment_memory_get_phys_addr(gst_buffer_peek_memory(buffer, 0));
  gst_structure_set(s,
                    "buffer-id", G_TYPE_INT64, phys_addr,
                    "buffer-name", G_TYPE_STRING, name.c_str(),
                    "buffer-offset", G_TYPE_INT64, static_cast<gint64>(0),
                    "frame-id", G_TYPE_INT64, static_cast<gint64>(frame_id),
                    "stream-id", G_TYPE_STRING, "0",
                    "timestamp", G_TYPE_UINT64, static_cast<guint64>(0),
                    nullptr);
#else
  (void)buffer;
  (void)frame_id;
  (void)opt;
#endif
}

void set_state_or_throw(GstElement* pipeline,
                        GstState target,
                        const char* where) {
  if (!pipeline) {
    throw std::runtime_error(std::string(where) + ": pipeline is null");
  }

  bool recovery_attempted = false;
  while (true) {
    GstStateChangeReturn r = gst_element_set_state(pipeline, target);
    if (r == GST_STATE_CHANGE_FAILURE) {
      sima::pipeline_internal::maybe_dump_dot(pipeline,
                                              std::string(where) + "_set_state_failure");
      throw std::runtime_error(std::string(where) + ": failed to set state");
    }

    GstState cur = GST_STATE_VOID_PENDING;
    GstState pend = GST_STATE_VOID_PENDING;
    gst_element_get_state(pipeline, &cur, &pend, 2 * GST_SECOND);

    sima::pipeline_internal::drain_bus(pipeline, nullptr, where);
    try {
      sima::pipeline_internal::throw_if_bus_error(pipeline, nullptr, where);
    } catch (const sima::PipelineError& e) {
      sima::PipelineReport rep = e.report();
      const bool allow_recover =
          sima::pipeline_internal::env_bool("SIMA_DISPATCHER_AUTO_RECOVER", true);
      if (sima::pipeline_internal::is_dispatcher_unavailable(rep) &&
          !recovery_attempted &&
          allow_recover) {
        recovery_attempted = true;
        sima::pipeline_internal::attempt_dispatcher_recovery(&rep, allow_recover);
        continue;
      }
      throw;
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      const bool allow_recover =
          sima::pipeline_internal::env_bool("SIMA_DISPATCHER_AUTO_RECOVER", true);
      if (sima::pipeline_internal::match_dispatcher_unavailable(msg) &&
          !recovery_attempted &&
          allow_recover) {
        recovery_attempted = true;
        sima::PipelineReport rep;
        rep.repro_note = msg;
        rep.error_code = sima::pipeline_internal::kDispatcherUnavailableError;
        sima::pipeline_internal::attempt_dispatcher_recovery(&rep, allow_recover);
        continue;
      }
      throw;
    }
    return;
  }
}

cv::Mat tensor_to_mat(const sima::NeatTensor& t) {
  if (!t.is_dense()) {
    throw std::runtime_error("ModelSession::run: empty tensor output");
  }

  if (t.dtype != sima::TensorDType::Float32) {
    throw std::runtime_error("ModelSession::run: only Float32 tensor supported");
  }

  sima::NeatTensor cpu = t.clone();
  sima::NeatMapping map = cpu.map(sima::NeatMapMode::Read);
  if (!map.data || map.size_bytes == 0) {
    throw std::runtime_error("ModelSession::run: empty tensor plane");
  }
  if (map.size_bytes % sizeof(float) != 0) {
    throw std::runtime_error("ModelSession::run: tensor byte size mismatch");
  }

  const size_t elems = map.size_bytes / sizeof(float);
  cv::Mat out(1, static_cast<int>(elems), CV_32FC1);
  std::memcpy(out.data, map.data, elems * sizeof(float));
  return out;
}

} // namespace

struct ModelSession::StreamState {
  sima::InputAppSrcOptions src_opt;
  InputCapsConfig caps;
  bool caps_set = false;

  GstElement* pipeline = nullptr;
  GstElement* appsrc = nullptr;
  GstElement* appsink = nullptr;

  InputBufferPoolGuard pool_guard;
  std::string pipeline_string;
};

ModelSession::ModelSession() = default;
ModelSession::ModelSession(ModelSession&&) noexcept = default;
ModelSession& ModelSession::operator=(ModelSession&&) noexcept = default;

ModelSession::~ModelSession() {
  close();
}

bool ModelSession::init(const std::string& tar_gz) {
  close();
  sima::nodes::groups::InferOptions opt;
  build_session(tar_gz, opt, /*tensor_mode=*/true);
  return initialized_;
}

bool ModelSession::init(const std::string& tar_gz,
                        int input_width,
                        int input_height,
                        const std::string& input_format,
                        bool normalize,
                        std::vector<float> channel_mean,
                        std::vector<float> channel_stddev) {
  close();
  sima::nodes::groups::InferOptions opt;
  opt.input_width = input_width;
  opt.input_height = input_height;
  opt.input_format = input_format;
  opt.normalize = normalize;
  opt.mean = std::move(channel_mean);
  opt.stddev = std::move(channel_stddev);
  build_session(tar_gz, opt, /*tensor_mode=*/false);
  return initialized_;
}

void ModelSession::build_session(const std::string& tar_gz,
                                 const sima::nodes::groups::InferOptions& opt,
                                 bool tensor_mode) {
  teardown_stream();
  tensor_mode_ = tensor_mode;
  initialized_ = false;
  last_error_.clear();

  std::string guard_error;
  auto guard = sima::pipeline_internal::acquire_simaai_guard(
      "ModelSession::init", std::string{}, /*force=*/true, &guard_error);
  if (!guard) {
    last_error_ = guard_error.empty()
        ? "ModelSession::init: failed to acquire single-owner guard"
        : guard_error;
    return;
  }

  sima::PipelineSession sess;
  sess.set_guard(guard);

  sima::InputAppSrcOptions src_opt;
  if (tensor_mode) {
    pack_ = std::make_unique<sima::mpk::ModelMPK>(tar_gz);
    src_opt = pack_->input_appsrc_options(/*tensor_mode=*/true);
    sess.add(sima::nodes::InputAppSrc(src_opt));
    sess.add(pack_->to_node_group(sima::mpk::ModelStage::Full));
  } else {
    pack_ = std::make_unique<sima::mpk::ModelMPK>(
        tar_gz,
        "video/x-raw",
        opt.input_format,
        opt.input_width,
        opt.input_height,
        0,
        opt.normalize,
        opt.mean,
        opt.stddev,
        opt.preproc_next_cpu,
        opt.upstream_name,
        opt.num_buffers_cvu,
        opt.num_buffers_mla,
        opt.queue_max_buffers,
        opt.queue_max_time_ns,
        opt.queue_leaky);
    src_opt = pack_->input_appsrc_options(/*tensor_mode=*/false);
    sess.add(sima::nodes::InputAppSrc(src_opt));
    sess.add(pack_->to_node_group(sima::mpk::ModelStage::Full));
  }

  sess.add(sima::nodes::OutputAppSink());
  session_ = std::move(sess);
  guard_ = std::move(guard);
  stream_ = std::make_unique<StreamState>();
  stream_->src_opt = src_opt;
  initialized_ = true;
}

void ModelSession::ensure_stream(const cv::Mat& input) {
  if (!stream_) {
    stream_ = std::make_unique<StreamState>();
    if (!pack_) {
      throw std::runtime_error("ModelSession::run_tensor: not initialized");
    }
    stream_->src_opt = pack_->input_appsrc_options(tensor_mode_);
  }

  InputCapsConfig cfg = infer_input_caps(stream_->src_opt, input);

  if (stream_->pipeline) {
    if (!caps_equal(cfg, stream_->caps)) {
      throw std::invalid_argument(
          "ModelSession::run_tensor: input caps changed; re-init ModelSession");
    }
    return;
  }

  sima::gst_init_once();
  sima::require_element("appsrc", "ModelSession::run_tensor");
  sima::require_element("appsink", "ModelSession::run_tensor");
  sima::require_element("identity", "ModelSession::run_tensor");

  const bool insert_boundaries =
      sima::pipeline_internal::env_bool("SIMA_GST_RUN_INSERT_BOUNDARIES", false);
  std::string pipeline_str = session_.to_gst(insert_boundaries);
  stream_->pipeline_string = pipeline_str;
  sima::pipeline_internal::enforce_single_mla_pipeline(
      "ModelSession::run_tensor", pipeline_str, this, "ModelSession");

  GError* err = nullptr;
  GstElement* pipeline = gst_parse_launch(pipeline_str.c_str(), &err);
  if (!pipeline) {
    std::string msg = err ? err->message : "unknown";
    if (err) g_error_free(err);
    throw std::runtime_error(
        "ModelSession::run_tensor: gst_parse_launch failed: " + msg +
        "\nPipeline:\n" + pipeline_str);
  }
  if (err) g_error_free(err);

  GstElement* appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysrc");
  if (!appsrc) {
    sima::pipeline_internal::stop_and_unref(pipeline);
    throw std::runtime_error(
        "ModelSession::run_tensor: appsrc 'mysrc' not found.\nPipeline:\n" +
        pipeline_str);
  }

  GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
  if (!appsink) {
    gst_object_unref(appsrc);
    sima::pipeline_internal::stop_and_unref(pipeline);
    throw std::runtime_error(
        "ModelSession::run_tensor: appsink 'mysink' not found.\nPipeline:\n" +
        pipeline_str);
  }

  sima::InputAppSrcOptions src_opt = stream_->src_opt;
  GstCaps* caps = build_input_caps(cfg, src_opt);
  gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
  gst_caps_unref(caps);

  src_opt.max_bytes = resolve_appsrc_max_bytes(src_opt, cfg);
  configure_appsrc(appsrc, src_opt);
  configure_appsink_for_input(appsink);

  try {
    set_state_or_throw(pipeline, GST_STATE_PLAYING, "ModelSession::run_tensor");
  } catch (...) {
    gst_object_unref(appsrc);
    gst_object_unref(appsink);
    sima::pipeline_internal::stop_and_unref(pipeline);
    throw;
  }

  stream_->pipeline = pipeline;
  stream_->appsrc = appsrc;
  stream_->appsink = appsink;
  stream_->caps = cfg;
  stream_->caps_set = true;
}

void ModelSession::teardown_stream() {
  if (!stream_) return;

  if (stream_->appsrc) {
    gst_object_unref(stream_->appsrc);
    stream_->appsrc = nullptr;
  }
  if (stream_->appsink) {
    gst_object_unref(stream_->appsink);
    stream_->appsink = nullptr;
  }
  if (stream_->pipeline) {
    sima::pipeline_internal::stop_and_unref(stream_->pipeline);
    stream_->pipeline = nullptr;
  }

  stream_.reset();
}

sima::NeatTensor ModelSession::run_tensor(const cv::Mat& input) {
  if (!initialized_) {
    throw std::runtime_error("ModelSession::run_tensor: not initialized");
  }

  ensure_stream(input);

  cv::Mat contiguous = input;
  if (!contiguous.isContinuous()) {
    contiguous = input.clone();
  }

  GstBuffer* buf =
      allocate_input_buffer(stream_->caps.bytes, stream_->src_opt, stream_->pool_guard);
  if (!buf) {
    throw std::runtime_error("ModelSession::run_tensor: failed to allocate GstBuffer");
  }

  GstMapInfo mi;
  if (!gst_buffer_map(buf, &mi, GST_MAP_WRITE)) {
    gst_buffer_unref(buf);
    throw std::runtime_error("ModelSession::run_tensor: failed to map GstBuffer");
  }

  std::memcpy(mi.data, contiguous.data, stream_->caps.bytes);
  gst_buffer_unmap(buf, &mi);
  maybe_add_simaai_meta(buf, next_input_frame_id(), stream_->src_opt);

  if (gst_app_src_push_buffer(GST_APP_SRC(stream_->appsrc), buf) != GST_FLOW_OK) {
    gst_buffer_unref(buf);
    throw std::runtime_error("ModelSession::run_tensor: appsrc push failed");
  }

  const int timeout_ms = std::max(
      10,
      std::atoi(
          sima::pipeline_internal::env_str("SIMA_GST_RUN_INPUT_TIMEOUT_MS", "10000").c_str()));

  auto sample_opt = sima::pipeline_internal::try_pull_sample_sliced(
      stream_->pipeline,
      stream_->appsink,
      timeout_ms,
      nullptr,
      "ModelSession::run_tensor");

  if (!sample_opt.has_value()) {
    throw std::runtime_error("ModelSession::run_tensor: timeout waiting for output");
  }

  std::unique_ptr<GstSample, decltype(&gst_sample_unref)> sample(*sample_opt,
                                                                 gst_sample_unref);
  GstCaps* out_caps = gst_sample_get_caps(sample.get());
  const GstStructure* st = out_caps ? gst_caps_get_structure(out_caps, 0) : nullptr;
  const char* media = st ? gst_structure_get_name(st) : nullptr;
  if (media && std::string(media).rfind("video/x-raw", 0) == 0) {
    const char* fmt = gst_structure_get_string(st, "format");
    if (fmt && std::string(fmt) == "NV12") {
      throw std::runtime_error("ModelSession::run_tensor: expected tensor output");
    }
  }

  return sima::from_gst_sample(sample.get());
}

cv::Mat ModelSession::run(const cv::Mat& input) {
  sima::NeatTensor t = run_tensor(input);
  return tensor_to_mat(t);
}

const sima::mpk::ModelMPK& ModelSession::model() const {
  if (!initialized_) {
    throw std::runtime_error("ModelSession::model: not initialized");
  }
  if (!pack_) {
    throw std::runtime_error("ModelSession::model: missing model");
  }
  return *pack_;
}

void ModelSession::close() {
  teardown_stream();
  initialized_ = false;
  tensor_mode_ = false;
  guard_.reset();
  last_error_.clear();
  session_ = sima::PipelineSession();
  pack_.reset();
}

} // namespace simaai
