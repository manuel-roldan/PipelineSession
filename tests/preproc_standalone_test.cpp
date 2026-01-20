#include "pipeline/PipelineSession.h"
#include "nodes/common/AppSink.h"
#include "nodes/io/InputAppSrc.h"
#include "nodes/sima/Preproc.h"

#include "test_utils.h"

#include <opencv2/core.hpp>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool env_bool(const char* key, bool def) {
  const char* v = std::getenv(key);
  if (!v) return def;
  return std::string(v) != "0";
}

int env_int(const char* key, int def) {
  const char* v = std::getenv(key);
  if (!v || !*v) return def;
  return std::atoi(v);
}

} // namespace

int main() {
  try {
    cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(64, 128, 192));
    if (!img.isContinuous()) img = img.clone();

    sima::PipelineSession p;

    sima::InputAppSrcOptions src_opt;
    src_opt.format = "RGB";
    src_opt.width = img.cols;
    src_opt.height = img.rows;
    src_opt.depth = 3;
    src_opt.is_live = true;
    src_opt.do_timestamp = true;
    src_opt.block = false;
    src_opt.use_simaai_pool = env_bool("SIMA_PREPROC_USE_POOL", true);
    src_opt.pool_min_buffers = env_int("SIMA_PREPROC_NUM_BUFFERS", 5);
    src_opt.pool_max_buffers = src_opt.pool_min_buffers;
    src_opt.buffer_name = "decoder";
    src_opt.max_bytes = 0;

    sima::PreprocOptions pre_opt;
    pre_opt.input_width = img.cols;
    pre_opt.input_height = img.rows;
    pre_opt.output_width = 640;
    pre_opt.output_height = 640;
    pre_opt.scaled_width = 640;
    pre_opt.scaled_height = 640;
    pre_opt.input_img_type = "RGB";
    pre_opt.output_img_type = "RGB";
    pre_opt.normalize = false;
    pre_opt.aspect_ratio = false;
    pre_opt.output_dtype = "EVXX_INT8";
    pre_opt.scaling_type = "BILINEAR";
    pre_opt.padding_type = "CENTER";
    pre_opt.next_cpu = "APU";
    pre_opt.upstream_name = "decoder";
    pre_opt.num_buffers = src_opt.pool_min_buffers;
    pre_opt.config_dir = "tmp";
    pre_opt.keep_config = env_bool("SIMA_PREPROC_KEEP_CONFIG", false);
    pre_opt.output_memory_order = {"output_rgb_image", "output_tessellated_image"};

    p.add(sima::nodes::InputAppSrc(src_opt));
    p.add(sima::nodes::Preproc(pre_opt));

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

    auto stream = p.run_input_stream(img, stream_opt);
    auto out = stream.push_and_pull(img, stream_opt.timeout_ms);
    stream.close();

    sima::FrameTensor owned;
    const sima::FrameTensor* tensor = nullptr;
    if (out.tensor.has_value()) {
      tensor = &out.tensor.value();
    } else if (out.tensor_ref.has_value()) {
      owned = out.tensor_ref.value().to_copy();
      tensor = &owned;
    }
    require(tensor != nullptr, "Preproc output missing tensor");
    require(tensor->width == 640 && tensor->height == 640, "Preproc size mismatch");
    require(tensor->format == "RGB", "Preproc format mismatch");
    require(tensor->dtype == sima::TensorDType::UInt8, "Preproc dtype mismatch");
    require(!tensor->planes.empty(), "Preproc tensor missing planes");

    const auto& plane = tensor->planes[0];
    const size_t expected = static_cast<size_t>(tensor->width) *
                            static_cast<size_t>(tensor->height) * 3;
    require(plane.size() >= expected, "Preproc plane too small");

    for (size_t i = 0; i < expected; i += 3) {
      if (plane[i] != 64 || plane[i + 1] != 128 || plane[i + 2] != 192) {
        throw std::runtime_error("Preproc output bytes mismatch");
      }
    }

    std::cout << "[OK] preproc_standalone_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
