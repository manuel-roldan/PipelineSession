#include "pipeline/PipelineSession.h"
#include "nodes/common/AppSink.h"
#include "nodes/io/InputAppSrc.h"

#include "test_utils.h"

#include <opencv2/core.hpp>

#include <cstring>
#include <iostream>
#include <optional>
#include <string>

namespace {

bool compare_rgb_tensor(const cv::Mat& img,
                        const sima::NeatTensor& t,
                        std::string& err) {
  if (t.dtype != sima::TensorDType::UInt8) {
    err = "dtype mismatch";
    return false;
  }
  if (!t.semantic.image.has_value() ||
      t.semantic.image->format != sima::NeatImageSpec::PixelFormat::RGB) {
    err = "format mismatch";
    return false;
  }
  if (t.shape.size() < 2) {
    err = "shape missing";
    return false;
  }
  const int h = static_cast<int>(t.shape[0]);
  const int w = static_cast<int>(t.shape[1]);
  if (w != img.cols || h != img.rows) {
    err = "size mismatch";
    return false;
  }

  sima::NeatMapping map = t.map(sima::NeatMapMode::Read);
  if (!map.data || map.size_bytes == 0) {
    err = "missing mapping";
    return false;
  }
  const int row_bytes = img.cols * img.channels();
  int64_t stride = row_bytes;
  if (!t.strides_bytes.empty()) {
    stride = t.strides_bytes[0];
  }
  if (stride < row_bytes) {
    err = "stride too small";
    return false;
  }
  for (int r = 0; r < img.rows; ++r) {
    const uint8_t* lhs = static_cast<const uint8_t*>(map.data) + r * stride;
    const uint8_t* rhs = img.ptr<uint8_t>(r);
    if (std::memcmp(lhs, rhs, row_bytes) != 0) {
      err = "byte mismatch";
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  try {
    constexpr int kW = 640;
    constexpr int kH = 640;
    const int iters = 50;

    cv::Mat img(kH, kW, CV_8UC3, cv::Scalar(64, 128, 192));
    if (!img.isContinuous()) img = img.clone();

    sima::PipelineSession p;
    sima::InputAppSrcOptions src_opt;
    src_opt.format = "RGB";
    src_opt.width = kW;
    src_opt.height = kH;
    src_opt.depth = 3;
    src_opt.is_live = true;
    src_opt.do_timestamp = true;
    src_opt.block = true;
    src_opt.use_simaai_pool = false;
    src_opt.max_bytes = static_cast<std::uint64_t>(img.total() * img.elemSize());
    p.add(sima::nodes::InputAppSrc(src_opt));

    sima::OutputAppSinkOptions sink_opt;
    sink_opt.sync = false;
    sink_opt.drop = false;
    sink_opt.max_buffers = 8;
    p.add(sima::nodes::OutputAppSink(sink_opt));

    sima::PipelineRunOptions run_opt;
    run_opt.input_queue = iters;
    run_opt.output_queue = iters;
    run_opt.drop = sima::DropPolicy::Block;
    run_opt.copy_output = true;
    run_opt.copy_input = false;
    run_opt.timeout_ms = -1;
    run_opt.appsink_max_buffers = sink_opt.max_buffers;

    auto run = p.build(img, run_opt);

    for (int i = 0; i < iters; ++i) {
      require(run.push(img), "async push failed");
    }
    run.close_input();

    int outputs = 0;
    bool checked = false;
    std::string err;
    while (true) {
      auto out = run.pull(20000);
      if (!out.has_value()) break;
      outputs += 1;
      if (!checked) {
        if (out->neat.has_value()) {
          require(compare_rgb_tensor(img, out->neat.value(), err), err);
        } else {
          require(false, "unexpected output kind");
        }
        checked = true;
      }
    }

    require(outputs == iters, "async output count mismatch");

    std::cout << "[OK] async_stream_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
