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
                        const sima::FrameTensor& t,
                        std::string& err) {
  if (t.dtype != sima::TensorDType::UInt8) {
    err = "dtype mismatch";
    return false;
  }
  if (t.format != "RGB") {
    err = "format mismatch";
    return false;
  }
  if (t.width != img.cols || t.height != img.rows) {
    err = "size mismatch";
    return false;
  }
  if (t.planes.empty()) {
    err = "missing plane";
    return false;
  }
  const auto& plane = t.planes[0];
  const int row_bytes = img.cols * img.channels();
  if (static_cast<int>(plane.size()) < row_bytes * img.rows) {
    err = "plane too small";
    return false;
  }
  for (int r = 0; r < img.rows; ++r) {
    const uint8_t* lhs = plane.data() + r * row_bytes;
    const uint8_t* rhs = img.ptr<uint8_t>(r);
    if (std::memcmp(lhs, rhs, row_bytes) != 0) {
      err = "byte mismatch";
      return false;
    }
  }
  return true;
}

bool compare_rgb_tensor(const cv::Mat& img,
                        const sima::FrameTensorRef& t,
                        std::string& err) {
  if (t.dtype != sima::TensorDType::UInt8) {
    err = "dtype mismatch";
    return false;
  }
  if (t.format != "RGB") {
    err = "format mismatch";
    return false;
  }
  if (t.width != img.cols || t.height != img.rows) {
    err = "size mismatch";
    return false;
  }
  if (t.planes.empty()) {
    err = "missing plane";
    return false;
  }
  const auto& plane = t.planes[0];
  const int row_bytes = img.cols * img.channels();
  if (plane.stride < row_bytes) {
    err = "stride too small";
    return false;
  }
  for (int r = 0; r < img.rows; ++r) {
    const uint8_t* lhs = plane.data + r * plane.stride;
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

    sima::AsyncOptions aopt;
    aopt.input_queue = iters;
    aopt.output_queue = iters;
    aopt.drop = sima::DropPolicy::Block;
    aopt.copy_output = true;
    aopt.copy_input = false;
    aopt.timeout_ms = -1;

    auto async = p.run_async(img, aopt);

    for (int i = 0; i < iters; ++i) {
      require(async.push(img), "async push failed");
    }
    async.close_input();

    int outputs = 0;
    bool checked = false;
    std::string err;
    while (true) {
      auto out = async.pull(20000);
      if (!out.has_value()) break;
      outputs += 1;
      if (!checked) {
        if (out->tensor.has_value()) {
          require(compare_rgb_tensor(img, out->tensor.value(), err), err);
        } else if (out->tensor_ref.has_value()) {
          require(compare_rgb_tensor(img, out->tensor_ref.value(), err), err);
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
