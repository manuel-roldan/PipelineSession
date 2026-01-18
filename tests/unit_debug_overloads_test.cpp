#include "sima/debug.h"
#include "nodes/groups/ImageInputGroup.h"

#include "test_utils.h"

#include <opencv2/core/mat.hpp>

#include <iostream>
#include <stdexcept>
#include <cstdlib>

int main(int argc, char** argv) {
  try {
    setenv("SIMA_DEBUG_OVERLOADS_LOG", "1", 1);

    if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " /path/to/image.jpg\n";
      return 2;
    }

    sima::nodes::groups::ImageInputGroupOptions opt;
    opt.path = argv[1];
    opt.imagefreeze_num_buffers = 2;
    opt.fps = 30;
    opt.use_videorate = true;
    opt.use_videoscale = true;
    opt.output_caps.enable = true;
    opt.output_caps.format = "NV12";
    opt.output_caps.width = 64;
    opt.output_caps.height = 64;
    opt.output_caps.fps = 30;

    sima::debug::DebugOptions dbg;
    dbg.timeout_ms = 3000;

    {
      cv::Mat rgb(4, 6, CV_8UC3, cv::Scalar(1, 2, 3));
      sima::InputAppSrcOptions src_opt;
      src_opt.format = "RGB";
      auto input = sima::nodes::InputAppSrc(rgb, src_opt, dbg);
      require(input.tensor.has_value(), "debug InputAppSrc missing tensor");
      require(input.tensor->width == 6 && input.tensor->height == 4,
              "debug InputAppSrc tensor shape mismatch");
      require(input.expected.format == "RGB", "debug InputAppSrc expected format mismatch");
    }

    auto out = sima::nodes::groups::ImageInputGroup(opt, sima::debug::output, dbg);
    require(out.tensor.has_value(), "debug output missing tensor");
    require(out.tensor->width == 64 && out.tensor->height == 64, "debug tensor shape mismatch");
    require(out.tensorizable, "debug output not tensorizable");
    require(!out.unknown, "debug output marked unknown");
    require(out.expected.width == 64 && out.expected.height == 64, "debug expected shape mismatch");

    auto stream = sima::nodes::groups::ImageInputGroup(opt, sima::debug::stream, dbg);
    require(static_cast<bool>(stream), "debug stream invalid");
    auto next = stream.next(3000);
    require(next.has_value(), "debug stream produced no output");
    require(next->tensor.has_value(), "debug stream output missing tensor");
    require(next->tensor->width == 64 && next->tensor->height == 64,
            "debug stream tensor shape mismatch");

    stream.close();
    std::cout << "[OK] unit_debug_overloads_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
