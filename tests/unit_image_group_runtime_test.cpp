#include "pipeline/PipelineSession.h"
#include "nodes/groups/ImageInputGroup.h"
#include "nodes/common/AppSink.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " /path/to/image.jpg\n";
      return 2;
    }

    sima::nodes::groups::ImageInputGroupOptions opt;
    opt.path = argv[1];
    opt.imagefreeze_num_buffers = 5;
    opt.fps = 30;
    opt.use_videoscale = true;
    opt.output_caps.width = 64;
    opt.output_caps.height = 64;

    sima::PipelineSession p;
    p.add(sima::nodes::groups::ImageInputGroup(opt));
    p.add(sima::nodes::OutputAppSink());

    bool got = false;
    p.set_frame_callback([&](const sima::FrameNV12Ref&) {
      got = true;
      return false;
    });
    p.run();
    require(got, "no frame from image group runtime test");
    std::cout << "[OK] unit_image_group_runtime_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
