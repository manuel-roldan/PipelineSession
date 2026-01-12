#include "pipeline/PipelineSession.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <iostream>
#include <stdexcept>

using namespace sima::nodes;

int main() {
  try {
    sima::PipelineSession p;
    p.gst("videotestsrc num-buffers=1");
    p.add(VideoConvert());
    p.add(CapsNV12SysMem(64, 64, 30));
    p.add(OutputAppSink());

    auto stream = p.run();
    auto ref = stream.next(2000);
    require(ref.has_value(), "no frame received");

    require(ref->width == 64, "width mismatch");
    require(ref->height == 64, "height mismatch");
    require(!ref->caps_string.empty(), "caps_string missing");
    require_contains(ref->caps_string, "video/x-raw", "caps_string unexpected");

    stream.close();

    std::cout << "[OK] unit_framestream_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
