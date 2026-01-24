#include "pipeline/PipelineSession.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <iostream>
#include <stdexcept>

using namespace sima::nodes;

int main() {
  try {
    sima::PipelineSession p;
    p.gst("videotestsrc num-buffers=1", sima::InputRole::Source);
    p.add(VideoConvert());
    p.add(CapsNV12SysMem(64, 64, 30));
    p.add(OutputAppSink());

    sima::NeatTensor ref{};
    bool got = false;
    p.set_tensor_callback([&](const sima::NeatTensor& f) {
      ref = f;
      got = true;
      return false;
    });
    p.run();

    require(got, "no frame received");
    require(ref.shape.size() >= 2, "shape missing");
    require(ref.shape[0] == 64, "height mismatch");
    require(ref.shape[1] == 64, "width mismatch");
    require(ref.semantic.image.has_value(), "image semantic missing");

    std::cout << "[OK] unit_framestream_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
