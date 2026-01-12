#include "pipeline/PipelineSession.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <iostream>
#include <stdexcept>

using namespace sima::nodes;

int main() {
  try {
    sima::PipelineSession p;
    p.gst("videotestsrc num-buffers=1 ! videoconvert ! video/x-raw,format=RGB");
    p.add(DebugPoint("a"));
    p.gst("boguselement");
    p.add(DebugPoint("b"));

    auto res = p.run_debug({.timeout_ms = 2000});

    require(res.taps.size() == 2, "expected two taps");
    require(res.taps[0].packet.has_value(), "tap a missing packet");
    require(res.taps[0].tensor.has_value(), "tap a missing tensor");
    require(res.taps[0].last_good_tensor.has_value(), "tap a missing last_good tensor");
    require(res.tensors.find("a") != res.tensors.end(), "tensor map missing tap a");
    require(!res.taps[1].error.empty(), "tap b should report error");

    std::cout << "[OK] run_debug_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
