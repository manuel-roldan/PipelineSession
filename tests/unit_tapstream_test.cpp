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
    p.add(DebugPoint("tap"));
    p.gst("fakesink sync=false");

    auto ts = p.run_tap("tap");
    auto pkt_opt = ts.next(2000);
    require(pkt_opt.has_value(), "no tap packet received");

    const auto& pkt = *pkt_opt;
    require(pkt.format == sima::TapFormat::NV12, "unexpected tap format");
    require(pkt.bytes.size() == 64 * 64 * 3 / 2, "unexpected tap size");

    ts.close();

    std::cout << "[OK] unit_tapstream_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
