#include "builder/NodeGroup.h"
#include "builder/OutputSpec.h"
#include "nodes/common/Caps.h"
#include "nodes/io/InputAppSrc.h"
#include "nodes/groups/GroupOutputSpec.h"

#include "test_utils.h"

#include <iostream>
#include <stdexcept>
#include <cstdlib>

int main() {
  try {
    setenv("SIMA_DEBUG_OUTPUTSPEC_LOG", "1", 1);

    sima::OutputSpec nv12;
    nv12.media_type = "video/x-raw";
    nv12.format = "NV12";
    nv12.width = 4;
    nv12.height = 2;
    const std::size_t nv12_bytes = sima::expected_byte_size(nv12);
    require(nv12_bytes == 12, "NV12 byte size mismatch");

    sima::InputAppSrcOptions appsrc_opt;
    appsrc_opt.media_type = "video/x-raw";
    appsrc_opt.format = "RGB";
    appsrc_opt.width = 10;
    appsrc_opt.height = 8;
    appsrc_opt.use_simaai_pool = false;

    auto appsrc = sima::nodes::InputAppSrc(appsrc_opt);
    auto caps = sima::nodes::CapsRaw("GRAY8", 10, 8, 30, sima::CapsMemory::SystemMemory);
    sima::NodeGroup group({appsrc, caps});

    sima::OutputSpec derived = sima::derive_output_spec(group);
    std::cerr << "[DBG] derived media=" << derived.media_type
              << " format=" << derived.format
              << " w=" << derived.width
              << " h=" << derived.height
              << " d=" << derived.depth
              << " layout=" << derived.layout
              << " dtype=" << derived.dtype
              << " mem=" << derived.memory
              << " bytes=" << derived.byte_size
              << "\n";
    require(derived.media_type == "video/x-raw", "derived media_type mismatch");
    require(derived.format == "GRAY8", "derived format mismatch");
    require(derived.width == 10 && derived.height == 8, "derived shape mismatch");
    require(derived.layout == "HW", "derived layout mismatch");
    require(derived.byte_size == 80, "derived byte_size mismatch");

    sima::nodes::groups::ImageInputGroupOptions img_opt;
    img_opt.output_caps.enable = true;
    img_opt.output_caps.format = "NV12";
    img_opt.output_caps.width = 32;
    img_opt.output_caps.height = 16;
    img_opt.output_caps.fps = 15;
    sima::OutputSpec img_spec = sima::nodes::groups::ImageInputGroupOutputSpec(img_opt);
    require(img_spec.format == "NV12", "image group spec format mismatch");
    require(img_spec.width == 32 && img_spec.height == 16, "image group spec shape mismatch");
    require(img_spec.byte_size == 32 * 16 * 3 / 2, "image group spec byte size mismatch");

    std::cout << "[OK] unit_outputspec_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
