#include "pipeline/PipelineSession.h"
#include "nodes/common/FileSrc.h"
#include "nodes/common/ImageDecode.h"
#include "nodes/common/ImageFreeze.h"
#include "nodes/common/DebugPoint.h"
#include "nodes/common/AppSink.h"

#include "test_utils.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    std::string image_path = "test.jpg";
    if (argc > 1) image_path = argv[1];

    sima::PipelineSession p;
    p.add(sima::nodes::FileSrc(image_path));
    p.add(sima::nodes::ImageDecode());
    p.add(sima::nodes::DebugPoint("io"));
    p.add(sima::nodes::ImageFreeze(1));
    p.add(sima::nodes::OutputAppSink());

    const std::string save_path = "pipeline_saved.json";
    p.save(save_path);

    sima::PipelineSession loaded = sima::PipelineSession::load(save_path);
    const std::string original = p.to_gst(false);
    const std::string restored = loaded.to_gst(false);

    require(!original.empty(), "Original pipeline string empty");
    require(original == restored, "Loaded pipeline string mismatch");

    std::cout << "[OK] unit_pipeline_io_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
