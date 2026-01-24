#include "pipeline/PipelineSession.h"
#include "nodes/groups/ImageInputGroup.h"

#include "test_utils.h"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    std::string path = "test.jpg";
    if (argc > 1) path = argv[1];

    sima::PipelineSession p;

    sima::nodes::groups::ImageInputGroupOptions opt;
    opt.path = path;
    opt.use_videoscale = true;
    opt.output_caps.width = 64;
    opt.output_caps.height = 64;

    p.add(sima::nodes::groups::ImageInputGroup(opt));

    sima::OutputTensorOptions out;
    out.format = "RGB";
    out.target_width = 64;
    out.target_height = 64;
    p.add_output_tensor(out);

    auto ts = p.run_tensor();
    auto frame = ts.next(/*timeout_ms=*/2000);
    require(frame.has_value(), "TensorStream missing frame");
    require(frame->semantic.image.has_value(), "TensorStream missing image semantic");
    require(frame->semantic.image->format == sima::NeatImageSpec::PixelFormat::RGB,
            "TensorStream format mismatch");
    require(frame->is_dense(), "TensorStream plane count mismatch");
    require(frame->shape.size() == 3, "TensorStream shape mismatch");
    require(frame->shape[0] == 64 && frame->shape[1] == 64,
            "TensorStream dims mismatch");

    ts.close();

    std::cout << "[OK] unit_tensorstream_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
