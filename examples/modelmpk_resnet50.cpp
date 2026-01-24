#include "pipeline/PipelineSession.h"
#include "mpk/ModelMPK.h"
#if defined(SIMA_WITH_OPENCV)
#include "mpk/ModelMPKOpenCV.h"
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  namespace mpk = sima::mpk;
  namespace nodes = sima::nodes;

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model.tar.gz> <image>\n";
    return 2;
  }

  const std::string tar_gz = argv[1];
  const std::string image_path = argv[2];

  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    std::cerr << "Failed to read image: " << image_path << "\n";
    return 3;
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  sima::PipelineSession p;

#if defined(SIMA_WITH_OPENCV)
  mpk::ModelMPKOptions mpk_opt(rgb);
#else
  sima::OutputSpec spec;
  spec.media_type = "video/x-raw";
  spec.format = "RGB";
  spec.width = rgb.cols;
  spec.height = rgb.rows;
  mpk::ModelMPKOptions mpk_opt = mpk::options_from_output_spec(spec);
#endif

  auto model = mpk::ModelMPK::load(tar_gz, mpk_opt);

  p.add(nodes::InputAppSrc(model.input_appsrc_options(/*tensor_mode=*/false)));
  p.add(model.to_node_group(mpk::ModelStage::Full));
  p.add(nodes::OutputAppSink());

  try {
    auto out = p.run(rgb);
    if (out.kind == sima::RunOutputKind::Tensor && out.tensor.has_value()) {
      std::cout << "OK: tensor output, format=" << out.format
                << " shape=" << out.tensor->shape.size() << "\n";
      return 0;
    }
    if (out.kind == sima::RunOutputKind::FrameNV12 && out.frame_nv12.has_value()) {
      std::cout << "OK: NV12 output " << out.frame_nv12->width
                << "x" << out.frame_nv12->height << "\n";
      return 0;
    }
    std::cerr << "Unexpected output kind\n";
    return 4;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 5;
  }
}
