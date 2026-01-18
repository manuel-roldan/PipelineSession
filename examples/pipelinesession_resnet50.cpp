#include "example_utils.h"

#include "mpk/ModelMPK.h"
#include "nodes/groups/ImageInputGroup.h"
#include "nodes/groups/GroupOutputSpec.h"
#include "nodes/groups/ModelGroups.h"
#include "pipeline/PipelineSession.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  constexpr const char* kGoldfishUrl =
      "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
      "n01443537_goldfish.JPEG";
  constexpr int kGoldfishId = 1; // ILSVRC2012 0-based index for "goldfish"
  constexpr int kInputWidth = 256;
  constexpr int kInputHeight = 256;
  constexpr int kInferFps = 30;

  std::string model_path;
  std::string image_path;
  std::string goldfish_url = kGoldfishUrl;
  float min_prob = 0.2f;

  std::string tmp;
  if (sima_examples::get_arg(argc, argv, "--model", tmp)) model_path = tmp;
  if (sima_examples::get_arg(argc, argv, "--image", tmp)) image_path = tmp;
  if (sima_examples::get_arg(argc, argv, "--goldfish-url", tmp)) goldfish_url = tmp;
  if (sima_examples::get_arg(argc, argv, "--min-prob", tmp)) min_prob = std::stof(tmp);

  if (model_path.empty()) {
    model_path = sima_examples::resolve_resnet50_tar();
  }
  if (model_path.empty()) {
    std::cerr << "Missing ResNet50 MPK tarball.\n";
    std::cerr << "Set SIMA_RESNET50_TAR or run 'sima-cli modelzoo get resnet_50'.\n";
    return 2;
  }

  if (image_path.empty()) {
    const fs::path out_path = sima_examples::default_goldfish_path();
    if (!sima_examples::download_file(goldfish_url, out_path)) {
      std::cerr << "Failed to download goldfish image.\n";
      std::cerr << "URL was: " << goldfish_url << "\n";
      return 3;
    }
    image_path = out_path.string();
  }

  std::cout << "Using model: " << model_path << "\n";
  std::cout << "Using image: " << image_path << "\n";

  sima::nodes::groups::ImageInputGroupOptions opt;
  opt.path = image_path;
  opt.imagefreeze_num_buffers = 8;
  opt.fps = kInferFps;
  opt.use_videorate = true;
  opt.use_videoscale = true;
  opt.output_caps.enable = true;
  opt.output_caps.format = "NV12";
  opt.output_caps.width = kInputWidth;
  opt.output_caps.height = kInputHeight;
  opt.output_caps.fps = kInferFps;
  opt.output_caps.memory = sima::CapsMemory::Any;
  opt.sima_decoder.enable = true;
  opt.sima_decoder.decoder_name = "decoder";
  opt.sima_decoder.raw_output = true;

  const sima::OutputSpec input_spec = sima::nodes::groups::ImageInputGroupOutputSpec(opt);
  auto model = sima::mpk::ModelMPK::load(
      model_path,
      sima::mpk::options_from_output_spec(input_spec));

  try {
    sima::PipelineSession p;
    p.add(sima::nodes::groups::ImageInputGroup(opt));
    p.add(sima::nodes::groups::Infer(model));
    p.add(sima::nodes::OutputAppSink());

    auto ts = p.run_tensor();
    auto t = sima_examples::pull_tensor_with_retry(ts, "image_group", 500, 20);
    ts.close();

    auto scores = sima_examples::scores_from_tensor(t, "image_group");
    sima_examples::check_top1(scores, kGoldfishId, min_prob, "image_group");
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 4;
  }

  return 0;
}
