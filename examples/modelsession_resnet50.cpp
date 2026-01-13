#include "ModelSession.hpp"
#include "example_utils.h"

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
  constexpr int kInferWidth = 224;
  constexpr int kInferHeight = 224;

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

  cv::Mat rgb;
  try {
    rgb = sima_examples::load_rgb_resized(image_path, kInferWidth, kInferHeight);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load image: " << e.what() << "\n";
    return 4;
  }

  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> stddev = {0.229f, 0.224f, 0.225f};

  simaai::ModelSession session;
  if (!session.init(model_path,
                    kInferWidth,
                    kInferHeight,
                    "RGB",
                    /*normalize=*/true,
                    mean,
                    stddev)) {
    std::cerr << "ModelSession init failed: " << session.last_error() << "\n";
    return 5;
  }

  try {
    auto t = session.run_tensor(rgb);
    auto scores = sima_examples::scores_from_tensor(t, "modelsession");
    sima_examples::check_top1(scores, kGoldfishId, min_prob, "modelsession");
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 6;
  }

  return 0;
}
