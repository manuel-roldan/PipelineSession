#pragma once

#include "pipeline/NeatTensor.h"
#include "pipeline/TensorStream.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace sima_examples {

bool get_arg(int argc, char** argv, const std::string& key, std::string& out);

std::filesystem::path default_goldfish_path();
bool download_file(const std::string& url, const std::filesystem::path& out_path);
std::string resolve_resnet50_tar();
std::string resolve_yolov8s_tar(const std::filesystem::path& root = {});
std::string resolve_yolov8s_tar_local_first(const std::filesystem::path& root = {},
                                            bool skip_download = false);
std::filesystem::path ensure_coco_sample(const std::filesystem::path& root = {});
std::string find_boxdecode_config(const std::filesystem::path& etc_dir);
std::string prepare_yolo_boxdecode_config(const std::string& src_path,
                                          const std::filesystem::path& root,
                                          int img_w,
                                          int img_h,
                                          float conf = 0.5f,
                                          float nms = 0.5f);

cv::Mat load_rgb_resized(const std::string& image_path, int w, int h);

struct ScoredIndex {
  int index = -1;
  float value = 0.0f;
  float prob = 0.0f;
};

std::vector<float> tensor_to_floats(const sima::NeatTensor& t);
std::vector<float> scores_from_tensor(const sima::NeatTensor& t, const std::string& label);
std::vector<ScoredIndex> topk_with_softmax(const std::vector<float>& v, int k);
void check_top1(const std::vector<float>& scores,
                int expected_id,
                float min_prob,
                const std::string& label);

sima::NeatTensor pull_tensor_with_retry(sima::TensorStream& ts,
                                        const std::string& label,
                                        int per_try_ms,
                                        int tries);

std::string h264_gst_pipeline(const std::filesystem::path& out_path,
                              int width,
                              int height,
                              double fps,
                              int bitrate_kbps = 4000);

bool open_h264_writer(cv::VideoWriter& writer,
                      const std::filesystem::path& out_path,
                      int width,
                      int height,
                      double fps,
                      int bitrate_kbps = 4000,
                      std::string* err = nullptr);

} // namespace sima_examples
