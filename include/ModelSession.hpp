#pragma once

#include "pipeline/PipelineSession.h"
#include "nodes/groups/ModelGroups.h"
#include "mpk/ModelMPK.h"

#include <opencv2/core/mat.hpp>

#include <string>
#include <vector>

namespace simaai {

class ModelSession {
public:
  ModelSession() = default;
  ~ModelSession();

  bool init(const std::string& tar_gz);
  bool init(const std::string& tar_gz,
            int input_width,
            int input_height,
            const std::string& input_format,
            bool normalize = false,
            std::vector<float> channel_mean = {},
            std::vector<float> channel_stddev = {});

  sima::FrameTensor run_tensor(const cv::Mat& input);
  cv::Mat run(const cv::Mat& input);
  void close();
  const std::string& last_error() const { return last_error_; }

private:
  void build_session(const std::string& tar_gz,
                     const sima::nodes::groups::InferOptions& opt,
                     bool tensor_mode);

  bool initialized_ = false;
  bool tensor_mode_ = false;
  sima::PipelineSession session_;
  sima::mpk::ModelMPK pack_;
  std::shared_ptr<void> guard_;
  std::string last_error_;
};

} // namespace simaai
