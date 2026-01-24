#pragma once

#include "pipeline/PipelineSession.h"
#include "nodes/groups/ModelGroups.h"
#include "mpk/ModelMPK.h"

#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>
#include <vector>

namespace simaai {

class ModelSession {
public:
  ModelSession();
  ~ModelSession();

  ModelSession(const ModelSession&) = delete;
  ModelSession& operator=(const ModelSession&) = delete;
  ModelSession(ModelSession&&) noexcept;
  ModelSession& operator=(ModelSession&&) noexcept;

  bool init(const std::string& tar_gz);
  bool init(const std::string& tar_gz,
            int input_width,
            int input_height,
            const std::string& input_format,
            bool normalize = false,
            std::vector<float> channel_mean = {},
            std::vector<float> channel_stddev = {});

  sima::NeatTensor run_tensor(const cv::Mat& input);
  cv::Mat run(const cv::Mat& input);
  void close();
  const std::string& last_error() const { return last_error_; }
  const sima::mpk::ModelMPK& model() const;
  bool initialized() const { return initialized_; }

private:
  void build_session(const std::string& tar_gz,
                     const sima::nodes::groups::InferOptions& opt,
                     bool tensor_mode);
  void ensure_stream(const cv::Mat& input);
  void teardown_stream();

  struct StreamState;

  bool initialized_ = false;
  bool tensor_mode_ = false;
  sima::PipelineSession session_;
  std::unique_ptr<sima::mpk::ModelMPK> pack_;
  std::shared_ptr<void> guard_;
  std::unique_ptr<StreamState> stream_;
  std::string last_error_;
};

} // namespace simaai
