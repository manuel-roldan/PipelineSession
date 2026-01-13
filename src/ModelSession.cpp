#include "ModelSession.hpp"

#include "nodes/io/InputAppSrc.h"
#include "pipeline/internal/SimaaiGuard.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace simaai {
namespace {

cv::Mat tensor_to_mat(const sima::FrameTensor& t) {
  if (t.planes.empty()) {
    throw std::runtime_error("ModelSession::run: empty tensor output");
  }

  if (t.dtype != sima::TensorDType::Float32) {
    throw std::runtime_error("ModelSession::run: only Float32 tensor supported");
  }

  const auto& plane = t.planes[0];
  if (plane.empty()) {
    throw std::runtime_error("ModelSession::run: empty tensor plane");
  }

  const size_t elems = plane.size() / sizeof(float);
  cv::Mat out(1, static_cast<int>(elems), CV_32FC1);
  std::memcpy(out.data, plane.data(), elems * sizeof(float));
  return out;
}

} // namespace

ModelSession::~ModelSession() {
  close();
}

bool ModelSession::init(const std::string& tar_gz) {
  close();
  sima::nodes::groups::InferOptions opt;
  build_session(tar_gz, opt, /*tensor_mode=*/true);
  return initialized_;
}

bool ModelSession::init(const std::string& tar_gz,
                        int input_width,
                        int input_height,
                        const std::string& input_format,
                        bool normalize,
                        std::vector<float> channel_mean,
                        std::vector<float> channel_stddev) {
  close();
  sima::nodes::groups::InferOptions opt;
  opt.input_width = input_width;
  opt.input_height = input_height;
  opt.input_format = input_format;
  opt.normalize = normalize;
  opt.mean = std::move(channel_mean);
  opt.stddev = std::move(channel_stddev);
  build_session(tar_gz, opt, /*tensor_mode=*/false);
  return initialized_;
}

void ModelSession::build_session(const std::string& tar_gz,
                                 const sima::nodes::groups::InferOptions& opt,
                                 bool tensor_mode) {
  tensor_mode_ = tensor_mode;
  initialized_ = false;
  last_error_.clear();

  std::string guard_error;
  auto guard = sima::pipeline_internal::acquire_simaai_guard(
      "ModelSession::init", std::string{}, /*force=*/true, &guard_error);
  if (!guard) {
    last_error_ = guard_error.empty()
        ? "ModelSession::init: failed to acquire single-owner guard"
        : guard_error;
    return;
  }

  sima::PipelineSession sess;
  sess.set_guard(guard);

  if (tensor_mode) {
    pack_ = sima::mpk::ModelMPK::load(tar_gz);
    sima::InputAppSrcOptions src_opt = pack_.input_appsrc_options(/*tensor_mode=*/true);
    sess.add(sima::nodes::InputAppSrc(src_opt));
    sess.add(pack_.to_node_group(sima::mpk::ModelStage::Full));
  } else {
    pack_ = sima::mpk::ModelMPK::load(
        tar_gz,
        sima::mpk::ModelMPKOptions{
            opt.normalize,
            opt.mean,
            opt.stddev,
            opt.input_width,
            opt.input_height,
            opt.input_format,
            0});
    sima::InputAppSrcOptions src_opt = pack_.input_appsrc_options(/*tensor_mode=*/false);
    sess.add(sima::nodes::InputAppSrc(src_opt));
    sess.add(pack_.to_node_group(sima::mpk::ModelStage::Full));
  }

  sess.add(sima::nodes::OutputAppSink());
  session_ = std::move(sess);
  guard_ = std::move(guard);
  initialized_ = true;
}

sima::FrameTensor ModelSession::run_tensor(const cv::Mat& input) {
  if (!initialized_) {
    throw std::runtime_error("ModelSession::run_tensor: not initialized");
  }

  auto out = session_.run(input);
  if (out.kind != sima::RunOutputKind::Tensor || !out.tensor.has_value()) {
    throw std::runtime_error("ModelSession::run_tensor: expected tensor output");
  }
  return *out.tensor;
}

cv::Mat ModelSession::run(const cv::Mat& input) {
  sima::FrameTensor t = run_tensor(input);
  return tensor_to_mat(t);
}

void ModelSession::close() {
  initialized_ = false;
  tensor_mode_ = false;
  guard_.reset();
  last_error_.clear();
  session_ = sima::PipelineSession();
  pack_ = sima::mpk::ModelMPK();
}

} // namespace simaai
