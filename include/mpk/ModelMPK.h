#pragma once

#include "sima/builder/NodeGroup.h"
#include "sima/nodes/io/InputAppSrc.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cv {
class Mat;
}

namespace sima::mpk {

enum class PipelineType : std::uint8_t { QuantTess, Preproc, CastTess };

enum class ModelStage { Preprocess, MlaOnly, Postprocess, Full };

struct ModelFragment {
  std::string gst;
  std::vector<std::string> elements;
  std::vector<std::string> config_paths;
};

class ModelMPK {
public:
  explicit ModelMPK(const std::string& tar_gz);
  ModelMPK(const std::string& tar_gz,
           const std::string& media_type,
           const std::string& format,
           int width,
           int height,
           int depth,
           bool normalize = false,
           std::vector<float> mean = {},
           std::vector<float> stddev = {},
           const std::string& preproc_next_cpu = {},
           const std::string& upstream_name = "decoder",
           int num_buffers_cvu = 4,
           int num_buffers_mla = 4,
           int queue_max_buffers = 0,
           int64_t queue_max_time_ns = -1,
           const std::string& queue_leaky = {});
#if defined(SIMA_WITH_OPENCV)
  ModelMPK(const std::string& tar_gz,
           const cv::Mat& mat,
           bool normalize = false,
           std::vector<float> mean = {},
           std::vector<float> stddev = {},
           const std::string& preproc_next_cpu = {},
           const std::string& upstream_name = "decoder",
           int num_buffers_cvu = 4,
           int num_buffers_mla = 4,
           int queue_max_buffers = 0,
           int64_t queue_max_time_ns = -1,
           const std::string& queue_leaky = {});
#endif

  const std::string& etc_dir() const { return etc_dir_; }
  PipelineType pipeline_type() const { return pipeline_type_; }
  std::string find_config_path_by_plugin(const std::string& plugin_id) const;
  std::string find_config_path_by_processor(const std::string& processor) const;

  ModelFragment fragment(ModelStage stage) const;
  std::string gst_fragment(ModelStage stage) const;
  sima::NodeGroup to_node_group(ModelStage stage) const;

  sima::InputAppSrcOptions input_appsrc_options(bool tensor_mode) const;

private:
  struct Config {
    bool normalize = false;
    std::vector<float> mean;
    std::vector<float> stddev;

    int input_width = 0;
    int input_height = 0;
    std::string input_format; // "RGB"/"BGR"/"GRAY"/"NV12"/"IYUV"
    int input_depth = 0;

    std::string preproc_next_cpu;
    std::string upstream_name = "decoder";

    int num_buffers_cvu = 4;
    int num_buffers_mla = 4;

    int queue_max_buffers = 0;
    int64_t queue_max_time_ns = -1;
    std::string queue_leaky;
  };

  void init(const std::string& tar_gz);
  void init_from_config(const std::string& tar_gz, Config cfg);

  std::string etc_dir_;
  Config options_;
  PipelineType pipeline_type_ = PipelineType::Preproc;
};

} // namespace sima::mpk
