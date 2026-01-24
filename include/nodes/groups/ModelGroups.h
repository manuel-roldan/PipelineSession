#pragma once

#include "mpk/ModelMPK.h"
#include "sima/builder/NodeGroup.h"

#include <string>
#include <vector>
#include <cstdint>

namespace sima::nodes::groups {

struct InferOptions {
  int input_width = 0;
  int input_height = 0;
  std::string input_format;
  bool normalize = false;
  std::vector<float> mean;
  std::vector<float> stddev;
  std::string upstream_name;
  std::string preproc_next_cpu;
  int num_buffers_cvu = 4;
  int num_buffers_mla = 4;
  int queue_max_buffers = 0;
  int64_t queue_max_time_ns = -1;
  std::string queue_leaky;
};

sima::NodeGroup preprocessing(const std::string& tar_gz, const InferOptions& opt = {});
sima::NodeGroup simple_infer(const std::string& tar_gz);
sima::NodeGroup postprocessing(const std::string& tar_gz);
sima::NodeGroup infer(const std::string& tar_gz);
sima::NodeGroup infer(const std::string& tar_gz, const InferOptions& opt);

sima::NodeGroup Preprocess(const sima::mpk::ModelMPK& model);
sima::NodeGroup MLA(const sima::mpk::ModelMPK& model);
sima::NodeGroup Postprocess(const sima::mpk::ModelMPK& model);
sima::NodeGroup Infer(const sima::mpk::ModelMPK& model);

} // namespace sima::nodes::groups
