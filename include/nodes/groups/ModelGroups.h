#pragma once

#include "mpk/ModelMPK.h"
#include "sima/builder/NodeGroup.h"

#include <string>
#include <vector>

namespace sima::nodes::groups {

struct InferOptions {
  int input_width = 0;
  int input_height = 0;
  std::string input_format;
  bool normalize = false;
  std::vector<float> mean;
  std::vector<float> stddev;
  std::string upstream_name;
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
