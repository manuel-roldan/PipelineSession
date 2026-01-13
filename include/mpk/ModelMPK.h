#pragma once

#include "sima/builder/NodeGroup.h"
#include "sima/nodes/io/InputAppSrc.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sima::mpk {

enum class PipelineType : std::uint8_t { QuantTess, Preproc, CastTess };

struct ModelMPKOptions {
  bool normalize = false;
  std::vector<float> mean;
  std::vector<float> stddev;

  int input_width = 0;
  int input_height = 0;
  std::string input_format; // "RGB"/"BGR"/"GRAY"/"NV12"/"IYUV"
  int input_depth = 0;

  // Override the upstream buffer name used in MPK JSONs (e.g. "decoder").
  std::string upstream_name;
};

enum class ModelStage { Preprocess, MlaOnly, Postprocess, Full };

struct ModelFragment {
  std::string gst;
  std::vector<std::string> elements;
};

class ModelMPK {
public:
  static ModelMPK load(const std::string& tar_gz, const ModelMPKOptions& opt = {});

  const std::string& etc_dir() const { return etc_dir_; }
  PipelineType pipeline_type() const { return pipeline_type_; }
  const ModelMPKOptions& options() const { return options_; }

  ModelFragment fragment(ModelStage stage) const;
  std::string gst_fragment(ModelStage stage) const;
  sima::NodeGroup to_node_group(ModelStage stage) const;

  sima::InputAppSrcOptions input_appsrc_options(bool tensor_mode) const;

private:
  std::string etc_dir_;
  ModelMPKOptions options_;
  PipelineType pipeline_type_ = PipelineType::Preproc;
};

} // namespace sima::mpk
