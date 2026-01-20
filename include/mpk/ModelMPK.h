#pragma once

#include "sima/builder/NodeGroup.h"
#include "sima/builder/OutputSpec.h"
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

  // Fast/throughput mode: apply defaults that match known high-FPS pipelines.
  bool fast_mode = false;
  // Skip inserting queues between model stages (CVU/MLA/boxdecode).
  bool disable_internal_queues = false;

  int input_width = 0;
  int input_height = 0;
  std::string input_format; // "RGB"/"BGR"/"GRAY"/"NV12"/"IYUV"
  int input_depth = 0;

  // Override next_cpu in preproc JSON (e.g., "APU", "MLA").
  std::string preproc_next_cpu;

  // Override the upstream buffer name used in MPK JSONs (e.g. "decoder").
  std::string upstream_name;

  // Pipeline depth for simaai process elements (0 = plugin default).
  int num_buffers_cvu = 0;
  int num_buffers_mla = 0;

  // Queue tuning between model stages (<=0 uses defaults).
  int queue_max_buffers = 0;
  int64_t queue_max_time_ns = -1;
  std::string queue_leaky;
};

ModelMPKOptions options_from_output_spec(const sima::OutputSpec& spec,
                                         const ModelMPKOptions& base = {});

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
