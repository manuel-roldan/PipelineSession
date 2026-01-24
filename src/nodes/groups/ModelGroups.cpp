#include "nodes/groups/ModelGroups.h"

#include "mpk/ModelMPK.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sima::nodes::groups {
namespace {

sima::mpk::ModelMPK load_pack(const std::string& tar_gz, const InferOptions& opt) {
  if (opt.input_format.empty()) {
    return sima::mpk::ModelMPK(tar_gz);
  }
  return sima::mpk::ModelMPK(
      tar_gz,
      "video/x-raw",
      opt.input_format,
      opt.input_width,
      opt.input_height,
      0,
      opt.normalize,
      opt.mean,
      opt.stddev,
      opt.preproc_next_cpu,
      opt.upstream_name,
      opt.num_buffers_cvu,
      opt.num_buffers_mla,
      opt.queue_max_buffers,
      opt.queue_max_time_ns,
      opt.queue_leaky);
}

} // namespace

sima::NodeGroup preprocessing(const std::string& tar_gz, const InferOptions& opt) {
  auto pack = load_pack(tar_gz, opt);
  return pack.to_node_group(sima::mpk::ModelStage::Preprocess);
}

sima::NodeGroup simple_infer(const std::string& tar_gz) {
  auto pack = sima::mpk::ModelMPK(tar_gz);
  return pack.to_node_group(sima::mpk::ModelStage::MlaOnly);
}

sima::NodeGroup postprocessing(const std::string& tar_gz) {
  auto pack = sima::mpk::ModelMPK(tar_gz);
  return pack.to_node_group(sima::mpk::ModelStage::Postprocess);
}

sima::NodeGroup infer(const std::string& tar_gz) {
  auto pack = sima::mpk::ModelMPK(tar_gz);
  return pack.to_node_group(sima::mpk::ModelStage::Full);
}

sima::NodeGroup infer(const std::string& tar_gz, const InferOptions& opt) {
  auto pack = load_pack(tar_gz, opt);
  return pack.to_node_group(sima::mpk::ModelStage::Full);
}

sima::NodeGroup Preprocess(const sima::mpk::ModelMPK& model) {
  return model.to_node_group(sima::mpk::ModelStage::Preprocess);
}

sima::NodeGroup MLA(const sima::mpk::ModelMPK& model) {
  return model.to_node_group(sima::mpk::ModelStage::MlaOnly);
}

sima::NodeGroup Postprocess(const sima::mpk::ModelMPK& model) {
  return model.to_node_group(sima::mpk::ModelStage::Postprocess);
}

sima::NodeGroup Infer(const sima::mpk::ModelMPK& model) {
  return model.to_node_group(sima::mpk::ModelStage::Full);
}

} // namespace sima::nodes::groups
