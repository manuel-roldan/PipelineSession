#include "nodes/groups/ModelGroups.h"

#include "mpk/ModelMPK.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sima::nodes::groups {
namespace {

sima::mpk::ModelMPKOptions to_pack_options(const InferOptions& opt) {
  sima::mpk::ModelMPKOptions out;
  out.normalize = opt.normalize;
  out.mean = opt.mean;
  out.stddev = opt.stddev;
  out.input_width = opt.input_width;
  out.input_height = opt.input_height;
  out.input_format = opt.input_format;
  out.upstream_name = opt.upstream_name;
  out.fast_mode = opt.fast_mode;
  out.preproc_next_cpu = opt.preproc_next_cpu;
  out.disable_internal_queues = opt.disable_internal_queues;
  out.num_buffers_cvu = opt.num_buffers_cvu;
  out.num_buffers_mla = opt.num_buffers_mla;
  out.queue_max_buffers = opt.queue_max_buffers;
  out.queue_max_time_ns = opt.queue_max_time_ns;
  out.queue_leaky = opt.queue_leaky;
  return out;
}

} // namespace

sima::NodeGroup preprocessing(const std::string& tar_gz, const InferOptions& opt) {
  auto pack = sima::mpk::ModelMPK::load(tar_gz, to_pack_options(opt));
  return pack.to_node_group(sima::mpk::ModelStage::Preprocess);
}

sima::NodeGroup simple_infer(const std::string& tar_gz) {
  auto pack = sima::mpk::ModelMPK::load(tar_gz);
  return pack.to_node_group(sima::mpk::ModelStage::MlaOnly);
}

sima::NodeGroup postprocessing(const std::string& tar_gz) {
  auto pack = sima::mpk::ModelMPK::load(tar_gz);
  return pack.to_node_group(sima::mpk::ModelStage::Postprocess);
}

sima::NodeGroup infer(const std::string& tar_gz) {
  auto pack = sima::mpk::ModelMPK::load(tar_gz);
  return pack.to_node_group(sima::mpk::ModelStage::Full);
}

sima::NodeGroup infer(const std::string& tar_gz, const InferOptions& opt) {
  auto pack = sima::mpk::ModelMPK::load(tar_gz, to_pack_options(opt));
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
