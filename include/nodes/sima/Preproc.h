#pragma once

#include "sima/builder/Node.h"
#include "sima/builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

struct PreprocOptions {
  int input_width = 1280;
  int input_height = 720;
  int output_width = 640;
  int output_height = 640;
  int scaled_width = 640;
  int scaled_height = 640;

  int input_channels = 3;
  int output_channels = 3;
  int batch_size = 1;

  bool normalize = true;
  bool aspect_ratio = true;
  bool tessellate = true;

  int tile_width = 128;
  int tile_height = 32;
  int tile_channels = 3;

  int input_offset = 0;
  int input_stride = 1;
  int output_stride = 1;

  int q_zp = -128;
  double q_scale = 255.06967737092486;

  std::vector<float> channel_mean = {0.0f, 0.0f, 0.0f};
  std::vector<float> channel_stddev = {1.0f, 1.0f, 1.0f};

  std::string input_img_type = "NV12";
  std::string output_img_type = "RGB";
  std::string output_dtype = "EVXX_INT8";
  std::string scaling_type = "BILINEAR";
  std::string padding_type = "CENTER";

  std::string graph_name = "preproc";
  std::string node_name = "preproc";
  std::string cpu = "CVU";
  std::string next_cpu = "MLA";
  std::string debug = "EVXX_DBG_DISABLED";

  std::string upstream_name = "decoder";
  std::string graph_input_name = "input_image";

  std::vector<std::string> output_memory_order;

  int num_buffers = 0;

  std::string config_path;
  std::string config_dir;
  bool keep_config = false;
};

class Preproc final : public Node, public OutputSpecProvider {
public:
  explicit Preproc(PreprocOptions opt = {});

  std::string kind() const override { return "Preproc"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const PreprocOptions& options() const { return opt_; }
  const std::string& config_path() const { return config_path_; }

private:
  struct PreprocConfigHolder;

  PreprocOptions opt_;
  std::shared_ptr<PreprocConfigHolder> config_holder_;
  std::string config_path_;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> Preproc(PreprocOptions opt = {});
} // namespace sima::nodes
