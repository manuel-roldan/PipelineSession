#pragma once

#include "sima/builder/Node.h"
#include "sima/builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

class H264Decode final : public Node, public OutputSpecProvider {
public:
  H264Decode(int sima_allocator_type = 2,
             std::string out_format = "NV12",
             std::string decoder_name = {},
             bool raw_output = false);
  std::string kind() const override { return "H264Decode"; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  int sima_allocator_type() const { return sima_allocator_type_; }
  const std::string& out_format() const { return out_format_; }
  const std::string& decoder_name() const { return decoder_name_; }
  bool raw_output() const { return raw_output_; }

private:
  int sima_allocator_type_ = 2;
  std::string out_format_ = "NV12";
  std::string decoder_name_;
  bool raw_output_ = false;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> H264Decode(int sima_allocator_type = 2,
                                       std::string out_format = "NV12",
                                       std::string decoder_name = {},
                                       bool raw_output = false);
} // namespace sima::nodes
