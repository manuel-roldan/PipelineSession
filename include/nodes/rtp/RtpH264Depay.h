#pragma once
#include "sima/builder/Node.h"
#include "sima/builder/OutputSpec.h"
#include <memory>
#include <vector>

namespace sima {

class RtpH264Depay final : public Node, public OutputSpecProvider {
public:
  explicit RtpH264Depay(int payload_type = 96);
  std::string kind() const override { return "RtpH264Depay"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  int payload_type() const { return payload_type_; }

private:
  int payload_type_ = 96;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> RtpH264Depay(int payload_type = 96);
std::shared_ptr<sima::Node> H264DepayParse(int payload_type = 96);
} // namespace sima::nodes
