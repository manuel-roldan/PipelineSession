#pragma once
#include "sima/builder/Node.h"
#include <memory>
#include <vector>

namespace sima {

class RtpH264Depay final : public Node {
public:
  std::string kind() const override { return "RtpH264Depay"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> RtpH264Depay();
} // namespace sima::nodes
