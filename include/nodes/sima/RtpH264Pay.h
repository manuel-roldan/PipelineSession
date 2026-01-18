#pragma once

#include "sima/builder/Node.h"
#include "sima/builder/OutputSpec.h"

#include <memory>
#include <vector>

namespace sima {

class RtpH264Pay final : public Node, public OutputSpecProvider {
public:
  RtpH264Pay(int pt = 96, int config_interval = 1);
  std::string kind() const override { return "RtpH264Pay"; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  int pt() const { return pt_; }
  int config_interval() const { return config_interval_; }

private:
  int pt_ = 96;
  int config_interval_ = 1;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> RtpH264Pay(int pt = 96, int config_interval = 1);
} // namespace sima::nodes
