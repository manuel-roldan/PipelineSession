#pragma once

#include "sima/builder/Node.h"

#include <memory>
#include <vector>

namespace sima {

class OutputAppSink final : public Node {
public:
  std::string kind() const override { return "OutputAppSink"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> OutputAppSink();
} // namespace sima::nodes
