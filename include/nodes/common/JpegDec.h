#pragma once

#include "sima/builder/Node.h"

#include <memory>
#include <vector>

namespace sima {

class JpegDec final : public Node {
public:
  std::string kind() const override { return "JpegDec"; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> JpegDec();
} // namespace sima::nodes
