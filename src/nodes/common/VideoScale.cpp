#include "nodes/common/VideoScale.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

std::string VideoScale::gst_fragment(int node_index) const {
  return "videoscale name=n" + std::to_string(node_index) + "_videoscale";
}

std::vector<std::string> VideoScale::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_videoscale"};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> VideoScale() {
  return std::make_shared<sima::VideoScale>();
}

} // namespace sima::nodes
