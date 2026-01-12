#include "nodes/common/VideoConvert.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

std::string VideoConvert::gst_fragment(int node_index) const {
  return "videoconvert name=n" + std::to_string(node_index) + "_videoconvert";
}

std::vector<std::string> VideoConvert::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_videoconvert"};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> VideoConvert() {
  return std::make_shared<sima::VideoConvert>();
}

} // namespace sima::nodes
