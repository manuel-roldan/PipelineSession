#include "nodes/common/ImageDecode.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

std::string ImageDecode::gst_fragment(int node_index) const {
  return "decodebin name=n" + std::to_string(node_index) + "_decodebin";
}

std::vector<std::string> ImageDecode::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_decodebin"};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> ImageDecode() {
  return std::make_shared<sima::ImageDecode>();
}

} // namespace sima::nodes
