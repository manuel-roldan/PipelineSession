#include "nodes/common/JpegDec.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

std::string JpegDec::gst_fragment(int node_index) const {
  return "jpegdec name=n" + std::to_string(node_index) + "_jpegdec";
}

std::vector<std::string> JpegDec::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_jpegdec"};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> JpegDec() {
  return std::make_shared<sima::JpegDec>();
}

} // namespace sima::nodes
