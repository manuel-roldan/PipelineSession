#include "nodes/io/UdpSink.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace sima {

std::string UdpSink::gst_fragment(int /*node_index*/) const {
  std::ostringstream ss;
  ss << "udpsink host=" << opt_.host
     << " port=" << opt_.port
     << " sync=" << (opt_.sync ? "true" : "false")
     << " async=" << (opt_.async ? "true" : "false");
  return ss.str();
}

std::vector<std::string> UdpSink::element_names(int /*node_index*/) const {
  return {};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> UdpSink(UdpSinkOptions opt) {
  return std::make_shared<sima::UdpSink>(std::move(opt));
}

} // namespace sima::nodes
