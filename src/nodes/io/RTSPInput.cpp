#include "nodes/io/RTSPInput.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sima {

RTSPInput::RTSPInput(std::string url, int latency_ms, bool tcp)
    : url_(std::move(url)), latency_ms_(latency_ms), tcp_(tcp) {}

std::string RTSPInput::gst_fragment(int node_index) const {
  const std::string el = "n" + std::to_string(node_index) + "_rtspsrc";
  std::ostringstream ss;
  ss << "rtspsrc name=" << el
     << " location=\"" << url_ << "\" "
     << "latency=" << latency_ms_ << " "
     << "protocols=" << (tcp_ ? "tcp" : "udp");
  return ss.str();
}

std::vector<std::string> RTSPInput::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_rtspsrc"};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> RTSPInput(std::string url, int latency_ms, bool tcp) {
  return std::make_shared<sima::RTSPInput>(std::move(url), latency_ms, tcp);
}

} // namespace sima::nodes
