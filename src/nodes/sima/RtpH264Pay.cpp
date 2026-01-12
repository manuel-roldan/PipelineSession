#include "nodes/sima/RtpH264Pay.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sima {

RtpH264Pay::RtpH264Pay(int pt, int config_interval)
    : pt_(pt), config_interval_(config_interval) {}

std::string RtpH264Pay::gst_fragment(int /*node_index*/) const {
  std::ostringstream ss;
  ss << "rtph264pay name=pay0 pt=" << pt_ << " config-interval=" << config_interval_;
  return ss.str();
}

std::vector<std::string> RtpH264Pay::element_names(int /*node_index*/) const {
  return {"pay0"};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> RtpH264Pay(int pt, int config_interval) {
  return std::make_shared<sima::RtpH264Pay>(pt, config_interval);
}

} // namespace sima::nodes
