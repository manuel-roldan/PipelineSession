#include "nodes/rtp/RtpH264Depay.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace sima {

RtpH264Depay::RtpH264Depay(int payload_type)
    : payload_type_(payload_type) {}

std::string RtpH264Depay::gst_fragment(int node_index) const {
  const std::string rtp = "n" + std::to_string(node_index) + "_rtp_caps";
  const std::string dep = "n" + std::to_string(node_index) + "_depay";
  const std::string par = "n" + std::to_string(node_index) + "_h264parse";
  const std::string hcc = "n" + std::to_string(node_index) + "_h264_caps";

  std::ostringstream ss;
  ss << "capsfilter name=" << rtp
     << " caps=\"application/x-rtp,media=video,encoding-name=H264,payload=" << payload_type_ << "\" "
     << "! rtph264depay name=" << dep
     << " ! h264parse name=" << par << " disable-passthrough=true "
     << "! capsfilter name=" << hcc
     << " caps=\"video/x-h264,stream-format=(string)byte-stream,alignment=(string)au\"";
  return ss.str();
}

std::vector<std::string> RtpH264Depay::element_names(int node_index) const {
  return {
      "n" + std::to_string(node_index) + "_rtp_caps",
      "n" + std::to_string(node_index) + "_depay",
      "n" + std::to_string(node_index) + "_h264parse",
      "n" + std::to_string(node_index) + "_h264_caps",
  };
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> RtpH264Depay(int payload_type) {
  return std::make_shared<sima::RtpH264Depay>(payload_type);
}

std::shared_ptr<sima::Node> H264DepayParse(int payload_type) {
  return std::make_shared<sima::RtpH264Depay>(payload_type);
}

} // namespace sima::nodes
