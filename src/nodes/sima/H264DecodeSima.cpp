#include "nodes/sima/H264DecodeSima.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sima {

H264Decode::H264Decode(int sima_allocator_type, std::string out_format)
    : sima_allocator_type_(sima_allocator_type),
      out_format_(std::move(out_format)) {}

std::string H264Decode::gst_fragment(int node_index) const {
  const std::string dec = "n" + std::to_string(node_index) + "_decoder";
  const std::string vc = "n" + std::to_string(node_index) + "_videoconvert";
  const std::string cap = "n" + std::to_string(node_index) + "_raw_caps";

  std::ostringstream caps;
  caps << "video/x-raw(memory:SystemMemory),format=" << out_format_;

  std::ostringstream ss;
  ss << "simaaidecoder name=" << dec
     << " sima-allocator-type=" << sima_allocator_type_
     << " ! videoconvert name=" << vc
     << " ! capsfilter name=" << cap << " caps=\"" << caps.str() << "\"";
  return ss.str();
}

std::vector<std::string> H264Decode::element_names(int node_index) const {
  return {
      "n" + std::to_string(node_index) + "_decoder",
      "n" + std::to_string(node_index) + "_videoconvert",
      "n" + std::to_string(node_index) + "_raw_caps",
  };
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> H264Decode(int sima_allocator_type, std::string out_format) {
  return std::make_shared<sima::H264Decode>(sima_allocator_type, std::move(out_format));
}

} // namespace sima::nodes
