#include "nodes/common/QtDemuxVideoPad.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sima {

QtDemuxVideoPad::QtDemuxVideoPad(int video_pad_index) : idx_(video_pad_index) {}

std::string QtDemuxVideoPad::gst_fragment(int node_index) const {
  const std::string base = "n" + std::to_string(node_index) + "_demux";
  std::ostringstream ss;
  ss << "qtdemux name=" << base << " " << base << ".video_" << idx_;
  return ss.str();
}

std::vector<std::string> QtDemuxVideoPad::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_demux"};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> QtDemuxVideoPad(int video_pad_index) {
  return std::make_shared<sima::QtDemuxVideoPad>(video_pad_index);
}

} // namespace sima::nodes
