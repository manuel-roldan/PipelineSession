#include "nodes/common/VideoRate.h"

#include <memory>
#include <string>
#include <vector>

#include "sima/builder/Node.h"

namespace {

class VideoRateNode final : public sima::Node {
public:
  std::string kind() const override { return "VideoRate"; }

  std::string gst_fragment(int node_index) const override {
    return "videorate name=n" + std::to_string(node_index) + "_videorate";
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_videorate"};
  }
};

} // namespace

namespace sima::nodes {

std::shared_ptr<sima::Node> VideoRate() {
  return std::make_shared<VideoRateNode>();
}

} // namespace sima::nodes
