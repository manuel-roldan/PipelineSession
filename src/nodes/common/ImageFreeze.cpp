#include "nodes/common/ImageFreeze.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "sima/builder/Node.h"

namespace {

class ImageFreezeNode final : public sima::Node {
public:
  explicit ImageFreezeNode(int num_buffers) : num_buffers_(num_buffers) {}

  std::string kind() const override { return "ImageFreeze"; }

  std::string gst_fragment(int node_index) const override {
    std::ostringstream ss;
    ss << "imagefreeze name=n" << node_index << "_imagefreeze";
    if (num_buffers_ > 0) ss << " num-buffers=" << num_buffers_;
    return ss.str();
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_imagefreeze"};
  }

private:
  int num_buffers_ = -1;
};

} // namespace

namespace sima::nodes {

std::shared_ptr<sima::Node> ImageFreeze(int num_buffers) {
  return std::make_shared<ImageFreezeNode>(num_buffers);
}

} // namespace sima::nodes
