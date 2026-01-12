#include "nodes/common/Queue.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

std::string Queue::gst_fragment(int node_index) const {
  return "queue name=n" + std::to_string(node_index) + "_queue";
}

std::vector<std::string> Queue::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_queue"};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> Queue() {
  return std::make_shared<sima::Queue>();
}

} // namespace sima::nodes
