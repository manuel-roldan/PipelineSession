#include "nodes/common/FileSrc.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sima {

FileSrc::FileSrc(std::string path) : path_(std::move(path)) {}

std::string FileSrc::gst_fragment(int node_index) const {
  const std::string el = "n" + std::to_string(node_index) + "_filesrc";
  std::ostringstream ss;
  ss << "filesrc name=" << el << " location=\"" << path_ << "\"";
  return ss.str();
}

std::vector<std::string> FileSrc::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_filesrc"};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> FileSrc(std::string path) {
  return std::make_shared<sima::FileSrc>(std::move(path));
}

} // namespace sima::nodes
