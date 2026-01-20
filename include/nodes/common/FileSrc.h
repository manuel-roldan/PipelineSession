#pragma once

#include "sima/builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

class FileSrc final : public Node {
public:
  explicit FileSrc(std::string path);

  std::string kind() const override { return "FileSrc"; }
  std::string user_label() const override { return path_; }
  InputRole input_role() const override { return InputRole::Source; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const std::string& path() const { return path_; }

private:
  std::string path_;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> FileSrc(std::string path);
} // namespace sima::nodes
