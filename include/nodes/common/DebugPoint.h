#pragma once

#include "sima/builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

class DebugPoint final : public Node {
public:
  explicit DebugPoint(std::string name = "dbg");
  std::string kind() const override { return "DebugPoint"; }
  std::string user_label() const override { return name_; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const std::string& name() const { return name_; }

private:
  std::string name_ = "dbg";
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> DebugPoint(std::string name);
} // namespace sima::nodes
