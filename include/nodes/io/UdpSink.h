#pragma once

#include "sima/builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

struct UdpSinkOptions {
  std::string host = "127.0.0.1";
  int port = 5000;
  bool sync = false;
  bool async = false;
};

class UdpSink final : public Node {
public:
  explicit UdpSink(UdpSinkOptions opt) : opt_(std::move(opt)) {}

  std::string kind() const override { return "UdpSink"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

private:
  UdpSinkOptions opt_;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> UdpSink(UdpSinkOptions opt = {});
} // namespace sima::nodes
