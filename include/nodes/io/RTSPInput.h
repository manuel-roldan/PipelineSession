#pragma once
#include "sima/builder/Node.h"
#include <memory>
#include <string>
#include <vector>

namespace sima {

class RTSPInput final : public Node {
public:
  RTSPInput(std::string url, int latency_ms = 200, bool tcp = true);

  std::string kind() const override { return "RTSPInput"; }
  std::string user_label() const override { return url_; }
  InputRole input_role() const override { return InputRole::Source; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const std::string& url() const { return url_; }
  int latency_ms() const { return latency_ms_; }
  bool tcp() const { return tcp_; }

private:
  std::string url_;
  int latency_ms_ = 200;
  bool tcp_ = true;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> RTSPInput(std::string url, int latency_ms = 200, bool tcp = true);
} // namespace sima::nodes
