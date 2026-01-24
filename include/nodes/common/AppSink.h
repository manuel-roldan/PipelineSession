#pragma once

#include "sima/builder/Node.h"

#include <memory>
#include <utility>
#include <vector>

namespace sima {

struct OutputAppSinkOptions {
  int max_buffers = 1;
  bool drop = false;
  bool sync = false;

  static OutputAppSinkOptions Latest();
  static OutputAppSinkOptions EveryFrame(int max_buffers = 30);
  static OutputAppSinkOptions Clocked(int max_buffers = 1);
};

class OutputAppSink final : public Node {
public:
  OutputAppSink() = default;
  explicit OutputAppSink(OutputAppSinkOptions opt) : opt_(std::move(opt)) {}

  const OutputAppSinkOptions& options() const { return opt_; }

  std::string kind() const override { return "OutputAppSink"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

private:
  OutputAppSinkOptions opt_;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> OutputAppSink(OutputAppSinkOptions opt = {});
} // namespace sima::nodes
