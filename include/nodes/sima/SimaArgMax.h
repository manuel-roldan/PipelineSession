#pragma once

#include "sima/builder/Node.h"
#include "sima/builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

struct SimaArgMaxOptions {
  std::string config_path;
  int sima_allocator_type = 2;
  bool silent = true;
  bool emit_signals = false;
  bool transmit = false;
};

class SimaArgMax final : public Node, public OutputSpecProvider {
public:
  explicit SimaArgMax(SimaArgMaxOptions opt = {});

  std::string kind() const override { return "SimaArgMax"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const SimaArgMaxOptions& options() const { return opt_; }

private:
  SimaArgMaxOptions opt_;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> SimaArgMax(SimaArgMaxOptions opt = {});
} // namespace sima::nodes
