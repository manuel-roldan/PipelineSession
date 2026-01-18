#pragma once

#include "sima/builder/Node.h"
#include "sima/builder/OutputSpec.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

struct SimaBoxDecodeOptions {
  std::string config_path;
  int sima_allocator_type = 2;
  bool silent = true;
  bool emit_signals = false;
  bool transmit = false;
};

class SimaBoxDecode final : public Node, public OutputSpecProvider {
public:
  explicit SimaBoxDecode(SimaBoxDecodeOptions opt = {});

  std::string kind() const override { return "SimaBoxDecode"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const SimaBoxDecodeOptions& options() const { return opt_; }

private:
  SimaBoxDecodeOptions opt_;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> SimaBoxDecode(SimaBoxDecodeOptions opt = {});
} // namespace sima::nodes
