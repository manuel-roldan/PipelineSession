#include "nodes/sima/SimaArgMax.h"

#include <sstream>
#include <string>
#include <utility>

namespace sima {

SimaArgMax::SimaArgMax(SimaArgMaxOptions opt) : opt_(std::move(opt)) {}

std::string SimaArgMax::gst_fragment(int node_index) const {
  std::ostringstream ss;
  ss << "simaaiargmax name=n" << node_index << "_argmax";
  if (!opt_.config_path.empty()) {
    ss << " config=\"" << opt_.config_path << "\"";
  }
  ss << " silent=" << (opt_.silent ? "true" : "false");
  ss << " emit-signals=" << (opt_.emit_signals ? "true" : "false");
  if (opt_.sima_allocator_type > 0) {
    ss << " sima-allocator-type=" << opt_.sima_allocator_type;
  }
  ss << " transmit=" << (opt_.transmit ? "true" : "false");
  return ss.str();
}

std::vector<std::string> SimaArgMax::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_argmax"};
}

OutputSpec SimaArgMax::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "ARGMAX";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "simaaiargmax";
  return out;
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> SimaArgMax(SimaArgMaxOptions opt) {
  return std::make_shared<sima::SimaArgMax>(std::move(opt));
}

} // namespace sima::nodes
