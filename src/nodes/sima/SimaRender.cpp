#include "nodes/sima/SimaRender.h"

#include <sstream>
#include <string>
#include <utility>

namespace sima {

SimaRender::SimaRender(SimaRenderOptions opt) : opt_(std::move(opt)) {}

std::string SimaRender::gst_fragment(int node_index) const {
  std::ostringstream ss;
  ss << "simaairender name=n" << node_index << "_render";
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

std::vector<std::string> SimaRender::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_render"};
}

OutputSpec SimaRender::output_spec(const OutputSpec& input) const {
  OutputSpec out = input;
  out.certainty = SpecCertainty::Hint;
  out.note = "simaairender";
  return out;
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> SimaRender(SimaRenderOptions opt) {
  return std::make_shared<sima::SimaRender>(std::move(opt));
}

} // namespace sima::nodes
