#include "nodes/sima/SimaBoxDecode.h"

#include <sstream>
#include <string>
#include <utility>

namespace sima {

SimaBoxDecode::SimaBoxDecode(SimaBoxDecodeOptions opt) : opt_(std::move(opt)) {}

std::string SimaBoxDecode::gst_fragment(int node_index) const {
  std::ostringstream ss;
  ss << "simaaiboxdecode name=n" << node_index << "_boxdecode";
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

std::vector<std::string> SimaBoxDecode::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_boxdecode"};
}

OutputSpec SimaBoxDecode::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "BBOX";
  out.memory = input.memory;
  out.certainty = SpecCertainty::Hint;
  out.note = "simaaiboxdecode";
  return out;
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> SimaBoxDecode(SimaBoxDecodeOptions opt) {
  return std::make_shared<sima::SimaBoxDecode>(std::move(opt));
}

} // namespace sima::nodes
