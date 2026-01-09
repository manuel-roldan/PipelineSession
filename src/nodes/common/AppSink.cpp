#include "nodes/common/AppSink.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

std::string OutputAppSink::gst_fragment(int /*node_index*/) const {
  return "appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true";
}

std::vector<std::string> OutputAppSink::element_names(int /*node_index*/) const {
  return {"mysink"};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> OutputAppSink() {
  return std::make_shared<sima::OutputAppSink>();
}

} // namespace sima::nodes
