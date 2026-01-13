#include "nodes/common/AppSink.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace sima {

OutputAppSinkOptions OutputAppSinkOptions::Latest() {
  return OutputAppSinkOptions{};
}

OutputAppSinkOptions OutputAppSinkOptions::EveryFrame(int max_buffers) {
  OutputAppSinkOptions opt;
  opt.drop = false;
  opt.sync = false;
  opt.max_buffers = (max_buffers < 0) ? 0 : max_buffers;
  return opt;
}

OutputAppSinkOptions OutputAppSinkOptions::Clocked(int max_buffers) {
  OutputAppSinkOptions opt;
  opt.drop = true;
  opt.sync = true;
  opt.max_buffers = (max_buffers < 0) ? 0 : max_buffers;
  return opt;
}

std::string OutputAppSink::gst_fragment(int /*node_index*/) const {
  const int max_buffers = (opt_.max_buffers < 0) ? 0 : opt_.max_buffers;

  std::ostringstream ss;
  ss << "appsink name=mysink emit-signals=false "
     << "sync=" << (opt_.sync ? "true" : "false") << " "
     << "max-buffers=" << max_buffers << " "
     << "drop=" << (opt_.drop ? "true" : "false");
  return ss.str();
}

std::vector<std::string> OutputAppSink::element_names(int /*node_index*/) const {
  return {"mysink"};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> OutputAppSink(OutputAppSinkOptions opt) {
  return std::make_shared<sima::OutputAppSink>(std::move(opt));
}

} // namespace sima::nodes
