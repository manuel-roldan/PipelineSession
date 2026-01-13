#include "nodes/io/InputAppSrc.h"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sima {
namespace {

const char* stream_type_string(int stream_type) {
  switch (stream_type) {
    case 1:
      return "seekable";
    case 2:
      return "random-access";
    case 0:
    default:
      return "stream";
  }
}

std::string build_caps_string(const InputAppSrcOptions& opt) {
  std::ostringstream caps;
  caps << opt.media_type;

  if (!opt.format.empty()) {
    caps << ",format=" << opt.format;
  }
  if (opt.width > 0) {
    caps << ",width=" << opt.width;
  }
  if (opt.height > 0) {
    caps << ",height=" << opt.height;
  }
  if (opt.depth > 0) {
    caps << ",depth=" << opt.depth;
  }

  return caps.str();
}

} // namespace

InputAppSrc::InputAppSrc(InputAppSrcOptions opt) : opt_(std::move(opt)) {}

std::string InputAppSrc::caps_string() const {
  return build_caps_string(opt_);
}

std::string InputAppSrc::gst_fragment(int /*node_index*/) const {
  std::ostringstream ss;
  ss << "appsrc name=mysrc";
  ss << " is-live=" << (opt_.is_live ? "true" : "false");
  ss << " format=time";
  ss << " do-timestamp=" << (opt_.do_timestamp ? "true" : "false");
  ss << " block=" << (opt_.block ? "true" : "false");
  ss << " stream-type=" << stream_type_string(opt_.stream_type);
  ss << " max-bytes=" << opt_.max_bytes;

  const std::string caps = caps_string();
  if (!caps.empty()) {
    ss << " caps=\"" << caps << "\"";
  }

  return ss.str();
}

std::vector<std::string> InputAppSrc::element_names(int /*node_index*/) const {
  return {"mysrc"};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> InputAppSrc(InputAppSrcOptions opt) {
  return std::make_shared<sima::InputAppSrc>(std::move(opt));
}

} // namespace sima::nodes
