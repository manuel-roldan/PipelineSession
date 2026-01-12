#include "nodes/common/DebugPoint.h"

#include <cctype>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sima {
namespace {

std::string sanitize_name(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (char c : in) {
    const bool ok =
        (c >= 'a' && c <= 'z') ||
        (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') ||
        (c == '_' || c == '-');
    out.push_back(ok ? c : '_');
  }
  if (out.empty()) out = "dbg";
  if (!out.empty() && (out[0] >= '0' && out[0] <= '9')) out = "_" + out;
  return out;
}

} // namespace

DebugPoint::DebugPoint(std::string name) : name_(std::move(name)) {
  if (name_.empty()) name_ = "dbg";
}

std::string DebugPoint::gst_fragment(int /*node_index*/) const {
  const std::string elname = "dbg_" + sanitize_name(name_);
  std::ostringstream ss;
  ss << "identity name=" << elname << " silent=true";
  return ss.str();
}

std::vector<std::string> DebugPoint::element_names(int /*node_index*/) const {
  return {"dbg_" + sanitize_name(name_)};
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> DebugPoint(std::string name) {
  return std::make_shared<sima::DebugPoint>(std::move(name));
}

} // namespace sima::nodes
