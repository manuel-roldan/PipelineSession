#include "nodes/common/Caps.h"

#include "builder/OutputSpec.h"

#include <cctype>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
  if (out.empty()) out = "gst";
  if (!out.empty() && (out[0] >= '0' && out[0] <= '9')) out = "_" + out;
  return out;
}

class GstNode final : public sima::Node {
public:
  explicit GstNode(std::string fragment, sima::InputRole role)
      : fragment_(std::move(fragment)), role_(role) {
    if (fragment_.empty()) fragment_ = "identity silent=true";
  }

  std::string kind() const override { return "GstNode"; }
  std::string user_label() const override { return fragment_; }
  sima::InputRole input_role() const override { return role_; }

  std::string gst_fragment(int node_index) const override {
    const std::string frag = trim_(fragment_);

    const bool has_name = (frag.find("name=") != std::string::npos);
    const bool looks_complex =
        (frag.find('!') != std::string::npos) ||
        (frag.find('(') != std::string::npos) ||
        (frag.find(')') != std::string::npos);

    if (has_name || looks_complex) return frag;

    std::string factory = first_token_(frag);
    const std::string elname =
        "n" + std::to_string(node_index) + "_" + sanitize_name(factory);
    return frag + " name=" + elname;
  }

  std::vector<std::string> element_names(int node_index) const override {
    const std::string frag = trim_(fragment_);

    const size_t pos = frag.find("name=");
    if (pos != std::string::npos) {
      size_t i = pos + 5;
      while (i < frag.size() && std::isspace(static_cast<unsigned char>(frag[i]))) i++;

      if (i < frag.size() && frag[i] == '"') {
        i++;
        size_t j = i;
        while (j < frag.size() && frag[j] != '"') j++;
        if (j > i) return {frag.substr(i, j - i)};
      } else {
        size_t j = i;
        while (j < frag.size() && !std::isspace(static_cast<unsigned char>(frag[j]))) j++;
        if (j > i) return {frag.substr(i, j - i)};
      }
      return {};
    }

    const bool looks_complex =
        (frag.find('!') != std::string::npos) ||
        (frag.find('(') != std::string::npos) ||
        (frag.find(')') != std::string::npos);
    if (looks_complex) return {};

    std::string factory = first_token_(frag);
    return {"n" + std::to_string(node_index) + "_" + sanitize_name(factory)};
  }

private:
  static std::string trim_(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) b++;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) e--;
    return s.substr(b, e - b);
  }

  static std::string first_token_(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) i++;
    size_t j = i;
    while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j]))) j++;
    return (j > i) ? s.substr(i, j - i) : "gst";
  }

  std::string fragment_;
  sima::InputRole role_ = sima::InputRole::None;
};

class CapsRawNode final : public sima::Node, public sima::OutputSpecProvider {
public:
  CapsRawNode(std::string format, int w, int h, int fps, sima::CapsMemory mem)
      : format_(std::move(format)), w_(w), h_(h), fps_(fps), mem_(mem) {}

  std::string kind() const override { return "CapsRaw"; }

  std::string gst_fragment(int node_index) const override {
    const std::string name = "n" + std::to_string(node_index) + "_caps";

    std::ostringstream caps;
    caps << "video/x-raw";
    if (mem_ == sima::CapsMemory::SystemMemory) caps << "(memory:SystemMemory)";
    if (!format_.empty()) caps << ",format=" << format_;
    if (w_ > 0) caps << ",width=" << w_;
    if (h_ > 0) caps << ",height=" << h_;
    if (fps_ > 0) caps << ",framerate=" << fps_ << "/1";

    std::ostringstream ss;
    ss << "capsfilter name=" << name << " caps=\"" << caps.str() << "\"";
    return ss.str();
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_caps"};
  }

  sima::OutputSpec output_spec(const sima::OutputSpec& input) const override {
    sima::OutputSpec out = input;
    out.media_type = "video/x-raw";
    if (!format_.empty()) out.format = format_;
    if (w_ > 0) out.width = w_;
    if (h_ > 0) out.height = h_;
    if (fps_ > 0) {
      out.fps_num = fps_;
      out.fps_den = 1;
    }
    out.memory = (mem_ == sima::CapsMemory::SystemMemory) ? "SystemMemory" : out.memory;
    out.dtype = "UInt8";
    if (out.format == "RGB" || out.format == "BGR") {
      out.layout = "HWC";
      out.depth = 3;
    }
    if (out.format == "GRAY8") {
      out.layout = "HW";
      out.depth = 1;
    }
    if (out.format == "NV12" || out.format == "I420") {
      out.layout = "Planar";
    }
    out.certainty = sima::SpecCertainty::Derived;
    out.note = "CapsRaw";
    out.byte_size = 0;
    out.byte_size = sima::expected_byte_size(out);
    return out;
  }

private:
  std::string format_;
  int w_ = -1;
  int h_ = -1;
  int fps_ = -1;
  sima::CapsMemory mem_ = sima::CapsMemory::Any;
};

} // namespace

namespace sima::nodes {

std::shared_ptr<sima::Node> Gst(std::string fragment, sima::InputRole role) {
  return std::make_shared<GstNode>(std::move(fragment), role);
}

std::shared_ptr<sima::Node> CapsRaw(std::string format,
                                    int width,
                                    int height,
                                    int fps,
                                    sima::CapsMemory memory) {
  return std::make_shared<CapsRawNode>(std::move(format), width, height, fps, memory);
}

std::shared_ptr<sima::Node> CapsNV12SysMem(int w, int h, int fps) {
  return CapsRaw("NV12", w, h, fps, sima::CapsMemory::SystemMemory);
}

std::shared_ptr<sima::Node> CapsI420(int w, int h, int fps, sima::CapsMemory memory) {
  return CapsRaw("I420", w, h, fps, memory);
}

} // namespace sima::nodes
