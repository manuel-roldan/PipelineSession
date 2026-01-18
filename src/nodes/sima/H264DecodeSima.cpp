#include "nodes/sima/H264DecodeSima.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sima {

H264Decode::H264Decode(int sima_allocator_type,
                       std::string out_format,
                       std::string decoder_name,
                       bool raw_output)
    : sima_allocator_type_(sima_allocator_type),
      out_format_(std::move(out_format)),
      decoder_name_(std::move(decoder_name)),
      raw_output_(raw_output) {}

std::string H264Decode::gst_fragment(int node_index) const {
  const std::string dec = decoder_name_.empty()
      ? ("n" + std::to_string(node_index) + "_decoder")
      : decoder_name_;
  const std::string vc = "n" + std::to_string(node_index) + "_videoconvert";
  const std::string cap = "n" + std::to_string(node_index) + "_raw_caps";

  if (raw_output_) {
    std::ostringstream ss;
    ss << "simaaidecoder name=" << dec
       << " sima-allocator-type=" << sima_allocator_type_;
    if (!decoder_name_.empty()) {
      ss << " op-buff-name=" << decoder_name_;
    }
    if (!out_format_.empty()) {
      std::string dec_fmt = out_format_;
      if (dec_fmt == "I420") dec_fmt = "YUV420P";
      if (dec_fmt == "NV12" || dec_fmt == "YUV420P") {
        ss << " dec-fmt=" << dec_fmt;
      }
    }
    return ss.str();
  }

  std::ostringstream caps;
  caps << "video/x-raw(memory:SystemMemory),format=" << out_format_;

  std::ostringstream ss;
  ss << "simaaidecoder name=" << dec
     << " sima-allocator-type=" << sima_allocator_type_;
  if (!decoder_name_.empty()) {
    ss << " op-buff-name=" << decoder_name_;
  }
  ss
     << " ! videoconvert name=" << vc
     << " ! capsfilter name=" << cap << " caps=\"" << caps.str() << "\"";
  return ss.str();
}

std::vector<std::string> H264Decode::element_names(int node_index) const {
  const std::string dec = decoder_name_.empty()
      ? ("n" + std::to_string(node_index) + "_decoder")
      : decoder_name_;
  if (raw_output_) {
    return {dec};
  }
  return {dec,
          "n" + std::to_string(node_index) + "_videoconvert",
          "n" + std::to_string(node_index) + "_raw_caps"};
}

OutputSpec H264Decode::output_spec(const OutputSpec& /*input*/) const {
  OutputSpec out;
  out.media_type = "video/x-raw";
  out.format = out_format_.empty() ? "NV12" : out_format_;
  out.layout = (out.format == "NV12" || out.format == "I420") ? "Planar" : "HWC";
  out.dtype = "UInt8";
  out.memory = raw_output_ ? "SimaAI" : "SystemMemory";
  out.certainty = SpecCertainty::Hint;
  out.note = "H264Decode output";
  out.byte_size = expected_byte_size(out);
  return out;
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> H264Decode(int sima_allocator_type,
                                       std::string out_format,
                                       std::string decoder_name,
                                       bool raw_output) {
  return std::make_shared<sima::H264Decode>(sima_allocator_type,
                                            std::move(out_format),
                                            std::move(decoder_name),
                                            raw_output);
}

} // namespace sima::nodes
