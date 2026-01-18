#include "nodes/sima/H264EncodeSima.h"

#include "gst/GstHelpers.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class H264EncodeSWNode final : public sima::Node {
public:
  explicit H264EncodeSWNode(int bitrate_kbps) : bitrate_kbps_(bitrate_kbps) {}
  std::string kind() const override { return "H264EncodeSW"; }

  std::string gst_fragment(int node_index) const override {
    std::string factory;
    std::string props;

    if (sima::element_exists("x264enc")) {
      factory = "x264enc";
      props =
          "tune=zerolatency speed-preset=ultrafast "
          "key-int-max=1 bframes=0 "
          "bitrate=" + std::to_string(bitrate_kbps_) + " "
          "byte-stream=true";
    } else if (sima::element_exists("openh264enc")) {
      factory = "openh264enc";
      props = "";
    } else if (sima::element_exists("avenc_h264")) {
      factory = "avenc_h264";
      props = "";
    } else {
      throw std::runtime_error(
          "H264EncodeSW: no software H264 encoder found. Install one of: "
          "x264enc (gst-plugins-ugly), openh264enc (gst-plugins-bad), avenc_h264 (gst-libav).");
    }

    std::ostringstream ss;
    ss << factory << " name=n" << node_index << "_swenc";
    if (!props.empty()) ss << " " << props;
    return ss.str();
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_swenc"};
  }

private:
  int bitrate_kbps_ = 400;
};

} // namespace

namespace sima {

H264EncodeSima::H264EncodeSima(int w, int h, int fps,
                               int bitrate_kbps,
                               std::string profile,
                               std::string level)
    : w_(w),
      h_(h),
      fps_(fps),
      bitrate_kbps_(bitrate_kbps),
      profile_(std::move(profile)),
      level_(std::move(level)) {}

std::string H264EncodeSima::gst_fragment(int node_index) const {
  std::ostringstream ss;
  ss << "simaaiencoder name=n" << node_index << "_encoder "
     << "enc-type=h264 "
     << "enc-profile=" << profile_ << " "
     << "enc-level=" << level_ << " "
     << "enc-fmt=NV12 "
     << "enc-width=" << w_ << " "
     << "enc-height=" << h_ << " "
     << "enc-frame-rate=" << fps_ << " "
     << "enc-bitrate=" << bitrate_kbps_ << " "
     << "enc-ip-mode=async "
     << "ip-rate-ctrl=false";
  return ss.str();
}

std::vector<std::string> H264EncodeSima::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_encoder"};
}

OutputSpec H264EncodeSima::output_spec(const OutputSpec& /*input*/) const {
  OutputSpec out;
  out.media_type = "video/x-h264";
  out.format = "H264";
  out.certainty = SpecCertainty::Hint;
  out.note = "H264 encoded stream";
  return out;
}

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> H264EncodeSima(int w, int h, int fps,
                                           int bitrate_kbps,
                                           std::string profile,
                                           std::string level) {
  return std::make_shared<sima::H264EncodeSima>(w, h, fps, bitrate_kbps,
                                                std::move(profile), std::move(level));
}

std::shared_ptr<sima::Node> H264EncodeSW(int bitrate_kbps) {
  return std::make_shared<H264EncodeSWNode>(bitrate_kbps);
}

} // namespace sima::nodes
