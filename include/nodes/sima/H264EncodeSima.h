#pragma once

#include "sima/builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

class H264EncodeSima final : public Node {
public:
  H264EncodeSima(int w, int h, int fps,
                 int bitrate_kbps = 400,
                 std::string profile = "baseline",
                 std::string level = "4.0");

  std::string kind() const override { return "H264EncodeSima"; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  int width() const { return w_; }
  int height() const { return h_; }
  int fps() const { return fps_; }
  int bitrate_kbps() const { return bitrate_kbps_; }
  const std::string& profile() const { return profile_; }
  const std::string& level() const { return level_; }

private:
  int w_ = 0;
  int h_ = 0;
  int fps_ = 30;

  int bitrate_kbps_ = 400;
  std::string profile_ = "baseline";
  std::string level_ = "4.0";
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> H264EncodeSima(int w, int h, int fps,
                                           int bitrate_kbps = 400,
                                           std::string profile = "baseline",
                                           std::string level = "4.0");

// picks x264enc/openh264enc/avenc_h264
std::shared_ptr<sima::Node> H264EncodeSW(int bitrate_kbps = 400);
} // namespace sima::nodes
