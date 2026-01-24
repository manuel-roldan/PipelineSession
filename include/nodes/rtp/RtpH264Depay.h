#pragma once
#include "sima/builder/Node.h"
#include "sima/builder/OutputSpec.h"
#include <memory>
#include <vector>

namespace sima {

class RtpH264Depay final : public Node, public OutputSpecProvider {
public:
  // payload_type <= 0 disables RTP payload filtering in caps.
  explicit RtpH264Depay(int payload_type = 96,
                        int h264_parse_config_interval = -1,
                        int h264_fps = -1,
                        int h264_width = -1,
                        int h264_height = -1);
  std::string kind() const override { return "RtpH264Depay"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  int payload_type() const { return payload_type_; }
  int h264_parse_config_interval() const { return h264_parse_config_interval_; }
  int h264_fps() const { return h264_fps_; }
  int h264_width() const { return h264_width_; }
  int h264_height() const { return h264_height_; }

private:
  int payload_type_ = 96;
  int h264_parse_config_interval_ = -1;
  int h264_fps_ = -1;
  int h264_width_ = -1;
  int h264_height_ = -1;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> RtpH264Depay(int payload_type = 96,
                                         int h264_parse_config_interval = -1,
                                         int h264_fps = -1,
                                         int h264_width = -1,
                                         int h264_height = -1);
std::shared_ptr<sima::Node> H264DepayParse(int payload_type = 96,
                                           int h264_parse_config_interval = -1,
                                           int h264_fps = -1,
                                           int h264_width = -1,
                                           int h264_height = -1);
} // namespace sima::nodes
