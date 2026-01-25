#pragma once

#include "sima/builder/NodeGroup.h"

#include <string>

namespace sima::nodes::groups {

struct UdpSinkOutputGroupOptions {
  std::string render_config;
  int width = 0;
  int height = 0;
  int fps = 30;
  int bitrate_kbps = 4000;
  int payload_type = 96;
  int config_interval = 1;
  std::string udp_host = "127.0.0.1";
  int udp_port = 5000;
  bool udp_sync = false;
  bool udp_async = false;
};

sima::NodeGroup UdpSinkOutputGroup(const UdpSinkOutputGroupOptions& opt);

} // namespace sima::nodes::groups
