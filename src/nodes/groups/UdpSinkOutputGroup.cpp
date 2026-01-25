#include "nodes/groups/UdpSinkOutputGroup.h"

#include "nodes/io/UdpSink.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/RtpH264Pay.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Queue.h"

#include <memory>
#include <vector>

namespace sima::nodes::groups {

sima::NodeGroup UdpSinkOutputGroup(const UdpSinkOutputGroupOptions& opt) {
  std::vector<std::shared_ptr<sima::Node>> nodes;

  std::string render_fragment =
      "simaai_sampledemux name=demux "
      "demux.bbox ! queue ! render.sink_0 "
      "demux.image ! queue ! render.sink_1 "
      "simaairender name=render config=\"" +
      opt.render_config + "\"";
  nodes.push_back(nodes::Gst(render_fragment));

  nodes.push_back(nodes::H264EncodeSima(opt.width, opt.height, opt.fps,
                                        opt.bitrate_kbps));
  nodes.push_back(nodes::H264Parse());
  nodes.push_back(nodes::RtpH264Pay(opt.payload_type, opt.config_interval));

  UdpSinkOptions udp_opt;
  udp_opt.host = opt.udp_host;
  udp_opt.port = opt.udp_port;
  udp_opt.sync = opt.udp_sync;
  udp_opt.async = opt.udp_async;
  nodes.push_back(nodes::UdpSink(udp_opt));

  return sima::NodeGroup(std::move(nodes));
}

} // namespace sima::nodes::groups
