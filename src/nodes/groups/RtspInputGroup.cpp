#include "nodes/groups/RtspInputGroup.h"

#include "nodes/common/Caps.h"
#include "nodes/common/DebugPoint.h"
#include "nodes/common/Queue.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264CapsFixup.h"
#include "nodes/rtp/RtpH264Depay.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoScale.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sima::nodes::groups {
namespace {

bool caps_enabled(const RtspInputGroupOptions::OutputCaps& c) {
  return c.enable || c.width > 0 || c.height > 0 || c.fps > 0;
}

} // namespace

sima::NodeGroup RtspInputGroup(const RtspInputGroupOptions& opt) {
  std::vector<std::shared_ptr<sima::Node>> nodes;

  nodes.push_back(nodes::RTSPInput(opt.url, opt.latency_ms, opt.tcp));
  if (opt.insert_queue) nodes.push_back(nodes::Queue());
  const bool use_auto_caps =
      opt.auto_caps_from_stream &&
      (opt.h264_fps <= 0 || opt.h264_width <= 0 || opt.h264_height <= 0);
  nodes.push_back(nodes::H264DepayParse(opt.payload_type,
                                        opt.h264_parse_config_interval,
                                        opt.h264_fps,
                                        opt.h264_width,
                                        opt.h264_height,
                                        /*enforce_h264_caps=*/!use_auto_caps));

  if (opt.debug.enable && opt.debug.encoded) {
    nodes.push_back(nodes::DebugPoint(opt.debug.encoded_name));
  }

  if (opt.insert_queue) nodes.push_back(nodes::Queue());

  if (use_auto_caps) {
    nodes.push_back(nodes::H264CapsFixup(opt.fallback_h264_fps,
                                         opt.fallback_h264_width,
                                         opt.fallback_h264_height));
  }

  nodes.push_back(nodes::H264Decode(opt.sima_allocator_type,
                                    opt.out_format,
                                    opt.decoder_name,
                                    opt.decoder_raw_output,
                                    opt.decoder_next_element));

  if (opt.debug.enable && opt.debug.decoded) {
    nodes.push_back(nodes::DebugPoint(opt.debug.decoded_name));
  }

  if (opt.use_videoconvert) nodes.push_back(nodes::VideoConvert());
  if (opt.use_videoscale) nodes.push_back(nodes::VideoScale());

  if (caps_enabled(opt.output_caps)) {
    const auto& c = opt.output_caps;
    nodes.push_back(nodes::CapsRaw(c.format, c.width, c.height, c.fps, c.memory));
  }

  if (!opt.extra_fragment.empty()) {
    nodes.push_back(nodes::Gst(opt.extra_fragment));
  }

  if (opt.debug.enable && opt.debug.normalized) {
    nodes.push_back(nodes::DebugPoint(opt.debug.normalized_name));
  }

  return sima::NodeGroup(std::move(nodes));
}

} // namespace sima::nodes::groups
