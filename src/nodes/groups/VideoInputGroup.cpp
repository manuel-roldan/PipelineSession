#include "nodes/groups/VideoInputGroup.h"

#include "nodes/common/Caps.h"
#include "nodes/common/DebugPoint.h"
#include "nodes/common/FileSrc.h"
#include "nodes/common/QtDemuxVideoPad.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoScale.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264Parse.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sima::nodes::groups {
namespace {

bool caps_enabled(const VideoInputGroupOptions::OutputCaps& c) {
  return c.enable || c.width > 0 || c.height > 0 || c.fps > 0;
}

} // namespace

sima::NodeGroup VideoInputGroup(const VideoInputGroupOptions& opt) {
  std::vector<std::shared_ptr<sima::Node>> nodes;

  nodes.push_back(nodes::FileSrc(opt.path));
  nodes.push_back(nodes::QtDemuxVideoPad(opt.demux_video_pad_index));

  if (opt.insert_queue) nodes.push_back(nodes::Queue());

  if (opt.parse_enforce_au) {
    nodes.push_back(nodes::H264ParseAu(opt.parse_config_interval));
  } else {
    nodes.push_back(nodes::H264Parse(opt.parse_config_interval));
  }

  if (opt.debug.enable && opt.debug.encoded) {
    nodes.push_back(nodes::DebugPoint(opt.debug.encoded_name));
  }

  if (opt.insert_queue) nodes.push_back(nodes::Queue());

  nodes.push_back(nodes::H264Decode(opt.sima_allocator_type, opt.out_format));

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
