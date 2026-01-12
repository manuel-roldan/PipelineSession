#include "nodes/groups/ImageInputGroup.h"

#include "nodes/common/Caps.h"
#include "nodes/common/DebugPoint.h"
#include "nodes/common/FileSrc.h"
#include "nodes/common/ImageDecode.h"
#include "nodes/common/ImageFreeze.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoRate.h"
#include "nodes/common/VideoScale.h"
#include "nodes/common/JpegDec.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sima::nodes::groups {
namespace {

bool caps_enabled(const ImageInputGroupOptions::OutputCaps& c) {
  return c.enable || c.width > 0 || c.height > 0 || c.fps > 0;
}

} // namespace

sima::NodeGroup ImageInputGroup(const ImageInputGroupOptions& opt) {
  std::vector<std::shared_ptr<sima::Node>> nodes;

  nodes.push_back(nodes::FileSrc(opt.path));

  switch (opt.decoder) {
    case ImageInputGroupOptions::Decoder::Auto:
      nodes.push_back(nodes::ImageDecode());
      break;
    case ImageInputGroupOptions::Decoder::ForceJpeg:
      nodes.push_back(nodes::JpegDec());
      break;
    case ImageInputGroupOptions::Decoder::ForcePng:
      nodes.push_back(nodes::Gst("pngdec"));
      break;
    case ImageInputGroupOptions::Decoder::Custom:
      nodes.push_back(nodes::Gst(opt.custom_decoder_fragment));
      break;
  }

  if (opt.debug.enable && opt.debug.decoded) {
    nodes.push_back(nodes::DebugPoint(opt.debug.decoded_name));
  }

  if (!opt.extra_fragment.empty()) {
    nodes.push_back(nodes::Gst(opt.extra_fragment));
  }

  nodes.push_back(nodes::ImageFreeze(opt.imagefreeze_num_buffers));

  if (opt.use_videorate) {
    nodes.push_back(nodes::VideoRate());
  }

  if (opt.use_videoconvert) {
    nodes.push_back(nodes::VideoConvert());
  }

  if (opt.use_videoscale) {
    nodes.push_back(nodes::VideoScale());
  }

  auto caps = opt.output_caps;
  if (caps.fps <= 0) {
    caps.fps = opt.fps;
  }

  if (caps_enabled(caps)) {
    nodes.push_back(nodes::CapsRaw(caps.format, caps.width, caps.height, caps.fps, caps.memory));
  }

  if (opt.debug.enable && opt.debug.normalized) {
    nodes.push_back(nodes::DebugPoint(opt.debug.normalized_name));
  }

  return sima::NodeGroup(std::move(nodes));
}

} // namespace sima::nodes::groups
