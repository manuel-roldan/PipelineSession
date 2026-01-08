#pragma once

#include "sima/builder/Node.h"

#include <memory>
#include <vector>

namespace sima {

// qtdemux exposes pads like demux.video_0, demux.audio_0, etc.
// This node both creates the demux element AND selects a video pad.
class QtDemuxVideoPad final : public Node {
public:
  explicit QtDemuxVideoPad(int video_pad_index = 0);

  std::string kind() const override { return "QtDemuxVideoPad"; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  int video_pad_index() const { return idx_; }

private:
  int idx_ = 0;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> QtDemuxVideoPad(int video_pad_index = 0);
} // namespace sima::nodes
