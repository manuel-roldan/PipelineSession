#pragma once

#include "sima/builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace sima {

class H264CapsFixup final : public Node {
public:
  H264CapsFixup(int fallback_fps, int fallback_width, int fallback_height);

  std::string kind() const override { return "H264CapsFixup"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  int fallback_fps() const { return fallback_fps_; }
  int fallback_width() const { return fallback_width_; }
  int fallback_height() const { return fallback_height_; }

private:
  int fallback_fps_ = 30;
  int fallback_width_ = 1280;
  int fallback_height_ = 720;
};

} // namespace sima

namespace sima::nodes {

std::shared_ptr<sima::Node> H264CapsFixup(int fallback_fps = 30,
                                          int fallback_width = 1280,
                                          int fallback_height = 720);

} // namespace sima::nodes
