#pragma once

#include "sima/builder/Node.h"

#include <memory>

namespace sima::nodes {
std::shared_ptr<sima::Node> ImageFreeze(int num_buffers = -1); // implemented as raw gst fragment (imagefreeze ...)
} // namespace sima::nodes
