#pragma once

#include "sima/builder/Node.h"
#include "sima/contracts/ContractTypes.h"

#include <memory>
#include <string>

namespace sima::nodes {

// Escape hatch (implemented via a raw gst fragment node)
std::shared_ptr<sima::Node> Gst(std::string fragment,
                                sima::InputRole role = sima::InputRole::None);

// Caps helpers (typed; always produce a capsfilter)
std::shared_ptr<sima::Node> CapsRaw(std::string format,
                                    int width = -1,
                                    int height = -1,
                                    int fps = -1,
                                    sima::CapsMemory memory = sima::CapsMemory::Any);

std::shared_ptr<sima::Node> CapsNV12SysMem(int w, int h, int fps);
std::shared_ptr<sima::Node> CapsI420(int w, int h, int fps, sima::CapsMemory memory = sima::CapsMemory::Any);

} // namespace sima::nodes
