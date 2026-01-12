#pragma once

#include "sima/builder/NodeGroup.h"
#include "sima/contracts/ContractTypes.h"
#include "sima/nodes/sima/H264Parse.h"

#include <string>

namespace sima::nodes::groups {

struct VideoInputGroupOptions {
  std::string path;
  int demux_video_pad_index = 0;
  bool insert_queue = true;

  int parse_config_interval = 1;
  bool parse_enforce_au = true;

  int sima_allocator_type = 2;
  std::string out_format = "NV12";

  bool use_videoconvert = false;
  bool use_videoscale = false;

  struct OutputCaps {
    bool enable = false;
    std::string format = "NV12";
    int width = -1;
    int height = -1;
    int fps = -1;
    sima::CapsMemory memory = sima::CapsMemory::SystemMemory;
  } output_caps;

  struct DebugPoints {
    bool enable = false;
    bool encoded = true;
    bool decoded = true;
    bool normalized = true;
    std::string encoded_name = "encoded";
    std::string decoded_name = "decoded";
    std::string normalized_name = "normalized";
  } debug;

  std::string extra_fragment;
};

sima::NodeGroup VideoInputGroup(const VideoInputGroupOptions& opt);

} // namespace sima::nodes::groups
