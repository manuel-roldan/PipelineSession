#pragma once

#include "sima/builder/NodeGroup.h"
#include "sima/contracts/ContractTypes.h"

#include <string>

namespace sima::nodes::groups {

struct RtspInputGroupOptions {
  std::string url;
  int latency_ms = 200;
  bool tcp = true;
  int payload_type = 96;
  int h264_parse_config_interval = -1;
  int h264_fps = -1;
  int h264_width = -1;
  int h264_height = -1;
  bool insert_queue = true;
  bool auto_caps_from_stream = true;
  int fallback_h264_fps = 30;
  int fallback_h264_width = 1280;
  int fallback_h264_height = 720;

  int sima_allocator_type = 2;
  std::string out_format = "NV12";
  std::string decoder_name;
  bool decoder_raw_output = true;
  std::string decoder_next_element;

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

sima::NodeGroup RtspInputGroup(const RtspInputGroupOptions& opt);

} // namespace sima::nodes::groups
