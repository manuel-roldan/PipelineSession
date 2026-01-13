#pragma once

#include "sima/builder/NodeGroup.h"
#include "sima/contracts/ContractTypes.h"

#include <string>

namespace sima::nodes::groups {

struct ImageInputGroupOptions {
  std::string path;
  int imagefreeze_num_buffers = -1;
  int fps = 30;

  bool use_videorate = true;
  bool use_videoconvert = true;
  bool use_videoscale = true;

  struct OutputCaps {
    bool enable = true;
    std::string format = "NV12";
    int width = -1;
    int height = -1;
    int fps = -1;
    sima::CapsMemory memory = sima::CapsMemory::SystemMemory;
  } output_caps;

  struct DebugPoints {
    bool enable = false;
    bool decoded = true;
    bool normalized = true;
    std::string decoded_name = "decoded";
    std::string normalized_name = "normalized";
  } debug;

  enum class Decoder {
    Auto = 0,      // decodebin (jpg/png auto)
    ForceJpeg,
    ForcePng,
    Custom,
  };

  Decoder decoder = Decoder::Auto;
  std::string custom_decoder_fragment;

  struct SimaDecoder {
    bool enable = false;
    int sima_allocator_type = 2;
    std::string decoder_name = "decoder";
    bool raw_output = false;
  } sima_decoder;

  // Optional raw fragment inserted before ImageFreeze (advanced use)
  std::string extra_fragment;
};

sima::NodeGroup ImageInputGroup(const ImageInputGroupOptions& opt);

} // namespace sima::nodes::groups
