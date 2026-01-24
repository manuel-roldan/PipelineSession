#pragma once

#include "pipeline/NeatTensorCore.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sima {

struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

struct BoxDecodeResult {
  std::vector<Box> boxes;
  std::vector<uint8_t> raw;
};

std::vector<Box> parse_bbox_bytes(const std::vector<uint8_t>& bytes,
                                  int img_w,
                                  int img_h,
                                  int expected_topk,
                                  bool strict);

BoxDecodeResult decode_bbox_tensor(const NeatTensor& tensor,
                                   int img_w,
                                   int img_h,
                                   int expected_topk,
                                   bool strict);

} // namespace sima
