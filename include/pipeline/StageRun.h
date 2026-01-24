#pragma once

#include "pipeline/DetectionTypes.h"
#include "pipeline/NeatTensorCore.h"

#include <string>

namespace cv {
class Mat;
} // namespace cv

namespace sima {
namespace mpk {
class ModelMPK;
} // namespace mpk

namespace stages {

struct BoxDecodeOptions {
  std::string decode_type = "";
  int original_width = 0;
  int original_height = 0;
  double detection_threshold = 0.0;
  double nms_iou_threshold = 0.0;
  int top_k = 0;
};

NeatTensor Preproc(const cv::Mat& input, const sima::mpk::ModelMPK& model);
NeatTensor MLA(const NeatTensor& input,
               const sima::mpk::ModelMPK& model);
BoxDecodeResult BoxDecode(const NeatTensor& input,
                          const sima::mpk::ModelMPK& model,
                          const BoxDecodeOptions& opt = {});

} // namespace stages
} // namespace sima
