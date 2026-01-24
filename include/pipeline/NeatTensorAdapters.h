#pragma once

#include "pipeline/NeatTensorCore.h"

#if defined(SIMA_WITH_OPENCV)
namespace cv {
class Mat;
} // namespace cv
#endif

typedef struct _GstSample GstSample;

namespace sima {

#if defined(SIMA_WITH_OPENCV)
NeatTensor from_cv_mat(const cv::Mat& mat, const std::optional<NeatImageSpec>& image = {});
#endif

NeatTensor from_gst_sample(GstSample* sample);

} // namespace sima
