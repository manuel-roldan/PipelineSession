#pragma once

#include "pipeline/NeatTensorCore.h"

#if defined(SIMA_WITH_OPENCV)
namespace sima {

NeatTensor from_cv_mat(const cv::Mat& mat,
                       NeatImageSpec::PixelFormat fmt = NeatImageSpec::PixelFormat::BGR,
                       bool read_only = true);

} // namespace sima
#endif
