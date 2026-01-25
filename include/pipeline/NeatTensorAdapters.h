#pragma once

#include "pipeline/NeatTensorCore.h"

#if defined(SIMA_WITH_OPENCV)
#include "pipeline/NeatTensorOpenCV.h"
#endif

typedef struct _GstSample GstSample;

namespace sima {

NeatTensor from_gst_sample(GstSample* sample);

} // namespace sima
