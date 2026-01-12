#pragma once

#include "sima/pipeline/TensorTypes.h"

typedef struct _GstSample GstSample;

namespace sima::pipeline_internal {

// Convert a GstSample into a zero-copy tensor view. This is the shared bridge
// used by TensorStream and run_debug; it keeps the sample alive via holder.
FrameTensorRef sample_to_tensor_ref(GstSample* sample);

} // namespace sima::pipeline_internal
