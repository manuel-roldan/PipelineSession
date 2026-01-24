#pragma once

#include "pipeline/NeatTensorCore.h"

typedef struct _GstSample GstSample;
typedef struct _GstBuffer GstBuffer;

namespace sima::pipeline_internal {

// Get a ref-counted GstBuffer from a tensor holder (e.g. GstSample holder).
// The returned buffer must be unref'd if it is not pushed into appsrc.
GstBuffer* buffer_from_tensor_holder(const std::shared_ptr<void>& holder);

// Copy a specific GstBuffer memory index into an owned NeatTensor.
NeatTensor copy_neat_tensor_from_sample_memory(const NeatTensor& ref, int memory_index);
std::shared_ptr<void> holder_from_neat_tensor(const NeatTensor& tensor);

} // namespace sima::pipeline_internal
