#pragma once

#include "pipeline/NeatTensorSpec.h"

typedef struct _GstCaps GstCaps;

namespace sima::pipeline_internal {

NeatTensorConstraint neat_constraint_from_caps(GstCaps* caps);
std::string neat_constraint_debug_string(const NeatTensorConstraint& constraint);

} // namespace sima::pipeline_internal
