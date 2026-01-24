#pragma once

#include "builder/NeatSpecProvider.h"
#include "builder/NodeGroup.h"
#include "pipeline/NeatTensorConversion.h"

#include <string>
#include <vector>

namespace sima {

struct NeatNegotiationResult {
  bool ok = true;
  std::string error;
  std::vector<NeatConversionTrace> trace;
};

NeatNegotiationResult validate_neat_pipeline(const NodeGroup& group,
                                             const NeatTensorConstraint& input,
                                             NeatConversionPolicy policy);

} // namespace sima
