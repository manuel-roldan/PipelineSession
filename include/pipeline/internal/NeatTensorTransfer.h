#pragma once

#include "pipeline/NeatTensorCore.h"

#include <cstddef>
#include <string>
#include <vector>

namespace sima::pipeline_internal {

struct NeatTransferPoolStats {
  std::size_t hits = 0;
  std::size_t misses = 0;
  std::size_t entries = 0;
};

NeatTransferPoolStats neat_transfer_pool_stats();

NeatTensor transfer_to_device(const NeatTensor& src,
                              const NeatDevice& target,
                              const std::vector<NeatSegment>* required_segments,
                              const std::vector<std::string>* required_segment_names);

NeatTensor transfer_to_cpu(const NeatTensor& src);

} // namespace sima::pipeline_internal
