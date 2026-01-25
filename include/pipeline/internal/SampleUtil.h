#pragma once

#include "sima/pipeline/PipelineOptions.h"

#include <memory>
#include <string>

namespace sima::pipeline_internal {

std::shared_ptr<void> make_sample_holder_from_bundle(const RunOutput& bundle,
                                                     std::string* err);

} // namespace sima::pipeline_internal
