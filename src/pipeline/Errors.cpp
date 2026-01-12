// src/pipeline/Errors.cpp
#include "sima/pipeline/Errors.h"

#include <utility>

namespace sima {

PipelineError::PipelineError(std::string msg, PipelineReport report)
  : std::runtime_error(std::move(msg)),
    report_(std::move(report)) {}

PipelineError::PipelineError(std::string msg)
  : std::runtime_error(std::move(msg)), report_{} {}

} // namespace sima
