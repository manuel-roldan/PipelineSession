#pragma once

#include "sima/pipeline/PipelineReport.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace sima {

// Exception that carries a structured report.
class PipelineError : public std::runtime_error {
public:
  PipelineError(std::string msg, PipelineReport report);
  const PipelineReport& report() const { return report_; }

private:
  PipelineReport report_;
};

} // namespace sima
