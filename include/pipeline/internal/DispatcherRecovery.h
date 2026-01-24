#pragma once

#include "pipeline/PipelineReport.h"

#include <string>

namespace sima::pipeline_internal {

constexpr const char* kDispatcherUnavailableError = "DispatcherUnavailable";

bool match_dispatcher_unavailable(const std::string& message);
bool is_dispatcher_unavailable(const PipelineReport& report);
bool attempt_dispatcher_recovery(PipelineReport* report, bool auto_recover);

} // namespace sima::pipeline_internal
