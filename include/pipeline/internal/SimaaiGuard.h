#pragma once

#include <memory>
#include <string>

namespace sima::pipeline_internal {

std::string simaai_single_owner_error(const std::string& where);
std::string simaai_single_use_error(const std::string& where);
bool pipeline_uses_simaai(const std::string& pipeline);
bool pipeline_uses_mla(const std::string& pipeline);

void enforce_single_mla_pipeline(const std::string& where,
                                 const std::string& pipeline,
                                 const void* owner,
                                 const char* owner_desc);

std::shared_ptr<void> acquire_simaai_guard(const std::string& where,
                                           const std::string& pipeline,
                                           bool force,
                                           std::string* err_out);

// Test-only hook to reset the guard to a known state.
void reset_simaai_guard_for_test();

} // namespace sima::pipeline_internal
