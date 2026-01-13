// src/pipeline/internal/SimaaiGuard.cpp
#include "pipeline/internal/SimaaiGuard.h"

#include <atomic>
#include <cstdlib>
#include <fstream>

namespace sima::pipeline_internal {
namespace {

static std::atomic<bool> g_simaai_in_use{false};
static std::atomic<bool> g_simaai_used_once{false};

} // namespace

std::string simaai_single_owner_error(const std::string& where) {
  return where +
      ": SimaAI pipelines are single-owner per process; spawn a child process to run another pipeline.";
}

std::string simaai_single_use_error(const std::string& where) {
  return where +
      ": SimaAI pipelines are single-use per process; spawn a child process for subsequent runs.";
}

bool pipeline_uses_simaai(const std::string& pipeline) {
  return pipeline.find("simaai") != std::string::npos;
}

std::shared_ptr<void> acquire_simaai_guard(const std::string& where,
                                           const std::string& pipeline,
                                           bool force,
                                           std::string* err_out) {
  if (!force && !pipeline_uses_simaai(pipeline)) return {};
  bool expected = false;
  if (!g_simaai_in_use.compare_exchange_strong(expected, true)) {
    if (err_out) *err_out = simaai_single_owner_error(where);
    return {};
  }
  if (g_simaai_used_once.load()) {
    g_simaai_in_use.store(false);
    if (err_out) *err_out = simaai_single_use_error(where);
    return {};
  }
  g_simaai_used_once.store(true);
  const char* signal_path = std::getenv("SIMA_GUARD_TEST_SIGNAL_PATH");
  if (signal_path && *signal_path) {
    std::ofstream out(signal_path, std::ios::out | std::ios::trunc);
    if (out) {
      out << "1\n";
    }
  }
  return std::shared_ptr<void>(new int(0), [](void* p) {
    delete static_cast<int*>(p);
    g_simaai_in_use.store(false);
  });
}

void reset_simaai_guard_for_test() {
  g_simaai_in_use.store(false);
  g_simaai_used_once.store(false);
}

} // namespace sima::pipeline_internal
