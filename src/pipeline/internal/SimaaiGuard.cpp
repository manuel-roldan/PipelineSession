// src/pipeline/internal/SimaaiGuard.cpp
#include "pipeline/internal/SimaaiGuard.h"

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>

namespace sima::pipeline_internal {
namespace {

static std::atomic<bool> g_simaai_in_use{false};
static std::atomic<bool> g_simaai_used_once{false};
static std::atomic<const void*> g_mla_owner{nullptr};
static std::mutex g_mla_mu;
static std::string g_mla_owner_desc;

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

bool pipeline_uses_mla(const std::string& pipeline) {
  return pipeline.find("simaaiprocessmla") != std::string::npos;
}

void enforce_single_mla_pipeline(const std::string& where,
                                 const std::string& pipeline,
                                 const void* owner,
                                 const char* owner_desc) {
  if (!pipeline_uses_mla(pipeline)) return;
  const void* expected = nullptr;
  if (g_mla_owner.compare_exchange_strong(expected, owner)) {
    std::lock_guard<std::mutex> lock(g_mla_mu);
    g_mla_owner_desc = owner_desc ? owner_desc : where;
    return;
  }
  if (g_mla_owner.load() == owner) return;

  std::string first_owner;
  {
    std::lock_guard<std::mutex> lock(g_mla_mu);
    first_owner = g_mla_owner_desc;
  }
  if (first_owner.empty()) first_owner = "another pipeline";
  throw std::runtime_error(
      where + ": MLA pipeline already owned by " + first_owner +
      "; only one pipeline per process may use MLA.");
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
  g_mla_owner.store(nullptr);
  std::lock_guard<std::mutex> lock(g_mla_mu);
  g_mla_owner_desc.clear();
}

} // namespace sima::pipeline_internal
