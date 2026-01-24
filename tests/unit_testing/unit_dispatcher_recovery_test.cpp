#include "pipeline/internal/DispatcherRecovery.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main() {
  try {
    const std::string err =
        "GST ERROR: Unable to connect to the server from dispatcher";
    sima::PipelineReport rep;
    if (sima::pipeline_internal::match_dispatcher_unavailable(err)) {
      rep.error_code = sima::pipeline_internal::kDispatcherUnavailableError;
    }

    bool called = false;
    auto cb = [&](const sima::PipelineReport&) { called = true; };
    if (sima::pipeline_internal::is_dispatcher_unavailable(rep)) {
      cb(rep);
    }

    if (!called) {
      throw std::runtime_error("dispatcher callback was not invoked");
    }

    std::cout << "[OK] unit_dispatcher_recovery_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
