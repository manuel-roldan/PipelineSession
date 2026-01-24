// src/pipeline/internal/DispatcherRecovery.cpp
#include "pipeline/internal/DispatcherRecovery.h"

#include "pipeline/internal/GstDiagnosticsUtil.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace sima::pipeline_internal {
namespace {

int run_cmd(const char* cmd) {
  const int rc = std::system(cmd);
  if (rc != 0) {
    std::fprintf(stderr, "[WARN] dispatcher_recovery: command failed rc=%d: %s\n", rc, cmd);
  }
  return rc;
}

void append_note(PipelineReport* report, const std::string& line) {
  if (!report) return;
  if (!report->repro_note.empty()) {
    report->repro_note += "\n";
  }
  report->repro_note += line;
}

} // namespace

bool match_dispatcher_unavailable(const std::string& message) {
  return message.find("Unable to connect to the server from dispatcher") != std::string::npos;
}

bool is_dispatcher_unavailable(const PipelineReport& report) {
  return report.error_code == kDispatcherUnavailableError;
}

bool attempt_dispatcher_recovery(PipelineReport* report, bool auto_recover) {
  append_note(report, "DispatcherUnavailable: auto recovery will run and you should retry.");
  if (!auto_recover) {
    append_note(report,
                "DispatcherUnavailable: auto recovery disabled; run the recovery commands and retry.");
    return false;
  }

  std::fprintf(stderr,
               "[WARN] dispatcher_recovery: attempting remoteproc reset + MLA init; "
               "retry the command after recovery.\n");

  bool ok = true;
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo stop > /sys/class/remoteproc/remoteproc0/state'") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo stop > /sys/class/remoteproc/remoteproc1/state'") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo start > /sys/class/remoteproc/remoteproc1/state'") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo start > /sys/class/remoteproc/remoteproc0/state'") == 0);

  ok &= (run_cmd(
    "printf '%s\n' edgeai | sudo -S -p '' sh -c 'for rp in /sys/class/remoteproc/remoteproc0 "
    "/sys/class/remoteproc/remoteproc1; do "
    "echo \"$rp: $(cat $rp/name) state=$(cat $rp/state)\"; done'") == 0);

  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' /usr/bin/init_mla_memory.sh") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' systemctl restart simaai-appcomplex.service") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' systemctl restart simaai-pipeline-manager.service") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' systemctl restart rctd.service") == 0);


  if (ok) {
    append_note(report,
                "DispatcherUnavailable: auto recovery completed; retry the command.");
  } else {
    append_note(report,
                "DispatcherUnavailable: auto recovery failed (sudo may require a password). "
                "Run the recovery commands manually and retry.");
  }
  return ok;
}

} // namespace sima::pipeline_internal
