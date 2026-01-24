#pragma once

#include "sima/pipeline/PipelineReport.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <glib.h>

namespace sima::pipeline_internal {

struct BoundaryFlowCounters {
  std::string boundary_name;
  int after_node_index = -1;
  int before_node_index = -1;

  std::atomic<uint64_t> in_buffers{0}, out_buffers{0};
  std::atomic<int64_t> last_in_pts_ns{-1}, last_out_pts_ns{-1};
  std::atomic<int64_t> last_in_wall_us{0}, last_out_wall_us{0};

  BoundaryFlowStats snapshot() const {
    BoundaryFlowStats s;
    s.boundary_name = boundary_name;
    s.after_node_index = after_node_index;
    s.before_node_index = before_node_index;

    s.in_buffers = in_buffers.load(std::memory_order_relaxed);
    s.out_buffers = out_buffers.load(std::memory_order_relaxed);
    s.last_in_pts_ns = last_in_pts_ns.load(std::memory_order_relaxed);
    s.last_out_pts_ns = last_out_pts_ns.load(std::memory_order_relaxed);
    s.last_in_wall_us = last_in_wall_us.load(std::memory_order_relaxed);
    s.last_out_wall_us = last_out_wall_us.load(std::memory_order_relaxed);
    return s;
  }
};

struct StageTimingStats {
  std::string stage_name;
  uint64_t samples = 0;
  uint64_t total_us = 0;
  uint64_t max_us = 0;
};

struct StageTimingCounters {
  std::string stage_name;
  std::atomic<uint64_t> samples{0};
  std::atomic<uint64_t> total_us{0};
  std::atomic<uint64_t> max_us{0};

  StageTimingStats snapshot() const {
    StageTimingStats s;
    s.stage_name = stage_name;
    s.samples = samples.load(std::memory_order_relaxed);
    s.total_us = total_us.load(std::memory_order_relaxed);
    s.max_us = max_us.load(std::memory_order_relaxed);
    return s;
  }
};

struct NextCpuDecision {
  int node_index = -1;
  std::string node_kind;
  std::string node_label;
  std::string next_cpu;
  bool applied = false;
};

struct DiagCtx {
  std::string pipeline_string;
  std::vector<NodeReport> node_reports;

  bool queue2_enabled = false;
  int queue2_depth = 0;
  std::vector<NextCpuDecision> next_cpu_decisions;

  mutable std::mutex bus_mu;
  std::vector<BusMessage> bus;

  std::vector<std::unique_ptr<BoundaryFlowCounters>> boundaries;
  std::vector<std::unique_ptr<StageTimingCounters>> stage_timings;

  static int64_t now_us() { return (int64_t)g_get_monotonic_time(); }

  void push_bus(const std::string& type,
                const std::string& src,
                const std::string& detail) {
    std::lock_guard<std::mutex> lk(bus_mu);
    bus.push_back(BusMessage{type, src, detail, now_us()});
  }

  PipelineReport snapshot_basic() const {
    PipelineReport rep;
    rep.pipeline_string = pipeline_string;
    rep.nodes = node_reports;

    {
      std::lock_guard<std::mutex> lk(bus_mu);
      rep.bus = bus;
    }

    rep.boundaries.reserve(boundaries.size());
    for (const auto& b : boundaries) {
      if (!b) continue;
      rep.boundaries.push_back(b->snapshot());
    }

    rep.repro_gst_launch = "gst-launch-1.0 -v '" + pipeline_string + "'";
    // rep.repro_env filled by caller if desired
    return rep;
  }
};

inline std::shared_ptr<DiagCtx> diag_as_ctx(const std::shared_ptr<void>& p) {
  if (!p) return {};
  // aliasing to the *same* DiagCtx type everywhere
  return std::shared_ptr<DiagCtx>(p, reinterpret_cast<DiagCtx*>(p.get()));
}

} // namespace sima::pipeline_internal
