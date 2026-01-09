// include/contracts/Validators.h
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "builder/Node.h"
#include "builder/NodeGroup.h"
#include "contracts/Contract.h"
#include "contracts/ContractRegistry.h"
#include "contracts/ValidationReport.h"

namespace sima {
namespace validators {

// -----------------------------
// Built-in Contract factories
// -----------------------------

/**
 * @brief Ensures NodeGroup is not empty.
 */
inline std::shared_ptr<Contract> NonEmptyPipeline() {
  class C final : public Contract {
  public:
    std::string id() const override { return "NonEmptyPipeline"; }
    std::string description() const override { return "Pipeline must contain at least one node."; }

    void validate(const NodeGroup& nodes, const ValidationContext& ctx, ValidationReport& r) const override {
      (void)ctx;
      if (nodes.empty()) {
        r.add_error(id(), "EMPTY_PIPELINE", "No nodes were added to the pipeline.");
      }
    }
  };
  return std::make_shared<C>();
}

/**
 * @brief Ensures there are no null node pointers.
 */
inline std::shared_ptr<Contract> NoNullNodes() {
  class C final : public Contract {
  public:
    std::string id() const override { return "NoNullNodes"; }
    std::string description() const override { return "All nodes must be non-null shared_ptr."; }

    void validate(const NodeGroup& nodes, const ValidationContext& ctx, ValidationReport& r) const override {
      (void)ctx;
      const auto& v = nodes.nodes();
      for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        if (!v[static_cast<std::size_t>(i)]) {
          r.add_error(id(), "NULL_NODE", "Null node pointer in NodeGroup.", i);
        }
      }
    }
  };
  return std::make_shared<C>();
}

/**
 * @brief Ensures OutputAppSink exists and is last when ctx.mode == Run.
 *
 * This is the builder-level version of the "sink last" contract described in the architecture.
 */
inline std::shared_ptr<Contract> SinkLastForRun(std::string sink_kind = "OutputAppSink") {
  class C final : public Contract {
  public:
    explicit C(std::string kind) : sink_kind_(std::move(kind)) {}
    std::string id() const override { return "SinkLastForRun"; }
    std::string description() const override {
      return "When running, the pipeline must end with the terminal appsink node.";
    }

    void validate(const NodeGroup& nodes, const ValidationContext& ctx, ValidationReport& r) const override {
      if (ctx.mode != ValidationContext::Mode::Run) return;
      const auto& v = nodes.nodes();
      if (v.empty()) return;

      int last_idx = static_cast<int>(v.size()) - 1;
      const auto& last = v.back();
      const std::string last_kind = last ? last->kind() : "";

      // Require sink kind last.
      if (!last || last_kind != sink_kind_) {
        r.add_error(id(), "SINK_NOT_LAST",
                    "Last node must be " + sink_kind_ + " for run().",
                    last_idx, last_kind, last ? last->user_label() : "");
      }

      // Disallow additional sinks earlier (best-effort sanity).
      for (int i = 0; i < last_idx; ++i) {
        const auto& n = v[static_cast<std::size_t>(i)];
        if (n && n->kind() == sink_kind_) {
          r.add_error(id(), "MULTIPLE_SINKS",
                      "Found " + sink_kind_ + " before the end of the pipeline.",
                      i, n->kind(), n->user_label());
        }
      }
    }

  private:
    std::string sink_kind_;
  };
  return std::make_shared<C>(std::move(sink_kind));
}

/**
 * @brief Ensures ctx.tap_name exists as a DebugPoint label when ctx.mode == Tap.
 *
 * This mirrors the architecture: run_tap(name) splits at DebugPoint(name).
 */
inline std::shared_ptr<Contract> TapPointExists(std::string debug_kind = "DebugPoint") {
  class C final : public Contract {
  public:
    explicit C(std::string k) : dbg_kind_(std::move(k)) {}
    std::string id() const override { return "TapPointExists"; }
    std::string description() const override { return "Tap mode requires a matching DebugPoint(name)."; }

    void validate(const NodeGroup& nodes, const ValidationContext& ctx, ValidationReport& r) const override {
      if (ctx.mode != ValidationContext::Mode::Tap) return;

      if (ctx.tap_name.empty()) {
        r.add_error(id(), "EMPTY_TAP_NAME", "Tap requested with empty point name.");
        return;
      }

      const auto& v = nodes.nodes();
      bool found = false;
      for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        const auto& n = v[static_cast<std::size_t>(i)];
        if (!n) continue;
        if (n->kind() == dbg_kind_ && n->user_label() == ctx.tap_name) {
          found = true;
          break;
        }
      }

      if (!found) {
        r.add_error(id(), "TAP_POINT_NOT_FOUND",
                    "No " + dbg_kind_ + "(\"" + ctx.tap_name + "\") found in pipeline.",
                    -1, dbg_kind_, ctx.tap_name);
      }
    }

  private:
    std::string dbg_kind_;
  };
  return std::make_shared<C>(std::move(debug_kind));
}

/**
 * @brief Warns if multiple DebugPoints share the same label (ambiguous taps).
 */
inline std::shared_ptr<Contract> UniqueDebugPointLabels(std::string debug_kind = "DebugPoint") {
  class C final : public Contract {
  public:
    explicit C(std::string k) : dbg_kind_(std::move(k)) {}
    std::string id() const override { return "UniqueDebugPointLabels"; }
    std::string description() const override { return "DebugPoint labels should be unique to avoid ambiguity."; }

    void validate(const NodeGroup& nodes, const ValidationContext& ctx, ValidationReport& r) const override {
      (void)ctx;
      const auto& v = nodes.nodes();
      std::unordered_map<std::string, int> first_idx;

      for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        const auto& n = v[static_cast<std::size_t>(i)];
        if (!n) continue;
        if (n->kind() != dbg_kind_) continue;

        const std::string label = n->user_label();
        if (label.empty()) continue;

        auto it = first_idx.find(label);
        if (it == first_idx.end()) {
          first_idx.emplace(label, i);
        } else {
          // Prefer warning unless strict=true and tap mode could be impacted.
          const auto sev = (ctx.strict && ctx.mode == ValidationContext::Mode::Tap)
                               ? ValidationSeverity::Error
                               : ValidationSeverity::Warning;

          ValidationIssue issue;
          issue.severity = sev;
          issue.contract_id = id();
          issue.code = "DUPLICATE_DEBUG_LABEL";
          issue.message = "Multiple DebugPoint nodes share label \"" + label +
                          "\" (first at node[" + std::to_string(it->second) + "]).";
          issue.node_index = i;
          issue.node_kind = n->kind();
          issue.node_label = label;
          r.add_issue(std::move(issue));
        }
      }
    }

  private:
    std::string dbg_kind_;
  };
  return std::make_shared<C>(std::move(debug_kind));
}

/**
 * @brief Ensures an RTSP source node exists when ctx.mode == Rtsp.
 *
 * Builder-level: we only check presence of AppSrcImage (or another configured kind).
 */
inline std::shared_ptr<Contract> RtspRequiresSource(std::string source_kind = "AppSrcImage") {
  class C final : public Contract {
  public:
    explicit C(std::string k) : src_kind_(std::move(k)) {}
    std::string id() const override { return "RtspRequiresSource"; }
    std::string description() const override { return "RTSP mode requires a server-side source node (e.g., AppSrcImage)."; }

    void validate(const NodeGroup& nodes, const ValidationContext& ctx, ValidationReport& r) const override {
      if (ctx.mode != ValidationContext::Mode::Rtsp) return;

      const auto& v = nodes.nodes();
      bool found = false;
      for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        const auto& n = v[static_cast<std::size_t>(i)];
        if (n && n->kind() == src_kind_) {
          found = true;
          break;
        }
      }

      if (!found) {
        r.add_error(id(), "RTSP_SOURCE_MISSING",
                    "RTSP mode requires a node of kind \"" + src_kind_ + "\".",
                    -1, src_kind_, "");
      }
    }

  private:
    std::string src_kind_;
  };
  return std::make_shared<C>(std::move(source_kind));
}

// -----------------------------
// Default registry
// -----------------------------

/**
 * @brief Reasonable default set of builder-level contracts.
 *
 * Keep this purely structural (no GStreamer).
 */
inline ContractRegistry DefaultRegistry() {
  ContractRegistry reg;
  reg.add(NonEmptyPipeline());
  reg.add(NoNullNodes());
  reg.add(UniqueDebugPointLabels());
  reg.add(SinkLastForRun());
  reg.add(TapPointExists());
  reg.add(RtspRequiresSource());
  return reg;
}

} // namespace validators
} // namespace sima
