#pragma once

#include "sima/contracts/ContractTypes.h"

#include <string>
#include <vector>

namespace sima {

enum class InputRole {
  None,
  Push,
  Source,
};

// =============================
// Builder Node API
// =============================
class Node {
public:
  virtual ~Node() = default;

  // Deterministic type label (used in reports).
  virtual std::string kind() const = 0;

  // Optional human label (e.g., DebugPoint name).
  virtual std::string user_label() const { return ""; }

  // Node fragment with deterministic element names (namespace = n<idx>_...).
  virtual std::string gst_fragment(int node_index) const = 0;

  // Deterministic list of element names this node creates.
  virtual std::vector<std::string> element_names(int node_index) const = 0;

  // Optional memory contract for this node (runner may still override).
  virtual MemoryContract memory_contract() const { return MemoryContract::AllowEitherButReport; }

  // Input role metadata used to validate run() vs run(input) usage.
  virtual InputRole input_role() const { return InputRole::None; }
};

} // namespace sima
