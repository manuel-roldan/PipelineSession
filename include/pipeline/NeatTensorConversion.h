#pragma once

#include "pipeline/NeatTensorCore.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sima {

enum class NeatConversionKind {
  Reinterpret = 0,
  View,
  Pack,
  Convert,
  Transfer,
};

enum class NeatConversionPolicy {
  Strict = 0,
  AllowWithTrace,
  AllowSilent,
};

struct NeatConversionCost {
  std::uint64_t bytes_copied = 0;
  int compute_class = 0; // 0=low, 1=med, 2=high
};

struct NeatConversionTrace {
  std::string stage;
  NeatConversionKind kind = NeatConversionKind::Reinterpret;
  std::string src_desc;
  std::string dst_desc;
  std::uint64_t bytes_copied = 0;
  std::uint64_t elapsed_us = 0;
  NeatConversionPolicy policy = NeatConversionPolicy::Strict;
};

struct NeatConversionTraceCollector {
  std::vector<NeatConversionTrace> traces;

  void add(NeatConversionTrace trace) { traces.push_back(std::move(trace)); }
  void clear() { traces.clear(); }
};

NeatConversionCost estimate_conversion_cost(NeatConversionKind kind,
                                            std::uint64_t bytes_copied);
bool conversion_allowed(NeatConversionPolicy policy, NeatConversionKind kind);

} // namespace sima
