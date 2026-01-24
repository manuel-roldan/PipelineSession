#include "pipeline/NeatTensorConversion.h"

namespace sima {

NeatConversionCost estimate_conversion_cost(NeatConversionKind kind,
                                            std::uint64_t bytes_copied) {
  NeatConversionCost cost;
  cost.bytes_copied = bytes_copied;
  switch (kind) {
    case NeatConversionKind::Reinterpret:
    case NeatConversionKind::View:
      cost.compute_class = 0;
      cost.bytes_copied = 0;
      break;
    case NeatConversionKind::Pack:
      cost.compute_class = 0;
      break;
    case NeatConversionKind::Convert:
      cost.compute_class = 1;
      break;
    case NeatConversionKind::Transfer:
      cost.compute_class = 2;
      break;
  }
  return cost;
}

bool conversion_allowed(NeatConversionPolicy policy, NeatConversionKind /*kind*/) {
  switch (policy) {
    case NeatConversionPolicy::Strict:
      return false;
    case NeatConversionPolicy::AllowWithTrace:
    case NeatConversionPolicy::AllowSilent:
      return true;
  }
  return false;
}

} // namespace sima
