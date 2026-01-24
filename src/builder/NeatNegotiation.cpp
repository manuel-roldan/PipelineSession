#include "builder/NeatNegotiation.h"

#include "builder/Node.h"

namespace sima {
namespace {

bool dtype_intersects(const std::vector<TensorDType>& a,
                      const std::vector<TensorDType>& b) {
  if (a.empty() || b.empty()) return true;
  for (auto da : a) {
    for (auto db : b) {
      if (da == db) return true;
    }
  }
  return false;
}

bool shape_compatible(const std::vector<int64_t>& a,
                      const std::vector<int64_t>& b) {
  if (a.empty() || b.empty()) return true;
  if (a.size() != b.size()) return true;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] >= 0 && b[i] >= 0 && a[i] != b[i]) return false;
  }
  return true;
}

bool constraint_compatible(const NeatTensorConstraint& cur,
                           const NeatTensorConstraint& expected) {
  if (!dtype_intersects(cur.dtypes, expected.dtypes)) return false;
  if (expected.rank >= 0 && cur.rank >= 0 && expected.rank != cur.rank) return false;
  if (!shape_compatible(cur.shape, expected.shape)) return false;
  if (expected.device.has_value() && cur.device.has_value()) {
    if (expected.device->type != cur.device->type ||
        expected.device->id != cur.device->id) {
      return false;
    }
  }
  if (expected.image_format.has_value() && cur.image_format.has_value()) {
    if (expected.image_format != cur.image_format) return false;
  }
  return true;
}

} // namespace

NeatNegotiationResult validate_neat_pipeline(const NodeGroup& group,
                                             const NeatTensorConstraint& input,
                                             NeatConversionPolicy /*policy*/) {
  NeatNegotiationResult out;
  NeatTensorConstraint current = input;

  for (const auto& node : group.nodes()) {
    if (!node) continue;
    const auto* provider = dynamic_cast<const NeatSpecProvider*>(node.get());
    if (!provider) continue;

    const NeatTensorConstraint expected = provider->expected_input_spec();
    if (!constraint_compatible(current, expected)) {
      out.ok = false;
      out.error = "NeatNegotiation: constraint mismatch at node " + node->kind();
      return out;
    }
    current = provider->output_spec(current);
  }

  return out;
}

} // namespace sima
