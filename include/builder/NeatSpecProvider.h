#pragma once

#include "pipeline/NeatTensorSpec.h"

namespace sima {

class NeatSpecProvider {
public:
  virtual ~NeatSpecProvider() = default;
  virtual NeatTensorConstraint expected_input_spec() const = 0;
  virtual NeatTensorConstraint output_spec(const NeatTensorConstraint& input) const = 0;
};

} // namespace sima
