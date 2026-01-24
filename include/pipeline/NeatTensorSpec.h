#pragma once

#include "pipeline/NeatTensorCore.h"

#include <optional>
#include <string>
#include <vector>

namespace sima {

struct NeatTensorConstraint {
  std::vector<TensorDType> dtypes;
  int rank = -1;
  std::vector<int64_t> shape; // use -1 for dynamic dims
  std::optional<NeatDevice> device;

  std::optional<NeatImageSpec::PixelFormat> image_format;
  bool allow_composite = true;

  bool matches(const NeatTensor& t) const {
    if (rank >= 0 && static_cast<int>(t.shape.size()) != rank) return false;
    if (!shape.empty() && shape.size() == t.shape.size()) {
      for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] >= 0 && t.shape[i] != shape[i]) return false;
      }
    }
    if (!dtypes.empty()) {
      bool ok = false;
      for (auto dt : dtypes) {
        if (dt == t.dtype) {
          ok = true;
          break;
        }
      }
      if (!ok) return false;
    }
    if (device.has_value()) {
      if (t.device.type != device->type || t.device.id != device->id) return false;
    }
    if (image_format.has_value()) {
      if (!t.semantic.image.has_value()) return false;
      if (t.semantic.image->format != *image_format) return false;
    }
    if (!allow_composite && t.is_composite()) return false;
    return true;
  }
};

} // namespace sima
