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
  std::vector<NeatDevice> allowed_devices;
  std::optional<NeatDevice> preferred_device;

  std::optional<NeatImageSpec::PixelFormat> image_format;
  std::vector<NeatSegment> required_segments;
  std::vector<std::string> required_segment_names;
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
    if (!allowed_devices.empty()) {
      bool ok = false;
      for (const auto& allowed : allowed_devices) {
        if (t.device.type == allowed.type && t.device.id == allowed.id) {
          ok = true;
          break;
        }
      }
      if (!ok) return false;
    }
    if (image_format.has_value()) {
      if (!t.semantic.image.has_value()) return false;
      if (t.semantic.image->format != *image_format) return false;
    }
    if (!required_segments.empty()) {
      if (!t.storage || t.storage->sima_segments.empty()) return false;
      if (t.storage->sima_segments.size() != required_segments.size()) return false;
      for (size_t i = 0; i < required_segments.size(); ++i) {
        if (t.storage->sima_segments[i].name != required_segments[i].name) return false;
        if (t.storage->sima_segments[i].size_bytes != required_segments[i].size_bytes) {
          return false;
        }
      }
    }
    if (!required_segment_names.empty()) {
      if (!t.storage || t.storage->sima_segments.empty()) return false;
      for (const auto& name : required_segment_names) {
        bool found = false;
        for (const auto& seg : t.storage->sima_segments) {
          if (seg.name == name) {
            found = true;
            break;
          }
        }
        if (!found) return false;
      }
    }
    if (!allow_composite && t.is_composite()) return false;
    return true;
  }
};

} // namespace sima
