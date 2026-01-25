// src/pipeline/internal/SampleUtil.cpp
#include "pipeline/internal/SampleUtil.h"

#include "InputStreamUtil.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/TensorUtil.h"

#include <gst/gst.h>

#include <cstdio>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>

namespace sima::pipeline_internal {
namespace {

constexpr const char* kSampleMetaName = "GstSimaSampleMeta";

const char* storage_kind_name(NeatStorageKind kind) {
  switch (kind) {
    case NeatStorageKind::CpuOwned: return "CpuOwned";
    case NeatStorageKind::CpuExternal: return "CpuExternal";
    case NeatStorageKind::GstSample: return "GstSample";
    case NeatStorageKind::DeviceHandle: return "DeviceHandle";
    case NeatStorageKind::Unknown:
    default: return "Unknown";
  }
}

const char* plane_role_name(NeatPlaneRole role) {
  switch (role) {
    case NeatPlaneRole::Y: return "Y";
    case NeatPlaneRole::U: return "U";
    case NeatPlaneRole::V: return "V";
    case NeatPlaneRole::UV: return "UV";
    case NeatPlaneRole::Unknown:
    default: return "Unknown";
  }
}

std::string join_dims(const std::vector<int64_t>& dims, char sep) {
  std::ostringstream ss;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i) ss << sep;
    ss << dims[i];
  }
  return ss.str();
}

bool sample_debug_enabled() {
  return env_bool("SIMA_SAMPLE_DEBUG", false);
}

bool sample_bytes_enabled() {
  return env_bool("SIMA_SAMPLE_BYTES", false);
}

void log_bundle_field(const RunOutput& field) {
  std::ostringstream ss;
  const std::string name = field.port_name.empty() ? "field" : field.port_name;
  ss << "[SAMPLE] field name=" << name;
  if (!field.caps_string.empty()) {
    ss << " caps=" << field.caps_string;
  }
  if (!field.neat.has_value()) {
    ss << " neat=<missing>";
    std::fprintf(stderr, "%s\n", ss.str().c_str());
    return;
  }
  const NeatTensor& t = *field.neat;
  ss << " " << t.debug_string();
  if (t.storage) {
    ss << " storage=" << storage_kind_name(t.storage->kind)
       << " size=" << t.storage->size_bytes;
  } else {
    ss << " storage=<none>";
  }
  for (size_t i = 0; i < t.planes.size(); ++i) {
    const NeatPlane& plane = t.planes[i];
    ss << " plane[" << i << "]="
       << plane_role_name(plane.role)
       << " shape=" << join_dims(plane.shape, 'x')
       << " strides=" << join_dims(plane.strides_bytes, ',')
       << " offset=" << plane.byte_offset;
  }
  std::fprintf(stderr, "%s\n", ss.str().c_str());
}

void log_bundle(const RunOutput& bundle) {
  std::ostringstream ss;
  ss << "[SAMPLE] bundle fields=" << bundle.fields.size()
     << " frame_id=" << bundle.frame_id;
  if (!bundle.stream_id.empty()) {
    ss << " stream_id=" << bundle.stream_id;
  }
  std::fprintf(stderr, "%s\n", ss.str().c_str());
  for (const auto& field : bundle.fields) {
    log_bundle_field(field);
  }
}

GstBuffer* buffer_from_neat_or_copy(const RunOutput& field, std::string* err) {
  if (!field.neat.has_value()) {
    if (err) *err = "Sample field missing tensor";
    return nullptr;
  }
  const NeatTensor& t = *field.neat;
  const char* name = field.port_name.empty() ? "field" : field.port_name.c_str();
  if (t.storage && t.storage->holder) {
    GstBuffer* buf = buffer_from_tensor_holder(t.storage->holder);
    if (buf) {
      if (sample_bytes_enabled()) {
        const size_t buf_bytes = static_cast<size_t>(gst_buffer_get_size(buf));
        std::fprintf(stderr,
                     "[SAMPLE] field name=%s source=holder bytes=%zu\n",
                     name,
                     buf_bytes);
      }
      return buf;
    }
  }

  const std::vector<uint8_t> payload = t.copy_payload_bytes();
  if (sample_bytes_enabled()) {
    std::fprintf(stderr,
                 "[SAMPLE] field name=%s source=copy bytes=%zu\n",
                 name,
                 payload.size());
  }
  GstBuffer* buf = gst_buffer_new_allocate(nullptr, payload.size(), nullptr);
  if (!buf) {
    if (err) *err = "Sample field buffer allocation failed";
    return nullptr;
  }
  if (!payload.empty()) {
    GstMapInfo map{};
    if (!gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
      gst_buffer_unref(buf);
      if (err) *err = "Sample field buffer map failed";
      return nullptr;
    }
    std::memcpy(map.data, payload.data(), payload.size());
    gst_buffer_unmap(buf, &map);
  }
  return buf;
}

bool add_field_to_list(GValue* list,
                       const RunOutput& field,
                       GstBuffer* buf,
                       const std::string& buffer_name) {
  if (!list || !buf) return false;
  const char* field_name = field.port_name.empty() ? "field" : field.port_name.c_str();
  const char* caps = field.caps_string.empty() ? nullptr : field.caps_string.c_str();

  GstStructure* entry = gst_structure_new(
      "simaai-sample-field",
      "name", G_TYPE_STRING, field_name,
      "buffer", GST_TYPE_BUFFER, buf,
      "buffer-name", G_TYPE_STRING, buffer_name.c_str(),
      nullptr);
  if (caps) {
    gst_structure_set(entry, "caps", G_TYPE_STRING, caps, nullptr);
  }

  GValue entry_val = G_VALUE_INIT;
  g_value_init(&entry_val, GST_TYPE_STRUCTURE);
  g_value_take_boxed(&entry_val, entry);

  gst_value_list_append_value(list, &entry_val);
  g_value_unset(&entry_val);
  return true;
}

} // namespace

std::shared_ptr<void> make_sample_holder_from_bundle(const RunOutput& bundle,
                                                     std::string* err) {
  if (bundle.kind != RunOutputKind::Bundle) {
    if (err) *err = "Sample bundle expected";
    return {};
  }
  if (bundle.fields.empty()) {
    if (err) *err = "Sample bundle has no fields";
    return {};
  }
  if (sample_debug_enabled()) {
    log_bundle(bundle);
  }

  GstBuffer* sample_buf = gst_buffer_new();
  if (!sample_buf) {
    if (err) *err = "Sample buffer allocation failed";
    return {};
  }

  GstCustomMeta* meta = gst_buffer_add_custom_meta(sample_buf, kSampleMetaName);
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s) {
    gst_buffer_unref(sample_buf);
    if (err) *err = "Sample meta attach failed";
    return {};
  }

  GValue list = G_VALUE_INIT;
  g_value_init(&list, GST_TYPE_LIST);

  for (const auto& field : bundle.fields) {
    std::string field_err;
    GstBuffer* buf = buffer_from_neat_or_copy(field, &field_err);
    if (!buf) {
      gst_buffer_unref(sample_buf);
      if (err) *err = field_err.empty() ? "Sample field buffer failed" : field_err;
      return {};
    }
    if (sample_bytes_enabled()) {
      const size_t buf_bytes = static_cast<size_t>(gst_buffer_get_size(buf));
      std::fprintf(stderr,
                   "[SAMPLE] bundle field=%s buffer-bytes=%zu\n",
                   field.port_name.empty() ? "field" : field.port_name.c_str(),
                   buf_bytes);
    }
    buf = gst_buffer_make_writable(buf);
    if (!buf) {
      gst_buffer_unref(sample_buf);
      if (err) *err = "Sample field buffer not writable";
      return {};
    }

    const std::string buffer_name =
        field.port_name.empty() ? std::string("field") : field.port_name;
    update_simaai_meta_fields(buf,
                              bundle.frame_id >= 0 ? std::optional<int64_t>(bundle.frame_id) : std::nullopt,
                              bundle.stream_id.empty() ? std::nullopt : std::optional<std::string>(bundle.stream_id),
                              buffer_name);

    if (!add_field_to_list(&list, field, buf, buffer_name)) {
      gst_buffer_unref(buf);
      gst_buffer_unref(sample_buf);
      if (err) *err = "Sample meta field insert failed";
      return {};
    }
    gst_buffer_unref(buf);
  }
  gst_structure_set_value(s, "fields", &list);
  g_value_unset(&list);

  GstSample* sample = gst_sample_new(sample_buf, nullptr, nullptr, nullptr);
  gst_buffer_unref(sample_buf);
  if (!sample) {
    if (err) *err = "Sample wrap failed";
    return {};
  }
  auto holder = std::shared_ptr<void>(
      gst_sample_ref(sample),
      [](void* p) { gst_sample_unref(static_cast<GstSample*>(p)); });
  gst_sample_unref(sample);
  return holder;
}

} // namespace sima::pipeline_internal
