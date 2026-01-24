#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "pipeline/TensorTypes.h"
#include "pipeline/NeatTensorCore.h"

namespace cv {
class Mat;
} // namespace cv

struct _GstBuffer;
struct _GstCaps;
struct _GstBufferPool;
using GstBuffer = struct _GstBuffer;
using GstCaps = struct _GstCaps;
using GstBufferPool = struct _GstBufferPool;

namespace sima {

struct InputAppSrcOptions;

struct InputCapsConfig {
  std::string media_type;
  std::string format;
  int width = -1;
  int height = -1;
  int depth = -1;
  size_t bytes = 0;
};

struct InputBufferPoolGuard {
  std::unique_ptr<GstBufferPool, void(*)(GstBufferPool*)> pool{
      nullptr, +[](GstBufferPool*) {}};
};

InputCapsConfig infer_input_caps(const InputAppSrcOptions& opt,
                                 const cv::Mat& input);
InputCapsConfig infer_input_caps(const InputAppSrcOptions& opt,
                                 const NeatTensor& input);
void validate_input_matches_caps(const InputCapsConfig& cfg,
                                 const InputAppSrcOptions& opt,
                                 const cv::Mat& input,
                                 const char* where);
void validate_input_matches_caps(const InputCapsConfig& cfg,
                                 const InputAppSrcOptions& opt,
                                 const NeatTensor& input,
                                 const char* where);
GstCaps* build_input_caps(const InputCapsConfig& cfg,
                          const InputAppSrcOptions& opt);
GstCaps* build_caps_with_override(const char* where,
                                  const std::string& media_type,
                                  const std::string& format,
                                  int width,
                                  int height,
                                  int depth,
                                  const std::string& caps_override);
GstBuffer* allocate_input_buffer(size_t bytes,
                                 const InputAppSrcOptions& opt,
                                 InputBufferPoolGuard& guard);
int64_t next_input_frame_id();
bool maybe_add_simaai_meta(GstBuffer* buffer,
                           int64_t frame_id,
                           const InputAppSrcOptions& opt);
void maybe_update_simaai_meta_name(GstBuffer* buffer, const std::string& name);
void dump_buffer_memories(GstBuffer* buffer, const char* label);
void dump_sima_meta(GstBuffer* buffer, const char* label);
GstBuffer* attach_simaai_meta_inplace(GstBuffer* buffer,
                                      const InputAppSrcOptions& opt,
                                      InputBufferPoolGuard& guard,
                                      const char* label);

} // namespace sima
