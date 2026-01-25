#pragma once

#include "sima/nodes/common/AppSink.h"
#include "sima/pipeline/PipelineReport.h"
#include "sima/pipeline/TapStream.h"
#include "sima/pipeline/TensorTypes.h"
#include "sima/pipeline/NeatTensor.h"
#include "sima/pipeline/NeatTensorCore.h"

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sima {

struct RtspServerOptions {
  std::string mount = "image";
  int port = 8554;
};

struct ValidateOptions {
  bool parse_launch = true;  // build gst pipeline and verify element naming contract
  bool enforce_names = true; // ensure no unnamed/foreign elements exist
};

struct RunDebugOptions {
  int timeout_ms = 10000;
};

struct RunInputOptions {
  bool copy_output = true;
  bool reuse_input_buffer = false;
  bool strict = true;
};

enum class PipelineOutputKind {
  Tensor,
};

enum class PipelineRunMode {
  Async,
  Sync,
};

struct PipelineSessionOptions {
  PipelineOutputKind output_kind = PipelineOutputKind::Tensor;
  int callback_timeout_ms = 1000;
  int throughput_depth = 8;
  // Insert queue2 between stages for async build(input) pipelines.
  bool enable_async_queue2 = true;
  bool auto_recover_dispatcher = true;
  std::function<void(const PipelineReport&)> on_dispatcher_error;
};

struct OutputTensorOptions {
  std::string format = "RGB";
  TensorLayout layout = TensorLayout::HWC;
  TensorDType dtype = TensorDType::UInt8;

  int target_width = -1;
  int target_height = -1;
  int target_fps = -1;

  bool use_videoconvert = true;
  bool use_videoscale = true;

  OutputAppSinkOptions sink;
};

struct RunDebugTap {
  std::string name;
  std::optional<TapPacket> packet;
  std::optional<NeatTensor> tensor;
  std::optional<NeatTensor> last_good_tensor;
  std::string error;
};

enum class RunOutputKind {
  Tensor,
  Bundle,
  Unknown,
};

enum class PullStatus {
  Ok,
  Timeout,
  Closed,
  Error,
};

struct PullError {
  std::string message;
  std::string code;
  std::optional<PipelineReport> report;
};

struct RunOutput {
  RunOutputKind kind = RunOutputKind::Unknown;
  bool owned = true;

  std::optional<NeatTensor> neat;
  std::vector<RunOutput> fields;

  std::string caps_string;
  std::string media_type;
  std::string payload_tag;
  std::string format; // Deprecated: use payload_tag.

  int64_t frame_id = -1;
  std::string stream_id;
  std::string port_name;
  int output_index = -1;
  int64_t input_seq = -1;
};

inline RunOutput make_tensor_output(const std::string& port_name, NeatTensor tensor) {
  RunOutput out;
  out.kind = RunOutputKind::Tensor;
  out.port_name = port_name;
  out.neat = std::move(tensor);
  return out;
}

inline RunOutput make_bundle_output(std::initializer_list<RunOutput> fields) {
  RunOutput out;
  out.kind = RunOutputKind::Bundle;
  out.fields = fields;
  return out;
}

using RunInputResult = RunOutput;

struct RunDebugResult {
  std::optional<NeatTensor> first_frame;
  std::vector<RunDebugTap> taps;
  std::unordered_map<std::string, NeatTensor> tensors;
  PipelineReport report;
};

} // namespace sima
