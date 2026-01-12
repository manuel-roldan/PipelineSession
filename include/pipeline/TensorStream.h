// include/pipeline/TensorStream.h
#pragma once

#include "sima/pipeline/PipelineReport.h"
#include "sima/pipeline/TensorTypes.h"

#include <memory>
#include <optional>
#include <string>

typedef struct _GstElement GstElement;

namespace sima {

class TensorStream {
public:
  TensorStream() = default;
  TensorStream(GstElement* pipeline, GstElement* appsink);

  TensorStream(const TensorStream&) = delete;
  TensorStream& operator=(const TensorStream&) = delete;

  TensorStream(TensorStream&& other) noexcept;
  TensorStream& operator=(TensorStream&& other) noexcept;

  ~TensorStream();

  explicit operator bool() const noexcept { return pipeline_ != nullptr && appsink_ != nullptr; }

  std::optional<FrameTensorRef> next(int timeout_ms);
  std::optional<FrameTensor> next_copy(int timeout_ms);

  PipelineReport report_snapshot(bool heavy) const;
  void close();

  void set_debug_pipeline(std::string name) { debug_pipeline_ = std::move(name); }
  void set_diag(std::shared_ptr<void> diag) { diag_ = std::move(diag); }

private:
  GstElement* pipeline_ = nullptr;
  GstElement* appsink_ = nullptr;

  std::string debug_pipeline_;
  std::shared_ptr<void> diag_;
};

} // namespace sima
