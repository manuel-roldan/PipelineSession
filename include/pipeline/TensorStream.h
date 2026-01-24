// include/pipeline/TensorStream.h
#pragma once

#include "sima/pipeline/PipelineReport.h"
#include "sima/pipeline/NeatTensorCore.h"

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

  std::optional<NeatTensor> next(int timeout_ms);
  std::optional<NeatTensor> next_copy(int timeout_ms);

  PipelineReport report_snapshot(bool heavy) const;
  void close();
  void kill() { close(); }

  void set_debug_pipeline(std::string name) { debug_pipeline_ = std::move(name); }
  void set_diag(std::shared_ptr<void> diag) { diag_ = std::move(diag); }
  void set_guard(std::shared_ptr<void> guard) { guard_ = std::move(guard); }

private:
  GstElement* pipeline_ = nullptr;
  GstElement* appsink_ = nullptr;

  std::string debug_pipeline_;
  std::shared_ptr<void> diag_;
  std::shared_ptr<void> guard_;
};

} // namespace sima
