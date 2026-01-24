// src/pipeline/TensorStream.cpp
#include "pipeline/TensorStream.h"

#include "pipeline/Errors.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/NeatTensorAdapters.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <utility>

namespace sima {
namespace {

using sima::pipeline_internal::diag_as_ctx;

} // namespace

TensorStream::TensorStream(GstElement* pipeline, GstElement* appsink)
    : pipeline_(pipeline), appsink_(appsink) {}

TensorStream::TensorStream(TensorStream&& other) noexcept {
  *this = std::move(other);
}

TensorStream& TensorStream::operator=(TensorStream&& other) noexcept {
  if (this == &other) return *this;

  close();

  pipeline_ = other.pipeline_;
  appsink_ = other.appsink_;
  debug_pipeline_ = std::move(other.debug_pipeline_);
  diag_ = std::move(other.diag_);
  guard_ = std::move(other.guard_);

  other.pipeline_ = nullptr;
  other.appsink_ = nullptr;

  return *this;
}

TensorStream::~TensorStream() {
  close();
}

void TensorStream::close() {
  auto diag = diag_as_ctx(diag_);

  if (pipeline_ && diag) {
    pipeline_internal::drain_bus(pipeline_, diag, "TensorStream::close(pre)");
  }

  if (appsink_) {
    pipeline_internal::stop_and_unref(appsink_);
    appsink_ = nullptr;
  }

  if (pipeline_) {
    pipeline_internal::stop_and_unref(pipeline_);
    pipeline_ = nullptr;
  }

  diag_.reset();
  debug_pipeline_.clear();
  guard_.reset();
}

std::optional<NeatTensor> TensorStream::next(int timeout_ms) {
  if (!appsink_) return std::nullopt;

  auto diag = diag_as_ctx(diag_);

  if (pipeline_ && diag) {
    pipeline_internal::drain_bus(pipeline_, diag, "TensorStream::next(pre)");
    pipeline_internal::throw_if_bus_error(pipeline_, diag, "TensorStream::next(pre)");
  }

  auto sample_opt =
      pipeline_internal::try_pull_sample_sliced(pipeline_, appsink_, timeout_ms, diag, "TensorStream::next");

  if (!sample_opt.has_value()) {
    if (pipeline_ && diag) {
      pipeline_internal::drain_bus(pipeline_, diag, "TensorStream::next(post-null)");
      pipeline_internal::throw_if_bus_error(pipeline_, diag, "TensorStream::next(post-null)");
    }
    return std::nullopt;
  }

  GstSample* sample = *sample_opt;
  NeatTensor out = from_gst_sample(sample);
  gst_sample_unref(sample);

  if (pipeline_ && diag) {
    pipeline_internal::drain_bus(pipeline_, diag, "TensorStream::next(post)");
    pipeline_internal::throw_if_bus_error(pipeline_, diag, "TensorStream::next(post)");
  }

  return out;
}

std::optional<NeatTensor> TensorStream::next_copy(int timeout_ms) {
  auto ref = next(timeout_ms);
  if (!ref.has_value()) return std::nullopt;
  return ref->clone();
}

PipelineReport TensorStream::report_snapshot(bool heavy) const {
  auto diag = diag_as_ctx(diag_);
  if (!diag) return {};
  (void)heavy;
  PipelineReport rep = diag->snapshot_basic();
  if (rep.pipeline_string.empty()) rep.pipeline_string = debug_pipeline_;
  return rep;
}

} // namespace sima
