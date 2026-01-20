#include "pipeline/AsyncStream.h"

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace sima {

struct AsyncStream::State {
  InputStream stream;
  AsyncOptions opt;
  InputStreamOptions stream_opt;

  std::mutex in_mu;
  std::condition_variable in_cv;
  std::deque<cv::Mat> in_queue;
  bool input_closed = false;

  std::mutex out_mu;
  std::condition_variable out_cv;
  std::deque<RunInputResult> out_queue;

  std::mutex latency_mu;
  std::deque<std::chrono::steady_clock::time_point> pending_times;
  std::uint64_t latency_count = 0;
  double latency_mean_ms = 0.0;
  double latency_min_ms = 0.0;
  double latency_max_ms = 0.0;
  bool latency_init = false;

  std::thread input_thread;
  std::atomic<bool> stop_requested{false};
  std::atomic<bool> input_thread_done{false};

  std::atomic<std::uint64_t> inputs_enqueued{0};
  std::atomic<std::uint64_t> inputs_dropped{0};
  std::atomic<std::uint64_t> inputs_pushed{0};
  std::atomic<std::uint64_t> outputs_ready{0};
  std::atomic<std::uint64_t> outputs_pulled{0};

  mutable std::mutex error_mu;
  std::string error;
};

namespace {

bool queue_full(const std::deque<cv::Mat>& q, int max) {
  return max > 0 && static_cast<int>(q.size()) >= max;
}

} // namespace

AsyncStream::AsyncStream(std::shared_ptr<State> state) : state_(std::move(state)) {}

AsyncStream::AsyncStream(AsyncStream&& other) noexcept : state_(std::move(other.state_)) {}

AsyncStream& AsyncStream::operator=(AsyncStream&& other) noexcept {
  if (this != &other) {
    close();
    state_ = std::move(other.state_);
  }
  return *this;
}

AsyncStream::~AsyncStream() {
  close();
}

AsyncStream AsyncStream::create(InputStream stream,
                                const AsyncOptions& opt,
                                const InputStreamOptions& stream_opt) {
  auto st = std::make_shared<State>();
  st->stream = std::move(stream);
  st->opt = opt;
  st->stream_opt = stream_opt;

  auto on_output = [st](RunInputResult out) {
    if (st->stop_requested.load()) return;

    std::chrono::steady_clock::time_point t0;
    bool have_ts = false;
    {
      std::lock_guard<std::mutex> lock(st->latency_mu);
      if (!st->pending_times.empty()) {
        t0 = st->pending_times.front();
        st->pending_times.pop_front();
        have_ts = true;
      }
    }
    if (have_ts) {
      const auto t1 = std::chrono::steady_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      std::lock_guard<std::mutex> lock(st->latency_mu);
      st->latency_count += 1;
      if (!st->latency_init) {
        st->latency_mean_ms = ms;
        st->latency_min_ms = ms;
        st->latency_max_ms = ms;
        st->latency_init = true;
      } else {
        const double n = static_cast<double>(st->latency_count);
        st->latency_mean_ms += (ms - st->latency_mean_ms) / n;
        st->latency_min_ms = std::min(st->latency_min_ms, ms);
        st->latency_max_ms = std::max(st->latency_max_ms, ms);
      }
    }

    {
      std::unique_lock<std::mutex> lock(st->out_mu);
      const int max = st->opt.output_queue;
      if (max > 0) {
        st->out_cv.wait(lock, [&]() {
          return st->stop_requested.load() ||
                 static_cast<int>(st->out_queue.size()) < max;
        });
      }
      if (st->stop_requested.load()) return;
      st->out_queue.push_back(std::move(out));
      st->outputs_ready.fetch_add(1, std::memory_order_relaxed);
    }
    st->out_cv.notify_one();
  };

  st->stream.start(on_output);

  st->input_thread = std::thread([st]() {
    while (true) {
      cv::Mat item;
      {
        std::unique_lock<std::mutex> lock(st->in_mu);
        st->in_cv.wait(lock, [&]() {
          return st->stop_requested.load() ||
                 st->input_closed ||
                 !st->in_queue.empty();
        });
        if (st->stop_requested.load()) break;
        if (st->in_queue.empty()) {
          if (st->input_closed) break;
          continue;
        }
        item = std::move(st->in_queue.front());
        st->in_queue.pop_front();
      }
      st->in_cv.notify_one();

      const auto t0 = std::chrono::steady_clock::now();
      {
        std::lock_guard<std::mutex> lock(st->latency_mu);
        st->pending_times.push_back(t0);
      }
      try {
        st->stream.push(item);
        st->inputs_pushed.fetch_add(1, std::memory_order_relaxed);
      } catch (const std::exception& e) {
        {
          std::lock_guard<std::mutex> lock(st->latency_mu);
          if (!st->pending_times.empty()) {
            st->pending_times.pop_back();
          }
        }
        std::lock_guard<std::mutex> lock(st->error_mu);
        st->error = e.what();
        st->stop_requested.store(true);
        break;
      }
    }

    if (!st->stop_requested.load() && st->input_closed) {
      try {
        st->stream.signal_eos();
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(st->error_mu);
        st->error = e.what();
        st->stop_requested.store(true);
      }
    }
    st->input_thread_done.store(true);
  });

  return AsyncStream(st);
}

AsyncStream::operator bool() const noexcept {
  return state_ && static_cast<bool>(state_->stream);
}

bool AsyncStream::running() const {
  return state_ && state_->stream.running();
}

std::string AsyncStream::last_error() const {
  if (!state_) return {};
  std::lock_guard<std::mutex> lock(state_->error_mu);
  return state_->error;
}

bool AsyncStream::push_impl(const cv::Mat& input, bool block) {
  if (!state_) {
    throw std::runtime_error("AsyncStream::push: stream is closed");
  }
  auto st = state_;
  {
    std::unique_lock<std::mutex> lock(st->in_mu);
    if (st->input_closed) return false;
    const int max = st->opt.input_queue;
    if (st->opt.drop == DropPolicy::Block) {
      if (!block) {
        if (queue_full(st->in_queue, max)) return false;
      } else {
        st->in_cv.wait(lock, [&]() {
          return st->stop_requested.load() || st->input_closed ||
                 !queue_full(st->in_queue, max);
        });
      }
    } else if (st->opt.drop == DropPolicy::DropNewest) {
      if (queue_full(st->in_queue, max)) {
        st->inputs_dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
    } else if (st->opt.drop == DropPolicy::DropOldest) {
      if (queue_full(st->in_queue, max)) {
        st->in_queue.pop_front();
        st->inputs_dropped.fetch_add(1, std::memory_order_relaxed);
      }
    }

    if (st->stop_requested.load() || st->input_closed) return false;

    if (st->opt.copy_input) {
      st->in_queue.push_back(input.clone());
    } else {
      st->in_queue.push_back(input);
    }
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->in_cv.notify_one();
  return true;
}

bool AsyncStream::push(const cv::Mat& input) {
  return push_impl(input, true);
}

bool AsyncStream::try_push(const cv::Mat& input) {
  return push_impl(input, false);
}

void AsyncStream::close_input() {
  if (!state_) return;
  auto st = state_;
  {
    std::lock_guard<std::mutex> lock(st->in_mu);
    st->input_closed = true;
  }
  st->in_cv.notify_all();
}

std::optional<RunInputResult> AsyncStream::pull(int timeout_ms) {
  if (!state_) return std::nullopt;
  auto st = state_;

  auto done = [&]() {
    return st->input_closed &&
           st->input_thread_done.load() &&
           st->outputs_pulled.load() >= st->inputs_pushed.load() &&
           st->out_queue.empty();
  };

  std::unique_lock<std::mutex> lock(st->out_mu);
  if (timeout_ms < 0) {
    st->out_cv.wait(lock, [&]() {
      return st->stop_requested.load() || !st->out_queue.empty() || done();
    });
  } else {
    const auto deadline = std::chrono::milliseconds(timeout_ms);
    if (!st->out_cv.wait_for(lock, deadline, [&]() {
          return st->stop_requested.load() || !st->out_queue.empty() || done();
        })) {
      throw std::runtime_error("AsyncStream::pull: timeout waiting for output");
    }
  }

  if (!st->out_queue.empty()) {
    RunInputResult out = std::move(st->out_queue.front());
    st->out_queue.pop_front();
    st->outputs_pulled.fetch_add(1, std::memory_order_relaxed);
    lock.unlock();
    st->out_cv.notify_one();
    return out;
  }

  if (done()) {
    lock.unlock();
    stop();
    return std::nullopt;
  }

  return std::nullopt;
}

AsyncStats AsyncStream::stats() const {
  AsyncStats out;
  if (!state_) return out;
  auto st = state_;
  out.inputs_enqueued = st->inputs_enqueued.load();
  out.inputs_dropped = st->inputs_dropped.load();
  out.inputs_pushed = st->inputs_pushed.load();
  out.outputs_ready = st->outputs_ready.load();
  out.outputs_pulled = st->outputs_pulled.load();
  {
    std::lock_guard<std::mutex> lock(st->latency_mu);
    if (st->latency_count > 0) {
      out.avg_latency_ms = st->latency_mean_ms;
      out.min_latency_ms = st->latency_min_ms;
      out.max_latency_ms = st->latency_max_ms;
    }
  }
  return out;
}

void AsyncStream::stop() {
  if (!state_) return;
  auto st = state_;
  st->stop_requested.store(true);
  st->in_cv.notify_all();
  st->out_cv.notify_all();
  if (st->input_thread.joinable()) {
    st->input_thread.join();
  }
  st->stream.stop();
}

void AsyncStream::close() {
  if (!state_) return;
  stop();
  state_->stream.close();
  state_.reset();
}

} // namespace sima
