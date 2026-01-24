#include "pipeline/PipelineRun.h"

#include "internal/InputStream.h"
#include "pipeline/Errors.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/DispatcherRecovery.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sima {

enum class InputKind {
  Mat,
  Neat,
  Holder,
};

struct InputItem {
  InputKind kind = InputKind::Mat;
  cv::Mat mat;
  NeatTensor neat;
  std::shared_ptr<void> holder;
};

struct PipelineRun::State {
  InputStream stream;
  PipelineRunOptions opt;
  InputStreamOptions stream_opt;
  bool supports_push = false;
  bool supports_pull = false;
  bool auto_recover_dispatcher = true;
  std::function<void(const PipelineReport&)> on_dispatcher_error;
  std::atomic<bool> recovery_attempted{false};

  std::mutex in_mu;
  std::condition_variable in_cv;
  std::deque<InputItem> in_queue;
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

  bool diag_enabled = false;
  std::atomic<bool> diag_logged{false};
  std::string diag_sysinfo;
};

namespace {

template <typename T>
bool queue_full(const std::deque<T>& q, int max) {
  return max > 0 && static_cast<int>(q.size()) >= max;
}

bool env_bool(const char* key, bool def_val) {
  const char* v = std::getenv(key);
  if (!v || !*v) return def_val;
  if (!std::strcmp(v, "1") || !std::strcmp(v, "true") || !std::strcmp(v, "TRUE") ||
      !std::strcmp(v, "yes") || !std::strcmp(v, "YES") ||
      !std::strcmp(v, "on")  || !std::strcmp(v, "ON")) {
    return true;
  }
  if (!std::strcmp(v, "0") || !std::strcmp(v, "false") || !std::strcmp(v, "FALSE") ||
      !std::strcmp(v, "no") || !std::strcmp(v, "NO") ||
      !std::strcmp(v, "off") || !std::strcmp(v, "OFF")) {
    return false;
  }
  return def_val;
}

int env_int(const char* key, int def_val) {
  const char* v = std::getenv(key);
  if (!v || !*v) return def_val;
  return std::atoi(v);
}

std::string read_first_line(const char* path) {
  std::ifstream in(path);
  if (!in.is_open()) return {};
  std::string line;
  std::getline(in, line);
  return line;
}

std::vector<std::string> tail_lines(const std::string& path, size_t max_lines) {
  std::ifstream in(path);
  if (!in.is_open()) return {};
  std::deque<std::string> buf;
  std::string line;
  while (std::getline(in, line)) {
    if (buf.size() == max_lines) buf.pop_front();
    buf.push_back(line);
  }
  return std::vector<std::string>(buf.begin(), buf.end());
}

bool contains_case_insensitive(const std::string& haystack, const std::string& needle) {
  if (needle.empty()) return true;
  const auto it = std::search(
      haystack.begin(), haystack.end(),
      needle.begin(), needle.end(),
      [](char a, char b) {
        return std::tolower(static_cast<unsigned char>(a)) ==
               std::tolower(static_cast<unsigned char>(b));
      });
  return it != haystack.end();
}

std::string collect_system_info() {
  std::ostringstream oss;
  const std::string governor =
      read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
  const std::string loadavg = read_first_line("/proc/loadavg");
  if (!governor.empty()) {
    oss << "cpu_governor=" << governor;
  }
  if (!loadavg.empty()) {
    if (!oss.str().empty()) oss << " ";
    oss << "loadavg=" << loadavg;
  }

  std::vector<std::string> rpmsg_lines;
  const std::vector<std::string> kern_tail = tail_lines("/var/log/kern.log", 200);
  const auto maybe_add_rpmsg = [&](const std::vector<std::string>& lines) {
    for (const auto& line : lines) {
      if (!contains_case_insensitive(line, "rpmsg")) continue;
      if (contains_case_insensitive(line, "error") ||
          contains_case_insensitive(line, "err") ||
          contains_case_insensitive(line, "fail")) {
        rpmsg_lines.push_back(line);
      }
    }
  };

  if (!kern_tail.empty()) {
    maybe_add_rpmsg(kern_tail);
  } else {
    const std::vector<std::string> syslog_tail = tail_lines("/var/log/syslog", 200);
    maybe_add_rpmsg(syslog_tail);
  }

  if (!rpmsg_lines.empty()) {
    if (!oss.str().empty()) oss << " ";
    oss << "rpmsg_errors=" << rpmsg_lines.size();
    const size_t show = std::min<size_t>(rpmsg_lines.size(), 3);
    for (size_t i = rpmsg_lines.size() - show; i < rpmsg_lines.size(); ++i) {
      oss << "\n  rpmsg: " << rpmsg_lines[i];
    }
  }

  return oss.str();
}

int parse_num_buffers_for(const std::string& pipeline, const std::string& plugin) {
  const std::string key = "num-buffers=";
  size_t pos = 0;
  while ((pos = pipeline.find(plugin, pos)) != std::string::npos) {
    size_t nb = pipeline.find(key, pos);
    if (nb == std::string::npos) {
      pos += plugin.size();
      continue;
    }
    nb += key.size();
    size_t end = nb;
    while (end < pipeline.size() &&
           std::isdigit(static_cast<unsigned char>(pipeline[end]))) {
      ++end;
    }
    if (end > nb) {
      return std::atoi(pipeline.substr(nb, end - nb).c_str());
    }
    pos = end;
  }
  return 0;
}

int parse_queue2_depth(const std::string& pipeline) {
  const std::string key = "queue2";
  const std::string depth_key = "max-size-buffers=";
  size_t pos = 0;
  while ((pos = pipeline.find(key, pos)) != std::string::npos) {
    size_t depth_pos = pipeline.find(depth_key, pos);
    if (depth_pos == std::string::npos) {
      pos += key.size();
      continue;
    }
    depth_pos += depth_key.size();
    size_t end = depth_pos;
    while (end < pipeline.size() &&
           std::isdigit(static_cast<unsigned char>(pipeline[end]))) {
      ++end;
    }
    if (end > depth_pos) {
      return std::atoi(pipeline.substr(depth_pos, end - depth_pos).c_str());
    }
    pos = end;
  }
  return 0;
}

void warn_no_warmup_once() {
  static std::atomic<bool> warned{false};
  if (warned.exchange(true)) return;
  std::printf("[WARN] PipelineRun::warmup: warm=0; throughput stability may vary without warmup.\n");
}

} // namespace

PipelineRun::PipelineRun(std::shared_ptr<State> state) : state_(std::move(state)) {}

PipelineRun::PipelineRun(PipelineRun&& other) noexcept : state_(std::move(other.state_)) {}

PipelineRun& PipelineRun::operator=(PipelineRun&& other) noexcept {
  if (this != &other) {
    close();
    state_ = std::move(other.state_);
  }
  return *this;
}

PipelineRun::~PipelineRun() {
  close();
}

PipelineRun PipelineRun::create(InputStream stream,
                                const PipelineRunOptions& opt,
                                const InputStreamOptions& stream_opt) {
  auto st = std::make_shared<State>();
  st->stream = std::move(stream);
  st->opt = opt;
  st->stream_opt = stream_opt;
  st->supports_push = st->stream.can_push();
  st->supports_pull = st->stream.can_pull();
  st->auto_recover_dispatcher = opt.auto_recover_dispatcher;
  st->on_dispatcher_error = opt.on_dispatcher_error;
  st->diag_enabled = env_bool("SIMA_ASYNC_TPUT_DIAG", false);
  if (st->diag_enabled) {
    st->diag_sysinfo = collect_system_info();
  }

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

  if (st->supports_pull) {
    st->stream.start(on_output);
  }

  if (!st->supports_push) {
    st->input_thread_done.store(true);
    return PipelineRun(st);
  }

  st->input_thread = std::thread([st]() {
    while (true) {
      InputItem item;
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
      if (st->supports_pull) {
        std::lock_guard<std::mutex> lock(st->latency_mu);
        st->pending_times.push_back(t0);
      }
      try {
        switch (item.kind) {
          case InputKind::Neat:
            st->stream.push(item.neat);
            break;
          case InputKind::Holder:
            st->stream.push_holder(item.holder);
            break;
          case InputKind::Mat:
          default:
            st->stream.push(item.mat);
            break;
        }
        st->inputs_pushed.fetch_add(1, std::memory_order_relaxed);
      } catch (const std::exception& e) {
        if (st->supports_pull) {
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

  return PipelineRun(st);
}

PipelineRun::operator bool() const noexcept {
  return state_ && static_cast<bool>(state_->stream);
}

bool PipelineRun::can_push() const {
  return state_ && state_->supports_push;
}

bool PipelineRun::can_pull() const {
  return state_ && state_->supports_pull;
}

bool PipelineRun::running() const {
  if (!state_) return false;
  if (state_->supports_pull) {
    return state_->stream.running();
  }
  return !state_->stop_requested.load();
}

std::string PipelineRun::last_error() const {
  if (!state_) return {};
  std::lock_guard<std::mutex> lock(state_->error_mu);
  if (!state_->error.empty()) return state_->error;
  return state_->stream.last_error();
}

std::string PipelineRun::diagnostics_summary() const {
  if (!state_) return {};
  return state_->stream.diagnostics_summary();
}

bool PipelineRun::push_impl(const cv::Mat& input, bool block) {
  if (!state_) {
    throw std::runtime_error("PipelineRun::push: stream is closed");
  }
  if (!state_->supports_push) {
    throw std::runtime_error(
        "PipelineRun::push: pipeline has no InputAppSrc (push not supported)");
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

    InputItem item;
    item.kind = InputKind::Mat;
    if (st->opt.copy_input) {
      item.mat = input.clone();
    } else {
      item.mat = input;
    }
    st->in_queue.push_back(std::move(item));
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->in_cv.notify_one();
  return true;
}

bool PipelineRun::push_impl(const NeatTensor& input, bool block) {
  if (!state_) {
    throw std::runtime_error("PipelineRun::push: stream is closed");
  }
  if (!state_->supports_push) {
    throw std::runtime_error(
        "PipelineRun::push: pipeline has no InputAppSrc (push not supported)");
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

    InputItem item;
    item.kind = InputKind::Neat;
    if (st->opt.copy_input) {
      item.neat = input.clone();
    } else {
      item.neat = input;
    }
    st->in_queue.push_back(std::move(item));
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->in_cv.notify_one();
  return true;
}

bool PipelineRun::push_holder_impl(const std::shared_ptr<void>& holder, bool block) {
  if (!state_) {
    throw std::runtime_error("PipelineRun::push_holder: stream is closed");
  }
  if (!state_->supports_push) {
    throw std::runtime_error(
        "PipelineRun::push_holder: pipeline has no InputAppSrc (push not supported)");
  }
  if (!holder) {
    throw std::invalid_argument("PipelineRun::push_holder: missing holder");
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

    InputItem item;
    item.kind = InputKind::Holder;
    item.holder = holder;
    st->in_queue.push_back(std::move(item));
    st->inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  st->in_cv.notify_one();
  return true;
}

bool PipelineRun::push(const cv::Mat& input) {
  return push_impl(input, true);
}

bool PipelineRun::try_push(const cv::Mat& input) {
  return push_impl(input, false);
}

bool PipelineRun::push(const NeatTensor& input) {
  return push_impl(input, true);
}

bool PipelineRun::try_push(const NeatTensor& input) {
  return push_impl(input, false);
}

bool PipelineRun::push_holder(const std::shared_ptr<void>& holder) {
  return push_holder_impl(holder, true);
}

bool PipelineRun::try_push_holder(const std::shared_ptr<void>& holder) {
  return push_holder_impl(holder, false);
}

void PipelineRun::close_input() {
  if (!state_) return;
  auto st = state_;
  {
    std::lock_guard<std::mutex> lock(st->in_mu);
    st->input_closed = true;
  }
  st->in_cv.notify_all();
}

std::optional<RunInputResult> PipelineRun::pull(int timeout_ms) {
  if (!state_) return std::nullopt;
  auto st = state_;
  if (!st->supports_pull) {
    throw std::runtime_error(
        "PipelineRun::pull: pipeline has no OutputAppSink (pull not supported)");
  }
  auto diag = st->stream.diag_ctx();
  const auto handle_stream_error = [&](const std::string& err) {
    PipelineReport rep = diag ? diag->snapshot_basic() : PipelineReport{};
    rep.repro_note = "PipelineRun::pull: " + err;
    if (pipeline_internal::match_dispatcher_unavailable(err)) {
      rep.error_code = pipeline_internal::kDispatcherUnavailableError;
    }
    if (pipeline_internal::is_dispatcher_unavailable(rep)) {
      const bool allow_recover =
          st->auto_recover_dispatcher &&
          pipeline_internal::env_bool("SIMA_DISPATCHER_AUTO_RECOVER", true);
      const bool first = !st->recovery_attempted.exchange(true);
      if (first && st->on_dispatcher_error) st->on_dispatcher_error(rep);
      if (first) {
        pipeline_internal::attempt_dispatcher_recovery(&rep, allow_recover);
      }
    }
    throw PipelineError(rep.repro_note, std::move(rep));
  };

  const std::string early_err = last_error();
  if (!early_err.empty()) {
    handle_stream_error(early_err);
  }

  auto done = [&]() {
    if (st->supports_push) {
      return st->input_closed &&
             st->input_thread_done.load() &&
             st->outputs_pulled.load() >= st->inputs_pushed.load() &&
             st->out_queue.empty();
    }
    return st->stop_requested.load() && st->out_queue.empty();
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
      throw std::runtime_error("PipelineRun::pull: timeout waiting for output");
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

  const bool is_done = done();
  lock.unlock();

  const std::string late_err = last_error();
  if (!late_err.empty()) {
    handle_stream_error(late_err);
  }

  if (is_done) {
    stop();
    return std::nullopt;
  }

  return std::nullopt;
}

RunInputResult PipelineRun::push_and_pull(const cv::Mat& input, int timeout_ms) {
  if (!push(input)) {
    throw std::runtime_error("PipelineRun::push_and_pull: push failed");
  }
  auto out = pull(timeout_ms);
  if (!out.has_value()) {
    throw std::runtime_error("PipelineRun::push_and_pull: no output");
  }
  return *out;
}

RunInputResult PipelineRun::push_and_pull(const NeatTensor& input, int timeout_ms) {
  if (!push(input)) {
    throw std::runtime_error("PipelineRun::push_and_pull: push failed");
  }
  auto out = pull(timeout_ms);
  if (!out.has_value()) {
    throw std::runtime_error("PipelineRun::push_and_pull: no output");
  }
  return *out;
}

RunInputResult PipelineRun::push_and_pull_holder(const std::shared_ptr<void>& holder,
                                                 int timeout_ms) {
  if (!push_holder(holder)) {
    throw std::runtime_error("PipelineRun::push_and_pull_holder: push failed");
  }
  auto out = pull(timeout_ms);
  if (!out.has_value()) {
    throw std::runtime_error("PipelineRun::push_and_pull_holder: no output");
  }
  return *out;
}

int PipelineRun::warmup(const cv::Mat& input, int warm, int timeout_ms) {
  if (warm < 0) {
    warm = env_int("SIMA_ASYNC_WARMUP", 0);
  }
  if (warm <= 0) {
    warn_no_warmup_once();
    return 0;
  }
  for (int i = 0; i < warm; ++i) {
    (void)push_and_pull(input, timeout_ms);
  }
  return warm;
}

PipelineRunStats PipelineRun::stats() const {
  PipelineRunStats out;
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

InputStreamStats PipelineRun::input_stats() const {
  if (!state_) return {};
  return state_->stream.stats();
}

PipelineRunDiagSnapshot PipelineRun::diag_snapshot() const {
  PipelineRunDiagSnapshot out;
  if (!state_) return out;
  const auto diag = state_->stream.diag_ctx();
  if (!diag) return out;

  out.stages.reserve(diag->stage_timings.size());
  for (const auto& stage : diag->stage_timings) {
    if (!stage) continue;
    const auto snap = stage->snapshot();
    PipelineRunStageStats s;
    s.stage_name = snap.stage_name;
    s.samples = snap.samples;
    s.total_us = snap.total_us;
    s.max_us = snap.max_us;
    out.stages.push_back(std::move(s));
  }

  out.boundaries.reserve(diag->boundaries.size());
  for (const auto& boundary : diag->boundaries) {
    if (!boundary) continue;
    out.boundaries.push_back(boundary->snapshot());
  }

  return out;
}

std::string PipelineRun::report(const PipelineRunReportOptions& opt) const {
  if (!state_) return {};
  auto st = state_;
  const auto diag = st->stream.diag_ctx();
  std::ostringstream oss;
  oss << "[REPORT] PipelineRun\n";

  if (opt.include_pipeline && diag && !diag->pipeline_string.empty()) {
    oss << "Pipeline:\n" << diag->pipeline_string << "\n";
  }

  if (diag) {
    if (opt.include_boundaries) {
      const std::string boundary = pipeline_internal::boundary_summary(diag);
      if (!boundary.empty()) oss << boundary;
    }
    if (opt.include_stage_timings) {
      const std::string stages = pipeline_internal::stage_timing_summary(diag);
      if (!stages.empty()) oss << stages;
    }
    if (opt.include_node_reports) {
      std::ostringstream mf;
      bool first = true;
      for (const auto& nr : diag->node_reports) {
        if (nr.kind != "ModelFragment" && nr.kind != "Preproc") continue;
        if (!first) mf << "; ";
        first = false;
        mf << nr.user_label;
        if (!nr.elements.empty()) {
          mf << " [";
          for (size_t i = 0; i < nr.elements.size(); ++i) {
            if (i) mf << ",";
            mf << nr.elements[i];
          }
          mf << "]";
        }
      }
      const std::string mf_str = mf.str();
      if (!mf_str.empty()) {
        oss << "Model fragments: " << mf_str << "\n";
      }
    }
    if (opt.include_next_cpu && !diag->next_cpu_decisions.empty()) {
      oss << "Next CPU decisions:\n";
      for (const auto& d : diag->next_cpu_decisions) {
        oss << "  - node=" << d.node_index
            << " kind=" << d.node_kind;
        if (!d.node_label.empty()) oss << " label=" << d.node_label;
        oss << " next_cpu=" << d.next_cpu
            << " applied=" << (d.applied ? "1" : "0") << "\n";
      }
    }
    if (opt.include_queue_depth || opt.include_num_buffers) {
      const std::string pipeline = diag->pipeline_string;
      if (opt.include_queue_depth) {
        const int queue2_depth = diag->queue2_enabled
            ? diag->queue2_depth
            : parse_queue2_depth(pipeline);
        if (!diag->queue2_enabled) {
          if (queue2_depth > 0) {
            oss << "queue2 depth=" << queue2_depth << " (manual)\n";
          } else {
            oss << "queue2 disabled\n";
          }
        } else if (queue2_depth > 0) {
          oss << "queue2 depth=" << queue2_depth << "\n";
        }
      }
      if (opt.include_num_buffers) {
        int num_cvu = parse_num_buffers_for(pipeline, "simaaiprocesscvu");
        if (num_cvu == 0) num_cvu = parse_num_buffers_for(pipeline, "processcvu");
        if (num_cvu == 0) num_cvu = parse_num_buffers_for(pipeline, "process_cvu");
        int num_mla = parse_num_buffers_for(pipeline, "simaaiprocessmla");
        if (num_mla == 0) num_mla = parse_num_buffers_for(pipeline, "processmla");
        if (num_mla == 0) num_mla = parse_num_buffers_for(pipeline, "process_mla");
        if (num_cvu > 0 || num_mla > 0) {
          oss << "num_buffers_cvu=" << num_cvu
              << " num_buffers_mla=" << num_mla << "\n";
        }
      }
    }
  }

  if (opt.include_run_stats) {
    const PipelineRunStats run_stats = stats();
    oss << "PipelineRunStats: inputs_enqueued=" << run_stats.inputs_enqueued
        << " inputs_dropped=" << run_stats.inputs_dropped
        << " inputs_pushed=" << run_stats.inputs_pushed
        << " outputs_ready=" << run_stats.outputs_ready
        << " outputs_pulled=" << run_stats.outputs_pulled
        << " avg_latency_ms=" << run_stats.avg_latency_ms
        << " min_latency_ms=" << run_stats.min_latency_ms
        << " max_latency_ms=" << run_stats.max_latency_ms << "\n";
  }
  if (opt.include_input_stats) {
    const InputStreamStats is = input_stats();
    oss << "InputStreamStats: push_count=" << is.push_count
        << " push_failures=" << is.push_failures
        << " pull_count=" << is.pull_count
        << " poll_count=" << is.poll_count
        << " avg_alloc_us=" << is.avg_alloc_us
        << " avg_map_us=" << is.avg_map_us
        << " avg_copy_us=" << is.avg_copy_us
        << " avg_push_us=" << is.avg_push_us
        << " avg_pull_wait_us=" << is.avg_pull_wait_us
        << " avg_decode_us=" << is.avg_decode_us << "\n";
  }
  if (opt.include_system_info && !st->diag_sysinfo.empty()) {
    oss << "System: " << st->diag_sysinfo << "\n";
  }

  return oss.str();
}

void PipelineRun::stop() {
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

void PipelineRun::close() {
  if (!state_) return;
  if (state_.use_count() > 1) {
    state_.reset();
    return;
  }
  stop();
  auto st = state_;
  if (st->diag_enabled && !st->diag_logged.exchange(true)) {
    auto log_diag = [&](auto& st_ref) {
      const auto diag = st_ref.stream.diag_ctx();
      std::ostringstream oss;
      oss << "[DIAG] async_tput\n";
      if (diag && !diag->pipeline_string.empty()) {
        oss << "Pipeline:\n" << diag->pipeline_string << "\n";
      }
      if (diag) {
        const std::string boundary = pipeline_internal::boundary_summary(diag);
        if (!boundary.empty()) oss << boundary;
        const std::string stages = pipeline_internal::stage_timing_summary(diag);
        if (!stages.empty()) oss << stages;

        std::ostringstream mf;
        bool first = true;
        for (const auto& nr : diag->node_reports) {
          if (nr.kind != "ModelFragment" && nr.kind != "Preproc") continue;
          if (!first) mf << "; ";
          first = false;
          mf << nr.user_label;
          if (!nr.elements.empty()) {
            mf << " [";
            for (size_t i = 0; i < nr.elements.size(); ++i) {
              if (i) mf << ",";
              mf << nr.elements[i];
            }
            mf << "]";
          }
        }
        const std::string mf_str = mf.str();
        if (!mf_str.empty()) {
          oss << "Model fragments: " << mf_str << "\n";
        }
      if (!diag->next_cpu_decisions.empty()) {
        oss << "Next CPU decisions:\n";
          for (const auto& d : diag->next_cpu_decisions) {
            oss << "  - node=" << d.node_index
                << " kind=" << d.node_kind;
            if (!d.node_label.empty()) oss << " label=" << d.node_label;
            oss << " next_cpu=" << d.next_cpu
                << " applied=" << (d.applied ? "1" : "0") << "\n";
    }
  }
}

      const std::string pipeline = diag ? diag->pipeline_string : std::string();
      const int queue2_depth = (diag && diag->queue2_enabled)
          ? diag->queue2_depth
          : parse_queue2_depth(pipeline);
      if (diag && !diag->queue2_enabled) {
        if (queue2_depth > 0) {
          oss << "queue2 depth=" << queue2_depth << " (manual)\n";
        } else {
          oss << "queue2 disabled\n";
        }
      } else if (queue2_depth > 0) {
        oss << "queue2 depth=" << queue2_depth << "\n";
      }

      int num_cvu = parse_num_buffers_for(pipeline, "simaaiprocesscvu");
      if (num_cvu == 0) num_cvu = parse_num_buffers_for(pipeline, "processcvu");
      if (num_cvu == 0) num_cvu = parse_num_buffers_for(pipeline, "process_cvu");
      int num_mla = parse_num_buffers_for(pipeline, "simaaiprocessmla");
      if (num_mla == 0) num_mla = parse_num_buffers_for(pipeline, "processmla");
      if (num_mla == 0) num_mla = parse_num_buffers_for(pipeline, "process_mla");
      if (num_cvu > 0 || num_mla > 0) {
        oss << "num_buffers_cvu=" << num_cvu << " num_buffers_mla=" << num_mla << "\n";
      }

      PipelineRunStats run_stats;
      run_stats.inputs_enqueued = st_ref.inputs_enqueued.load();
      run_stats.inputs_dropped = st_ref.inputs_dropped.load();
      run_stats.inputs_pushed = st_ref.inputs_pushed.load();
      run_stats.outputs_ready = st_ref.outputs_ready.load();
      run_stats.outputs_pulled = st_ref.outputs_pulled.load();
      {
        std::lock_guard<std::mutex> lock(st_ref.latency_mu);
        if (st_ref.latency_count > 0) {
          run_stats.avg_latency_ms = st_ref.latency_mean_ms;
          run_stats.min_latency_ms = st_ref.latency_min_ms;
          run_stats.max_latency_ms = st_ref.latency_max_ms;
        }
      }

      oss << "PipelineRunStats: inputs_enqueued=" << run_stats.inputs_enqueued
          << " inputs_dropped=" << run_stats.inputs_dropped
          << " inputs_pushed=" << run_stats.inputs_pushed
          << " outputs_ready=" << run_stats.outputs_ready
          << " outputs_pulled=" << run_stats.outputs_pulled
          << " avg_latency_ms=" << run_stats.avg_latency_ms
          << " min_latency_ms=" << run_stats.min_latency_ms
          << " max_latency_ms=" << run_stats.max_latency_ms << "\n";

      const InputStreamStats is = st_ref.stream.stats();
      oss << "InputStreamStats: push_count=" << is.push_count
          << " push_failures=" << is.push_failures
          << " pull_count=" << is.pull_count
          << " poll_count=" << is.poll_count
          << " avg_alloc_us=" << is.avg_alloc_us
          << " avg_map_us=" << is.avg_map_us
          << " avg_copy_us=" << is.avg_copy_us
          << " avg_push_us=" << is.avg_push_us
          << " avg_pull_wait_us=" << is.avg_pull_wait_us
          << " avg_decode_us=" << is.avg_decode_us << "\n";

      if (!st_ref.diag_sysinfo.empty()) {
        oss << "System: " << st_ref.diag_sysinfo << "\n";
      }

      std::printf("%s", oss.str().c_str());
    };
    log_diag(*st);
  }
  st->stream.close();
  state_.reset();
}

} // namespace sima
