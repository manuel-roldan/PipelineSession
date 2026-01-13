#include "pipeline/PipelineSession.h"

#include "test_utils.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace sima::nodes;

namespace {

void sleep_ms(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

} // namespace

int main() {
  try {
    // Default: latest frame, no clock sync.
    {
      sima::PipelineSession p;
      p.gst("videotestsrc num-buffers=1");
      p.add(VideoConvert());
      p.add(CapsNV12SysMem(64, 64, 30));
      p.add(OutputAppSink());

      const std::string gst = p.to_gst();
      require_contains(
          gst,
          "appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true",
          "default OutputAppSink settings mismatch");
    }

    // Latest/drop: only the most recent buffer should remain.
    {
      sima::PipelineSession p;
      p.gst("videotestsrc num-buffers=5");
      p.add(VideoConvert());
      p.add(CapsNV12SysMem(64, 64, 30));
      p.add(OutputAppSink());

      auto stream = p.run();
      sleep_ms(300);

      auto first = stream.next(2000);
      require(first.has_value(), "latest/drop: expected at least one frame");

      auto extra = stream.next(100);
      require(!extra.has_value(), "latest/drop: expected drops with max-buffers=1");

      stream.close();
    }

    // Every-frame: keep all buffers (bounded).
    {
      sima::PipelineSession p;
      p.gst("videotestsrc num-buffers=5");
      p.add(VideoConvert());
      p.add(CapsNV12SysMem(64, 64, 30));
      p.add(OutputAppSink(sima::OutputAppSinkOptions::EveryFrame(5)));

      auto stream = p.run();
      sleep_ms(300);

      for (int i = 0; i < 5; ++i) {
        auto frame = stream.next(2000);
        require(frame.has_value(), "every-frame: missing frame");
      }

      auto extra = stream.next(100);
      require(!extra.has_value(), "every-frame: expected exactly 5 frames");

      stream.close();
    }

    // add_output_tensor forwards sink options.
    {
      sima::PipelineSession p;
      p.gst("videotestsrc num-buffers=1");

      sima::OutputTensorOptions out;
      out.target_width = 64;
      out.target_height = 64;
      out.sink = sima::OutputAppSinkOptions::EveryFrame(3);
      p.add_output_tensor(out);

      const std::string gst = p.to_gst();
      require_contains(gst, "max-buffers=3", "add_output_tensor should forward sink options");
      require_contains(gst, "drop=false", "add_output_tensor should forward drop=false");
    }

    // Clocked: verify sync=true in the pipeline string.
    {
      sima::PipelineSession p;
      p.gst("videotestsrc num-buffers=1");
      p.add(VideoConvert());
      p.add(CapsNV12SysMem(64, 64, 30));
      p.add(OutputAppSink(sima::OutputAppSinkOptions::Clocked()));

      const std::string gst = p.to_gst();
      require_contains(gst, "sync=true", "clocked OutputAppSink should set sync=true");
    }

    std::cout << "[OK] output_appsink_policy_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
