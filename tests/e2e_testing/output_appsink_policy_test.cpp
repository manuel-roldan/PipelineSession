#include "pipeline/PipelineSession.h"

#include "test_utils.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace sima::nodes;

int main() {
  try {
    // Default: latest frame, no clock sync.
    {
      sima::PipelineSession p;
      p.gst("videotestsrc num-buffers=1", sima::InputRole::Source);
      p.add(VideoConvert());
      p.add(CapsNV12SysMem(64, 64, 30));
      p.add(OutputAppSink());

      const std::string gst = p.to_gst();
      require_contains(
          gst,
          "appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=false",
          "default OutputAppSink settings mismatch");
    }

    // Latest/drop: only the most recent buffer should remain.
    {
      sima::PipelineSession p;
      p.gst("videotestsrc num-buffers=5", sima::InputRole::Source);
      p.add(VideoConvert());
      p.add(CapsNV12SysMem(64, 64, 30));
      sima::OutputAppSinkOptions opt;
      opt.drop = true;
      p.add(OutputAppSink(opt));

      int got = 0;
      p.set_tensor_callback([&](const sima::NeatTensor&) {
        ++got;
        return got < 1;
      });
      p.run();
      require(got == 1, "latest/drop: expected exactly one frame");
    }

    // Every-frame: keep all buffers (bounded).
    {
      sima::PipelineSession p;
      p.gst("videotestsrc num-buffers=5", sima::InputRole::Source);
      p.add(VideoConvert());
      p.add(CapsNV12SysMem(64, 64, 30));
      p.add(OutputAppSink(sima::OutputAppSinkOptions::EveryFrame(5)));

      int got = 0;
      p.set_tensor_callback([&](const sima::NeatTensor&) {
        ++got;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        return true;
      });
      p.run();
      require(got == 5, "every-frame: expected exactly 5 frames");
    }

    // add_output_tensor forwards sink options.
    {
      sima::PipelineSession p;
      p.gst("videotestsrc num-buffers=1", sima::InputRole::Source);

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
      p.gst("videotestsrc num-buffers=1", sima::InputRole::Source);
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
