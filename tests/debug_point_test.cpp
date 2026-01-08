#include "pipeline/PipelineSession.h"

#include <gst/gst.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sima::nodes;

static void usage(const char* prog) {
  std::cerr
    << "Usage: " << prog << " --image <path.jpg> --w <even> --h <even>\n"
    << "Example: " << prog << " --image test.jpg --w 256 --h 256\n";
}

static bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) { out = argv[i + 1]; return true; }
  }
  return false;
}

static int get_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string s;
  if (!get_arg(argc, argv, key, s)) return def;
  return std::stoi(s);
}

static void require(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

static std::string i420_caps_wh_fps(int w, int h, int fps) {
  return "video/x-raw,format=I420,width=" + std::to_string(w) +
         ",height=" + std::to_string(h) +
         ",framerate=" + std::to_string(fps) + "/1";
}

static std::string capsfilter_caps(const std::string& caps) {
  return std::string("capsfilter caps=\"") + caps + "\"";
}

static void drain_until_eos(sima::TapStream& ts, int per_try_ms = 200, int max_empty = 5) {
  int empty = 0;
  while (empty < max_empty) {
    auto opt = ts.next(per_try_ms);
    if (!opt) {
      empty++;
      continue;
    }
    empty = 0; // still getting buffers
  }
}

static void check_h264_packet(const sima::TapPacket& pkt, const std::string& name) {
  require(!pkt.bytes.empty(),
          "DebugPoint '" + name + "': got empty bytes. memory_mappable=" +
            std::string(pkt.memory_mappable ? "true" : "false") +
            " reason=" + pkt.non_mappable_reason +
            " features=" + pkt.memory_features);

  require(pkt.caps_string.find("video/x-h264") != std::string::npos,
          "DebugPoint '" + name + "': expected video/x-h264 caps, got: " + pkt.caps_string);
}

static void check_raw_size_1p5(const sima::TapPacket& pkt, const std::string& name, int w, int h) {
  require(!pkt.bytes.empty(),
          "DebugPoint '" + name + "': got empty bytes. memory_mappable=" +
            std::string(pkt.memory_mappable ? "true" : "false") +
            " reason=" + pkt.non_mappable_reason +
            " features=" + pkt.memory_features);

  const size_t expected = (size_t)w * (size_t)h * 3 / 2;
  require(pkt.bytes.size() == expected,
          "DebugPoint '" + name + "': unexpected raw size. Got " + std::to_string(pkt.bytes.size()) +
          " expected " + std::to_string(expected) + ". Caps: " + pkt.caps_string);
}

static sima::TapPacket pull_with_retry(sima::TapStream& ts,
                                       const std::string& name,
                                       int per_try_ms,
                                       int tries) {
  for (int i = 0; i < tries; ++i) {
    auto pkt_opt = ts.next(per_try_ms);
    if (pkt_opt.has_value()) return *pkt_opt;
  }
  throw std::runtime_error("DebugPoint '" + name + "': no packet received (timeout/EOS).");
}

int main(int argc, char** argv) {
  std::string image;
  if (!get_arg(argc, argv, "--image", image)) {
    usage(argv[0]);
    return 2;
  }

  int w = get_int_arg(argc, argv, "--w", 256);
  int h = get_int_arg(argc, argv, "--h", 256);

  const int fps = 30;
  const int bitrate_kbps = 400;

  // NV12 requires even dimensions
  if ((w & 1) || (h & 1)) {
    std::cerr << "[ERR] w/h must be even for NV12. Got w=" << w << " h=" << h << "\n";
    return 2;
  }

  try {
    sima::PipelineSession p;

    // -----------------------------
    // Source still image -> raw stream
    // -----------------------------
    p.add(FileSrc(image));
    p.add(JpegDec());

    // Turn single decoded frame into a short stream and then EOS.
    // (We want deterministic EOS so repeated run_tap() doesn't hang.)
    p.gst("imagefreeze num-buffers=16");

    // Normalize to NV12/SystemMemory at requested w/h/fps (NEW helper)
    p.add(VideoConvert());
    p.add(VideoScale());
    p.gst("videorate");
    p.add(CapsNV12SysMem(w, h, fps));
    p.add(Queue());

    // -----------------------------
    // Encode #1 (SIMA) -> parse -> DebugPoint enc1
    // -----------------------------
    p.add(H264EncodeSima(w, h, fps, bitrate_kbps, "baseline", "4.0"));
    p.add(H264Parse(/*config_interval=*/1));
    p.add(DebugPoint("enc1"));

    // -----------------------------
    // Decode (SIMA) -> force NV12/SystemMemory -> DebugPoint dec1
    // -----------------------------
    p.add(Queue());
    p.add(H264Decode(/*sima_allocator_type=*/2, /*out_format=*/"NV12"));

    // Ensure what we tap is mappable SystemMemory NV12
    p.add(VideoConvert());
    p.add(CapsNV12SysMem(w, h, fps));
    p.add(DebugPoint("dec1"));

    // -----------------------------
    // Convert to I420 -> DebugPoint pre_enc2
    // -----------------------------
    p.add(Queue());
    p.add(VideoConvert());
    // If you later add a CapsI420 helper, replace this p.gst() with it.
    p.gst(capsfilter_caps(i420_caps_wh_fps(w, h, fps)));
    p.add(DebugPoint("pre_enc2"));

    // -----------------------------
    // "Encoder #2" (also SIMA, to avoid x264enc dependency)
    // We convert back to NV12/SystemMemory before SIMA encoder.
    // -----------------------------
    p.add(Queue());
    p.add(VideoConvert());
    p.add(CapsNV12SysMem(w, h, fps));
    p.add(H264EncodeSima(w, h, fps, bitrate_kbps, "baseline", "4.0"));
    p.add(H264Parse(/*config_interval=*/1));

    // Explicit sink so the pipeline always has a terminal element.
    p.gst("fakesink sync=false");

    // ---- Tap enc1 ----
    {
      auto ts = p.run_tap("enc1");
      std::cout << "[DBG] Pipeline for 'enc1': " << ts.debug_pipeline() << "\n";

      const auto pkt = pull_with_retry(ts, "enc1", 500, 30);
      check_h264_packet(pkt, "enc1");

      std::cout << "[OK] DebugPoint 'enc1' received " << pkt.bytes.size()
                << " bytes. Caps: " << pkt.caps_string << "\n";

      drain_until_eos(ts);
      ts.close();
    }

    // ---- Tap dec1 ----
    {
      auto ts = p.run_tap("dec1");
      std::cout << "[DBG] Pipeline for 'dec1': " << ts.debug_pipeline() << "\n";

      const auto pkt = pull_with_retry(ts, "dec1", 500, 30);
      check_raw_size_1p5(pkt, "dec1", w, h);

      std::cout << "[OK] DebugPoint 'dec1' received " << pkt.bytes.size()
                << " bytes. Caps: " << pkt.caps_string << "\n";

      drain_until_eos(ts);
      ts.close();
    }

    // ---- Tap pre_enc2 ----
    {
      auto ts = p.run_tap("pre_enc2");
      std::cout << "[DBG] Pipeline for 'pre_enc2': " << ts.debug_pipeline() << "\n";

      const auto pkt = pull_with_retry(ts, "pre_enc2", 500, 30);
      check_raw_size_1p5(pkt, "pre_enc2", w, h);

      if (pkt.caps_string.find("I420") == std::string::npos) {
        std::cerr << "[WARN] DebugPoint 'pre_enc2': expected I420 caps; got: " << pkt.caps_string << "\n";
      }

      std::cout << "[OK] DebugPoint 'pre_enc2' received " << pkt.bytes.size()
                << " bytes. Caps: " << pkt.caps_string << "\n";

      drain_until_eos(ts);
      ts.close();
    }

    std::cout << "[ALL OK] DebugPoint tests passed.\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
