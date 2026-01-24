#include "nodes/common/Caps.h"
#include "nodes/common/DebugPoint.h"
#include "pipeline/PipelineSession.h"
#include "pipeline/Errors.h"

#include <gst/gst.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>


using namespace sima::nodes;

static std::vector<uint8_t> copy_nv12_from_neat(const sima::NeatTensor& t,
                                                int& out_w,
                                                int& out_h) {
  if (!t.is_nv12()) {
    throw std::runtime_error("Expected NV12 NeatTensor output");
  }
  out_h = t.height();
  out_w = t.width();
  if (out_w <= 0 || out_h <= 0) {
    throw std::runtime_error("NV12 tensor invalid dimensions");
  }
  return t.copy_nv12_contiguous();
}

static void dump_nv12_raw(const std::vector<uint8_t>& nv12, const std::string& path_nv12) {
  std::ofstream ofs(path_nv12, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(nv12.data()),
            static_cast<std::streamsize>(nv12.size()));
  std::cerr << "[DUMP] " << path_nv12 << " (" << nv12.size() << " bytes)\n";
}

static cv::Mat nv12_to_bgr(const std::vector<uint8_t>& nv12, int w, int h) {
  const size_t expected = static_cast<size_t>(w) * static_cast<size_t>(h) * 3 / 2;
  if (nv12.size() < expected) {
    throw std::runtime_error("NV12 buffer too small: got " +
                             std::to_string(nv12.size()) + " expected >= " +
                             std::to_string(expected));
  }

  cv::Mat yuv(h + h / 2, w, CV_8UC1, const_cast<uint8_t*>(nv12.data()));
  cv::Mat bgr;
  cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
  return bgr;
}

static void save_bgr(const cv::Mat& bgr, const std::string& path) {
  if (!cv::imwrite(path, bgr)) throw std::runtime_error("OpenCV imwrite failed: " + path);
  std::cerr << "[SAVE] " << path << "\n";
}

struct Metrics {
  double mae = 0.0;
  double mse = 0.0;
  double psnr = 0.0;
  double max_abs = 0.0;
};

static Metrics compare_bgr(const cv::Mat& a, const cv::Mat& b) {
  if (a.empty() || b.empty()) throw std::runtime_error("compare_bgr: empty image");
  if (a.size() != b.size() || a.type() != b.type()) throw std::runtime_error("compare_bgr: mismatch");

  cv::Mat diff;
  cv::absdiff(a, b, diff);

  cv::Scalar mean_abs = cv::mean(diff);
  const double mae = (mean_abs[0] + mean_abs[1] + mean_abs[2]) / 3.0;

  cv::Mat diff_f;
  diff.convertTo(diff_f, CV_32F);
  diff_f = diff_f.mul(diff_f);
  cv::Scalar mean_sq = cv::mean(diff_f);
  const double mse = (mean_sq[0] + mean_sq[1] + mean_sq[2]) / 3.0;

  const double psnr = (mse <= 1e-12) ? 1e9 : (10.0 * std::log10((255.0 * 255.0) / mse));

  cv::Mat diff_gray;
  cv::cvtColor(diff, diff_gray, cv::COLOR_BGR2GRAY);
  double minv = 0.0, maxv = 0.0;
  cv::minMaxLoc(diff_gray, &minv, &maxv);

  return {mae, mse, psnr, maxv};
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
static double get_double_arg(int argc, char** argv, const std::string& key, double def) {
  std::string s;
  if (!get_arg(argc, argv, key, s)) return def;
  return std::stod(s);
}

static std::atomic<bool> g_keep_running{true};
static void on_sigint(int) { g_keep_running.store(false); }

static void wait_forever_until_ctrl_c() {
  std::signal(SIGINT, on_sigint);
  std::cerr << "[SERVER] Press Ctrl+C to stop.\n";
  while (g_keep_running.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

struct DecodedResult {
  bool ok = false;
  sima::NeatTensor tensor{};
};

enum class TapLocation {
  None,
  Encoded,
  Decoder,
  Convert,
};

static const char* tap_label(TapLocation tap) {
  switch (tap) {
    case TapLocation::Encoded: return "encoded";
    case TapLocation::Decoder: return "decoder";
    case TapLocation::Convert: return "convert";
    default: return "none";
  }
}

static std::string to_lower_copy(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

static TapLocation tap_from_string(const std::string& raw) {
  const std::string s = to_lower_copy(raw);
  if (s == "encoded") return TapLocation::Encoded;
  if (s == "decoder") return TapLocation::Decoder;
  if (s == "convert" || s == "videoconvert") return TapLocation::Convert;
  throw std::runtime_error("Invalid --debug-taps entry: " + raw +
                           " (expected encoded|decoder|convert)");
}

static std::vector<TapLocation> parse_debug_taps(const std::string& raw) {
  std::vector<TapLocation> taps;
  std::string token;
  for (size_t i = 0; i <= raw.size(); ++i) {
    const char c = (i < raw.size()) ? raw[i] : ',';
    if (c == ',') {
      if (!token.empty()) {
        taps.push_back(tap_from_string(token));
        token.clear();
      }
      continue;
    }
    if (!std::isspace(static_cast<unsigned char>(c))) token.push_back(c);
  }
  return taps;
}

enum class ClientStage {
  NonSimaDecode = 0,
  NonSimaDecodeCaps = 1,
  SimaDecoderOnly = 2,
  Current = 3,
};

static const char* client_stage_name(ClientStage stage) {
  switch (stage) {
    case ClientStage::NonSimaDecode: return "non-sima-decode";
    case ClientStage::NonSimaDecodeCaps: return "non-sima-decode+caps";
    case ClientStage::SimaDecoderOnly: return "sima-decoder-only";
    case ClientStage::Current: return "current";
  }
  return "unknown";
}

static ClientStage client_stage_from_int(int value) {
  switch (value) {
    case 0: return ClientStage::NonSimaDecode;
    case 1: return ClientStage::NonSimaDecodeCaps;
    case 2: return ClientStage::SimaDecoderOnly;
    case 3: return ClientStage::Current;
    default: throw std::runtime_error("Invalid --client-stage: " + std::to_string(value));
  }
}

struct ClientDecodeOptions {
  int fps = 30;
  int h264_width = -1;
  int h264_height = -1;
  std::string sw_decoder;
  int sima_allocator = 2;
  std::string sima_next;
  bool no_sysmem_caps = false;
};

static void add_rtp_depay_parse_chain(sima::PipelineSession& p,
                                      int payload_type,
                                      int fps,
                                      int h264_width,
                                      int h264_height,
                                      bool add_h264_caps) {
  std::string rtp_caps = "application/x-rtp,media=video,encoding-name=H264";
  if (payload_type > 0) {
    rtp_caps += ",payload=" + std::to_string(payload_type);
  }
  p.add(Gst("capsfilter caps=\"" + rtp_caps + "\""));
  p.add(Gst("rtph264depay wait-for-keyframe=true"));
  p.add(Gst("h264parse disable-passthrough=true config-interval=1"));
  if (add_h264_caps) {
    std::ostringstream caps;
    caps << "video/x-h264,parsed=true,stream-format=(string)byte-stream,alignment=(string)au";
    if (h264_width > 0 && h264_height > 0) {
      caps << ",width=(int)" << h264_width << ",height=(int)" << h264_height;
    }
    if (fps > 0) {
      caps << ",framerate=(fraction)" << fps << "/1";
    }
    p.add(Gst("capsfilter caps=\"" + caps.str() + "\""));
  }
}

static void add_raw_caps(sima::PipelineSession& p, bool sysmem) {
  p.add(CapsRaw("NV12", -1, -1, -1,
                sysmem ? sima::CapsMemory::SystemMemory : sima::CapsMemory::Any));
}

static void add_sw_decode_chain(sima::PipelineSession& p,
                                const std::string& decoder_element,
                                bool sysmem_caps,
                                TapLocation tap) {
  p.add(Gst(decoder_element));
  if (tap == TapLocation::Decoder) {
    p.add(DebugPoint(tap_label(tap)));
  }
  p.add(Gst("videoconvert"));
  add_raw_caps(p, sysmem_caps);
  if (tap == TapLocation::Convert) {
    p.add(DebugPoint(tap_label(tap)));
  }
}

static void add_sima_manual_decode_chain(sima::PipelineSession& p,
                                         const ClientDecodeOptions& opt,
                                         TapLocation tap) {
  std::ostringstream ss;
  ss << "simaaidecoder sima-allocator-type=" << opt.sima_allocator
     << " dec-fmt=NV12";
  if (!opt.sima_next.empty()) {
    ss << " next-element=" << opt.sima_next;
    if (opt.sima_next == "APU") {
      ss << " apu-mem-pool=malloc";
    }
  }
  p.add(Gst(ss.str()));
  if (tap == TapLocation::Decoder) {
    p.add(DebugPoint(tap_label(tap)));
  }
  p.add(Gst("videoconvert"));
  add_raw_caps(p, !opt.no_sysmem_caps);
  if (tap == TapLocation::Convert) {
    p.add(DebugPoint(tap_label(tap)));
  }
}

static void build_decode_pipeline(sima::PipelineSession& p,
                                  const std::string& url,
                                  const ClientDecodeOptions& opt,
                                  ClientStage stage,
                                  TapLocation tap,
                                  bool add_output_sink) {
  p.add(RTSPInput(url, /*latency_ms=*/200, /*tcp=*/true));

  const TapLocation decode_tap =
      (tap == TapLocation::Encoded) ? TapLocation::None : tap;

  if (stage == ClientStage::NonSimaDecode) {
    add_rtp_depay_parse_chain(p, /*payload_type=*/96, opt.fps,
                              opt.h264_width, opt.h264_height, false);
    if (tap == TapLocation::Encoded) p.add(DebugPoint(tap_label(tap)));
    add_sw_decode_chain(p, opt.sw_decoder, !opt.no_sysmem_caps, decode_tap);
  } else if (stage == ClientStage::NonSimaDecodeCaps) {
    add_rtp_depay_parse_chain(p, /*payload_type=*/96, opt.fps,
                              opt.h264_width, opt.h264_height, true);
    if (tap == TapLocation::Encoded) p.add(DebugPoint(tap_label(tap)));
    add_sw_decode_chain(p, opt.sw_decoder, !opt.no_sysmem_caps, decode_tap);
  } else if (stage == ClientStage::SimaDecoderOnly) {
    add_rtp_depay_parse_chain(p, /*payload_type=*/96, opt.fps,
                              opt.h264_width, opt.h264_height, true);
    if (tap == TapLocation::Encoded) p.add(DebugPoint(tap_label(tap)));
    add_sima_manual_decode_chain(p, opt, decode_tap);
  } else {
    p.add(H264DepayParse(/*payload_type=*/96,
                         /*h264_parse_config_interval=*/1,
                         /*h264_fps=*/opt.fps,
                         /*h264_width=*/opt.h264_width,
                         /*h264_height=*/opt.h264_height));
    if (tap == TapLocation::Encoded) p.add(DebugPoint(tap_label(tap)));

    const bool force_raw_chain =
        (decode_tap != TapLocation::None) || opt.no_sysmem_caps;
    if (force_raw_chain) {
      p.add(H264Decode(/*sima_allocator_type=*/opt.sima_allocator,
                       /*out_format=*/"NV12",
                       /*decoder_name=*/"",
                       /*raw_output=*/true,
                       /*next_element=*/opt.sima_next));
      if (decode_tap == TapLocation::Decoder) {
        p.add(DebugPoint(tap_label(decode_tap)));
      }
      p.add(Gst("videoconvert"));
      add_raw_caps(p, !opt.no_sysmem_caps);
      if (decode_tap == TapLocation::Convert) {
        p.add(DebugPoint(tap_label(decode_tap)));
      }
    } else {
      p.add(H264Decode(/*sima_allocator_type=*/opt.sima_allocator,
                       /*out_format=*/"NV12",
                       /*decoder_name=*/"",
                       /*raw_output=*/false,
                       /*next_element=*/opt.sima_next));
    }
  }

  if (add_output_sink) {
    sima::OutputAppSinkOptions sink_opt;
    sink_opt.drop = true;
    p.add(OutputAppSink(sink_opt));
  }
}

static DecodedResult run_decode_rtsp_stage(const std::string& url,
                                           const ClientDecodeOptions& opt,
                                           ClientStage stage,
                                           bool debug) {
  sima::PipelineSessionOptions ps_opt;
  ps_opt.callback_timeout_ms = 12000;
  sima::PipelineSession p(ps_opt);
  build_decode_pipeline(p, url, opt, stage, TapLocation::None, /*add_output_sink=*/true);

  if (debug) {
    std::cerr << "[CLIENT] stage=" << client_stage_name(stage) << "\n";
    std::cerr << "[CLIENT] pipeline:\n" << p.to_gst() << "\n";
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(600));

  sima::NeatTensor captured{};
  bool got_any = false;
  bool printed_first = false;

  p.set_tensor_callback([&](const sima::NeatTensor& tensor) {
    captured = tensor;
    got_any = true;
    if (debug && !printed_first) {
      const int64_t h = tensor.shape.size() > 0 ? tensor.shape[0] : 0;
      const int64_t w = tensor.shape.size() > 1 ? tensor.shape[1] : 0;
      const char* fmt = "UNKNOWN";
      if (tensor.semantic.image.has_value()) {
        switch (tensor.semantic.image->format) {
          case sima::NeatImageSpec::PixelFormat::NV12: fmt = "NV12"; break;
          case sima::NeatImageSpec::PixelFormat::I420: fmt = "I420"; break;
          case sima::NeatImageSpec::PixelFormat::RGB: fmt = "RGB"; break;
          case sima::NeatImageSpec::PixelFormat::BGR: fmt = "BGR"; break;
          case sima::NeatImageSpec::PixelFormat::GRAY8: fmt = "GRAY8"; break;
          case sima::NeatImageSpec::PixelFormat::UNKNOWN: fmt = "UNKNOWN"; break;
        }
      }
      std::cerr << "[CLIENT] frame format=" << fmt
                << " w=" << w
                << " h=" << h
                << " planes=" << tensor.planes.size()
                << "\n";
      printed_first = true;
    }
    return false;
  });
  try {
    p.run();
  } catch (const sima::PipelineError& e) {
    if (debug) {
      std::cerr << "[CLIENT] stage=" << client_stage_name(stage)
                << " PipelineError: " << e.what() << "\n";
      std::cerr << "[CLIENT] report_json=" << e.report().to_json() << "\n";
    }
    throw;
  }

  if (!got_any) {
    return {false, {}};
  }
  return {true, captured};
}

static ClientDecodeOptions make_client_options(ClientStage stage,
                                               int fps,
                                               int h264_width,
                                               int h264_height,
                                               const std::string& sw_decoder,
                                               bool no_sysmem_caps,
                                               int dec_allocator,
                                               bool dec_allocator_set,
                                               const std::string& dec_next,
                                               bool dec_next_set) {
  ClientDecodeOptions opt;
  opt.fps = fps;
  opt.h264_width = h264_width;
  opt.h264_height = h264_height;
  opt.sw_decoder = sw_decoder;
  opt.no_sysmem_caps = no_sysmem_caps;

  if (stage == ClientStage::SimaDecoderOnly) {
    opt.sima_allocator = dec_allocator_set ? dec_allocator : 1;
    opt.sima_next = dec_next_set ? dec_next : "APU";
  } else if (stage == ClientStage::Current) {
    opt.sima_allocator = dec_allocator_set ? dec_allocator : 2;
    opt.sima_next = dec_next_set ? dec_next : "";
  } else {
    opt.sima_allocator = dec_allocator_set ? dec_allocator : 2;
    opt.sima_next = dec_next_set ? dec_next : "";
  }

  return opt;
}

static void print_debug_tap(const sima::RunDebugTap& tap) {
  std::cerr << "[TAP] name=" << tap.name;
  if (!tap.error.empty()) {
    std::cerr << " error=" << tap.error << "\n";
    return;
  }
  if (!tap.packet.has_value()) {
    std::cerr << " packet=none\n";
    return;
  }
  const sima::TapPacket& pkt = *tap.packet;
  std::cerr << " caps=\"" << pkt.caps_string << "\"";
  std::cerr << " mem=\"" << pkt.memory_features << "\"";
  if (!pkt.video.format.empty()) {
    std::cerr << " fmt=" << pkt.video.format;
  }
  if (pkt.video.width > 0 && pkt.video.height > 0) {
    std::cerr << " " << pkt.video.width << "x" << pkt.video.height;
  }
  if (pkt.video.fps_num > 0) {
    std::cerr << " fps=" << pkt.video.fps_num << "/" << pkt.video.fps_den;
  }
  std::cerr << " keyframe=" << (pkt.keyframe ? "true" : "false");
  std::cerr << " mappable=" << (pkt.memory_mappable ? "true" : "false");
  if (!pkt.memory_mappable && !pkt.non_mappable_reason.empty()) {
    std::cerr << " reason=" << pkt.non_mappable_reason;
  }
  std::cerr << " bytes=" << pkt.bytes.size() << "\n";
}

static void run_debug_tap(const std::string& url,
                          const ClientDecodeOptions& opt,
                          ClientStage stage,
                          TapLocation tap,
                          int timeout_ms,
                          bool print_pipeline,
                          bool verbose) {
  if (tap == TapLocation::None) return;

  sima::PipelineSession p;
  build_decode_pipeline(p, url, opt, stage, tap, /*add_output_sink=*/false);

  std::cerr << "[TAP] stage=" << client_stage_name(stage)
            << " tap=" << tap_label(tap) << "\n";
  if (print_pipeline) {
    std::cerr << "[TAP] pipeline:\n" << p.to_gst() << "\n";
  }

  sima::RunDebugOptions dbg_opt;
  dbg_opt.timeout_ms = timeout_ms;
  auto dbg = p.run_debug(dbg_opt);

  if (dbg.taps.empty()) {
    std::cerr << "[TAP] no taps returned\n";
  } else {
    for (const auto& t : dbg.taps) {
      print_debug_tap(t);
    }
  }

  const bool tap_error = dbg.taps.empty() ||
                         (!dbg.taps[0].packet.has_value() && !dbg.taps[0].error.empty());
  if (tap_error && !dbg.report.repro_note.empty()) {
    std::cerr << "[TAP] report_note=" << dbg.report.repro_note << "\n";
  }
  if ((verbose || tap_error) && !dbg.report.pipeline_string.empty()) {
    std::cerr << "[TAP] report_json=" << dbg.report.to_json() << "\n";
  }
}

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "Usage: " << argv[0]
                << " /path/to/static.jpg [--server-only] [--debug]"
                << " [--client-stage N] [--client-seq] [--sw-decoder ELEM]"
                << " [--debug-taps encoded,decoder,convert]"
                << " [--dec-allocator N] [--dec-next STR] [--no-sysmem-caps]"
                << " [--rtsp-url URL]"
                << " [--enc-w N] [--enc-h N] [--mae X] [--psnr Y]\n";
      return 2;
    }

    std::string image_path = argv[1];
    bool server_only = false;
    bool debug = false;
    bool client_seq = false;
    int client_stage = 3;
    std::string sw_decoder = "decodebin";
    int enc_w_override = -1;
    int enc_h_override = -1;
    bool no_sysmem_caps = false;
    std::string debug_taps_raw;
    int dec_allocator = 2;
    bool dec_allocator_set = false;
    std::string dec_next;
    bool dec_next_set = false;
    std::string rtsp_url;

    for (int i = 2; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--server-only") server_only = true;
      else if (a == "--debug") debug = true;
      else if (a == "--client-seq") client_seq = true;
      else if (a == "--client-stage" && i + 1 < argc) client_stage = std::stoi(argv[++i]);
      else if (a == "--sw-decoder" && i + 1 < argc) sw_decoder = argv[++i];
      else if (a == "--debug-taps" && i + 1 < argc) debug_taps_raw = argv[++i];
      else if (a == "--dec-allocator" && i + 1 < argc) {
        dec_allocator = std::stoi(argv[++i]);
        dec_allocator_set = true;
      } else if (a == "--dec-next" && i + 1 < argc) {
        dec_next = argv[++i];
        dec_next_set = true;
      } else if (a == "--no-sysmem-caps") {
        no_sysmem_caps = true;
      } else if (a == "--rtsp-url" && i + 1 < argc) {
        rtsp_url = argv[++i];
      } else if (a == "--enc-w" && i + 1 < argc) {
        enc_w_override = std::stoi(argv[++i]);
      } else if (a == "--enc-h" && i + 1 < argc) {
        enc_h_override = std::stoi(argv[++i]);
      }
    }

    const double mae_thr = get_double_arg(argc, argv, "--mae", 25.0);
    const double psnr_thr = get_double_arg(argc, argv, "--psnr", 22.0);

    std::vector<TapLocation> debug_taps;
    if (!debug_taps_raw.empty()) {
      debug_taps = parse_debug_taps(debug_taps_raw);
    }

    // Content dims for this test
    const int content_w = 224;
    const int content_h = 224;

    // Encoder padding defaults
    int enc_w = (enc_w_override > 0) ? enc_w_override : 256;
    int enc_h = (enc_h_override > 0) ? enc_h_override : 256;

    // -------------------------
    // Server session (RTSP)
    // -------------------------
    const int fps = 30;

    std::string server_url;
    bool server_started = false;
    sima::RtspServerHandle server;
    sima::PipelineSession s;

    if (!rtsp_url.empty()) {
      server_url = rtsp_url;
      std::cerr << "[INFO] Using external RTSP URL: " << server_url << "\n";
    } else {
      s.add(AppSrcImage(image_path, content_w, content_h, enc_w, enc_h, fps));
      s.add(H264EncodeSima(enc_w, enc_h, fps, /*bitrate_kbps=*/400, "baseline", "4.0"));
      s.add(H264Parse(/*config_interval=*/1));
      s.add(RtpH264Pay(/*pt=*/96, /*config_interval=*/1));

      server = s.run_rtsp({ .mount = "image", .port = 8554 });
      server_started = true;
      server_url = server.url();
      std::cerr << "[INFO] RTSP URL: " << server_url << "\n";
      std::cerr << "[INFO] Server pipeline: " << s.last_pipeline() << "\n";
    }

    if (server_only) {
      if (!rtsp_url.empty()) {
        throw std::runtime_error("--server-only cannot be used with --rtsp-url");
      }
      wait_forever_until_ctrl_c();
      if (server_started) server.stop();
      return 0;
    }

    // -------------------------
    // Client session (decode)
    // -------------------------
    DecodedResult dec{};
    ClientStage max_stage = client_stage_from_int(client_stage);
    const int debug_timeout_ms = 8000;
    if (client_seq) {
      for (int s = 0; s <= static_cast<int>(max_stage); ++s) {
        const ClientStage stage = client_stage_from_int(s);
        if (debug) {
          std::cerr << "[CLIENT] running stage=" << client_stage_name(stage) << "\n";
        }
        const ClientDecodeOptions opt = make_client_options(stage,
                                                            fps,
                                                            enc_w,
                                                            enc_h,
                                                            sw_decoder,
                                                            no_sysmem_caps,
                                                            dec_allocator,
                                                            dec_allocator_set,
                                                            dec_next,
                                                            dec_next_set);
        if (!debug_taps.empty()) {
          for (TapLocation tap : debug_taps) {
            run_debug_tap(server_url,
                          opt,
                          stage,
                          tap,
                          debug_timeout_ms,
                          debug,
                          debug);
          }
        }
        dec = run_decode_rtsp_stage(server_url, opt, stage, debug);
        if (!dec.ok) {
          if (server_started) server.stop();
          std::cerr << "[FAIL] Stage " << client_stage_name(stage)
                    << " could not decode any frame from RTSP.\n";
          return 3;
        }
        std::cerr << "[OK] Stage " << client_stage_name(stage) << " decoded a frame.\n";
      }
    } else {
      const ClientStage stage = max_stage;
      const ClientDecodeOptions opt = make_client_options(stage,
                                                          fps,
                                                          enc_w,
                                                          enc_h,
                                                          sw_decoder,
                                                          no_sysmem_caps,
                                                          dec_allocator,
                                                          dec_allocator_set,
                                                          dec_next,
                                                          dec_next_set);
      if (!debug_taps.empty()) {
        for (TapLocation tap : debug_taps) {
          run_debug_tap(server_url,
                        opt,
                        stage,
                        tap,
                        debug_timeout_ms,
                        debug,
                        debug);
        }
      }
      dec = run_decode_rtsp_stage(server_url, opt, stage, debug);
      if (!dec.ok) {
        if (server_started) server.stop();
        std::cerr << "[FAIL] Stage " << client_stage_name(stage)
                  << " could not decode any frame from RTSP.\n";
        return 3;
      }
    }

    int out_w = 0;
    int out_h = 0;
    std::vector<uint8_t> nv12 = copy_nv12_from_neat(dec.tensor, out_w, out_h);
    dump_nv12_raw(nv12, "decoded_sima.nv12");

    cv::Mat bgr_full = nv12_to_bgr(nv12, out_w, out_h);
    save_bgr(bgr_full, "decoded_sima_full.jpg");

    // Crop back to content if encoded padded
    cv::Mat bgr_crop;
    if (out_w == enc_w && out_h == enc_h &&
        (enc_w != content_w || enc_h != content_h)) {

      int off_x = (enc_w - content_w) / 2;
      int off_y = (enc_h - content_h) / 2;
      off_x &= ~1;
      off_y &= ~1;

      bgr_crop = bgr_full(cv::Rect(off_x, off_y, content_w, content_h)).clone();
    } else {
      bgr_crop = bgr_full.clone();
      if (bgr_crop.cols != content_w || bgr_crop.rows != content_h) {
        cv::resize(bgr_crop, bgr_crop, cv::Size(content_w, content_h), 0, 0, cv::INTER_LINEAR);
      }
    }
    save_bgr(bgr_crop, "decoded_sima_crop.jpg");

    // Reference: OpenCV decode + resize to content
    cv::Mat ref = cv::imread(image_path, cv::IMREAD_COLOR);
    if (ref.empty()) {
      server.stop();
      throw std::runtime_error("OpenCV imread failed: " + image_path);
    }
    cv::Mat ref_rs;
    int interp = (content_w < ref.cols || content_h < ref.rows) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(ref, ref_rs, cv::Size(content_w, content_h), 0, 0, interp);
    save_bgr(ref_rs, "reference_content.jpg");

    Metrics m = compare_bgr(bgr_crop, ref_rs);

    std::cerr << "[METRICS] MAE=" << m.mae
              << "  PSNR=" << m.psnr
              << " dB  MaxAbs=" << m.max_abs << "\n";

    if (m.mae > mae_thr || m.psnr < psnr_thr) {
      std::cerr << "[FAIL] Encoding/decoding mismatch: requires MAE <= " << mae_thr
                << " and PSNR >= " << psnr_thr << " dB\n";
      server.stop();
      return 4;
    }

    std::cout << "[OK] encoding_decoding_test passed.\n";
    if (server_started) server.stop();
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "[FATAL] " << e.what() << "\n";
    return 1;
  }
}
