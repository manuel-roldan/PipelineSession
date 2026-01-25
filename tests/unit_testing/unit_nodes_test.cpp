#include "nodes/common/AppSink.h"
#include "nodes/common/Caps.h"
#include "nodes/common/DebugPoint.h"
#include "nodes/common/FileSrc.h"
#include "nodes/common/JpegDec.h"
#include "nodes/common/QtDemuxVideoPad.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoScale.h"
#include "nodes/common/VideoRate.h"
#include "nodes/common/ImageDecode.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/RtpH264Depay.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/RtpH264Pay.h"
#include "gst/GstHelpers.h"

#include "test_utils.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static const char* decoder_element_name() {
  return "neatdecoder";
}

int main() {
  try {
    auto caps = sima::nodes::CapsNV12SysMem(128, 64, 30);
    require_contains(caps->gst_fragment(3), "capsfilter name=n3_caps", "Caps fragment missing name");

    auto dbg = sima::nodes::DebugPoint("a b");
    require_contains(dbg->gst_fragment(0), "dbg_a_b", "DebugPoint name not sanitized");

    auto fs = sima::nodes::FileSrc("path.jpg");
    require_contains(fs->gst_fragment(1), "filesrc name=n1_filesrc", "FileSrc name mismatch");

    auto jd = sima::nodes::JpegDec();
    require_contains(jd->gst_fragment(2), "jpegdec name=n2_jpegdec", "JpegDec name mismatch");

    auto q = sima::nodes::Queue();
    require_contains(q->gst_fragment(5), "queue name=n5_queue", "Queue name mismatch");

    auto vc = sima::nodes::VideoConvert();
    require_contains(vc->gst_fragment(6), "videoconvert name=n6_videoconvert", "VideoConvert name mismatch");

    auto vs = sima::nodes::VideoScale();
    require_contains(vs->gst_fragment(7), "videoscale name=n7_videoscale", "VideoScale name mismatch");

    auto vr = sima::nodes::VideoRate();
    require_contains(vr->gst_fragment(8), "videorate name=n8_videorate", "VideoRate name mismatch");

    auto demux = sima::nodes::QtDemuxVideoPad(0);
    require_contains(demux->gst_fragment(4), "qtdemux name=n4_demux", "QtDemux name mismatch");

    auto rtsp = sima::nodes::RTSPInput("rtsp://example", 200, true);
    require_contains(rtsp->gst_fragment(1), "rtspsrc name=n1_rtspsrc", "RTSPInput name mismatch");

    auto depay = sima::nodes::H264DepayParse(97);
    require_contains(depay->gst_fragment(2), "rtph264depay name=n2_depay", "Depay fragment mismatch");
    require_contains(depay->gst_fragment(2), "payload=97", "Depay payload mismatch");

    auto dec = sima::nodes::H264Decode(2, "NV12");
    const std::string dec_expect = std::string(decoder_element_name()) + " name=n1_decoder";
    require_contains(dec->gst_fragment(1), dec_expect, "Decode fragment mismatch");

    auto enc = sima::nodes::H264EncodeSima(64, 64, 30, 400, "baseline", "4.0");
    require_contains(enc->gst_fragment(0), "neatencoder name=n0_encoder", "Encode fragment mismatch");

    sima::H264ParseOptions opt;
    opt.enforce_caps = true;
    opt.alignment = sima::H264ParseOptions::Alignment::AU;
    auto parse = sima::nodes::H264Parse(opt);
    require_contains(parse->gst_fragment(3), "h264parse name=n3_h264parse", "Parse fragment mismatch");
    require_contains(parse->gst_fragment(3), "capsfilter name=n3_h264_caps", "Parse capsfilter missing");

    auto pay = sima::nodes::RtpH264Pay(96, 1);
    require_contains(pay->gst_fragment(9), "rtph264pay name=pay0", "Pay fragment mismatch");

    auto sink = sima::nodes::OutputAppSink();
    require_contains(sink->gst_fragment(0), "appsink name=mysink", "AppSink fragment mismatch");

    auto idec = sima::nodes::ImageDecode();
    require_contains(idec->gst_fragment(2), "decodebin name=n2_decodebin", "ImageDecode fragment mismatch");

    bool has_sw = sima::element_exists("x264enc") ||
                  sima::element_exists("openh264enc") ||
                  sima::element_exists("avenc_h264");
    if (has_sw) {
      auto sw = sima::nodes::H264EncodeSW(400);
      require_contains(sw->gst_fragment(1), "name=n1_swenc", "H264EncodeSW name mismatch");
    }

    std::cout << "[OK] unit_nodes_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
