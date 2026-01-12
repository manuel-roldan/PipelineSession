#include "nodes/groups/ImageInputGroup.h"
#include "nodes/groups/VideoInputGroup.h"
#include "nodes/groups/RtspInputGroup.h"

#include "nodes/common/Caps.h"
#include "nodes/common/DebugPoint.h"
#include "nodes/common/FileSrc.h"
#include "nodes/common/ImageDecode.h"
#include "nodes/common/ImageFreeze.h"
#include "nodes/common/QtDemuxVideoPad.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoRate.h"
#include "nodes/common/VideoScale.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/RtpH264Depay.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264Parse.h"

#include "test_utils.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void compare_groups(const sima::NodeGroup& a, const sima::NodeGroup& b) {
  const auto& an = a.nodes();
  const auto& bn = b.nodes();
  require(an.size() == bn.size(), "NodeGroup size mismatch");

  for (size_t i = 0; i < an.size(); ++i) {
    require(an[i] && bn[i], "Null node in group comparison");
    require(an[i]->kind() == bn[i]->kind(), "Node kind mismatch");
    require(an[i]->user_label() == bn[i]->user_label(), "Node label mismatch");
    require(an[i]->gst_fragment(static_cast<int>(i)) ==
                bn[i]->gst_fragment(static_cast<int>(i)),
            "gst_fragment mismatch");
    require(an[i]->element_names(static_cast<int>(i)) ==
                bn[i]->element_names(static_cast<int>(i)),
            "element_names mismatch");
  }
}

} // namespace

int main() {
  try {
    // ----------------------------
    // Image group (auto decode)
    // ----------------------------
    sima::nodes::groups::ImageInputGroupOptions io;
    io.path = "test.jpg";
    io.imagefreeze_num_buffers = 5;
    io.fps = 30;
    io.debug.enable = true;
    io.debug.decoded_name = "decoded";
    io.debug.normalized_name = "normalized";
    io.output_caps.width = 64;
    io.output_caps.height = 64;

    auto group_img = sima::nodes::groups::ImageInputGroup(io);

    std::vector<std::shared_ptr<sima::Node>> manual_img;
    manual_img.push_back(sima::nodes::FileSrc(io.path));
    manual_img.push_back(sima::nodes::ImageDecode());
    manual_img.push_back(sima::nodes::DebugPoint(io.debug.decoded_name));
    manual_img.push_back(sima::nodes::ImageFreeze(io.imagefreeze_num_buffers));
    manual_img.push_back(sima::nodes::VideoRate());
    manual_img.push_back(sima::nodes::VideoConvert());
    manual_img.push_back(sima::nodes::VideoScale());
    int img_fps = io.output_caps.fps > 0 ? io.output_caps.fps : io.fps;
    manual_img.push_back(sima::nodes::CapsRaw(io.output_caps.format,
                                               io.output_caps.width,
                                               io.output_caps.height,
                                               img_fps,
                                               io.output_caps.memory));
    manual_img.push_back(sima::nodes::DebugPoint(io.debug.normalized_name));

    compare_groups(group_img, sima::NodeGroup(std::move(manual_img)));

    // ----------------------------
    // Video group
    // ----------------------------
    sima::nodes::groups::VideoInputGroupOptions vo;
    vo.path = "video.mp4";
    vo.demux_video_pad_index = 0;
    vo.insert_queue = true;
    vo.parse_config_interval = 1;
    vo.parse_enforce_au = true;
    vo.debug.enable = true;
    vo.output_caps.enable = false;

    auto group_vid = sima::nodes::groups::VideoInputGroup(vo);

    std::vector<std::shared_ptr<sima::Node>> manual_vid;
    manual_vid.push_back(sima::nodes::FileSrc(vo.path));
    manual_vid.push_back(sima::nodes::QtDemuxVideoPad(vo.demux_video_pad_index));
    manual_vid.push_back(sima::nodes::Queue());
    manual_vid.push_back(sima::nodes::H264ParseAu(vo.parse_config_interval));
    manual_vid.push_back(sima::nodes::DebugPoint(vo.debug.encoded_name));
    manual_vid.push_back(sima::nodes::Queue());
    manual_vid.push_back(sima::nodes::H264Decode(vo.sima_allocator_type, vo.out_format));
    manual_vid.push_back(sima::nodes::DebugPoint(vo.debug.decoded_name));
    manual_vid.push_back(sima::nodes::DebugPoint(vo.debug.normalized_name));

    compare_groups(group_vid, sima::NodeGroup(std::move(manual_vid)));

    // ----------------------------
    // RTSP group
    // ----------------------------
    sima::nodes::groups::RtspInputGroupOptions ro;
    ro.url = "rtsp://example";
    ro.latency_ms = 123;
    ro.tcp = true;
    ro.payload_type = 97;
    ro.insert_queue = true;
    ro.debug.enable = true;

    auto group_rtsp = sima::nodes::groups::RtspInputGroup(ro);

    std::vector<std::shared_ptr<sima::Node>> manual_rtsp;
    manual_rtsp.push_back(sima::nodes::RTSPInput(ro.url, ro.latency_ms, ro.tcp));
    manual_rtsp.push_back(sima::nodes::Queue());
    manual_rtsp.push_back(sima::nodes::H264DepayParse(ro.payload_type));
    manual_rtsp.push_back(sima::nodes::DebugPoint(ro.debug.encoded_name));
    manual_rtsp.push_back(sima::nodes::Queue());
    manual_rtsp.push_back(sima::nodes::H264Decode(ro.sima_allocator_type, ro.out_format));
    manual_rtsp.push_back(sima::nodes::DebugPoint(ro.debug.decoded_name));
    manual_rtsp.push_back(sima::nodes::DebugPoint(ro.debug.normalized_name));

    compare_groups(group_rtsp, sima::NodeGroup(std::move(manual_rtsp)));

    std::cout << "[OK] unit_groups_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
