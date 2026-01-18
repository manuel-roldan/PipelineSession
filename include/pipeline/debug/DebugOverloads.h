#pragma once

#include "sima/pipeline/DebugTypes.h"

#include "sima/mpk/ModelMPK.h"
#include "sima/nodes/io/InputAppSrc.h"
#include "sima/nodes/groups/GroupOutputSpec.h"
#include "sima/nodes/groups/ImageInputGroup.h"
#include "sima/nodes/groups/ModelGroups.h"
#include "sima/nodes/groups/RtspInputGroup.h"
#include "sima/nodes/groups/VideoInputGroup.h"

namespace cv {
class Mat;
} // namespace cv

namespace sima::nodes::groups {

debug::DebugOutput ImageInputGroup(const ImageInputGroupOptions& opt,
                                   debug::OutputTag,
                                   const debug::DebugOptions& dbg = {});
debug::DebugStream ImageInputGroup(const ImageInputGroupOptions& opt,
                                   debug::StreamTag,
                                   const debug::DebugOptions& dbg = {});

debug::DebugOutput RtspInputGroup(const RtspInputGroupOptions& opt,
                                  debug::OutputTag,
                                  const debug::DebugOptions& dbg = {});
debug::DebugStream RtspInputGroup(const RtspInputGroupOptions& opt,
                                  debug::StreamTag,
                                  const debug::DebugOptions& dbg = {});

debug::DebugOutput VideoInputGroup(const VideoInputGroupOptions& opt,
                                   debug::OutputTag,
                                   const debug::DebugOptions& dbg = {});
debug::DebugStream VideoInputGroup(const VideoInputGroupOptions& opt,
                                   debug::StreamTag,
                                   const debug::DebugOptions& dbg = {});

debug::DebugOutput Infer(const debug::DebugOutput& input,
                         const sima::mpk::ModelMPK& model,
                         const debug::DebugOptions& dbg = {});
debug::DebugStream Infer(const debug::DebugStream& input,
                         const sima::mpk::ModelMPK& model,
                         const debug::DebugOptions& dbg = {});

debug::DebugOutput Preprocess(const debug::DebugOutput& input,
                              const sima::mpk::ModelMPK& model,
                              const debug::DebugOptions& dbg = {});
debug::DebugStream Preprocess(const debug::DebugStream& input,
                              const sima::mpk::ModelMPK& model,
                              const debug::DebugOptions& dbg = {});

debug::DebugOutput MLA(const debug::DebugOutput& input,
                       const sima::mpk::ModelMPK& model,
                       const debug::DebugOptions& dbg = {});
debug::DebugStream MLA(const debug::DebugStream& input,
                       const sima::mpk::ModelMPK& model,
                       const debug::DebugOptions& dbg = {});

debug::DebugOutput Postprocess(const debug::DebugOutput& input,
                               const sima::mpk::ModelMPK& model,
                               const debug::DebugOptions& dbg = {});
debug::DebugStream Postprocess(const debug::DebugStream& input,
                               const sima::mpk::ModelMPK& model,
                               const debug::DebugOptions& dbg = {});

} // namespace sima::nodes::groups

namespace sima::nodes {

debug::DebugOutput InputAppSrc(const cv::Mat& input,
                               const sima::InputAppSrcOptions& opt = {},
                               const debug::DebugOptions& dbg = {});

} // namespace sima::nodes
