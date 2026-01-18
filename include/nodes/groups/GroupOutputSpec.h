#pragma once

#include "sima/builder/OutputSpec.h"
#include "sima/nodes/groups/ImageInputGroup.h"
#include "sima/nodes/groups/RtspInputGroup.h"
#include "sima/nodes/groups/VideoInputGroup.h"

namespace sima::nodes::groups {

OutputSpec ImageInputGroupOutputSpec(const ImageInputGroupOptions& opt);
OutputSpec RtspInputGroupOutputSpec(const RtspInputGroupOptions& opt);
OutputSpec VideoInputGroupOutputSpec(const VideoInputGroupOptions& opt);

} // namespace sima::nodes::groups
