#pragma once

#include "sima/builder/ConfigJsonConsumer.h"
#include "sima/builder/ConfigJsonOverride.h"
#include "sima/builder/ConfigJsonProvider.h"
#include "sima/builder/Node.h"
#include "sima/builder/OutputSpec.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sima {

namespace mpk {
class ModelMPK;
} // namespace mpk
} // namespace sima

namespace sima {

struct BoxDecodeOptionsInternal;
struct BoxDecodeConfigHolder;

class SimaBoxDecode final : public Node,
                            public OutputSpecProvider,
                            public ConfigJsonProvider,
                            public ConfigJsonOverride,
                            public ConfigJsonConsumer {
public:
  explicit SimaBoxDecode(const sima::mpk::ModelMPK& model,
                         const std::string& decode_type = "",
                         int original_width = 0,
                         int original_height = 0,
                         double detection_threshold = 0.0,
                         double nms_iou_threshold = 0.0,
                         int top_k = 0);

  std::string kind() const override { return "SimaBoxDecode"; }
  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const std::string& config_path() const { return config_path_; }
  const nlohmann::json* config_json() const override;
  bool override_config_json(const std::function<void(nlohmann::json&)>& edit,
                            const std::string& tag) override;
  void apply_upstream_config(const nlohmann::json& upstream,
                             const std::string& upstream_kind) override;

private:
  std::unique_ptr<BoxDecodeOptionsInternal> opt_;
  std::shared_ptr<BoxDecodeConfigHolder> config_holder_;
  std::string config_path_;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> SimaBoxDecode(const sima::mpk::ModelMPK& model,
                                          const std::string& decode_type = "",
                                          int original_width = 0,
                                          int original_height = 0,
                                          double detection_threshold = 0.0,
                                          double nms_iou_threshold = 0.0,
                                          int top_k = 0);
} // namespace sima::nodes
