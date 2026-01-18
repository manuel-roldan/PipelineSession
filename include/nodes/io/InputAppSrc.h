#pragma once

#include "sima/builder/Node.h"
#include "sima/builder/OutputSpec.h"

#include <memory>
#include <cstdint>
#include <string>
#include <vector>

namespace sima {

struct InputAppSrcOptions {
  std::string media_type = "video/x-raw";
  std::string format;
  int width = -1;
  int height = -1;
  int depth = -1;

  bool is_live = true;
  bool do_timestamp = true;
  bool block = false;
  int stream_type = 0; // GST_APP_STREAM_TYPE_STREAM
  std::uint64_t max_bytes = 0;

  bool use_simaai_pool = true;
  int pool_min_buffers = 5;
  int pool_max_buffers = 5;
};

class InputAppSrc final : public Node, public OutputSpecProvider {
public:
  explicit InputAppSrc(InputAppSrcOptions opt);

  std::string kind() const override { return "InputAppSrc"; }
  std::string user_label() const override { return "mysrc"; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const InputAppSrcOptions& options() const { return opt_; }
  std::string caps_string() const;

private:
  InputAppSrcOptions opt_;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> InputAppSrc(InputAppSrcOptions opt = {});
} // namespace sima::nodes
