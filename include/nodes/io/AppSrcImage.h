#pragma once

#include "sima/builder/Node.h"
#include "sima/builder/OutputSpec.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sima {

class AppSrcImage final : public Node, public OutputSpecProvider {
public:
  AppSrcImage(std::string image_path,
              int content_w,
              int content_h,
              int enc_w,
              int enc_h,
              int fps);

  std::string kind() const override { return "AppSrcImage"; }
  std::string user_label() const override { return image_path_; }

  std::string gst_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const std::string& image_path() const { return image_path_; }
  int content_w() const { return content_w_; }
  int content_h() const { return content_h_; }
  int enc_w() const { return enc_w_; }
  int enc_h() const { return enc_h_; }
  int fps() const { return fps_; }
  const std::shared_ptr<std::vector<uint8_t>>& nv12_enc() const { return nv12_enc_; }

private:
  std::string image_path_;
  int content_w_ = 0;
  int content_h_ = 0;
  int enc_w_ = 0;
  int enc_h_ = 0;
  int fps_ = 30;

  std::shared_ptr<std::vector<uint8_t>> nv12_enc_;
};

} // namespace sima

namespace sima::nodes {
std::shared_ptr<sima::Node> AppSrcImage(std::string image_path,
                                        int content_w,
                                        int content_h,
                                        int enc_w,
                                        int enc_h,
                                        int fps);
} // namespace sima::nodes
