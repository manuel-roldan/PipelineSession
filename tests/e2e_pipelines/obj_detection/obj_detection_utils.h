#pragma once

#include "test_utils.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace objdet {

struct MatchResult {
  bool ok = false;
  std::string note;
};

struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

struct ExpectedBox {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  int class_id = -1;
};

struct BoxSummary {
  int count = 0;
  float min_score = 0.0f;
  float max_score = 0.0f;
};

std::string format_xyxy(float x1, float y1, float x2, float y2);
std::string format_box(const Box& b);
std::string format_expected(const ExpectedBox& b);

float box_iou_xyxy(float ax1, float ay1, float ax2, float ay2,
                   float bx1, float by1, float bx2, float by2);
float box_iou(const ExpectedBox& exp, const Box& pred);

std::vector<ExpectedBox> expected_people_boxes();

MatchResult match_expected_boxes(const std::vector<Box>& boxes,
                                 const std::vector<ExpectedBox>& expected,
                                 float min_score,
                                 float min_iou);

std::vector<Box> parse_boxes_strict(const std::vector<uint8_t>& bytes,
                                    int img_w,
                                    int img_h,
                                    int expected_topk,
                                    bool debug);

std::vector<Box> parse_boxes_lenient(const std::vector<uint8_t>& bytes,
                                     int img_w,
                                     int img_h,
                                     int expected_topk);

BoxSummary summarize_boxes(const std::vector<Box>& boxes, float min_score);

void draw_boxes(cv::Mat& img,
                const std::vector<Box>& boxes,
                float min_score,
                const cv::Scalar& color,
                const std::string& label_prefix);

void draw_expected_boxes(cv::Mat& img, const std::vector<ExpectedBox>& expected);

void save_overlay_boxes(const cv::Mat& img,
                        const std::vector<Box>& boxes,
                        const std::vector<ExpectedBox>& expected,
                        float min_score,
                        const std::filesystem::path& out_path);

} // namespace objdet
