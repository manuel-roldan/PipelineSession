#include "pipeline/NeatTensorAdapters.h"

#include "test_utils.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

#include <iostream>

int main() {
  try {
#if !defined(SIMA_WITH_OPENCV)
    std::cout << "[SKIP] unit_neattensor_cvmat_test requires SIMA_WITH_OPENCV\n";
    return 0;
#else
    cv::Mat img(4, 4, CV_8UC3);
    img.setTo(cv::Scalar(1, 2, 3));

    cv::Rect roi(1, 1, 2, 2);
    cv::Mat view = img(roi);

    sima::NeatTensor t = sima::from_cv_mat(view);
    require(t.shape.size() == 3, "shape rank mismatch");
    require(t.shape[0] == 2 && t.shape[1] == 2 && t.shape[2] == 3, "shape mismatch");
    require(!t.is_contiguous(), "ROI view should be non-contiguous");

    auto map = t.map(sima::NeatMapMode::Read);
    require(map.data != nullptr, "mapping failed");
    require(map.data == view.data, "expected zero-copy ROI mapping");

    std::cout << "[OK] unit_neattensor_cvmat_test passed\n";
    return 0;
#endif
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
