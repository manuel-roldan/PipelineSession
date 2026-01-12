#include "nodes/io/AppSrcImage.h"

#include "test_utils.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " /path/to/image.jpg\n";
      return 2;
    }

    std::string path = argv[1];
    sima::AppSrcImage node(path, 64, 64, 64, 64, 30);

    require(node.nv12_enc() && !node.nv12_enc()->empty(), "nv12_enc missing");
    auto frag = node.gst_fragment(0);
    require_contains(frag, "appsrc name=mysrc", "AppSrcImage fragment missing appsrc");
    require_contains(frag, "width=64", "AppSrcImage fragment missing width");
    require_contains(frag, "height=64", "AppSrcImage fragment missing height");

    std::cout << "[OK] unit_appsrcimage_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
