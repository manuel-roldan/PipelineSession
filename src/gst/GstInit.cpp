// src/gst/GstInit.cpp
#include "gst/GstInit.h"

#include <gst/gst.h>

#include <mutex>

namespace sima {

void gst_init_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    int argc = 0;
    char** argv = nullptr;
    gst_init(&argc, &argv);
  });
}

} // namespace sima
