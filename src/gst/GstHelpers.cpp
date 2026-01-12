// src/gst/GstHelpers.cpp
#include "gst/GstHelpers.h"

#include "gst/GstInit.h"

#include <gst/gst.h>

#include <sstream>
#include <stdexcept>

namespace sima {

bool element_exists(const char* factory) {
  gst_init_once();
  GstElementFactory* f = gst_element_factory_find(factory);
  if (f) {
    gst_object_unref(f);
    return true;
  }
  return false;
}

void require_element(const char* factory, const char* context) {
  if (!element_exists(factory)) {
    std::ostringstream ss;
    ss << (context ? context : "<unknown>")
       << ": required GStreamer element not found: " << factory;
    throw std::runtime_error(ss.str());
  }
}

} // namespace sima
