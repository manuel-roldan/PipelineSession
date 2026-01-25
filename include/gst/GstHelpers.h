// include/gst/GstHelpers.h
#pragma once

namespace sima {

bool element_exists(const char* factory);

void require_element(const char* factory, const char* context);
void require_neatdecoder(const char* context);

} // namespace sima
