// src/gst/GstInit.cpp
#include "gst/GstInit.h"

#include <gst/gst.h>
#include <glib.h>

#include <mutex>
#include <cstring>
#include <cstdlib>

namespace sima {

namespace {

void json_log_suppressor(const gchar* domain,
                         GLogLevelFlags level,
                         const gchar* message,
                         gpointer user_data) {
  if (message &&
      (std::strstr(message, "json_object_get_array_member") ||
       std::strstr(message, "json_object_get_string_member"))) {
    return;
  }
  g_log_default_handler(domain, level, message, user_data);
}

bool env_bool(const char* key, bool def_val) {
  const char* v = std::getenv(key);
  if (!v || !*v) return def_val;
  if (!std::strcmp(v, "1") || !std::strcmp(v, "true") || !std::strcmp(v, "TRUE") ||
      !std::strcmp(v, "yes") || !std::strcmp(v, "YES") ||
      !std::strcmp(v, "on")  || !std::strcmp(v, "ON")) {
    return true;
  }
  if (!std::strcmp(v, "0") || !std::strcmp(v, "false") || !std::strcmp(v, "FALSE") ||
      !std::strcmp(v, "no") || !std::strcmp(v, "NO") ||
      !std::strcmp(v, "off") || !std::strcmp(v, "OFF")) {
    return false;
  }
  return def_val;
}

} // namespace

void gst_init_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    int argc = 0;
    char** argv = nullptr;
    gst_init(&argc, &argv);

    if (env_bool("SIMA_GST_SUPPRESS_JSON_WARNINGS", true)) {
      g_log_set_handler("Json",
                        static_cast<GLogLevelFlags>(G_LOG_LEVEL_CRITICAL | G_LOG_LEVEL_WARNING),
                        json_log_suppressor,
                        nullptr);
    }
  });
}

} // namespace sima
