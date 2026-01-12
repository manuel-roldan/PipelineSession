#pragma once

#include <stdexcept>
#include <string>

inline void require(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

inline void require_contains(const std::string& haystack,
                             const std::string& needle,
                             const std::string& msg) {
  if (haystack.find(needle) == std::string::npos) {
    throw std::runtime_error(msg + " (missing: " + needle + ")");
  }
}
