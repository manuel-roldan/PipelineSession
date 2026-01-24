#include "pipeline/TessellatedTensor.h"

#include <cctype>

namespace sima {
namespace {

std::string upper_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

bool is_uint8_format(const std::string& fmt) {
  return upper_copy(fmt) == "UINT8";
}

} // namespace

bool is_tessellated_int8_format(const std::string& fmt) {
  if (fmt.empty()) return false;
  const std::string up = upper_copy(fmt);
  if (up == "INT8" || up == "EVXX_INT8" || up == "EV74_INT8") return true;
  if (up.find("INT8") != std::string::npos && !is_uint8_format(up)) return true;
  return false;
}

bool is_tessellated_bf16_format(const std::string& fmt) {
  if (fmt.empty()) return false;
  const std::string up = upper_copy(fmt);
  return up.find("BF16") != std::string::npos || up.find("BFLOAT16") != std::string::npos;
}

} // namespace sima
