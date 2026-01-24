#include "e2e_pipelines/e2e_utils.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace sima_e2e {
namespace {

std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

bool move_to_tmp(const fs::path& src, const fs::path& dst) {
  std::error_code ec;
  fs::create_directories(dst.parent_path(), ec);
  ec.clear();
  fs::rename(src, dst, ec);
  if (!ec) return true;

  ec.clear();
  fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
  if (ec) return false;
  fs::remove(src, ec);
  return true;
}

std::string resolve_yolov8s_tar_local_first(const fs::path& root_in, bool skip_download) {
  const fs::path root = root_in.empty() ? fs::current_path() : root_in;
  const fs::path tmp_tar = root / "tmp" / "yolo_v8s_mpk.tar.gz";

  const char* env = std::getenv("SIMA_YOLO_TAR");
  if (env && *env && fs::exists(env)) {
    return std::string(env);
  }

  const fs::path direct_tar = root / "yolo_v8s_mpk.tar.gz";
  if (fs::exists(direct_tar)) return direct_tar.string();

  if (fs::exists(tmp_tar)) return tmp_tar.string();

  const char* home = std::getenv("HOME");
  const fs::path home_path = home ? fs::path(home) : fs::path();
  const std::vector<fs::path> search_dirs = {
      root,
      fs::current_path(),
      root / "tmp",
      home_path / ".simaai",
      home_path / ".simaai" / "modelzoo",
      home_path / ".sima" / "modelzoo",
      "/data/simaai/modelzoo",
  };

  const std::vector<std::string> names = {
      "yolo_v8s_mpk.tar.gz",
      "yolo-v8s_mpk.tar.gz",
      "yolov8s_mpk.tar.gz",
      "yolov8_s_mpk.tar.gz",
  };

  for (const auto& dir : search_dirs) {
    if (dir.empty()) continue;
    for (const auto& name : names) {
      fs::path candidate = dir / name;
      if (fs::exists(candidate) && move_to_tmp(candidate, tmp_tar)) {
        return tmp_tar.string();
      }
    }
  }

  if (!skip_download) {
    const int rc = std::system("sima-cli modelzoo get yolo_v8s");
    if (rc == 0 && fs::exists(tmp_tar)) return tmp_tar.string();
  }

  for (const auto& dir : search_dirs) {
    if (dir.empty()) continue;
    for (const auto& name : names) {
      fs::path candidate = dir / name;
      if (fs::exists(candidate) && move_to_tmp(candidate, tmp_tar)) {
        return tmp_tar.string();
      }
    }
  }

  return "";
}

} // namespace

bool download_file(const std::string& url, const fs::path& out_path) {
  if (fs::exists(out_path)) {
    std::error_code ec;
    if (fs::file_size(out_path, ec) > 0 && !ec) return true;
  }

  std::error_code ec;
  fs::create_directories(out_path.parent_path(), ec);

  const std::string qurl = shell_quote(url);
  const std::string qout = shell_quote(out_path.string());

  std::string cmd = "curl -L --fail --silent --show-error -o " + qout + " " + qurl;
  if (std::system(cmd.c_str()) == 0) return true;

  cmd = "wget -O " + qout + " " + qurl;
  if (std::system(cmd.c_str()) == 0) return true;

  std::error_code rm_ec;
  fs::remove(out_path, rm_ec);
  return false;
}

std::string resolve_yolov8s_tar(const fs::path& root) {
  return resolve_yolov8s_tar_local_first(root, false);
}

fs::path ensure_coco_sample(const fs::path& root_in) {
  const fs::path root = root_in.empty() ? fs::current_path() : root_in;
  const char* url_env = std::getenv("SIMA_COCO_URL");
  const std::string url = (url_env && *url_env)
      ? std::string(url_env)
      : "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg";
  const fs::path out_path = root / "tmp" / "coco_sample.jpg";
  if (!download_file(url, out_path)) return {};
  return out_path;
}

} // namespace sima_e2e
