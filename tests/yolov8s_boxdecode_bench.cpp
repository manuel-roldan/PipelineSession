#include "pipeline/PipelineSession.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/InputAppSrc.h"
#include "nodes/common/Caps.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "mpk/ModelMPK.h"

#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool env_bool(const char* key, bool def = false) {
  const char* v = std::getenv(key);
  if (!v) return def;
  return std::string(v) != "0";
}

int env_int(const char* key, int def) {
  const char* v = std::getenv(key);
  if (!v || !*v) return def;
  return std::atoi(v);
}

int env_opt_int(const char* key) {
  const char* v = std::getenv(key);
  if (!v || !*v) return -1;
  return std::atoi(v);
}

std::string env_str(const char* key, const std::string& def) {
  const char* v = std::getenv(key);
  if (!v) return def;
  return std::string(v);
}

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

bool download_file(const std::string& url, const fs::path& out_path) {
  std::error_code ec;
  fs::create_directories(out_path.parent_path(), ec);

  const std::string qurl = shell_quote(url);
  const std::string qout = shell_quote(out_path.string());
  const int timeout_s = env_int("SIMA_COCO_DL_TIMEOUT_S", 20);
  const int retries = std::max(0, env_int("SIMA_COCO_DL_RETRIES", 2));

  std::string cmd = "curl -L --fail --silent --show-error"
                    " --connect-timeout 5"
                    " --max-time " + std::to_string(timeout_s) +
                    " --retry " + std::to_string(retries) +
                    " -o " + qout + " " + qurl;
  if (std::system(cmd.c_str()) == 0) return true;

  cmd = "wget -O " + qout +
        " --timeout=" + std::to_string(timeout_s) +
        " --tries=" + std::to_string(std::max(1, retries + 1)) +
        " " + qurl;
  if (std::system(cmd.c_str()) == 0) return true;

  fs::remove(out_path, ec);
  return false;
}

bool extract_instances_json(const fs::path& zip_path, const fs::path& out_path) {
  std::error_code ec;
  fs::create_directories(out_path.parent_path(), ec);
  const std::string qzip = shell_quote(zip_path.string());
  const std::string qout = shell_quote(out_path.string());
  const std::string cmd =
      "unzip -p " + qzip + " annotations/instances_val2017.json > " + qout;
  return std::system(cmd.c_str()) == 0;
}

std::string resolve_yolov8s_tar(const fs::path& root) {
  const fs::path tmp_tar = root / "tmp" / "yolo_v8s_mpk.tar.gz";

  const int rc = std::system("sima-cli modelzoo get yolo_v8s");
  require(rc == 0, "sima-cli modelzoo get yolo_v8s failed");

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
      if (fs::exists(candidate)) {
        std::error_code ec;
        fs::create_directories(tmp_tar.parent_path(), ec);
        fs::copy_file(candidate, tmp_tar, fs::copy_options::overwrite_existing, ec);
        if (!ec) return tmp_tar.string();
      }
    }
  }

  return "";
}

std::string find_boxdecode_config(const fs::path& etc_dir) {
  if (!fs::exists(etc_dir)) return "";
  auto contains_lower = [](const std::string& hay, const std::string& needle) {
    auto lower = [](std::string s) {
      for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      return s;
    };
    const std::string h = lower(hay);
    const std::string n = lower(needle);
    return h.find(n) != std::string::npos;
  };

  for (const auto& entry : fs::directory_iterator(etc_dir)) {
    if (!entry.is_regular_file()) continue;
    const fs::path p = entry.path();
    if (p.extension() == ".json" && contains_lower(p.filename().string(), "boxdecode")) {
      return p.string();
    }
  }

  for (const auto& entry : fs::directory_iterator(etc_dir)) {
    if (!entry.is_regular_file()) continue;
    const fs::path p = entry.path();
    if (p.extension() != ".json") continue;

    std::ifstream in(p);
    if (!in.is_open()) continue;
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (contains_lower(content, "boxdecode")) {
      return p.string();
    }
  }

  return "";
}

bool is_bbox_tensor(const sima::RunInputResult& out) {
  if (out.kind != sima::RunOutputKind::Tensor) return false;
  if (out.tensor.has_value()) return out.tensor->format == "BBOX";
  if (out.tensor_ref.has_value()) return out.tensor_ref->format == "BBOX";
  return false;
}

struct BoxdecodeConfig {
  std::string path;
};

BoxdecodeConfig prepare_boxdecode_config(const fs::path& src,
                                         const fs::path& root,
                                         int img_w,
                                         int img_h) {
  std::ifstream in(src);
  require(in.is_open(), "Failed to open boxdecode config: " + src.string());

  nlohmann::json j;
  in >> j;

  j["original_width"] = img_w;
  j["original_height"] = img_h;
  j["detection_threshold"] = 0.05;
  j["decode_type"] = "yolov8";
  if (j.contains("debug") && j["debug"].is_string()) {
    j["debug"] = false;
  }
  if (j.contains("system") && j["system"].is_object()) {
    j["system"]["debug"] = 0;
  }

  const fs::path out_path = root / "tmp" / "boxdecode_bench.json";
  std::ofstream out(out_path);
  require(out.is_open(), "Failed to write boxdecode runtime config");
  out << j.dump(2);

  BoxdecodeConfig cfg;
  cfg.path = out_path.string();
  return cfg;
}

std::vector<fs::path> collect_coco_images(const fs::path& root, size_t count) {
  const fs::path images_dir = root / "tmp" / "coco1k";
  std::error_code ec;
  fs::create_directories(images_dir, ec);

  const fs::path ann_zip = root / "tmp" / "annotations_trainval2017.zip";
  const fs::path ann_json = root / "tmp" / "instances_val2017.json";
  if (!fs::exists(ann_json)) {
    const std::string url =
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip";
    require(download_file(url, ann_zip), "Failed to download COCO annotations zip");
    require(extract_instances_json(ann_zip, ann_json),
            "Failed to extract instances_val2017.json");
  }

  std::ifstream in(ann_json);
  require(in.is_open(), "Failed to open instances_val2017.json");
  nlohmann::json j;
  in >> j;
  require(j.contains("images") && j["images"].is_array(),
          "instances_val2017.json missing images array");

  int category_id = env_opt_int("SIMA_COCO_CATEGORY_ID");
  if (category_id < 0) category_id = 1; // default to "person"
  const bool filter_by_category = (category_id > 0);

  std::unordered_set<int64_t> allowed_ids;
  if (filter_by_category) {
    if (!j.contains("annotations") || !j["annotations"].is_array()) {
      require(false, "instances_val2017.json missing annotations array");
    }
    for (const auto& ann : j["annotations"]) {
      if (!ann.contains("image_id")) continue;
      if (ann.contains("category_id")) {
        if (ann["category_id"].get<int>() != category_id) continue;
      }
      allowed_ids.insert(ann["image_id"].get<int64_t>());
    }
  }

  auto parse_coco_id = [](const fs::path& p) -> std::optional<int64_t> {
    const std::string name = p.stem().string();
    if (name.empty()) return std::nullopt;
    int64_t id = 0;
    for (char c : name) {
      if (c < '0' || c > '9') return std::nullopt;
      id = id * 10 + (c - '0');
    }
    return id;
  };

  std::vector<fs::path> cached;
  for (const auto& entry : fs::directory_iterator(images_dir)) {
    if (!entry.is_regular_file()) continue;
    const fs::path p = entry.path();
    if (p.extension() != ".jpg") continue;
    if (filter_by_category) {
      auto id = parse_coco_id(p);
      if (!id.has_value() || allowed_ids.find(*id) == allowed_ids.end()) {
        continue;
      }
    }
    cached.push_back(p);
  }
  if (cached.size() >= count) {
    std::sort(cached.begin(), cached.end());
    cached.resize(count);
    return cached;
  }

  const std::string base_url = "http://images.cocodataset.org/val2017/";
  const bool debug_log = env_bool("SIMA_COCO_DEBUG");
  size_t downloaded = cached.size();

  std::vector<fs::path> images = cached;
  images.reserve(count);

  for (const auto& item : j["images"]) {
    if (!item.contains("file_name")) continue;
    if (filter_by_category) {
      if (!item.contains("id")) continue;
      const int64_t id = item["id"].get<int64_t>();
      if (allowed_ids.find(id) == allowed_ids.end()) continue;
    }
    const std::string file = item["file_name"].get<std::string>();
    const fs::path out_path = images_dir / file;
    if (!fs::exists(out_path)) {
      const std::string url = base_url + file;
      if (!download_file(url, out_path)) continue;
      ++downloaded;
      if (debug_log && (downloaded % 50 == 0)) {
        std::cerr << "[DBG] downloaded=" << downloaded << "\n";
      }
    }
    images.push_back(out_path);
    if (images.size() >= count) break;
  }

  if (filter_by_category && images.size() < count) {
    std::cerr << "[WARN] Only found " << images.size()
              << " images for category_id=" << category_id
              << "; falling back to unfiltered COCO images.\n";
    images.clear();
    for (const auto& item : j["images"]) {
      if (!item.contains("file_name")) continue;
      const std::string file = item["file_name"].get<std::string>();
      const fs::path out_path = images_dir / file;
      if (!fs::exists(out_path)) {
        const std::string url = base_url + file;
        if (!download_file(url, out_path)) continue;
      }
      images.push_back(out_path);
      if (images.size() >= count) break;
    }
  }

  require(images.size() >= count, "Failed to download enough COCO images");
  return images;
}

double percentile(std::vector<double> values, double pct) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double idx = pct * (values.size() - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(values.size() - 1, lo + 1);
  const double t = idx - static_cast<double>(lo);
  return values[lo] * (1.0 - t) + values[hi] * t;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (!env_bool("SIMA_COCO_BENCH")) {
      std::cout << "[SKIP] SIMA_COCO_BENCH not set\n";
      return 0;
    }

    const bool timings = env_bool("SIMA_COCO_TIMINGS");
    const bool debug_log = env_bool("SIMA_COCO_DEBUG");
    const bool use_run = env_bool("SIMA_COCO_USE_RUN", false);
    const bool stage_timings = env_bool("SIMA_COCO_STAGE_TIMINGS");
    const bool fast_mode = env_bool("SIMA_COCO_FAST", false);
    const int stats_limit_env = env_int("SIMA_COCO_STATS_SAMPLES", 0);
    const size_t stats_limit =
        (stats_limit_env > 0) ? static_cast<size_t>(stats_limit_env) : 0;
    const bool stop_after_stats = env_bool("SIMA_COCO_STOP_AFTER_STATS");
    if (debug_log) {
      setenv("SIMA_INPUTSTREAM_DEBUG", "1", 1);
    }
    if (stage_timings) {
      setenv("SIMA_GST_STAGE_TIMINGS", "1", 1);
      setenv("SIMA_GST_BOUNDARY_PROBES", "1", 1);
    }
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const int bench_samples = std::max(1, env_int("SIMA_COCO_SAMPLES", 200));
    const size_t kBenchCount = static_cast<size_t>(bench_samples);
    const int warmup_env = env_int("SIMA_COCO_WARMUP", 1);
    const size_t kWarmupCount =
        std::min(kBenchCount, static_cast<size_t>(std::max(0, warmup_env)));
    const int kInputW = 640;
    const int kInputH = 640;

    std::chrono::steady_clock::time_point t_download_start{};
    if (timings) t_download_start = std::chrono::steady_clock::now();
    std::vector<fs::path> image_paths;
    const std::string fixed_image = env_str("SIMA_COCO_IMAGE_PATH", "");
    if (!fixed_image.empty()) {
      const fs::path p = fs::path(fixed_image);
      require(fs::exists(p), "SIMA_COCO_IMAGE_PATH does not exist: " + fixed_image);
      image_paths.assign(kBenchCount, p);
    } else {
      image_paths = collect_coco_images(root, kBenchCount);
    }
    std::chrono::steady_clock::time_point t_download_end{};
    if (timings) t_download_end = std::chrono::steady_clock::now();
    const std::string tar_gz = resolve_yolov8s_tar(root);
    require(!tar_gz.empty(), "Failed to locate yolo_v8s MPK tarball");

    const int max_inflight =
        std::max(1, env_int("SIMA_COCO_MAX_INFLIGHT", 8));

    sima::OutputSpec input_spec;
    input_spec.media_type = "video/x-raw";
    input_spec.format = "BGR";
    input_spec.width = kInputW;
    input_spec.height = kInputH;
    input_spec.depth = 3;
    std::chrono::steady_clock::time_point t_model_start{};
    if (timings) t_model_start = std::chrono::steady_clock::now();
    sima::mpk::ModelMPKOptions mpk_opt = sima::mpk::options_from_output_spec(input_spec);
    mpk_opt.fast_mode = fast_mode;
    mpk_opt.disable_internal_queues = true;
    const bool env_num_cvu = std::getenv("SIMA_COCO_NUM_BUFFERS_CVU") != nullptr;
    const bool env_num_mla = std::getenv("SIMA_COCO_NUM_BUFFERS_MLA") != nullptr;
    const int default_buffers = env_int("SIMA_COCO_NUM_BUFFERS_DEFAULT", 1);
    int num_cvu = env_int("SIMA_COCO_NUM_BUFFERS_CVU", default_buffers);
    int num_mla = env_int("SIMA_COCO_NUM_BUFFERS_MLA", default_buffers);
    if (use_run) {
      if (num_cvu > 1) {
        std::cerr << "[WARN] SIMA_COCO_USE_RUN=1 prefers num_buffers_cvu<=1; "
                  << "clamping " << num_cvu << " -> 1 to avoid stalls.\n";
        num_cvu = 1;
      }
      if (num_mla > 1) {
        std::cerr << "[WARN] SIMA_COCO_USE_RUN=1 prefers num_buffers_mla<=1; "
                  << "clamping " << num_mla << " -> 1 to avoid stalls.\n";
        num_mla = 1;
      }
      if (!env_num_cvu) num_cvu = 0;
      if (!env_num_mla) num_mla = 0;
      if ((env_num_cvu && num_cvu == 1) || (env_num_mla && num_mla == 1)) {
        std::cerr << "[WARN] SIMA_COCO_USE_RUN=1 using num_buffers=1 "
                  << "may still be slower than defaults; set to 0 to use plugin defaults.\n";
      }
    } else {
      if (num_cvu > 1) {
        std::cerr << "[WARN] SIMA_COCO_USE_RUN=0 with num_buffers_cvu>1 can stall; "
                  << "use 0 to rely on plugin defaults.\n";
      }
      if (num_mla > 1) {
        std::cerr << "[WARN] SIMA_COCO_USE_RUN=0 with num_buffers_mla>1 can stall; "
                  << "use 0 to rely on plugin defaults.\n";
      }
    }
    mpk_opt.num_buffers_cvu = num_cvu;
    mpk_opt.num_buffers_mla = num_mla;
    const bool env_queue_buf = std::getenv("SIMA_COCO_QUEUE_BUFFERS") != nullptr;
    const bool env_queue_time = std::getenv("SIMA_COCO_QUEUE_TIME_NS") != nullptr;
    const bool env_queue_leaky = std::getenv("SIMA_COCO_QUEUE_LEAKY") != nullptr;
    if (use_run) {
      if (env_queue_buf) {
        std::cerr << "[WARN] SIMA_COCO_USE_RUN=1 ignoring SIMA_COCO_QUEUE_BUFFERS "
                  << "to avoid stalls in sync run(input).\n";
      }
      if (env_queue_time) {
        std::cerr << "[WARN] SIMA_COCO_USE_RUN=1 ignoring SIMA_COCO_QUEUE_TIME_NS "
                  << "to avoid stalls in sync run(input).\n";
      }
      if (env_queue_leaky) {
        std::cerr << "[WARN] SIMA_COCO_USE_RUN=1 ignoring SIMA_COCO_QUEUE_LEAKY "
                  << "to avoid stalls in sync run(input).\n";
      }
      mpk_opt.queue_max_buffers = 0;
      mpk_opt.queue_max_time_ns = -1;
      mpk_opt.queue_leaky.clear();
    } else {
      mpk_opt.queue_max_buffers = env_int("SIMA_COCO_QUEUE_BUFFERS", 16);
      mpk_opt.queue_max_time_ns = static_cast<int64_t>(env_int("SIMA_COCO_QUEUE_TIME_NS", 0));
      mpk_opt.queue_leaky = env_str("SIMA_COCO_QUEUE_LEAKY", "no");
    }
    auto model = sima::mpk::ModelMPK::load(tar_gz, mpk_opt);
    std::chrono::steady_clock::time_point t_model_end{};
    if (timings) t_model_end = std::chrono::steady_clock::now();

    const std::string config_path = find_boxdecode_config(model.etc_dir());
    require(!config_path.empty(), "Failed to locate simaaiboxdecode config JSON");
    std::chrono::steady_clock::time_point t_cfg_start{};
    if (timings) t_cfg_start = std::chrono::steady_clock::now();
    const BoxdecodeConfig runtime_config =
        prepare_boxdecode_config(config_path, root, kInputW, kInputH);
    std::chrono::steady_clock::time_point t_cfg_end{};
    if (timings) t_cfg_end = std::chrono::steady_clock::now();

    std::chrono::steady_clock::time_point t_load_start{};
    if (timings) t_load_start = std::chrono::steady_clock::now();
    std::vector<cv::Mat> inputs;
    inputs.reserve(kBenchCount);
    std::unordered_map<std::string, cv::Mat> cache;
    for (const auto& path : image_paths) {
      const std::string key = path.string();
      cv::Mat img;
      auto it = cache.find(key);
      if (it != cache.end()) {
        img = it->second;
      } else {
        img = cv::imread(key, cv::IMREAD_COLOR);
        require(!img.empty(), "Failed to read image: " + key);
        if (img.cols != kInputW || img.rows != kInputH) {
          cv::resize(img, img, cv::Size(kInputW, kInputH), 0.0, 0.0, cv::INTER_LINEAR);
        }
        if (!img.isContinuous()) img = img.clone();
        cache.emplace(key, img);
      }
      inputs.push_back(std::move(img));
    }
    std::chrono::steady_clock::time_point t_load_end{};
    if (timings) t_load_end = std::chrono::steady_clock::now();

    sima::PipelineSession p;
    sima::InputAppSrcOptions src_opt;
    src_opt.format = "BGR";
    src_opt.width = kInputW;
    src_opt.height = kInputH;
    src_opt.is_live = true;
    src_opt.do_timestamp = true;
    src_opt.block = false;
    src_opt.use_simaai_pool = true;
    const size_t bytes_per_frame = inputs.front().total() * inputs.front().elemSize();
    if (use_run) {
      src_opt.max_bytes = 0;
    } else {
      // Bound appsrc by expected in-flight depth to avoid runaway buffering.
      src_opt.max_bytes = static_cast<std::uint64_t>(bytes_per_frame * max_inflight);
    }

    const int is_live_env = env_opt_int("SIMA_COCO_IS_LIVE");
    if (is_live_env >= 0) src_opt.is_live = (is_live_env != 0);
    const int do_ts_env = env_opt_int("SIMA_COCO_DO_TIMESTAMP");
    if (do_ts_env >= 0) src_opt.do_timestamp = (do_ts_env != 0);
    const int block_env = env_opt_int("SIMA_COCO_BLOCK");
    if (block_env >= 0) src_opt.block = (block_env != 0);
    const int pool_env = env_opt_int("SIMA_COCO_USE_POOL");
    if (pool_env >= 0) src_opt.use_simaai_pool = (pool_env != 0);
    const int max_bytes_env = env_opt_int("SIMA_COCO_MAX_BYTES");
    if (max_bytes_env >= 0) {
      src_opt.max_bytes = static_cast<std::uint64_t>(max_bytes_env);
    }
    if (!use_run) {
      const int pool_min_env = env_opt_int("SIMA_COCO_POOL_MIN");
      const int pool_max_env = env_opt_int("SIMA_COCO_POOL_MAX");
      src_opt.pool_min_buffers =
          (pool_min_env >= 0) ? pool_min_env : max_inflight;
      src_opt.pool_max_buffers =
          (pool_max_env >= 0) ? pool_max_env : max_inflight;
    }

    p.add(sima::nodes::InputAppSrc(src_opt));
    p.add(sima::nodes::groups::Preprocess(model));
    const bool enable_stage_queues = env_bool("SIMA_COCO_STAGE_QUEUES", false);
    const int stage_queue_depth =
        std::max(1, env_int("SIMA_COCO_STAGE_QUEUE_SIZE", max_inflight));
    auto make_queue = [&](const std::string& name) {
      const std::string frag = "queue name=" + name +
          " max-size-buffers=" + std::to_string(stage_queue_depth) +
          " max-size-bytes=0 max-size-time=0 leaky=no silent=true";
      return sima::nodes::Gst(frag);
    };
    if (enable_stage_queues) {
      p.add(make_queue("q_preproc"));
    }
    p.add(sima::nodes::groups::MLA(model));
    if (enable_stage_queues) {
      p.add(make_queue("q_mla"));
    }
    sima::SimaBoxDecodeOptions box_opt;
    box_opt.config_path = runtime_config.path;
    p.add(sima::nodes::SimaBoxDecode(box_opt));
    const bool add_sink_queue = env_bool("SIMA_COCO_SINK_QUEUE", true);
    if (!use_run && add_sink_queue) {
      const int sink_queue_depth =
          std::max(1, env_int("SIMA_COCO_SINK_QUEUE_SIZE", max_inflight));
      const std::string frag = "queue name=q_sink"
          " max-size-buffers=" + std::to_string(sink_queue_depth) +
          " max-size-bytes=0 max-size-time=0 leaky=no silent=true";
      p.add(sima::nodes::Gst(frag));
    }
    sima::OutputAppSinkOptions sink_opt;
    const int default_sink_buffers =
        env_int("SIMA_COCO_SINK_BUFFERS_DEFAULT", 1);
    sink_opt.sync = env_bool("SIMA_COCO_APPSINK_SYNC", true);
    sink_opt.drop = env_bool("SIMA_COCO_APPSINK_DROP", false);
    sink_opt.max_buffers =
        env_int("SIMA_COCO_SINK_BUFFERS", default_sink_buffers);
    p.add(sima::nodes::OutputAppSink(sink_opt));

    sima::RunInputOptions run_opt;
    run_opt.copy_output = env_bool("SIMA_COCO_COPY_OUTPUT", false);
    run_opt.reuse_input_buffer = env_bool("SIMA_COCO_REUSE_INPUT_BUFFER", false);
    run_opt.strict = env_bool("SIMA_COCO_STRICT", true);

    const size_t warmup = std::min(kWarmupCount, kBenchCount);
    const size_t total_frames = warmup + kBenchCount;
    const bool sync_pull = env_bool("SIMA_COCO_SYNC_PULL");

    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::chrono::steady_clock::time_point> sent;
    std::vector<double> push_ms;
    std::vector<double> lat_ms;
    lat_ms.reserve(kBenchCount);
    push_ms.reserve(kBenchCount);
    std::atomic<size_t> received{0};
    std::atomic<bool> stop_requested{false};
    std::atomic<bool> stats_done{false};
    std::string callback_error;
    std::chrono::steady_clock::time_point start_time{};
    std::chrono::steady_clock::time_point end_time{};
    std::chrono::steady_clock::time_point stats_end_time{};
    std::chrono::steady_clock::time_point first_output{};
    const int log_every = std::max(1, env_int("SIMA_COCO_LOG_EVERY", 50));

    sima::InputStream stream;
    sima::InputStreamOptions stream_opt;
    std::chrono::steady_clock::time_point t_pipe_start{};
    std::chrono::steady_clock::time_point t_pipe_end{};
    if (!use_run) {
      stream_opt.timeout_ms = env_int("SIMA_COCO_TIMEOUT_MS", 10000);
      stream_opt.appsink_sync = env_bool("SIMA_COCO_APPSINK_SYNC", true);
      stream_opt.appsink_drop = env_bool("SIMA_COCO_APPSINK_DROP", false);
      stream_opt.appsink_max_buffers =
          env_int("SIMA_COCO_SINK_BUFFERS",
                  env_int("SIMA_COCO_SINK_BUFFERS_DEFAULT", 1));
      stream_opt.copy_output = env_bool("SIMA_COCO_COPY_OUTPUT", false);
      stream_opt.reuse_input_buffer = env_bool("SIMA_COCO_REUSE_INPUT_BUFFER", true);
      const int sink_sync_env = env_opt_int("SIMA_COCO_APPSINK_SYNC");
      if (sink_sync_env >= 0) stream_opt.appsink_sync = (sink_sync_env != 0);
      stream_opt.enable_timings = timings;
      stream_opt.preflight_run = env_bool("SIMA_COCO_PREFLIGHT_RUN");

      if (timings) t_pipe_start = std::chrono::steady_clock::now();
      stream = p.run_input_stream(inputs.front(), stream_opt);
      if (timings) t_pipe_end = std::chrono::steady_clock::now();
    }

    const fs::path gst_path = root / "tmp" / "coco_bench_pipeline.gst";
    {
      std::ofstream gst_out(gst_path);
      if (gst_out.is_open()) {
        gst_out << p.to_gst(false) << "\n";
      }
    }
    if (debug_log) {
      std::cerr << "[DBG] gst_pipeline_path=" << gst_path << "\n";
      std::cerr << "[DBG] appsrc block=" << (src_opt.block ? "true" : "false")
                << " is_live=" << (src_opt.is_live ? "true" : "false")
                << " do_timestamp=" << (src_opt.do_timestamp ? "true" : "false")
                << " max_bytes=" << src_opt.max_bytes
                << " sima_pool=" << (src_opt.use_simaai_pool ? "true" : "false")
                << " pool_min=" << src_opt.pool_min_buffers
                << " pool_max=" << src_opt.pool_max_buffers << "\n";
      std::cerr << "[DBG] bench_mode=" << (use_run ? "run" : "stream") << "\n";
      if (!use_run) {
        std::cerr << "[DBG] appsink sync=" << (stream_opt.appsink_sync ? "true" : "false")
                  << " drop=" << (stream_opt.appsink_drop ? "true" : "false")
                  << " max_buffers=" << stream_opt.appsink_max_buffers << "\n";
      }
      std::cerr << "[DBG] num_buffers_cvu=" << mpk_opt.num_buffers_cvu
                << " num_buffers_mla=" << mpk_opt.num_buffers_mla
                << " queue_max_buffers=" << mpk_opt.queue_max_buffers
                << " queue_max_time_ns=" << mpk_opt.queue_max_time_ns
                << " queue_leaky=" << mpk_opt.queue_leaky << "\n";
    }

    if (use_run) {
      for (size_t i = 0; i < total_frames; ++i) {
        const auto t0 = timings ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
        if (i == warmup && timings) start_time = t0;
        const size_t idx = (i < kBenchCount) ? i : (i % kBenchCount);
        sima::RunInputResult out;
        try {
          out = p.run(inputs[idx], run_opt);
        } catch (const std::exception& e) {
          callback_error = e.what();
          break;
        }
        if (!is_bbox_tensor(out)) {
          callback_error = "Unexpected output in run() bench";
          break;
        }
        const auto t1 = timings ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
        const size_t r = ++received;
        if (r == 1 && timings) first_output = t1;
        if (r > warmup && timings) {
          const std::chrono::duration<double, std::milli> dt = t1 - t0;
          lat_ms.push_back(dt.count());
          end_time = t1;
          if (stats_limit > 0 && lat_ms.size() == stats_limit) {
            stats_end_time = t1;
            stats_done.store(true);
            if (stop_after_stats) break;
          }
        }
      }
    } else if (sync_pull) {
      for (size_t i = 0; i < total_frames; ++i) {
        const auto t0 = timings ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
        if (i == warmup && timings) start_time = t0;
        const size_t idx = (i < kBenchCount) ? i : (i % kBenchCount);
        sima::RunInputResult out;
        try {
          out = stream.push_and_pull(inputs[idx], stream_opt.timeout_ms);
        } catch (const std::exception& e) {
          callback_error = e.what();
          break;
        }
        if (!is_bbox_tensor(out)) {
          callback_error = "Unexpected output in sync bench";
          break;
        }
        const auto t1 = timings ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
        const size_t r = ++received;
        if (r == 1 && timings) first_output = t1;
        if (r > warmup && timings) {
          const std::chrono::duration<double, std::milli> dt = t1 - t0;
          lat_ms.push_back(dt.count());
          end_time = t1;
          if (stats_limit > 0 && lat_ms.size() == stats_limit) {
            stats_end_time = t1;
            stats_done.store(true);
            if (stop_after_stats) break;
          }
        }
      }
    } else {
      stream.start([&](sima::RunInputResult out) {
        const auto now = timings ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
        std::unique_lock<std::mutex> lock(mu);
        if (!is_bbox_tensor(out)) {
          callback_error = "Unexpected output in benchmark stream";
          stop_requested.store(true);
          lock.unlock();
          cv.notify_one();
          return;
        }
        if (sent.empty()) {
          callback_error = "Output received without matching input";
          stop_requested.store(true);
          lock.unlock();
          cv.notify_one();
          return;
        }
        const auto t0 = sent.front();
        sent.pop_front();

        const size_t r = ++received;
        if (r == 1 && timings) {
          first_output = now;
        }
        if (r > warmup && timings) {
          const std::chrono::duration<double, std::milli> dt = now - t0;
          lat_ms.push_back(dt.count());
          end_time = now;
          if (stats_limit > 0 && lat_ms.size() == stats_limit) {
            stats_end_time = now;
            stats_done.store(true);
            if (stop_after_stats) {
              stop_requested.store(true);
              lock.unlock();
              cv.notify_one();
              return;
            }
          }
        }
        if (debug_log && (r % static_cast<size_t>(log_every) == 0)) {
          std::cerr << "[DBG] received=" << r
                    << " in_flight=" << sent.size();
          if (timings && !lat_ms.empty()) {
            std::cerr << " lat_ms=" << lat_ms.back();
          }
          std::cerr << "\n";
        }
        lock.unlock();
        if (r == total_frames) {
          cv.notify_one();
        }
      });

    std::thread producer([&]() {
        const int retry_ms = std::max(1, env_int("SIMA_COCO_PUSH_RETRY_MS", 1000));
        for (size_t i = 0; i < total_frames && !stop_requested.load(); ++i) {
          const auto t0 = timings ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
          {
            std::lock_guard<std::mutex> lock(mu);
            sent.push_back(t0);
            if (i == warmup && timings) start_time = t0;
          }
          const size_t idx = (i < kBenchCount) ? i : (i % kBenchCount);
          const auto deadline = std::chrono::steady_clock::now() +
              std::chrono::milliseconds(retry_ms);
          while (!stop_requested.load()) {
            if (stream.try_push(inputs[idx])) break;
            if (std::chrono::steady_clock::now() > deadline) {
              std::lock_guard<std::mutex> lock(mu);
              callback_error = "Input push stalled";
              stop_requested.store(true);
              cv.notify_one();
              break;
            }
            // Minimal backoff to avoid burning CPU if downstream is momentarily full.
            std::this_thread::sleep_for(std::chrono::microseconds(50));
          }
          const auto t1 = timings ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
          if (timings && i >= warmup && i < total_frames) {
            const std::chrono::duration<double, std::milli> dt = t1 - t0;
            std::lock_guard<std::mutex> lock(mu);
            push_ms.push_back(dt.count());
          }
        }
      });

      const int global_timeout_s = env_int("SIMA_COCO_GLOBAL_TIMEOUT_S", 120);
      const int stall_timeout_s = env_int("SIMA_COCO_STALL_TIMEOUT_S", 15);
      {
        std::unique_lock<std::mutex> lock(mu);
        auto last_progress = std::chrono::steady_clock::now();
        size_t last_count = 0;
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(global_timeout_s);

        while (received.load() < total_frames && callback_error.empty()) {
          cv.wait_for(lock, std::chrono::seconds(1));
          const size_t curr = received.load();
          if (stop_after_stats && stats_done.load()) {
            break;
          }
          if (curr != last_count) {
            last_count = curr;
            last_progress = std::chrono::steady_clock::now();
          }
          const auto now = std::chrono::steady_clock::now();
          if (now - last_progress > std::chrono::seconds(stall_timeout_s)) {
            callback_error = "Benchmark stalled waiting for outputs";
            stop_requested.store(true);
            break;
          }
          if (now > deadline) {
            callback_error = "Benchmark timed out waiting for outputs";
            stop_requested.store(true);
            break;
          }
        }
      }
      producer.join();
    }
    if (stream) {
      stream.stop();
    }

    if (!callback_error.empty() && stream) {
      const std::string diag = stream.diagnostics_summary();
      if (!diag.empty()) {
        std::cerr << "[DBG] diagnostics\n" << diag << "\n";
      }
    }
    if (use_run && env_bool("SIMA_RUN_INPUT_DIAG")) {
      const std::string diag = p.last_input_diagnostics();
      if (!diag.empty()) {
        std::cerr << "[DBG] diagnostics\n" << diag << "\n";
      }
    }
    if (stage_timings) {
      std::string diag;
      if (use_run) {
        diag = p.last_input_diagnostics();
      } else if (stream) {
        diag = stream.diagnostics_summary();
      }
      if (!diag.empty()) {
        std::cout << "[STAGE] diagnostics\n" << diag << "\n";
      }
    }

    double total_ms = 0.0;
    double mean_ms = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double mean_push_ms = 0.0;
    double throughput = 0.0;
    if (timings) {
      const size_t stats_count =
          (stats_limit > 0) ? std::min(stats_limit, lat_ms.size()) : lat_ms.size();
      const auto stats_end =
          (stats_limit > 0 && stats_end_time != std::chrono::steady_clock::time_point{})
              ? stats_end_time
              : end_time;
      total_ms = std::chrono::duration<double, std::milli>(stats_end - start_time).count();
      if (stats_count > 0) {
        const auto begin = lat_ms.begin();
        const auto end = lat_ms.begin() + static_cast<std::ptrdiff_t>(stats_count);
        const double sum = std::accumulate(begin, end, 0.0);
        mean_ms = sum / static_cast<double>(stats_count);
        std::vector<double> subset(begin, end);
        p50 = percentile(subset, 0.50);
        p95 = percentile(subset, 0.95);
      }
      const size_t push_count =
          (stats_limit > 0) ? std::min(stats_limit, push_ms.size()) : push_ms.size();
      if (push_count > 0) {
        mean_push_ms = std::accumulate(push_ms.begin(),
                                       push_ms.begin() + static_cast<std::ptrdiff_t>(push_count),
                                       0.0) /
                       static_cast<double>(push_count);
      }
      throughput = (stats_count == 0 || total_ms <= 0.0)
          ? 0.0
          : (static_cast<double>(stats_count) * 1000.0 / total_ms);
    }

    sima::InputStreamStats stats;
    if (use_run) {
      stats = p.last_input_stats();
    } else if (stream) {
      stats = stream.stats();
    }

    double dl_ms = 0.0;
    double model_ms = 0.0;
    double load_ms = 0.0;
    double pipe_ms = 0.0;
    double cfg_ms = 0.0;
    double warmup_first_ms = 0.0;
    if (timings) {
      dl_ms = std::chrono::duration<double, std::milli>(t_download_end - t_download_start).count();
      model_ms = std::chrono::duration<double, std::milli>(t_model_end - t_model_start).count();
      load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
      pipe_ms = std::chrono::duration<double, std::milli>(t_pipe_end - t_pipe_start).count();
      cfg_ms = std::chrono::duration<double, std::milli>(t_cfg_end - t_cfg_start).count();
      if (first_output != std::chrono::steady_clock::time_point{} &&
          start_time != std::chrono::steady_clock::time_point{}) {
        warmup_first_ms =
            std::chrono::duration<double, std::milli>(first_output - start_time).count();
      }
    }

    if (!callback_error.empty() && timings) {
      std::cerr << "[DBG] timing_summary"
                << " download_ms=" << dl_ms
                << " model_ms=" << model_ms
                << " config_ms=" << cfg_ms
                << " load_ms=" << load_ms
                << " pipeline_setup_ms=" << pipe_ms
                << " warmup_first_ms=" << warmup_first_ms
                << " mean_ms=" << mean_ms
                << " throughput_fps=" << throughput
                << " push_count=" << stats.push_count
                << " pull_count=" << stats.pull_count
                << " avg_alloc_us=" << stats.avg_alloc_us
                << " avg_map_us=" << stats.avg_map_us
                << " avg_copy_us=" << stats.avg_copy_us
                << " avg_push_us=" << stats.avg_push_us
                << " avg_pull_wait_us=" << stats.avg_pull_wait_us
                << " avg_decode_us=" << stats.avg_decode_us << "\n";
    }

    require(callback_error.empty(), callback_error);
    if (stream) {
      const std::string stream_err = stream.last_error();
      require(stream_err.empty(), "Stream error: " + stream_err);
    }
    if (timings) {
      const size_t stats_count =
          (stats_limit > 0) ? std::min(stats_limit, lat_ms.size()) : lat_ms.size();
      require(stats_count > 0, "Benchmark did not receive outputs");
      if (!stop_after_stats) {
        require(lat_ms.size() == kBenchCount, "Benchmark did not receive all outputs");
      }
    }

    const fs::path csv_path = root / "tmp" / "coco_bench.csv";
    if (timings) {
      std::ofstream csv(csv_path);
      require(csv.is_open(), "Failed to write benchmark CSV");
      csv << "index,lat_ms,push_ms\n";
      for (size_t i = 0; i < lat_ms.size(); ++i) {
        const double p = (i < push_ms.size()) ? push_ms[i] : 0.0;
        csv << i << "," << lat_ms[i] << "," << p << "\n";
      }
    }

    std::cout << "[OK] COCO bench\n"
              << "  samples=" << kBenchCount << " warmup=" << warmup;
    if (stats_limit > 0) {
      std::cout << " stats=" << stats_limit;
    }
    std::cout << "\n"
              << "  timings=" << (timings ? "on" : "off") << "\n";
    if (timings) {
      std::cout << "  download_ms=" << dl_ms
                << " model_ms=" << model_ms
                << " config_ms=" << cfg_ms
                << " load_ms=" << load_ms
                << " pipeline_setup_ms=" << pipe_ms
                << " warmup_first_ms=" << warmup_first_ms << "\n"
                << "  mean_ms=" << mean_ms
                << " p50_ms=" << p50
                << " p95_ms=" << p95
                << " mean_push_ms=" << mean_push_ms
                << " throughput_fps=" << throughput << "\n"
                << "  stream_stats"
                << " push=" << stats.push_count
                << " push_fail=" << stats.push_failures
                << " pull=" << stats.pull_count
                << " polls=" << stats.poll_count
                << " avg_alloc_us=" << stats.avg_alloc_us
                << " avg_map_us=" << stats.avg_map_us
                << " avg_copy_us=" << stats.avg_copy_us
                << " avg_push_us=" << stats.avg_push_us
                << " avg_pull_wait_us=" << stats.avg_pull_wait_us
                << " avg_decode_us=" << stats.avg_decode_us << "\n"
                << "  csv=" << csv_path << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
