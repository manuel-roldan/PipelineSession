#include "pipeline/NeatTensorCore.h"

#include "test_utils.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    auto storage = sima::make_cpu_owned_storage(16);
    require(storage && storage->size_bytes == 16, "storage size mismatch");

    sima::NeatTensor t;
    t.storage = storage;
    t.dtype = sima::TensorDType::UInt8;
    t.shape = {4, 4};
    t.strides_bytes = {4, 1};
    t.device = {sima::NeatDeviceType::CPU, 0};
    t.read_only = false;

    require(t.is_dense(), "expected dense tensor");
    require(t.is_contiguous(), "expected contiguous tensor");

    {
      auto map = t.map(sima::NeatMapMode::ReadWrite);
      require(map.data != nullptr, "map failed");
      std::memset(map.data, 0xAB, map.size_bytes);
    }

    sima::NeatTensor copy = t.clone();
    require(copy.is_contiguous(), "clone not contiguous");
    require(copy.storage && copy.storage->size_bytes == 16, "clone size mismatch");
    {
      auto map = copy.map(sima::NeatMapMode::Read);
      require(map.data != nullptr, "clone map failed");
      const uint8_t* bytes = static_cast<const uint8_t*>(map.data);
      require(bytes[0] == 0xAB, "clone data mismatch");
    }

    sima::NeatTensor non;
    auto storage_nc = sima::make_cpu_owned_storage(8);
    {
      auto map = storage_nc->map(sima::NeatMapMode::Write);
      require(map.data != nullptr, "non-contig map failed");
      uint8_t* buf = static_cast<uint8_t*>(map.data);
      buf[0] = 1;
      buf[1] = 2;
      buf[2] = 0;
      buf[3] = 0;
      buf[4] = 3;
      buf[5] = 4;
      buf[6] = 0;
      buf[7] = 0;
    }
    non.storage = storage_nc;
    non.dtype = sima::TensorDType::UInt8;
    non.shape = {2, 2};
    non.strides_bytes = {4, 1};
    non.device = {sima::NeatDeviceType::CPU, 0};
    non.read_only = false;
    require(!non.is_contiguous(), "expected non-contiguous");
    {
      sima::NeatTensor non_copy = non.clone();
      require(non_copy.is_contiguous(), "non-contiguous clone not contiguous");
      auto map = non_copy.map(sima::NeatMapMode::Read);
      require(map.data != nullptr, "non-contig clone map failed");
      const uint8_t* buf = static_cast<const uint8_t*>(map.data);
      require(buf[0] == 1 && buf[1] == 2 && buf[2] == 3 && buf[3] == 4,
              "non-contig clone data mismatch");
    }

    {
      const int w = 4;
      const int h = 2;
      auto storage_i420 = sima::make_cpu_owned_storage(12);
      auto map = storage_i420->map(sima::NeatMapMode::Write);
      require(map.data != nullptr, "i420 map failed");
      uint8_t* buf = static_cast<uint8_t*>(map.data);
      for (int i = 0; i < 8; ++i) buf[i] = static_cast<uint8_t>(i);
      buf[8] = 100;
      buf[9] = 101;
      buf[10] = 200;
      buf[11] = 201;

      sima::NeatTensor i420;
      i420.storage = storage_i420;
      i420.dtype = sima::TensorDType::UInt8;
      i420.layout = sima::TensorLayout::HW;
      i420.shape = {h, w};
      i420.device = {sima::NeatDeviceType::CPU, 0};
      i420.read_only = true;
      i420.semantic.image = sima::NeatImageSpec{sima::NeatImageSpec::PixelFormat::I420, ""};
      sima::NeatPlane y;
      y.role = sima::NeatPlaneRole::Y;
      y.shape = {h, w};
      y.strides_bytes = {w, 1};
      y.byte_offset = 0;
      sima::NeatPlane u;
      u.role = sima::NeatPlaneRole::U;
      u.shape = {h / 2, w / 2};
      u.strides_bytes = {w / 2, 1};
      u.byte_offset = 8;
      sima::NeatPlane v;
      v.role = sima::NeatPlaneRole::V;
      v.shape = {h / 2, w / 2};
      v.strides_bytes = {w / 2, 1};
      v.byte_offset = 10;
      i420.planes = {y, u, v};

      require(i420.i420_required_bytes() == 12, "i420 required bytes mismatch");
      auto mapped = i420.map_i420_read();
      require(mapped.has_value(), "i420 map_i420_read failed");
      require(mapped->view.width == w && mapped->view.height == h, "i420 dims mismatch");

      std::vector<uint8_t> out = i420.copy_i420_contiguous();
      require(out.size() == 12, "i420 copy size mismatch");
      require(out[0] == 0 && out[7] == 7, "i420 Y data mismatch");
      require(out[8] == 100 && out[9] == 101, "i420 U data mismatch");
      require(out[10] == 200 && out[11] == 201, "i420 V data mismatch");
      std::string err;
      require(i420.validate(&err), "i420 validate failed: " + err);
    }

    std::cout << "[OK] unit_neattensor_core_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
