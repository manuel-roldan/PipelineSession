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
    non.storage = storage;
    non.dtype = sima::TensorDType::UInt8;
    non.shape = {2, 2};
    non.strides_bytes = {4, 1};
    require(!non.is_contiguous(), "expected non-contiguous");
    bool threw = false;
    try {
      (void)non.clone();
    } catch (const std::exception&) {
      threw = true;
    }
    require(threw, "non-contiguous clone should throw");

    std::cout << "[OK] unit_neattensor_core_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
