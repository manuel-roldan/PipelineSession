cmake_minimum_required(VERSION 3.16)
project(PipelineSession LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(PkgConfig REQUIRED)

pkg_check_modules(GST REQUIRED
  gstreamer-1.0
  gstreamer-app-1.0
  gstreamer-video-1.0
  gstreamer-rtsp-server-1.0
  glib-2.0
)

pkg_check_modules(OPENCV REQUIRED opencv4)

add_library(pipelinesession STATIC
  src/PipelineSession.cpp
)

target_include_directories(pipelinesession
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GST_INCLUDE_DIRS}
    ${OPENCV_INCLUDE_DIRS}
)

target_compile_options(pipelinesession PRIVATE ${GST_CFLAGS_OTHER} ${OPENCV_CFLAGS_OTHER})
target_link_directories(pipelinesession PRIVATE ${GST_LIBRARY_DIRS} ${OPENCV_LIBRARY_DIRS})
target_link_libraries(pipelinesession PUBLIC ${GST_LIBRARIES} ${OPENCV_LIBRARIES})

# -------------------------
# Tests
# -------------------------
add_executable(debug_point_test tests/debug_point_test.cpp)
target_link_libraries(debug_point_test PRIVATE pipelinesession)

add_executable(file_read_test tests/file_read_test.cpp)
target_link_libraries(file_read_test PRIVATE pipelinesession)

add_executable(encoding_decoding_test tests/encoding_decoding_test.cpp)
target_link_libraries(encoding_decoding_test PRIVATE pipelinesession)

enable_testing()

# Your repo has test.jpg in the root (per your ls)
add_test(NAME debug_point_test
  COMMAND debug_point_test --image ${CMAKE_SOURCE_DIR}/test.jpg --w 256 --h 256
)

add_test(NAME file_read_test
  COMMAND file_read_test --image ${CMAKE_SOURCE_DIR}/test.jpg --w 256 --h 256
)

add_test(NAME encoding_decoding_test
  COMMAND encoding_decoding_test ${CMAKE_SOURCE_DIR}/test.jpg
)