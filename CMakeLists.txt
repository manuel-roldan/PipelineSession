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

find_library(SIMAAI_ALLOCATOR_LIB gstsimaallocator)
find_library(SIMAAI_BUFFERPOOL_LIB gstsimaaibufferpool)

add_library(pipelinesession STATIC
  src/pipeline/PipelineSession.cpp
  src/pipeline/Errors.cpp
  src/pipeline/PipelineReport.cpp
  src/pipeline/TapStream.cpp
  src/pipeline/FrameStream.cpp
  src/pipeline/TensorTypes.cpp
  src/pipeline/TensorStream.cpp
  src/pipeline/TensorUtil.cpp
  src/pipeline/internal/GstDiagnosticsUtil.cpp
  src/pipeline/internal/SimaaiGuard.cpp
  src/mpk/ModelMPK.cpp
  src/ModelSession.cpp
  src/gst/GstBusWatch.cpp
  src/gst/GstInit.cpp
  src/gst/GstHelpers.cpp
  src/gst/GstParseLaunch.cpp
  src/gst/GstPadTap.cpp
  src/gst/GstIntrospection.cpp
  src/nodes/common/AppSink.cpp
  src/nodes/common/Caps.cpp
  src/nodes/common/DebugPoint.cpp
  src/nodes/common/FileSrc.cpp
  src/nodes/common/JpegDec.cpp
  src/nodes/common/QtDemuxVideoPad.cpp
  src/nodes/common/Queue.cpp
  src/nodes/common/VideoConvert.cpp
  src/nodes/common/VideoScale.cpp
  src/nodes/common/VideoRate.cpp
  src/nodes/common/ImageFreeze.cpp
  src/nodes/common/ImageDecode.cpp
  src/nodes/io/RTSPInput.cpp
  src/nodes/io/AppSrcImage.cpp
  src/nodes/io/InputAppSrc.cpp
  src/nodes/rtp/RtpH264Depay.cpp
  src/nodes/sima/H264DecodeSima.cpp
  src/nodes/sima/H264EncodeSima.cpp
  src/nodes/sima/H264Parse.cpp
  src/nodes/sima/RtpH264Pay.cpp
  src/nodes/groups/ImageInputGroup.cpp
  src/nodes/groups/VideoInputGroup.cpp
  src/nodes/groups/RtspInputGroup.cpp
  src/nodes/groups/ModelGroups.cpp
)

target_include_directories(pipelinesession
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pipeline
    ${GST_INCLUDE_DIRS}
    ${OPENCV_INCLUDE_DIRS}
)

target_compile_options(pipelinesession PRIVATE ${GST_CFLAGS_OTHER} ${OPENCV_CFLAGS_OTHER})
target_link_directories(pipelinesession PRIVATE ${GST_LIBRARY_DIRS} ${OPENCV_LIBRARY_DIRS})
target_link_libraries(pipelinesession PUBLIC ${GST_LIBRARIES} ${OPENCV_LIBRARIES})

if (SIMAAI_ALLOCATOR_LIB)
  target_link_libraries(pipelinesession PUBLIC ${SIMAAI_ALLOCATOR_LIB})
endif()
if (SIMAAI_BUFFERPOOL_LIB)
  target_link_libraries(pipelinesession PUBLIC ${SIMAAI_BUFFERPOOL_LIB})
endif()

# -------------------------
# Examples
# -------------------------
add_executable(modelsession_resnet50_example
  examples/modelsession_resnet50.cpp
  examples/example_utils.cpp
)
target_link_libraries(modelsession_resnet50_example PRIVATE pipelinesession)

add_executable(pipelinesession_resnet50_example
  examples/pipelinesession_resnet50.cpp
  examples/example_utils.cpp
)
target_link_libraries(pipelinesession_resnet50_example PRIVATE pipelinesession)

# -------------------------
# Tests
# -------------------------
add_executable(debug_point_test tests/debug_point_test.cpp)
target_link_libraries(debug_point_test PRIVATE pipelinesession)

add_executable(file_read_test tests/file_read_test.cpp)
target_link_libraries(file_read_test PRIVATE pipelinesession)

add_executable(encoding_decoding_test tests/encoding_decoding_test.cpp)
target_link_libraries(encoding_decoding_test PRIVATE pipelinesession)

add_executable(unit_builder_test tests/unit_builder_test.cpp)
target_link_libraries(unit_builder_test PRIVATE pipelinesession)

add_executable(unit_contracts_test tests/unit_contracts_test.cpp)
target_link_libraries(unit_contracts_test PRIVATE pipelinesession)

add_executable(unit_nodes_test tests/unit_nodes_test.cpp)
target_link_libraries(unit_nodes_test PRIVATE pipelinesession)

add_executable(unit_gst_helpers_test tests/unit_gst_helpers_test.cpp)
target_link_libraries(unit_gst_helpers_test PRIVATE pipelinesession)

add_executable(unit_framestream_test tests/unit_framestream_test.cpp)
target_link_libraries(unit_framestream_test PRIVATE pipelinesession)

add_executable(unit_tapstream_test tests/unit_tapstream_test.cpp)
target_link_libraries(unit_tapstream_test PRIVATE pipelinesession)

add_executable(run_debug_test tests/run_debug_test.cpp)
target_link_libraries(run_debug_test PRIVATE pipelinesession)

add_executable(unit_appsrcimage_test tests/unit_appsrcimage_test.cpp)
target_link_libraries(unit_appsrcimage_test PRIVATE pipelinesession)

add_executable(unit_groups_test tests/unit_groups_test.cpp)
target_link_libraries(unit_groups_test PRIVATE pipelinesession)

add_executable(unit_image_group_runtime_test tests/unit_image_group_runtime_test.cpp)
target_link_libraries(unit_image_group_runtime_test PRIVATE pipelinesession)

add_executable(unit_tensorstream_test tests/unit_tensorstream_test.cpp)
target_link_libraries(unit_tensorstream_test PRIVATE pipelinesession)

add_executable(modelsession_resnet50_guard_test tests/modelsession_resnet50_guard_test.cpp)
target_link_libraries(modelsession_resnet50_guard_test PRIVATE pipelinesession)

add_executable(unit_pipeline_io_test tests/unit_pipeline_io_test.cpp)
target_link_libraries(unit_pipeline_io_test PRIVATE pipelinesession)

add_executable(resnet50_modelpack_test tests/resnet50_modelpack_test.cpp)
target_link_libraries(resnet50_modelpack_test PRIVATE pipelinesession)

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

add_test(NAME unit_builder_test
  COMMAND unit_builder_test
)

add_test(NAME unit_contracts_test
  COMMAND unit_contracts_test
)

add_test(NAME unit_nodes_test
  COMMAND unit_nodes_test
)

add_test(NAME unit_gst_helpers_test
  COMMAND unit_gst_helpers_test
)

add_test(NAME unit_framestream_test
  COMMAND unit_framestream_test
)

add_test(NAME unit_tapstream_test
  COMMAND unit_tapstream_test
)

add_test(NAME run_debug_test
  COMMAND run_debug_test
)

add_test(NAME unit_appsrcimage_test
  COMMAND unit_appsrcimage_test ${CMAKE_SOURCE_DIR}/test.jpg
)

add_test(NAME unit_groups_test
  COMMAND unit_groups_test
)

add_test(NAME unit_image_group_runtime_test
  COMMAND unit_image_group_runtime_test ${CMAKE_SOURCE_DIR}/test.jpg
)

add_test(NAME unit_tensorstream_test
  COMMAND unit_tensorstream_test ${CMAKE_SOURCE_DIR}/test.jpg
)

add_test(NAME modelsession_resnet50_guard_test
  COMMAND modelsession_resnet50_guard_test
)

add_test(NAME unit_pipeline_io_test
  COMMAND unit_pipeline_io_test ${CMAKE_SOURCE_DIR}/test.jpg
)

add_test(NAME resnet50_modelpack_direct
  COMMAND resnet50_modelpack_test --goldfish --only-direct
)

add_test(NAME resnet50_modelpack_jpeg
  COMMAND resnet50_modelpack_test --goldfish --only-jpeg
)

add_test(NAME resnet50_modelpack_rtsp
  COMMAND resnet50_modelpack_test --goldfish --only-rtsp
)
