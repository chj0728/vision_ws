cmake_minimum_required(VERSION 3.18)
project(trt_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" TRT_SYSTEM_PROCESSOR)
if(TRT_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(TRT_PLATFORM "orin-aarch64")
  set(TRT_MULTIARCH "aarch64-linux-gnu")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures")
  endif()
elseif(TRT_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  set(TRT_PLATFORM "x86_64")
  set(TRT_MULTIARCH "x86_64-linux-gnu")
else()
  message(FATAL_ERROR "Unsupported platform architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
enable_language(CUDA)
message(STATUS "${PROJECT_NAME}: platform=${TRT_PLATFORM}, CUDA architectures=${CMAKE_CUDA_ARCHITECTURES}")

find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)
find_package(SQLite3 REQUIRED)

set(TENSORRT_ROOT "/usr" CACHE PATH "TensorRT installation prefix")
find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS
    "${TENSORRT_ROOT}/include/${TRT_MULTIARCH}"
    "${TENSORRT_ROOT}/include"
    "/usr/local/TensorRT/include"
  REQUIRED
)
find_library(TENSORRT_NVINFER_LIBRARY
  NAMES nvinfer
  HINTS
    "${TENSORRT_ROOT}/lib/${TRT_MULTIARCH}"
    "${TENSORRT_ROOT}/lib"
    "/usr/local/TensorRT/lib"
    "/usr/local/TensorRT/lib64"
  REQUIRED
)
find_library(TENSORRT_PLUGIN_LIBRARY
  NAMES nvinfer_plugin
  HINTS
    "${TENSORRT_ROOT}/lib/${TRT_MULTIARCH}"
    "${TENSORRT_ROOT}/lib"
    "/usr/local/TensorRT/lib"
    "/usr/local/TensorRT/lib64"
  REQUIRED
)
find_library(UUID_LIBRARY NAMES uuid REQUIRED)

add_library(${PROJECT_NAME} SHARED
  include/arcface_trt/arcface_trt.cpp
  include/arcface_trt/face_database.cpp
  include/gender_trt/gender_trt.cpp
  include/scrfd_trt/scrfd_gpu_ops.cu
  include/scrfd_trt/scrfd_trt.cpp
  include/sixdrepnet_trt/sixdrepnet_trt.cpp
  include/yolo_trt/yolo_engine.cpp
  include/yolo_trt/yolo_gpu_preprocess.cu
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/arcface_trt>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/gender_trt>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/scrfd_trt>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/sixdrepnet_trt>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/yolo_trt>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/arcface_trt>
    $<INSTALL_INTERFACE:include/gender_trt>
    $<INSTALL_INTERFACE:include/scrfd_trt>
    $<INSTALL_INTERFACE:include/sixdrepnet_trt>
    $<INSTALL_INTERFACE:include/yolo_trt>
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    CUDA::cudart
    SQLite::SQLite3
    ${UUID_LIBRARY}
    ${TENSORRT_NVINFER_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    ${OpenCV_LIBS}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(
  DIRECTORY include/
  DESTINATION include
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(CUDAToolkit OpenCV SQLite3)
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})

ament_package()
