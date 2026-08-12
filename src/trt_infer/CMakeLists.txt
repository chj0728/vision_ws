cmake_minimum_required(VERSION 3.18)
project(trt_infer LANGUAGES CXX)

# 基础编译标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找完整 CUDA Toolkit，避免误用缺少 cicc 的独立 nvcc
set(TRT_CUDA_ROOT "" CACHE PATH "CUDA Toolkit root containing bin/nvcc and nvvm/bin/cicc")
if(NOT TRT_CUDA_ROOT)
  set(_trt_cuda_candidates)

  if(CMAKE_CUDA_COMPILER)
    get_filename_component(_trt_cuda_bin_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(_trt_cuda_compiler_root "${_trt_cuda_bin_dir}" DIRECTORY)
    list(APPEND _trt_cuda_candidates "${_trt_cuda_compiler_root}")
  endif()

  file(GLOB _trt_cuda_versioned_roots LIST_DIRECTORIES true "/usr/local/cuda-*")
  list(SORT _trt_cuda_versioned_roots COMPARE NATURAL ORDER DESCENDING)
  set(_trt_cuda_candidates
    ${_trt_cuda_candidates}
    "$ENV{CUDA_HOME}"
    "$ENV{CUDA_PATH}"
    "/usr/local/cuda"
    ${_trt_cuda_versioned_roots}
  )

  foreach(_cuda_root IN LISTS _trt_cuda_candidates)
    if(_cuda_root
       AND EXISTS "${_cuda_root}/bin/nvcc"
       AND EXISTS "${_cuda_root}/nvvm/bin/cicc")
      set(TRT_CUDA_ROOT "${_cuda_root}" CACHE PATH
        "CUDA Toolkit root containing bin/nvcc and nvvm/bin/cicc" FORCE)
      break()
    endif()
  endforeach()
endif()

if(NOT TRT_CUDA_ROOT
   OR NOT EXISTS "${TRT_CUDA_ROOT}/bin/nvcc"
   OR NOT EXISTS "${TRT_CUDA_ROOT}/nvvm/bin/cicc")
  message(FATAL_ERROR
    "Complete CUDA Toolkit not found. Set TRT_CUDA_ROOT to a directory containing "
    "bin/nvcc and nvvm/bin/cicc, for example /usr/local/cuda-12.8.")
endif()

set(CMAKE_CUDA_COMPILER "${TRT_CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
set(CUDAToolkit_ROOT "${TRT_CUDA_ROOT}" CACHE PATH "CUDA Toolkit root" FORCE)

# Orin 固定使用 SM 8.7；x86 默认面向 RTX 50 系列，可通过命令行覆盖
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" TRT_SYSTEM_PROCESSOR)
if(TRT_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(TRT_PLATFORM "orin-aarch64")
  set(TRT_MULTIARCH "aarch64-linux-gnu")
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures" FORCE)
  endif()
elseif(TRT_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  set(TRT_PLATFORM "x86_64")
  set(TRT_MULTIARCH "x86_64-linux-gnu")
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
  endif()
else()
  message(FATAL_ERROR "Unsupported platform architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
enable_language(CUDA)
message(STATUS
  "${PROJECT_NAME}: platform=${TRT_PLATFORM}, CUDA=${TRT_CUDA_ROOT}, "
  "CUDA architectures=${CMAKE_CUDA_ARCHITECTURES}")

find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 优先使用 Ubuntu/ROS 系统 OpenCV，避免误选由其他 CUDA 版本编译的 /usr/local OpenCV
set(TRT_OPENCV_DIR "" CACHE PATH "Directory containing OpenCVConfig.cmake")
if(NOT TRT_OPENCV_DIR)
  set(_trt_opencv_candidates
    "/usr/lib/${TRT_MULTIARCH}/cmake/opencv4"
    "/usr/lib/cmake/opencv4"
    "/usr/share/OpenCV"
  )
  foreach(_opencv_dir IN LISTS _trt_opencv_candidates)
    if(EXISTS "${_opencv_dir}/OpenCVConfig.cmake")
      set(TRT_OPENCV_DIR "${_opencv_dir}")
      break()
    endif()
  endforeach()
endif()

if(TRT_OPENCV_DIR)
  if(NOT EXISTS "${TRT_OPENCV_DIR}/OpenCVConfig.cmake")
    message(FATAL_ERROR "OpenCVConfig.cmake not found in TRT_OPENCV_DIR=${TRT_OPENCV_DIR}")
  endif()
  set(OpenCV_DIR "${TRT_OPENCV_DIR}" CACHE PATH "OpenCV package directory" FORCE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)
find_package(SQLite3 REQUIRED)
message(STATUS "${PROJECT_NAME}: OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")

# TensorRT 在 x86 与 Jetson 上使用不同的 Debian multiarch 目录
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

# 将全部推理实现编译为供其他 ROS 2 功能包链接的共享库
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

# 安装共享库、公共头文件并导出 ament CMake target
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
