cmake_minimum_required(VERSION 3.18)
project(orin_trt_infer_ros LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(remote_infer_bridge_cpp REQUIRED)

option(ORIN_TRT_LINK_OPENCV_45 "Link libopencv 4.5d to match cv_bridge (Ubuntu/Jammy + Humble)" ON)
set(_orin_trt_opencv_via_find OFF)
if(ORIN_TRT_LINK_OPENCV_45 AND UNIX AND NOT APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    set(_ocv_arch "aarch64-linux-gnu")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_ocv_arch "x86_64-linux-gnu")
  else()
    set(_ocv_arch "${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
  endif()
  set(_ocv_libdir "/usr/lib/${_ocv_arch}")
  find_library(_OCV45_CORE opencv_core.4.5d PATHS "${_ocv_libdir}" NO_DEFAULT_PATH)
  find_library(_OCV45_IMGPROC opencv_imgproc.4.5d PATHS "${_ocv_libdir}" NO_DEFAULT_PATH)
  find_library(_OCV45_IMGCODECS opencv_imgcodecs.4.5d PATHS "${_ocv_libdir}" NO_DEFAULT_PATH)
  if(_OCV45_CORE AND _OCV45_IMGPROC AND _OCV45_IMGCODECS)
    set(OpenCV_LIBS "${_OCV45_CORE};${_OCV45_IMGPROC};${_OCV45_IMGCODECS}")
    set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
    message(STATUS "orin_trt_infer_ros: OpenCV 4.5d from ${_ocv_libdir}")
  else()
    set(_orin_trt_opencv_via_find ON)
  endif()
else()
  set(_orin_trt_opencv_via_find ON)
endif()
if(_orin_trt_opencv_via_find)
  find_package(OpenCV REQUIRED)
  message(STATUS "orin_trt_infer_ros: find_package OpenCV ${OpenCV_VERSION}")
endif()

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  set(ORIN_CUDART CUDA::cudart)
else()
  find_library(ORIN_CUDART cudart REQUIRED PATHS /usr/local/cuda/lib64 /usr/local/cuda-12.6/lib64)
endif()

set(TENSORRT_ROOT "/usr")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include/aarch64-linux-gnu")
  set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib/aarch64-linux-gnu")
else()
  set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include/x86_64-linux-gnu")
  set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib/x86_64-linux-gnu")
endif()

set(YOLO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/yolo_trt")

add_executable(trt_infer_ros_node
  src/trt_infer_ros_node.cpp
  ${YOLO_DIR}/src/yolo_engine.cpp
  ${YOLO_DIR}/src/yolo_gpu_preprocess.cu
)
set_property(TARGET trt_infer_ros_node PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(trt_infer_ros_node PRIVATE
  ${YOLO_DIR}/include
  ${TENSORRT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(trt_infer_ros_node
  rclcpp
  sensor_msgs
  cv_bridge
  message_filters
  remote_infer_bridge_cpp
)
target_link_libraries(trt_infer_ros_node
  ${TENSORRT_LIB_DIR}/libnvinfer.so
  ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so
  ${ORIN_CUDART}
  ${OpenCV_LIBS}
)

install(TARGETS trt_infer_ros_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
