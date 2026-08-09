cmake_minimum_required(VERSION 3.18)
project(human_face_fusion_ros LANGUAGES CXX CUDA)

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
find_package(remote_infer_bridge_cpp REQUIRED)

option(HFF_LINK_OPENCV_45 "Link libopencv 4.5d to match cv_bridge (Jammy + Humble)" ON)
set(_hff_opencv_via_find OFF)
if(HFF_LINK_OPENCV_45 AND UNIX AND NOT APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    set(_ocv_arch "aarch64-linux-gnu")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_ocv_arch "x86_64-linux-gnu")
  else()
    set(_ocv_arch "${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
  endif()
  set(_ocv_libdir "/usr/lib/${_ocv_arch}")
  find_library(_OCV45_CORE      opencv_core.4.5d      PATHS "${_ocv_libdir}" NO_DEFAULT_PATH)
  find_library(_OCV45_IMGPROC   opencv_imgproc.4.5d   PATHS "${_ocv_libdir}" NO_DEFAULT_PATH)
  find_library(_OCV45_IMGCODECS opencv_imgcodecs.4.5d PATHS "${_ocv_libdir}" NO_DEFAULT_PATH)
  find_library(_OCV45_HIGHGUI   opencv_highgui.4.5d   PATHS "${_ocv_libdir}" NO_DEFAULT_PATH)
  if(_OCV45_CORE AND _OCV45_IMGPROC AND _OCV45_IMGCODECS AND _OCV45_HIGHGUI)
    set(OpenCV_LIBS "${_OCV45_CORE};${_OCV45_IMGPROC};${_OCV45_IMGCODECS};${_OCV45_HIGHGUI}")
    set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
    message(STATUS "human_face_fusion_ros: OpenCV 4.5d from ${_ocv_libdir}")
  else()
    set(_hff_opencv_via_find ON)
  endif()
else()
  set(_hff_opencv_via_find ON)
endif()
if(_hff_opencv_via_find)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
  message(STATUS "human_face_fusion_ros: find_package OpenCV ${OpenCV_VERSION}")
endif()

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  set(HFF_CUDART CUDA::cudart)
else()
  find_library(HFF_CUDART cudart REQUIRED PATHS /usr/local/cuda/lib64 /usr/local/cuda-12.6/lib64)
endif()

set(TENSORRT_ROOT "/usr")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include/aarch64-linux-gnu")
  set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib/aarch64-linux-gnu")
else()
  set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include/x86_64-linux-gnu")
  set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib/x86_64-linux-gnu")
endif()

get_filename_component(_HFF_WS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

set(SCRFD_DIR "${_HFF_WS_SRC}/orin_trt_infer_ros/scrfd_trt")
if(NOT EXISTS "${SCRFD_DIR}/src/scrfd_trt.cpp")
  message(FATAL_ERROR "human_face_fusion_ros: SCRFD not found: ${SCRFD_DIR}")
endif()

set(SIXDREPNET_DIR "${_HFF_WS_SRC}/orin_trt_infer_ros/sixdrepnet_trt")
if(NOT EXISTS "${SIXDREPNET_DIR}/src/sixdrepnet_trt.cpp")
  message(FATAL_ERROR "human_face_fusion_ros: SixDRepNet not found: ${SIXDREPNET_DIR}")
endif()

set(ARCFACE_DIR "${_HFF_WS_SRC}/orin_trt_infer_ros/arcface_trt")
if(NOT EXISTS "${ARCFACE_DIR}/src/arcface_trt.cpp")
  message(FATAL_ERROR "human_face_fusion_ros: ArcFace not found: ${ARCFACE_DIR}")
endif()

set(GENDER_DIR "${_HFF_WS_SRC}/orin_trt_infer_ros/gender_trt")
if(NOT EXISTS "${GENDER_DIR}/src/gender_trt.cpp")
  message(FATAL_ERROR "human_face_fusion_ros: GenderTRT not found: ${GENDER_DIR}")
endif()

# ── Find system libraries ─────────────────────────────────────────────────────
find_package(Freetype REQUIRED)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(_SYS_LIB_DIR "/usr/lib/aarch64-linux-gnu")
else()
  set(_SYS_LIB_DIR "/usr/lib/x86_64-linux-gnu")
endif()

find_library(HFF_SQLITE3 sqlite3 PATHS "${_SYS_LIB_DIR}" NO_DEFAULT_PATH
             DOC "SQLite3 library for face database")
if(NOT HFF_SQLITE3)
  find_library(HFF_SQLITE3 sqlite3 REQUIRED)
endif()

find_library(HFF_UUID uuid PATHS "${_SYS_LIB_DIR}" NO_DEFAULT_PATH
             DOC "libuuid for UUID v4 generation")
if(NOT HFF_UUID)
  find_library(HFF_UUID uuid REQUIRED)
endif()

# ── Shared includes and libs ──────────────────────────────────────────────────
set(_HFF_COMMON_INCLUDES
  ${SCRFD_DIR}/include
  ${SIXDREPNET_DIR}/include
  ${ARCFACE_DIR}/include
  ${GENDER_DIR}/include
  ${TENSORRT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

set(_HFF_COMMON_LIBS
  ${TENSORRT_LIB_DIR}/libnvinfer.so
  ${HFF_CUDART}
  ${OpenCV_LIBS}
  Freetype::Freetype
  ${HFF_SQLITE3}
  ${HFF_UUID}
)

# ── human_face_fusion_node ────────────────────────────────────────────────────
add_executable(human_face_fusion_node
  src/human_face_fusion_node.cpp
  ${SCRFD_DIR}/src/scrfd_trt.cpp
  ${SCRFD_DIR}/src/scrfd_gpu_ops.cu
  ${SIXDREPNET_DIR}/src/sixdrepnet_trt.cpp
  ${ARCFACE_DIR}/src/arcface_trt.cpp
  ${ARCFACE_DIR}/src/face_database.cpp
  ${GENDER_DIR}/src/gender_trt.cpp
)
target_include_directories(human_face_fusion_node PRIVATE ${_HFF_COMMON_INCLUDES})
ament_target_dependencies(human_face_fusion_node
  rclcpp sensor_msgs std_msgs cv_bridge remote_infer_bridge_cpp)
target_link_libraries(human_face_fusion_node ${_HFF_COMMON_LIBS})

install(TARGETS human_face_fusion_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
