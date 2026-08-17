cmake_minimum_required(VERSION 3.8)
project(trt_infer_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# cv_bridge 查找 OpenCV 前必须锁定与当前平台兼容的系统版本
set(TRT_OPENCV_DIR "" CACHE PATH "Directory containing OpenCVConfig.cmake")
if(TRT_OPENCV_DIR)
  if(NOT EXISTS "${TRT_OPENCV_DIR}/OpenCVConfig.cmake")
    message(FATAL_ERROR "OpenCVConfig.cmake not found in TRT_OPENCV_DIR=${TRT_OPENCV_DIR}")
  endif()
  set(OpenCV_DIR "${TRT_OPENCV_DIR}" CACHE PATH "OpenCV package directory" FORCE)
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rmw REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trt_infer REQUIRED)
find_package(trt_infer_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

# --------------- trt_infer_ros_component -------------------------
add_library(trt_infer_ros_component SHARED
  src/trt_infer_ros_component.cpp
)
target_compile_features(trt_infer_ros_component PUBLIC cxx_std_17)
target_include_directories(trt_infer_ros_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(trt_infer_ros_component
  cv_bridge
  message_filters
  OpenCV
  rclcpp
  rclcpp_components
  rmw
  sensor_msgs
  trt_infer
  trt_infer_msgs
)
rclcpp_components_register_nodes(trt_infer_ros_component "trt_infer_ros::TrtInferComponent")

add_executable(trt_infer_ros_node src/trt_infer_ros_node.cpp)
target_compile_features(trt_infer_ros_node PUBLIC cxx_std_17)
target_link_libraries(trt_infer_ros_node trt_infer_ros_component)
ament_target_dependencies(trt_infer_ros_node rclcpp)
# ---------------------------------------------------------------------

# ------------------ perception_ros_component -------------------------
add_library(perception_ros_component SHARED
  src/perception_ros_component.cpp
  src/perception_pipeline.cpp
  src/pipeline/yolo_pipeline.cpp
)
target_compile_features(perception_ros_component PUBLIC cxx_std_17)
target_include_directories(perception_ros_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(perception_ros_component
  cv_bridge
  message_filters
  OpenCV
  rclcpp
  rclcpp_components
  rmw
  sensor_msgs
  trt_infer
  trt_infer_msgs
  yaml-cpp
)
target_link_libraries(perception_ros_component yaml-cpp)
rclcpp_components_register_nodes(perception_ros_component "perception_ros_component::PerceptionRosComponent")
add_executable(perception_ros_node src/perception_ros_node.cpp)
target_compile_features(perception_ros_node PUBLIC cxx_std_17)
target_link_libraries(perception_ros_node perception_ros_component)
ament_target_dependencies(perception_ros_node rclcpp)
# ---------------------------------------------------------------------

install(TARGETS
  trt_infer_ros_component
  trt_infer_ros_node
  perception_ros_component
  perception_ros_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
