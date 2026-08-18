cmake_minimum_required(VERSION 3.8)
project(trt_infer_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 默认从“工作空间/src/当前包”反推工作空间根目录，也可通过
# -DTRT_WORKSPACE_ROOT=/path/to/workspace 在配置阶段覆盖。
get_filename_component(DEFAULT_WORKSPACE_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE
)
set(TRT_WORKSPACE_ROOT "${DEFAULT_WORKSPACE_ROOT}" CACHE PATH
  "Path to the ROS 2 workspace root"
)

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

# 共用的 ROS 及推理依赖
set(COMMON_COMPONENT_DEPENDENCIES
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

# 为组件统一设置语言标准、头文件路径和公共依赖
function(configure_component target)
  target_compile_features(${target} PUBLIC cxx_std_17)
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(${target} ${COMMON_COMPONENT_DEPENDENCIES})
endfunction()

# ------------- trt_infer_ros_component -----------------------
add_library(trt_infer_ros_component SHARED src/trt_infer_ros_component.cpp)
configure_component(trt_infer_ros_component)
rclcpp_components_register_nodes(trt_infer_ros_component "trt_infer_ros::TrtInferComponent")

add_executable(trt_infer_ros_node src/trt_infer_ros_node.cpp)
target_compile_features(trt_infer_ros_node PUBLIC cxx_std_17)
target_link_libraries(trt_infer_ros_node trt_infer_ros_component)
ament_target_dependencies(trt_infer_ros_node rclcpp)
# -------------------------------------------------------------

# --------- perception_ros_component -------------------------
add_library(perception_ros_component SHARED
  src/perception_ros_component.cpp
  src/perception_pipeline.cpp
  src/pipeline/yolo_pipeline.cpp
)
configure_component(perception_ros_component)
ament_target_dependencies(perception_ros_component yaml-cpp)
target_link_libraries(perception_ros_component yaml-cpp)
# 将工作空间路径作为字符串宏传给需要加载模型的源文件
target_compile_definitions(perception_ros_component PRIVATE
  TRT_WORKSPACE_ROOT="${TRT_WORKSPACE_ROOT}"
)
rclcpp_components_register_nodes(perception_ros_component "perception_ros_component::PerceptionRosComponent")

add_executable(perception_ros_node src/perception_ros_node.cpp)
target_compile_features(perception_ros_node PUBLIC cxx_std_17)
target_link_libraries(perception_ros_node perception_ros_component)
ament_target_dependencies(perception_ros_node rclcpp)
# ------------------------------------------------------------

# 安装组件库、可执行程序、头文件及运行配置
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

  # 当前源码尚未统一添加版权与许可证头，因此暂时跳过对应检查
  set(ament_cmake_copyright_FOUND TRUE)

  # cpplint 依赖 Git 仓库及完整的版权头，条件具备后可移除该设置
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
