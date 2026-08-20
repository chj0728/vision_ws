cmake_minimum_required(VERSION 3.16)
project(trt_infer_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PersonPerception.msg"
  "msg/ScenePerceptionResult.msg"
  "srv/UpdatePersonName.srv"

  "msg/BodyDetection.msg"
  "msg/BoundingBox.msg"
  "msg/FaceDetection.msg"
  "msg/FaceRecog.msg"
  "msg/HeadPose.msg"
  "msg/PerceptionResult.msg"
  "msg/PersonMeta.msg"

  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime std_msgs)
ament_package()
