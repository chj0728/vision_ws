#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "trt_infer_ros_component.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<trt_infer_ros::TrtInferComponent>());
  rclcpp::shutdown();
  return 0;
}
