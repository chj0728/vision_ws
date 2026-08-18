#include <rclcpp/rclcpp.hpp>

#include "perception_ros_component.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(
      std::make_shared<perception_ros_component::PerceptionRosComponent>());
  rclcpp::shutdown();
  return 0;
}