#include <rclcpp/rclcpp.hpp>

#include "perception_ros_component.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  // single-threaded spin
  // rclcpp::spin(
  //     std::make_shared<perception_ros_component::PerceptionRosComponent>());
  // rclcpp::shutdown();

  // multi-threaded spin using MultiThreadedExecutor
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<perception_ros_component::PerceptionRosComponent>();
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();

  return 0;
}