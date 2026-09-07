"""Launch perception as a node or load it into an existing container."""

import launch
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_composition = LaunchConfiguration("use_composition")
    target_container = LaunchConfiguration("target_container")
    ros_config_path = PathJoinSubstitution(
        [FindPackageShare("trt_infer_ros"), "config", "ros.yaml"]
    )
    pipeline_config_path = PathJoinSubstitution(
        [FindPackageShare("trt_infer_ros"), "config", "pipeline.yaml"]
    )
    parameters = [
        ros_config_path,
        {"pipeline_config_path": pipeline_config_path},
    ]

    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_composition",
                default_value="false",
                description="Load perception into an existing component container.",
            ),
            DeclareLaunchArgument(
                "target_container",
                default_value="perception_container",
                description="Container used when use_composition is true.",
            ),
            Node(
                condition=UnlessCondition(use_composition),
                package="trt_infer_ros",
                executable="perception_ros_node",
                name="perception_ros_node",
                output="screen",
                parameters=parameters,
            ),
            LoadComposableNodes(
                condition=IfCondition(use_composition),
                target_container=target_container,
                composable_node_descriptions=[
                    ComposableNode(
                        package="trt_infer_ros",
                        plugin=(
                            "perception_ros_component::PerceptionRosComponent"
                        ),
                        name="perception_ros_node",
                        parameters=parameters,
                    )
                ],
            ),
        ]
    )
