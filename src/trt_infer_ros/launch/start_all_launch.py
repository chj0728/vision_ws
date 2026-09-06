import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    ros_config_path = PathJoinSubstitution(
        [FindPackageShare("trt_infer_ros"), "config", "ros.yaml"]
    )
    pipeline_config_path = PathJoinSubstitution(
        [FindPackageShare("trt_infer_ros"), "config", "pipeline.yaml"]
    )
    use_composition = LaunchConfiguration("use_composition", default="True")

    return launch.LaunchDescription(
        [
            # -------------- 全局环境变量设置（影响所有后续节点）------------------
            SetEnvironmentVariable(name="RCUTILS_COLORIZED_OUTPUT", value="0"),
            SetEnvironmentVariable(
                name="RCUTILS_CONSOLE_OUTPUT_FORMAT",
                value="[{severity}][{time}]-[{name}-{line_number}]: {message}",
            ),
            DeclareLaunchArgument(
                "use_composition",
                default_value="True",
                description="Whether to use composed nodes",
            ),
            # use composition
            GroupAction(
                condition=IfCondition(use_composition),
                actions=[
                    ComposableNodeContainer(
                        name="perception_container",
                        namespace="",
                        package="rclcpp_components",
                        executable="component_container_mt",
                        composable_node_descriptions=[],
                        output="screen",
                    ),
                    ## 1. Load Orbbec Camera composable node
                    TimerAction(
                        period=2.0,
                        actions=[
                            LoadComposableNodes(
                                target_container="perception_container",
                                composable_node_descriptions=[
                                    ComposableNode(
                                        package="orbbec_camera",
                                        plugin="orbbec_camera::OBCameraNodeDriver",
                                        name="camera",
                                        parameters=[
                                            {
                                                # "camera_name": "camera",
                                                "depth_registration": "true",
                                                "enable_ir": "false",
                                                "enable_point_cloud": "false",
                                                "color_width": "640",
                                                "color_height": "400",
                                                "depth_width": "640",
                                                "depth_height": "400",
                                            },
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    ## 2. Load Perception ROS composable node
                    TimerAction(
                        period=10.0,
                        actions=[
                            LoadComposableNodes(
                                target_container="perception_container",
                                composable_node_descriptions=[
                                    ComposableNode(
                                        package="trt_infer_ros",
                                        plugin="perception_ros_component::PerceptionRosComponent",
                                        name="perception_ros_node",
                                        parameters=[
                                            ros_config_path,
                                            {
                                                "pipeline_config_path": pipeline_config_path.perform(
                                                    launch.LaunchContext()
                                                ),
                                            },
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            # not using composition
            GroupAction(
                condition=IfCondition(PythonExpression(["not ", use_composition])),
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            os.path.join(
                                get_package_share_directory("trt_infer_ros"),
                                "launch",
                                "camera.launch.py",
                            )
                        )
                    ),
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            os.path.join(
                                get_package_share_directory("trt_infer_ros"),
                                "launch",
                                "perception.launch.py",
                            )
                        ),
                        launch_arguments={"use_composition": "False"}.items(),
                    ),
                ],
            ),
        ]
    )
