"""Launch the camera and perception together."""

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
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer


def custom_launch_path(filename):
    return os.path.join(
        get_package_share_directory("trt_infer_ros"), "launch", filename
    )


def include_custom_launch(filename, use_composition):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(custom_launch_path(filename)),
        launch_arguments={
            "use_composition": use_composition,
            "target_container": "perception_container",
        }.items(),
    )


def generate_launch_description():
    use_composition = LaunchConfiguration("use_composition")

    return launch.LaunchDescription(
        [
            SetEnvironmentVariable(name="RCUTILS_COLORIZED_OUTPUT", value="0"),
            SetEnvironmentVariable(
                name="RCUTILS_CONSOLE_OUTPUT_FORMAT",
                value="[{severity}][{time}]-[{name}-{line_number}]: {message}",
            ),
            DeclareLaunchArgument(
                "use_composition",
                default_value="true",
                description=("Run the camera and perception in perception_container."),
            ),
            ComposableNodeContainer(
                condition=IfCondition(use_composition),
                name="perception_container",
                namespace="",
                package="rclcpp_components",
                executable="component_container_mt",
                output="screen",
            ),
            GroupAction(
                actions=[
                    TimerAction(
                        period=2.0,
                        actions=[
                            include_custom_launch(
                                "custom_gemini2L.launch.py", use_composition
                            )
                        ],
                    ),
                    TimerAction(
                        period=10.0,
                        actions=[
                            include_custom_launch(
                                "custom_perception.launch.py", use_composition
                            )
                        ],
                    ),
                ]
            ),
        ]
    )
