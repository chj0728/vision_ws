"""Load an Orbbec Gemini2L camera component into an existing container."""

import launch
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def convert_value(value):
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value


def load_camera_component(context):
    parameter_names = (
        "depth_registration",
        "enable_ir",
        "enable_point_cloud",
        "color_width",
        "color_height",
        "depth_width",
        "depth_height",
    )
    parameters = {
        name: convert_value(LaunchConfiguration(name).perform(context))
        for name in parameter_names
    }

    return [
        LoadComposableNodes(
            target_container=LaunchConfiguration("target_container"),
            composable_node_descriptions=[
                ComposableNode(
                    package="orbbec_camera",
                    plugin="orbbec_camera::OBCameraNodeDriver",
                    name="camera",
                    namespace=LaunchConfiguration("camera_name"),
                    parameters=[parameters],
                ),
            ],
        )
    ]


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("target_container", default_value="perception_container"),
        DeclareLaunchArgument("camera_name", default_value="camera"),
        DeclareLaunchArgument("depth_registration", default_value="true"),
        DeclareLaunchArgument("enable_ir", default_value="false"),
        DeclareLaunchArgument("enable_point_cloud", default_value="false"),
        DeclareLaunchArgument("color_width", default_value="640"),
        DeclareLaunchArgument("color_height", default_value="400"),
        DeclareLaunchArgument("depth_width", default_value="640"),
        DeclareLaunchArgument("depth_height", default_value="400"),
    ]
    return launch.LaunchDescription(
        arguments + [OpaqueFunction(function=load_camera_component)]
    )
