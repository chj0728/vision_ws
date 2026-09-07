"""Load an Orbbec Gemini2L camera component into an existing container."""

import launch
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
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
        "enable_color",
        "color_width",
        "color_height",
        "color_fps",
        "color_format",
        "enable_ir",
        "enable_point_cloud",
        "depth_width",
        "depth_height",
        "depth_fps",
        "depth_format",
        "enable_depth",
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


def create_camera_container(context):
    parameter_names = (
        "depth_registration",
        "enable_color",
        "color_width",
        "color_height",
        "color_fps",
        "color_format",
        "enable_ir",
        "enable_point_cloud",
        "depth_width",
        "depth_height",
        "depth_fps",
        "depth_format",
        "enable_depth",
    )
    parameters = {
        name: convert_value(LaunchConfiguration(name).perform(context))
        for name in parameter_names
    }

    return [
        ComposableNodeContainer(
            name=LaunchConfiguration("target_container"),
            namespace="",
            package="rclcpp_components",
            executable="component_container_mt",
            composable_node_descriptions=[
                ComposableNode(
                    package="orbbec_camera",
                    plugin="orbbec_camera::OBCameraNodeDriver",
                    name="camera",
                    namespace=LaunchConfiguration("camera_name"),
                    parameters=[parameters],
                ),
            ],
            output="screen",
        )
    ]


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("target_container", default_value="perception_container"),
        DeclareLaunchArgument("create_container", default_value="false"),
        DeclareLaunchArgument("camera_name", default_value="camera"),
        DeclareLaunchArgument("depth_registration", default_value="true"),
        DeclareLaunchArgument("enable_color", default_value="true"),
        DeclareLaunchArgument("color_width", default_value="640"),
        DeclareLaunchArgument("color_height", default_value="400"),
        DeclareLaunchArgument("color_fps", default_value="30"),
        DeclareLaunchArgument("color_format", default_value="MJPG"),
        DeclareLaunchArgument("enable_ir", default_value="false"),
        DeclareLaunchArgument("enable_point_cloud", default_value="false"),
        DeclareLaunchArgument("depth_width", default_value="640"),
        DeclareLaunchArgument("depth_height", default_value="400"),
        DeclareLaunchArgument("depth_fps", default_value="30"),
        DeclareLaunchArgument("depth_format", default_value="Y16"),
        DeclareLaunchArgument("enable_depth", default_value="true"),
    ]
    return launch.LaunchDescription(
        arguments
        + [
            OpaqueFunction(
                function=lambda context: (
                    create_camera_container(context)
                    if LaunchConfiguration("create_container").perform(context).lower()
                    == "true"
                    else load_camera_component(context)
                )
            )
        ]
    )
