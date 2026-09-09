"""Launch the Orbbec Gemini2L camera as a node or a component."""

import launch
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode

# 1280 800 15hz
# 640 400 30hz
CAMERA_PARAMETER_DEFAULTS = {
    "depth_registration": "true",
    "enable_color": "true",
    "color_width": "1280",
    "color_height": "800",
    "color_fps": "15",
    "color_format": "MJPG",
    "enable_ir": "false",
    "enable_point_cloud": "false",
    "enable_colored_point_cloud": "false",
    "depth_width": "1280",
    "depth_height": "800",
    "depth_fps": "15",
    "depth_format": "Y16",
    "enable_depth": "true",
}


def convert_value(value):
    """Convert launch argument strings to ROS parameter values."""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value


def launch_camera(context):
    parameters = {
        name: convert_value(LaunchConfiguration(name).perform(context))
        for name in CAMERA_PARAMETER_DEFAULTS
    }

    if LaunchConfiguration("use_composition").perform(context).lower() == "true":
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
                    )
                ],
            )
        ]

    return [
        Node(
            package="orbbec_camera",
            executable="orbbec_camera_node",
            name="camera",
            namespace=LaunchConfiguration("camera_name"),
            parameters=[parameters],
            output="screen",
        )
    ]


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument(
            "use_composition",
            default_value="false",
            description="Load the camera into an existing component container.",
        ),
        DeclareLaunchArgument(
            "target_container",
            default_value="perception_container",
            description="Container used when use_composition is true.",
        ),
        DeclareLaunchArgument("camera_name", default_value="camera"),
        *[
            DeclareLaunchArgument(name, default_value=default)
            for name, default in CAMERA_PARAMETER_DEFAULTS.items()
        ],
    ]

    return launch.LaunchDescription(
        arguments + [OpaqueFunction(function=launch_camera)]
    )
