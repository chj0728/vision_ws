"""
Camera launch file for trt_infer_ros.

Returns:
    launch.LaunchDescription: The launch description for the camera node.

1. 加载 orbbec_camera 官方 gemini2L.launch.py，启动 Orbbec 相机节点。
2. 设置日志格式为 [严重级别][时间]-[节点名-行号]: 日志内容
"""

import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    return launch.LaunchDescription(
        [  # -------------- 全局环境变量设置（影响所有后续节点）------------------
            SetEnvironmentVariable(name="RCUTILS_COLORIZED_OUTPUT", value="0"),
            SetEnvironmentVariable(
                name="RCUTILS_CONSOLE_OUTPUT_FORMAT",
                value="[{severity}][{time}]-[{name}-{line_number}]: {message}",
            ),
            # ------------- orbbec_camera gemini2L.launch.py ----------------
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("orbbec_camera"),
                        "launch",
                        "gemini2L.launch.py",
                    )
                ),
                launch_arguments={
                    "camera_name": "camera",
                    "depth_registration": "true",
                    "enable_ir": "false",
                    "enable_point_cloud": "false",
                    "color_width": "640",
                    "color_height": "400",
                    "depth_width": "640",
                    "depth_height": "400",
                }.items(),
            ),
        ]
    )
