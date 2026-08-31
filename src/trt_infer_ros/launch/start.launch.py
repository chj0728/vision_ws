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
        [
            # -------------- 全局环境变量设置（影响所有后续节点）------------------
            SetEnvironmentVariable(name="RCUTILS_COLORIZED_OUTPUT", value="0"),
            SetEnvironmentVariable(
                name="RCUTILS_CONSOLE_OUTPUT_FORMAT",
                value="[{severity}][{time}]-[{name}-{line_number}]: {message}",
            ),
            # ------------- trt_infer_ros camera.launch.py ----------------
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("trt_infer_ros"),
                        "launch",
                        "camera.launch.py",
                    )
                )
            ),
            # ------------- trt_infer_ros trt_infer_ros.launch.py ----------------
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("trt_infer_ros"),
                        "launch",
                        "perception.launch.py",
                    )
                ),
                launch_arguments={"use_composition": "True"}.items(),
            ),
        ]
    )
