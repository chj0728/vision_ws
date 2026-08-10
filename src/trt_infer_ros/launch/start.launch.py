import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    return launch.LaunchDescription(
        [
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
        ]
    )
