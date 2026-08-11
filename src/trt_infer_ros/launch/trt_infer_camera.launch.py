"""
Camera launch file for trt_infer_ros.

Returns:
    launch.LaunchDescription: The launch description for the camera node.

1. 加载 orbbec_camera 官方 gemini2L.launch.py，启动 Orbbec 相机节点。
2. 设置日志目录为工作空间下的 logs/camera/时间戳 目录
3. 设置日志格式为 [严重级别][时间]-[节点名-行号]: 日志内容
"""

import datetime
import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    # 使用 COLCON_PREFIX_PATH 环境变量
    colcon_prefix_path = os.environ.get("COLCON_PREFIX_PATH", "")
    if colcon_prefix_path:
        # 取第一个路径作为工作空间目录
        work_space_dir = os.path.dirname(colcon_prefix_path.split(":")[0])
    else:
        # 回退到方案1
        trt_infer_ros_dir = get_package_share_directory("trt_infer_ros")
        work_space_dir = os.path.dirname(os.path.dirname(trt_infer_ros_dir))

    # 创建带时间戳的日志目录
    timestamp = datetime.datetime.now(
        tz=datetime.timezone(datetime.timedelta(hours=8))
    ).strftime("%Y-%m-%d_%H-%M-%S")

    camera_log_dir = os.path.join(work_space_dir, "logs", "camera", timestamp)
    print("Camera log directory:", camera_log_dir)

    return launch.LaunchDescription(
        [  # -------------- 全局环境变量设置（影响所有后续节点）------------------
            SetEnvironmentVariable(name="RCUTILS_COLORIZED_OUTPUT", value="1"),
            SetEnvironmentVariable(name="ROS_LOG_DIR", value=camera_log_dir),
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
                }.items(),
            ),
        ]
    )
