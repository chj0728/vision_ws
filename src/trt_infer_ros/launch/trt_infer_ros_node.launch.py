"""
launch trt_infer_ros node（YOLO: 人体+深度）

1. 设置日志目录为工作空间下的 logs/trt_infer_ros_node/时间戳 目录
2. 设置日志格式为 [严重级别][时间]-[节点名-行号]: 日志内容
"""

import datetime
import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PythonExpression,
)
from launch_ros.actions import (
    LoadComposableNodes,
)
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    use_composition = LaunchConfiguration("use_composition")

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

    log_dir = os.path.join(work_space_dir, "logs", "trt_infer_ros_node", timestamp)
    print("trt_infer_ros_node log directory:", log_dir)

    models_dir = os.path.join(work_space_dir, "models")
    print("models directory:", models_dir)

    yolo_engine_path = os.path.join(models_dir, "yolo26m_fp16.engine")
    print("yolo engine path:", yolo_engine_path)

    return launch.LaunchDescription(
        [  # -------------- 全局环境变量设置（影响所有后续节点）------------------
            SetEnvironmentVariable(name="RCUTILS_COLORIZED_OUTPUT", value="1"),
            SetEnvironmentVariable(name="ROS_LOG_DIR", value=log_dir),
            SetEnvironmentVariable(
                name="RCUTILS_CONSOLE_OUTPUT_FORMAT",
                value="[{severity}][{time}]-[{name}-{line_number}]: {message}",
            ),
            DeclareLaunchArgument(
                "use_composition",
                default_value="True",
                description="Whether to use component composition.",
            ),
            DeclareLaunchArgument("model_path", default_value=yolo_engine_path),
            DeclareLaunchArgument(
                "image_topic", default_value="/camera/color/image_raw"
            ),
            DeclareLaunchArgument(
                "depth_topic", default_value="/camera/depth/image_raw"
            ),
            DeclareLaunchArgument(
                "detections_topic", default_value="/remote_infer/detections"
            ),
            DeclareLaunchArgument("conf_threshold", default_value="0.45"),
            DeclareLaunchArgument("bbox_coord_space", default_value="letterbox"),
            DeclareLaunchArgument("engine_input_h", default_value="0"),
            DeclareLaunchArgument("engine_input_w", default_value="0"),
            DeclareLaunchArgument("max_distance_m", default_value="5.0"),
            DeclareLaunchArgument("only_human_class", default_value="true"),
            DeclareLaunchArgument("human_class_id", default_value="0"),
            DeclareLaunchArgument("depth_sync_exact", default_value="false"),
            DeclareLaunchArgument("depth_sync_queue_size", default_value="40"),
            DeclareLaunchArgument("depth_scale_to_meters", default_value="0.001"),
            DeclareLaunchArgument("depth_roi_y0", default_value="0.52"),
            DeclareLaunchArgument("depth_roi_y1", default_value="0.98"),
            DeclareLaunchArgument("depth_roi_x_margin", default_value="0.2"),
            DeclareLaunchArgument("min_depth_m", default_value="0.08"),
            DeclareLaunchArgument("max_depth_read_m", default_value="25.0"),
            DeclareLaunchArgument("depth_percentile", default_value="0.5"),
            DeclareLaunchArgument("depth_trim_close_ratio", default_value="0.0"),
            DeclareLaunchArgument("distance_ema_enable", default_value="false"),
            DeclareLaunchArgument("distance_ema_alpha", default_value="0.35"),
            # launch nodes normally if not use composition
            GroupAction(
                condition=IfCondition(PythonExpression(["not ", use_composition])),
                actions=[
                    ## trt_infer_ros_node
                    launch_ros.actions.Node(
                        package="trt_infer_ros",
                        executable="trt_infer_ros_node",
                        name="trt_infer_ros_node",
                        output="screen",
                        parameters=[
                            {
                                "model_path": LaunchConfiguration("model_path"),
                                "image_topic": LaunchConfiguration("image_topic"),
                                "depth_topic": LaunchConfiguration("depth_topic"),
                                "detections_topic": LaunchConfiguration(
                                    "detections_topic"
                                ),
                                "conf_threshold": LaunchConfiguration("conf_threshold"),
                                "bbox_coord_space": LaunchConfiguration(
                                    "bbox_coord_space"
                                ),
                                "engine_input_h": LaunchConfiguration("engine_input_h"),
                                "engine_input_w": LaunchConfiguration("engine_input_w"),
                                "max_distance_m": LaunchConfiguration("max_distance_m"),
                                "only_human_class": LaunchConfiguration(
                                    "only_human_class"
                                ),
                                "human_class_id": LaunchConfiguration("human_class_id"),
                                "depth_sync_exact": LaunchConfiguration(
                                    "depth_sync_exact"
                                ),
                                "depth_sync_queue_size": LaunchConfiguration(
                                    "depth_sync_queue_size"
                                ),
                                "depth_scale_to_meters": LaunchConfiguration(
                                    "depth_scale_to_meters"
                                ),
                                "depth_roi_y0": LaunchConfiguration("depth_roi_y0"),
                                "depth_roi_y1": LaunchConfiguration("depth_roi_y1"),
                                "depth_roi_x_margin": LaunchConfiguration(
                                    "depth_roi_x_margin"
                                ),
                                "min_depth_m": LaunchConfiguration("min_depth_m"),
                                "max_depth_read_m": LaunchConfiguration(
                                    "max_depth_read_m"
                                ),
                                "depth_percentile": LaunchConfiguration(
                                    "depth_percentile"
                                ),
                                "depth_trim_close_ratio": LaunchConfiguration(
                                    "depth_trim_close_ratio"
                                ),
                                "distance_ema_enable": LaunchConfiguration(
                                    "distance_ema_enable"
                                ),
                                "distance_ema_alpha": LaunchConfiguration(
                                    "distance_ema_alpha"
                                ),
                            }
                        ],
                    ),
                ],
            ),
            # use LoadComposableNodes to launch node into a component container if use composition
            TimerAction(
                period=3.0,  # 延迟3秒
                actions=[
                    LoadComposableNodes(
                        condition=IfCondition(use_composition),
                        target_container="/camera/camera_container",
                        composable_node_descriptions=[
                            ComposableNode(
                                package="trt_infer_ros",
                                plugin="trt_infer_ros::TrtInferComponent",
                                name="trt_infer_ros_node",
                                parameters=[
                                    {
                                        "model_path": LaunchConfiguration("model_path"),
                                        "image_topic": LaunchConfiguration(
                                            "image_topic"
                                        ),
                                        "depth_topic": LaunchConfiguration(
                                            "depth_topic"
                                        ),
                                        "detections_topic": LaunchConfiguration(
                                            "detections_topic"
                                        ),
                                        "conf_threshold": LaunchConfiguration(
                                            "conf_threshold"
                                        ),
                                        "bbox_coord_space": LaunchConfiguration(
                                            "bbox_coord_space"
                                        ),
                                        "engine_input_h": LaunchConfiguration(
                                            "engine_input_h"
                                        ),
                                        "engine_input_w": LaunchConfiguration(
                                            "engine_input_w"
                                        ),
                                        "max_distance_m": LaunchConfiguration(
                                            "max_distance_m"
                                        ),
                                        "only_human_class": LaunchConfiguration(
                                            "only_human_class"
                                        ),
                                        "human_class_id": LaunchConfiguration(
                                            "human_class_id"
                                        ),
                                        "depth_sync_exact": LaunchConfiguration(
                                            "depth_sync_exact"
                                        ),
                                        "depth_sync_queue_size": LaunchConfiguration(
                                            "depth_sync_queue_size"
                                        ),
                                        "depth_scale_to_meters": LaunchConfiguration(
                                            "depth_scale_to_meters"
                                        ),
                                        "depth_roi_y0": LaunchConfiguration(
                                            "depth_roi_y0"
                                        ),
                                        "depth_roi_y1": LaunchConfiguration(
                                            "depth_roi_y1"
                                        ),
                                        "depth_roi_x_margin": LaunchConfiguration(
                                            "depth_roi_x_margin"
                                        ),
                                        "min_depth_m": LaunchConfiguration(
                                            "min_depth_m"
                                        ),
                                        "max_depth_read_m": LaunchConfiguration(
                                            "max_depth_read_m"
                                        ),
                                        "depth_percentile": LaunchConfiguration(
                                            "depth_percentile"
                                        ),
                                        "depth_trim_close_ratio": LaunchConfiguration(
                                            "depth_trim_close_ratio"
                                        ),
                                        "distance_ema_enable": LaunchConfiguration(
                                            "distance_ema_enable"
                                        ),
                                        "distance_ema_alpha": LaunchConfiguration(
                                            "distance_ema_alpha"
                                        ),
                                    }
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )
