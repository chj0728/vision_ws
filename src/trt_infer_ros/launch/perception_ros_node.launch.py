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
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import (
    LoadComposableNodes,
)
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    ros_config_path = PathJoinSubstitution(
        [FindPackageShare("trt_infer_ros"), "config", "ros.yaml"]
    )
    pipeline_config_path = PathJoinSubstitution(
        [FindPackageShare("trt_infer_ros"), "config", "pipeline.yaml"]
    )

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

    log_dir = os.path.join(work_space_dir, "logs", "perception_ros_node", timestamp)
    print("perception_ros_node log directory:", log_dir)

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
            DeclareLaunchArgument(
                "hard_sync",
                default_value="False",
                description="Whether to use exact synchronization.",
            ),
            DeclareLaunchArgument(
                "sync_queue_size",
                default_value="10",
                description="Queue size for message synchronization.",
            ),
            DeclareLaunchArgument(
                "processing_rate_hz",
                default_value="1.0",
                description="Maximum RGB-D processing rate in Hz.",
            ),
            DeclareLaunchArgument(
                "color_image_topic",
                default_value="/camera/rgb/image_color",
                description="Color image topic.",
            ),
            DeclareLaunchArgument(
                "depth_image_topic",
                default_value="/camera/depth/image",
                description="Depth image topic.",
            ),
            # launch nodes normally if not use composition
            GroupAction(
                condition=IfCondition(PythonExpression(["not ", use_composition])),
                actions=[
                    ## ----------------- perception_ros_node ------------------
                    launch_ros.actions.Node(
                        package="trt_infer_ros",
                        executable="perception_ros_node",
                        name="perception_ros_node",
                        output="screen",
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
            # use LoadComposableNodes to launch node into a component container if use composition
            TimerAction(
                period=3.0,  # 延迟3秒
                actions=[
                    GroupAction(
                        condition=IfCondition(use_composition),
                        actions=[
                            LoadComposableNodes(
                                target_container="/camera/camera_container",
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
        ]
    )
