import launch
import launch_ros.actions
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

    use_composition = LaunchConfiguration("use_composition", default="False")

    # models_dir = os.path.join(work_space_dir, "models")
    # print("models directory:", models_dir)

    # yolo_engine_path = os.path.join(models_dir, "yolo26m_fp16.engine")
    # print("yolo engine path:", yolo_engine_path)

    return launch.LaunchDescription(
        [  # -------------- 全局环境变量设置（影响所有后续节点）------------------
            SetEnvironmentVariable(name="RCUTILS_COLORIZED_OUTPUT", value="0"),
            SetEnvironmentVariable(
                name="RCUTILS_CONSOLE_OUTPUT_FORMAT",
                value="[{severity}][{time}]-[{name}:{line_number}]: {message}",
            ),
            DeclareLaunchArgument(
                "use_composition",
                default_value="False",
                description="Whether to use component composition.",
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
                period=10.0,  # 延迟10秒
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
