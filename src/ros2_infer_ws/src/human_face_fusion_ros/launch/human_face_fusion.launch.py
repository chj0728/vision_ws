"""融合节点：需已运行 trt_infer_ros_node。SCRFD 引擎默认使用本仓库 models/。"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    _home = os.path.expanduser("~")
    default_scrfd = os.path.join(_home, "ros2_infer_ws", "models", "scrfd_2.5g_bnkps_shape640x640.trt")
    return LaunchDescription(
        [
            DeclareLaunchArgument("engine_path", default_value=default_scrfd),
            DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("detections_topic", default_value="/remote_infer/detections"),
            DeclareLaunchArgument("fused_topic", default_value="/human_face_fusion/image_viz"),
            DeclareLaunchArgument("publish_fused", default_value="true"),
            DeclareLaunchArgument("show_window", default_value="true"),
            DeclareLaunchArgument("nms_threshold", default_value="0.4"),
            DeclareLaunchArgument("scrfd_preprocess", default_value="insightface"),
            DeclareLaunchArgument("color_sub_qos", default_value="sensor_data"),
            DeclareLaunchArgument("detection_stale_ms", default_value="500"),
            DeclareLaunchArgument("person_head_height_ratio", default_value="0.58"),
            DeclareLaunchArgument("person_head_width_pad_ratio", default_value="0.24"),
            DeclareLaunchArgument(
                "person_head_top_expand_ratio",
                default_value="0.14",
                description="人体框顶边上扩（相对人体高度），补偿人体检测顶边偏低、避免裁掉额头",
            ),
            DeclareLaunchArgument("face_roi_min_side", default_value="48"),
            DeclareLaunchArgument("face_max_person_rois", default_value="8"),
            DeclareLaunchArgument(
                "align_detection_dims",
                default_value="true",
                description="检测消息分辨率与彩图不一致时缩放 persons",
            ),
            DeclareLaunchArgument(
                "face_persons_fallback_ms",
                default_value="350",
                description="persons 因深度间歇为空时，多少 ms 内仍用上一帧非空 persons 做人脸 ROI（0=关闭）",
            ),
            DeclareLaunchArgument(
                "roi_scrfd_prob_threshold",
                default_value="0.38",
                description="头肩 ROI 内 SCRFD 置信度阈值（略降可提升小脸检出）",
            ),
            DeclareLaunchArgument(
                "roi_scrfd_nms_threshold",
                default_value="0.45",
                description="头肩 ROI 内 SCRFD 的 NMS（略高于 nms_threshold 可减少同一脸上的重复抑制）",
            ),
            DeclareLaunchArgument("also_subscribe_orbbec_composable_color", default_value="true"),
            DeclareLaunchArgument("viz_show_legend", default_value="false"),
            DeclareLaunchArgument("log_frame_timing", default_value="false"),
            DeclareLaunchArgument("log_frame_timing_period_ms", default_value="1000"),
            # 6DRepNet head pose
            DeclareLaunchArgument(
                "head_pose_enable",
                default_value="false",
                description="启用 6DRepNet 头部姿态估计（需提供 head_pose_engine_path）",
            ),
            DeclareLaunchArgument(
                "head_pose_engine_path",
                default_value="",
                description="6DRepNet TensorRT 引擎绝对路径（.trt）",
            ),
            DeclareLaunchArgument(
                "head_pose_min_face_px",
                default_value="36",
                description="执行头部姿态估计所需的最小人脸边长（像素）",
            ),
            DeclareLaunchArgument(
                "head_pose_expand_ratio",
                default_value="0.12",
                description="送入 6DRepNet 前在每侧扩展人脸 bbox 的比例",
            ),
            DeclareLaunchArgument(
                "head_pose_skip_yaw_deg",
                default_value="85.0",
                description="|yaw| 超过此角度（度）时跳过画轴（极端侧脸）",
            ),
            # Face engagement classification
            DeclareLaunchArgument("publish_engagement", default_value="true",
                                  description="是否发布 ScenePerceptionResult（engagement_topic）；head_pose 关闭时仍有全量人体+人脸框，参与度为 NOT_ENGAGED 占位"),
            DeclareLaunchArgument("engagement_topic", default_value="/human_face_fusion/scene_perception"),
            DeclareLaunchArgument("yaw_engaged_deg",   default_value="30.0",
                                  description="ENGAGED 状态的 |yaw| 上限（度）"),
            DeclareLaunchArgument("pitch_engaged_deg", default_value="25.0",
                                  description="ENGAGED 状态的 |pitch| 上限（度）"),
            DeclareLaunchArgument("yaw_attention_deg",   default_value="55.0",
                                  description="ATTENTION 状态的 |yaw| 上限（度）"),
            DeclareLaunchArgument("pitch_attention_deg", default_value="40.0",
                                  description="ATTENTION 状态的 |pitch| 上限（度）"),
            DeclareLaunchArgument("engaged_max_distance_m",   default_value="2.5",
                                  description="ENGAGED 状态的最大距离（米）"),
            DeclareLaunchArgument("attention_max_distance_m", default_value="4.0",
                                  description="ATTENTION 状态的最大距离（米）"),
            Node(
                package="human_face_fusion_ros",
                executable="human_face_fusion_node",
                name="human_face_fusion_node",
                output="screen",
                parameters=[
                    {
                        "engine_path": LaunchConfiguration("engine_path"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "detections_topic": LaunchConfiguration("detections_topic"),
                        "fused_topic": LaunchConfiguration("fused_topic"),
                        "publish_fused": LaunchConfiguration("publish_fused"),
                        "show_window": LaunchConfiguration("show_window"),
                        "nms_threshold": LaunchConfiguration("nms_threshold"),
                        "scrfd_preprocess": LaunchConfiguration("scrfd_preprocess"),
                        "color_sub_qos": LaunchConfiguration("color_sub_qos"),
                        "detection_stale_ms": LaunchConfiguration("detection_stale_ms"),
                        "person_head_height_ratio": LaunchConfiguration("person_head_height_ratio"),
                        "person_head_width_pad_ratio": LaunchConfiguration("person_head_width_pad_ratio"),
                        "person_head_top_expand_ratio": LaunchConfiguration("person_head_top_expand_ratio"),
                        "face_roi_min_side": LaunchConfiguration("face_roi_min_side"),
                        "face_max_person_rois": LaunchConfiguration("face_max_person_rois"),
                        "align_detection_dims": LaunchConfiguration("align_detection_dims"),
                        "face_persons_fallback_ms": LaunchConfiguration("face_persons_fallback_ms"),
                        "roi_scrfd_prob_threshold": LaunchConfiguration("roi_scrfd_prob_threshold"),
                        "roi_scrfd_nms_threshold": LaunchConfiguration("roi_scrfd_nms_threshold"),
                        "also_subscribe_orbbec_composable_color": LaunchConfiguration(
                            "also_subscribe_orbbec_composable_color"
                        ),
                        "viz_show_legend": LaunchConfiguration("viz_show_legend"),
                        "log_frame_timing": LaunchConfiguration("log_frame_timing"),
                        "log_frame_timing_period_ms": LaunchConfiguration("log_frame_timing_period_ms"),
                        "head_pose_enable": LaunchConfiguration("head_pose_enable"),
                        "head_pose_engine_path": LaunchConfiguration("head_pose_engine_path"),
                        "head_pose_min_face_px": LaunchConfiguration("head_pose_min_face_px"),
                        "head_pose_expand_ratio": LaunchConfiguration("head_pose_expand_ratio"),
                        "head_pose_skip_yaw_deg": LaunchConfiguration("head_pose_skip_yaw_deg"),
                        "publish_engagement":     LaunchConfiguration("publish_engagement"),
                        "engagement_topic":       LaunchConfiguration("engagement_topic"),
                        "yaw_engaged_deg":        LaunchConfiguration("yaw_engaged_deg"),
                        "pitch_engaged_deg":      LaunchConfiguration("pitch_engaged_deg"),
                        "yaw_attention_deg":        LaunchConfiguration("yaw_attention_deg"),
                        "pitch_attention_deg":      LaunchConfiguration("pitch_attention_deg"),
                        "engaged_max_distance_m":   LaunchConfiguration("engaged_max_distance_m"),
                        "attention_max_distance_m": LaunchConfiguration("attention_max_distance_m"),
                    }
                ],
            ),
        ]
    )
