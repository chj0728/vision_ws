"""人体+深度+YOLO：同步彩图与深度，发布 HumanFrameResult / PersonDistances。可视化请用 human_face_fusion_ros。"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    _default_engine = os.path.join(
        os.path.expanduser("~"), "ros2_infer_ws", "models", "yolo26m_fp16.engine"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("model_path", default_value=_default_engine),
            DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("depth_topic", default_value="/camera/depth/image_raw"),
            DeclareLaunchArgument("detections_topic", default_value="/remote_infer/detections"),
            DeclareLaunchArgument("person_distances_topic", default_value="/remote_infer/person_distances"),
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
            DeclareLaunchArgument("min_depth_samples", default_value="6"),
            DeclareLaunchArgument("min_depth_m", default_value="0.08"),
            DeclareLaunchArgument("max_depth_read_m", default_value="25.0"),
            DeclareLaunchArgument("depth_percentile", default_value="0.5"),
            DeclareLaunchArgument("depth_trim_close_ratio", default_value="0.0"),
            DeclareLaunchArgument("distance_ema_enable", default_value="false"),
            DeclareLaunchArgument("distance_ema_alpha", default_value="0.35"),
            DeclareLaunchArgument("log_frame_timing", default_value="false"),
            DeclareLaunchArgument("log_frame_timing_period_ms", default_value="1000"),
            Node(
                package="orin_trt_infer_ros",
                executable="trt_infer_ros_node",
                name="trt_infer_ros_node",
                output="screen",
                parameters=[
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        "detections_topic": LaunchConfiguration("detections_topic"),
                        "person_distances_topic": LaunchConfiguration("person_distances_topic"),
                        "conf_threshold": LaunchConfiguration("conf_threshold"),
                        "bbox_coord_space": LaunchConfiguration("bbox_coord_space"),
                        "engine_input_h": LaunchConfiguration("engine_input_h"),
                        "engine_input_w": LaunchConfiguration("engine_input_w"),
                        "max_distance_m": LaunchConfiguration("max_distance_m"),
                        "only_human_class": LaunchConfiguration("only_human_class"),
                        "human_class_id": LaunchConfiguration("human_class_id"),
                        "depth_sync_exact": LaunchConfiguration("depth_sync_exact"),
                        "depth_sync_queue_size": LaunchConfiguration("depth_sync_queue_size"),
                        "depth_scale_to_meters": LaunchConfiguration("depth_scale_to_meters"),
                        "depth_roi_y0": LaunchConfiguration("depth_roi_y0"),
                        "depth_roi_y1": LaunchConfiguration("depth_roi_y1"),
                        "depth_roi_x_margin": LaunchConfiguration("depth_roi_x_margin"),
                        "min_depth_samples": LaunchConfiguration("min_depth_samples"),
                        "min_depth_m": LaunchConfiguration("min_depth_m"),
                        "max_depth_read_m": LaunchConfiguration("max_depth_read_m"),
                        "depth_percentile": LaunchConfiguration("depth_percentile"),
                        "depth_trim_close_ratio": LaunchConfiguration("depth_trim_close_ratio"),
                        "distance_ema_enable": LaunchConfiguration("distance_ema_enable"),
                        "distance_ema_alpha": LaunchConfiguration("distance_ema_alpha"),
                        "log_frame_timing": LaunchConfiguration("log_frame_timing"),
                        "log_frame_timing_period_ms": LaunchConfiguration("log_frame_timing_period_ms"),
                    }
                ],
            ),
        ]
    )
