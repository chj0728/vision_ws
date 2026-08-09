"""一体启动：trt_infer_dds（人体+深度）+ human_face_fusion（同屏人脸识别+追踪+性别）。相机需另起。"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    orin_share = get_package_share_directory("orin_trt_infer_ros")
    trt_launch  = os.path.join(orin_share, "launch", "trt_infer_dds.launch.py")

    _home = os.path.expanduser("~")
    ws    = os.path.join(_home, "ros2_infer_ws")

    default_yolo      = os.path.join(ws, "models", "yolo26m_fp16.engine")
    default_scrfd     = os.path.join(ws, "models", "scrfd_2.5g_bnkps_shape640x640.trt")
    default_arcface   = os.path.join(ws, "models", "w600k_r50_b16_gpu0_fp16.engine")
    default_genderage = os.path.join(ws, "models", "genderage.engine")
    default_face_db   = os.path.join(ws, "data", "face_db.sqlite3")

    return LaunchDescription([
        # ── Camera / body detection ───────────────────────────────────────────
        DeclareLaunchArgument("model_path",       default_value=default_yolo),
        DeclareLaunchArgument("image_topic",      default_value="/camera/color/image_raw"),
        DeclareLaunchArgument("depth_topic",      default_value="/camera/depth/image_raw"),
        DeclareLaunchArgument("detections_topic", default_value="/remote_infer/detections"),
        DeclareLaunchArgument("also_subscribe_orbbec_composable_color", default_value="false"),
        DeclareLaunchArgument("color_sub_qos", default_value="reliable",
                              description="彩色图订阅 QoS：reliable 或 sensor_data"),

        # ── Fusion visualisation ─────────────────────────────────────────────
        DeclareLaunchArgument("engine_path",      default_value=default_scrfd,
                              description="SCRFD .trt 引擎路径"),
        DeclareLaunchArgument("publish_fused",    default_value="true"),
        DeclareLaunchArgument("show_window",      default_value="true"),
        DeclareLaunchArgument("viz_show_legend",  default_value="false"),

        # ── Timing logs ───────────────────────────────────────────────────────
        DeclareLaunchArgument("body_log_frame_timing",            default_value="false"),
        DeclareLaunchArgument("body_log_frame_timing_period_ms",  default_value="1000"),
        DeclareLaunchArgument("fusion_log_frame_timing",          default_value="false"),
        DeclareLaunchArgument("fusion_log_frame_timing_period_ms",default_value="1000"),

        # ── SCRFD ROI ─────────────────────────────────────────────────────────
        DeclareLaunchArgument("person_head_height_ratio",    default_value="0.58"),
        DeclareLaunchArgument("person_head_width_pad_ratio", default_value="0.24"),
        DeclareLaunchArgument("person_head_top_expand_ratio",default_value="0.14"),
        DeclareLaunchArgument("face_roi_min_side",           default_value="48"),
        DeclareLaunchArgument("face_max_person_rois",        default_value="8"),
        DeclareLaunchArgument("align_detection_dims",        default_value="true"),
        DeclareLaunchArgument("face_persons_fallback_ms",    default_value="350"),
        DeclareLaunchArgument("roi_scrfd_prob_threshold",    default_value="0.38"),
        DeclareLaunchArgument("roi_scrfd_nms_threshold",     default_value="0.45"),
        DeclareLaunchArgument("nms_threshold",               default_value="0.4"),

        # ── 6DRepNet head pose ────────────────────────────────────────────────
        DeclareLaunchArgument("head_pose_enable",      default_value="false"),
        DeclareLaunchArgument("head_pose_engine_path", default_value=""),
        DeclareLaunchArgument("head_pose_min_face_px", default_value="36"),
        DeclareLaunchArgument("head_pose_expand_ratio",default_value="0.12"),
        DeclareLaunchArgument("head_pose_skip_yaw_deg",default_value="85.0"),

        # ── ArcFace recognition ───────────────────────────────────────────────
        DeclareLaunchArgument("face_recog_enable",      default_value="false",
                              description="启用 ArcFace 人脸识别 + 自动注册"),
        DeclareLaunchArgument("face_recog_engine_path", default_value=default_arcface,
                              description="ArcFace .trt 引擎绝对路径"),
        DeclareLaunchArgument("face_db_path",           default_value=default_face_db,
                              description="人脸库 SQLite 路径（不存在时自动创建）"),
        DeclareLaunchArgument("face_recog_threshold",   default_value="0.45",
                              description="识别余弦相似度阈值"),

        # ── Registration quality gate ─────────────────────────────────────────
        DeclareLaunchArgument("recog_min_face_px",      default_value="64",
                              description="注册/识别最小人脸尺寸（像素）"),
        DeclareLaunchArgument("recog_min_face_conf",    default_value="0.75",
                              description="注册/识别 SCRFD 最小置信度"),
        DeclareLaunchArgument("recog_max_yaw_deg",      default_value="30.0"),
        DeclareLaunchArgument("recog_max_pitch_deg",    default_value="25.0"),
        DeclareLaunchArgument("recog_min_track_frames", default_value="20",
                              description="注册所需 track 最小稳定帧数"),
        DeclareLaunchArgument("recog_interval_frames",  default_value="150",
                              description="已识别 track 重验间隔（帧）"),
        DeclareLaunchArgument("recog_emb_buffer_size",  default_value="5",
                              description="注册/首次识别所需积累 embedding 数"),

        # ── IoU tracker ───────────────────────────────────────────────────────
        DeclareLaunchArgument("tracker_iou_threshold",  default_value="0.30",
                              description="body bbox IoU 匹配阈值"),
        DeclareLaunchArgument("tracker_max_age_frames", default_value="30",
                              description="track 最大无匹配存活帧数"),

        # ── Gender detection ──────────────────────────────────────────────────
        DeclareLaunchArgument("gender_enable",      default_value="false",
                              description="启用 GenderAge 性别检测"),
        DeclareLaunchArgument("gender_engine_path", default_value=default_genderage,
                              description="genderage .engine 路径"),
        DeclareLaunchArgument("gender_swap_labels", default_value="false",
                              description="当模型男女标签相反时进行交换"),
        DeclareLaunchArgument("gender_min_conf",    default_value="0.70",
                              description="性别最小置信度，低于该值输出 UNKNOWN"),
        DeclareLaunchArgument("gender_recheck_interval_frames", default_value="90",
                              description="性别已确定后的重判间隔（帧）"),

        # ── Engagement ────────────────────────────────────────────────────────
        DeclareLaunchArgument("publish_engagement",       default_value="true"),
        DeclareLaunchArgument("engagement_topic",         default_value="/human_face_fusion/scene_perception"),
        DeclareLaunchArgument("yaw_engaged_deg",          default_value="30.0"),
        DeclareLaunchArgument("pitch_engaged_deg",        default_value="25.0"),
        DeclareLaunchArgument("yaw_attention_deg",        default_value="55.0"),
        DeclareLaunchArgument("pitch_attention_deg",      default_value="40.0"),
        DeclareLaunchArgument("engaged_max_distance_m",   default_value="2.5"),
        DeclareLaunchArgument("attention_max_distance_m", default_value="4.0"),

        # ── Include trt_infer ─────────────────────────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(trt_launch),
            launch_arguments={
                "model_path":               LaunchConfiguration("model_path"),
                "image_topic":              LaunchConfiguration("image_topic"),
                "depth_topic":              LaunchConfiguration("depth_topic"),
                "detections_topic":         LaunchConfiguration("detections_topic"),
                "log_frame_timing":         LaunchConfiguration("body_log_frame_timing"),
                "log_frame_timing_period_ms": LaunchConfiguration("body_log_frame_timing_period_ms"),
            }.items(),
        ),

        # ── human_face_fusion_node ────────────────────────────────────────────
        Node(
            package="human_face_fusion_ros",
            executable="human_face_fusion_node",
            name="human_face_fusion_node",
            output="screen",
            parameters=[{
                "engine_path":                   LaunchConfiguration("engine_path"),
                "image_topic":                   LaunchConfiguration("image_topic"),
                "detections_topic":              LaunchConfiguration("detections_topic"),
                "also_subscribe_orbbec_composable_color": LaunchConfiguration("also_subscribe_orbbec_composable_color"),
                "color_sub_qos":                  LaunchConfiguration("color_sub_qos"),
                "publish_fused":                 LaunchConfiguration("publish_fused"),
                "show_window":                   LaunchConfiguration("show_window"),
                "viz_show_legend":               LaunchConfiguration("viz_show_legend"),
                "log_frame_timing":              LaunchConfiguration("fusion_log_frame_timing"),
                "log_frame_timing_period_ms":    LaunchConfiguration("fusion_log_frame_timing_period_ms"),
                "person_head_height_ratio":      LaunchConfiguration("person_head_height_ratio"),
                "person_head_width_pad_ratio":   LaunchConfiguration("person_head_width_pad_ratio"),
                "person_head_top_expand_ratio":  LaunchConfiguration("person_head_top_expand_ratio"),
                "face_roi_min_side":             LaunchConfiguration("face_roi_min_side"),
                "face_max_person_rois":          LaunchConfiguration("face_max_person_rois"),
                "align_detection_dims":          LaunchConfiguration("align_detection_dims"),
                "face_persons_fallback_ms":      LaunchConfiguration("face_persons_fallback_ms"),
                "roi_scrfd_prob_threshold":      LaunchConfiguration("roi_scrfd_prob_threshold"),
                "roi_scrfd_nms_threshold":       LaunchConfiguration("roi_scrfd_nms_threshold"),
                "nms_threshold":                 LaunchConfiguration("nms_threshold"),
                "head_pose_enable":              LaunchConfiguration("head_pose_enable"),
                "head_pose_engine_path":         LaunchConfiguration("head_pose_engine_path"),
                "head_pose_min_face_px":         LaunchConfiguration("head_pose_min_face_px"),
                "head_pose_expand_ratio":        LaunchConfiguration("head_pose_expand_ratio"),
                "head_pose_skip_yaw_deg":        LaunchConfiguration("head_pose_skip_yaw_deg"),
                "face_recog_enable":             LaunchConfiguration("face_recog_enable"),
                "face_recog_engine_path":        LaunchConfiguration("face_recog_engine_path"),
                "face_db_path":                  LaunchConfiguration("face_db_path"),
                "face_recog_threshold":          LaunchConfiguration("face_recog_threshold"),
                "recog_min_face_px":             LaunchConfiguration("recog_min_face_px"),
                "recog_min_face_conf":           LaunchConfiguration("recog_min_face_conf"),
                "recog_max_yaw_deg":             LaunchConfiguration("recog_max_yaw_deg"),
                "recog_max_pitch_deg":           LaunchConfiguration("recog_max_pitch_deg"),
                "recog_min_track_frames":        LaunchConfiguration("recog_min_track_frames"),
                "recog_interval_frames":         LaunchConfiguration("recog_interval_frames"),
                "recog_emb_buffer_size":         LaunchConfiguration("recog_emb_buffer_size"),
                "tracker_iou_threshold":         LaunchConfiguration("tracker_iou_threshold"),
                "tracker_max_age_frames":        LaunchConfiguration("tracker_max_age_frames"),
                "gender_enable":                 LaunchConfiguration("gender_enable"),
                "gender_engine_path":            LaunchConfiguration("gender_engine_path"),
                "gender_swap_labels":            LaunchConfiguration("gender_swap_labels"),
                "gender_min_conf":               LaunchConfiguration("gender_min_conf"),
                "gender_recheck_interval_frames": LaunchConfiguration("gender_recheck_interval_frames"),
                "publish_engagement":            LaunchConfiguration("publish_engagement"),
                "engagement_topic":              LaunchConfiguration("engagement_topic"),
                "yaw_engaged_deg":               LaunchConfiguration("yaw_engaged_deg"),
                "pitch_engaged_deg":             LaunchConfiguration("pitch_engaged_deg"),
                "yaw_attention_deg":             LaunchConfiguration("yaw_attention_deg"),
                "pitch_attention_deg":           LaunchConfiguration("pitch_attention_deg"),
                "engaged_max_distance_m":        LaunchConfiguration("engaged_max_distance_m"),
                "attention_max_distance_m":      LaunchConfiguration("attention_max_distance_m"),
            }],
        ),
    ])
