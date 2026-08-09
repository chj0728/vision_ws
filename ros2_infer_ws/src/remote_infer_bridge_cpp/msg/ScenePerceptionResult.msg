# 每帧全场景感知结果 —— human_face_fusion_node 对外唯一输出话题。
#
# 数据流说明：
#   trt_infer_ros_node  → 发布（仅填帧元信息 + persons[i].body_* + distance）→ /remote_infer/detections
#   human_face_fusion_node → 订阅 → 补全 persons[i] 的人脸/头姿/参与度字段 → 发布完整消息
#
# 参与度状态常量（有序：值越大 = 参与程度越高）：
uint8 NOT_ENGAGED = 0   # 未参与：距离超出注意阈值，或朝向偏离注意锥
uint8 ATTENTION   = 1   # 注意中：在注意锥内且距离 ≤ attention_max_distance_m，但未满足 ENGAGED 条件
uint8 ENGAGED     = 2   # 交流中：朝向在交流锥内（|yaw| ≤ yaw_engaged_deg 且 |pitch| ≤ pitch_engaged_deg）且距离 ≤ engaged_max_distance_m

# ── 帧元信息 ──────────────────────────────────────────────────────────────────
std_msgs/Header header          # ROS 时间戳 + frame_id，与彩色图像帧对齐
uint32  image_width             # 本帧彩图宽度（像素）
uint32  image_height            # 本帧彩图高度（像素）
float32 body_pipeline_ms        # trt_infer 侧人体+深度流水线耗时（ms），用于全流程日志

# ── 逐人感知数据 ──────────────────────────────────────────────────────────────
# 每个元素包含该人的人体框、距离、人脸（可选）、头姿（可选）和参与度
PersonPerception[] persons

# ── 帧级汇总（下游状态机最常用字段，无需遍历 persons 数组）──────────────────
uint8   best_engagement             # 本帧所有人中最高的参与度状态
uint32  engaged_count               # 处于 ENGAGED 状态的人数
uint32  attention_count             # 处于 ATTENTION 状态的人数
float32 closest_engaged_distance    # 最近 ENGAGED 人的距离（米）；-1 表示无 ENGAGED 人
float32 closest_attention_distance  # 最近 ATTENTION 人的距离（米）；-1 表示无 ATTENTION 人
