# 单人感知结果：人体框（必填）+ 追踪ID + 人脸/头姿/参与度/性别/身份（fusion 阶段补全）。
#
# 生命周期：
#   trt_infer_ros_node  发布时：仅填 body_* / distance，has_face=false，track_id=-1
#   human_face_fusion_node 补全后：track_id 由 IoU 追踪器赋值；其余字段在 has_face=true 时有效

# ── 性别常量 ──────────────────────────────────────────────────────────────────
uint8 GENDER_UNKNOWN = 0
uint8 GENDER_MALE    = 1
uint8 GENDER_FEMALE  = 2

# ── 追踪（fusion 阶段赋值）────────────────────────────────────────────────────
int32  track_id      # IoU 追踪器会话 ID；-1 表示未关联追踪

# ── 人体检测（YOLO + 深度，必填）────────────────────────────────────────────
int32   body_x      # 人体 bbox 左上角 x（像素）
int32   body_y      # 人体 bbox 左上角 y（像素）
int32   body_w      # 人体 bbox 宽度（像素）
int32   body_h      # 人体 bbox 高度（像素）
float32 body_conf   # YOLO 置信度 [0, 1]
float32 distance    # 深度估计距离（米）；-1 表示无效或超出范围

# ── 人脸检测（SCRFD；has_face=true 时有效）───────────────────────────────────
bool    has_face    # 是否在该人体的头部 ROI 内检测到人脸
int32   face_x      # 人脸 bbox 左上角 x（像素）
int32   face_y      # 人脸 bbox 左上角 y（像素）
int32   face_w      # 人脸 bbox 宽度（像素）
int32   face_h      # 人脸 bbox 高度（像素）
float32 face_conf   # SCRFD 置信度 [0, 1]

# ── 头部姿态（6DRepNet；has_face=true 且 head_pose_enable=true 时有效）────────
float32 yaw         # 偏航角（度）；正值 = 向右转，负值 = 向左转
float32 pitch       # 俯仰角（度）；正值 = 低头，负值 = 抬头
float32 roll        # 翻滚角（度）；正值 = 向右侧倾

# ── 参与度（has_face=true 时有效；常量定义在 ScenePerceptionResult）────────────
uint8   engagement  # NOT_ENGAGED(0) / ATTENTION(1) / ENGAGED(2)

# ── 人脸识别（ArcFace；has_face=true 且 face_recog_enable=true 时有效）────────
string       person_uuid         # 人物 UUID v4；""表示未识别或库中无此人
string       person_name         # 数据库中存储的姓名；""表示尚未命名
float32      face_recog_conf     # 余弦相似度 [0,1]；0 表示未识别或未开启
float32[512] face_embedding      # L2 归一化特征向量；全零表示未提取

# ── 性别检测（GenderAge；has_face=true 且 gender_enable=true 时有效）─────────
uint8   gender       # GENDER_UNKNOWN(0) / GENDER_MALE(1) / GENDER_FEMALE(2)
float32 gender_conf  # 置信度 [0,1]；0 表示未检测
