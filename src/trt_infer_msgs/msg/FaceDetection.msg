# |--> 人脸检测（SCRFD）
#   |--> FaceRecog 人脸识别（ArcFace）
#   |--> HeadPose 头部姿态（6DRepNet）
#   |--> GenderAge 性别年龄（GenderAge）
bool has_face             # 是否检测到人脸
float32 face_confidence   # SCRFD 置信度 [0, 1]

BoundingBox face_bbox     # 人脸边界框