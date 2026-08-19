# |--> 人脸检测（SCRFD）
#   |--> FaceRecog 人脸识别（ArcFace）
#   |--> HeadPose 头部姿态（6DRepNet）
#   |--> GenderAge 性别年龄（GenderAge）
BoundingBox face_bbox

float32 face_confidence   # SCRFD 置信度 [0, 1]