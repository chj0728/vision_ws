int32  track_id      # IoU 追踪器会话 ID；-1 表示未关联追踪

BodyDetection body_detection # 人体检测（YOLO + 深度）

FaceDetection face_detection # 人脸检测（SCRFD）

FaceRecog face_recog         # 人脸识别（ArcFace）

HeadPose head_pose           # 头部姿态（6DRepNet）