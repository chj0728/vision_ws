std_msgs/Header header

uint32  image_width             # 本帧彩图宽度（像素）
uint32  image_height            # 本帧彩图高度（像素）

float32 body_detection_ms          # YOLO (人体检测 + 距离) pipeline 耗时（毫秒）
float32 face_detection_ms          # SCRFD (人脸检测) pipeline 耗时（毫秒）
float32 head_pose_ms               # 6DRepNet (头部姿态) pipeline耗时（毫秒）
float32 face_recog_ms              # ArcFace (人脸识别) pipeline 耗时（毫秒）

PersonMeta[] persons