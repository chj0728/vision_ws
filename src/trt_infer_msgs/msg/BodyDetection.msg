# ── 人体检测（YOLO + 深度）
BoundingBox body_bbox

float32 body_confidence   # YOLO 置信度 [0, 1]
float32 body_distance     # 深度估计距离（米）；-1 表示无效或超出范围