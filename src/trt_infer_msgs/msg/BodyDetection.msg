# ── 人体检测（YOLO + 深度）
float32 body_distance     # 深度估计距离（米）；-1 表示无效或超出范围
float32 body_confidence   # YOLO 置信度 [0, 1]

BoundingBox body_bbox     # 人体边界框