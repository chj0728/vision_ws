std_msgs/Header header

uint32  image_width             # 本帧彩图宽度（像素）
uint32  image_height            # 本帧彩图高度（像素）

float32 body_pipeline_ms          # YOLO (人体检测 + 距离) pipeline 耗时（毫秒）

PersonMeta[] persons