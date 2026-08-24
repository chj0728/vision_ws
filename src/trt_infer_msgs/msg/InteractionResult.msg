# 参与度状态常量（有序：值越大 = 参与程度越高）：
# status = INVALID_STATUS < ATTENTION_STATUS < TALKING_STATUS
uint8 INVALID_STATUS     = 0   # 无参与：不在注意锥内或距离 > attention_max_distance_m
uint8 ATTENTION_STATUS   = 1   # 注意中：在注意锥内且距离 ≤ attention_max_distance_m，但未满足 TALKING 条件
uint8 TALKING_STATUS     = 2   # 交流中：朝向在交流锥内（|yaw| ≤ talking_yaw_deg 且 |pitch| ≤ talking_pitch_deg）且距离 ≤ talking_max_distance_m

std_msgs/Header header

uint8 best_status   # 本帧所有人中最高的参与度状态
uint8 invalid_status_num   # 本帧所有人中参与度状态为 INVALID 的人数
uint8 attention_status_num # 本帧所有人中参与度状态为 ATTENTION 的人数
uint8 talking_status_num   # 本帧所有人中参与度状态为 TALKING 的人数

float32 closest_attention_distance  # 本帧所有人中参与度状态为 ATTENTION 的人中，距离最近的人的距离（单位：米）；-1 表示无 ATTENTION 人
float32 closest_talking_distance    # 本帧所有人中参与度状态为 TALKING 的人中，距离最近的人的距离（单位：米）；-1 表示无 TALKING 人
